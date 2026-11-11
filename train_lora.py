#!/usr/bin/env python3
"""LoRA-дообучение маленькой Qwen на CPU (команды Minecraft Bedrock).

Датасет (jsonl), автоопределение формата:
  {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "metadata": {"category": "..."}}
  {"instruction": "...", "input": "...", "output": "..."} | {"prompt": "...", "completion": "..."} | {"text": "..."}

Запуск:   python train_lora.py --data data.jsonl --out out
Проверка: python train_lora.py --data data.jsonl --out out_test --limit 200   (несколько минут, смотри ток/с)
Продолжить после остановки: добавить --resume
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import resource
import shutil

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
ap.add_argument("--out", default="out")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--accum", type=int, default=8, help="накопление градиента (эффективный батч)")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--save_every", type=int, default=25, help="сохранять адаптер каждые N шагов оптимизатора")
ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
ap.add_argument("--only_command", action=argparse.BooleanOptionalAction, default=True,
                help="учить только команду (первая строка ответа), без пояснения")
ap.add_argument("--system", default="Minecraft Bedrock 1.21.51. Отвечай только командами.",
                help="короткий системный промпт (пустая строка = стандартный Qwen). Тот же нужен при использовании!")
ap.add_argument("--val_frac", type=float, default=0.05, help="доля отложенных примеров на категорию")
ap.add_argument("--min_per_category", type=int, default=20,
                help="редкие категории дублируются до этого числа в обучении (0 = выключить)")
ap.add_argument("--limit", type=int, default=0, help="взять только N обучающих примеров (быстрая проверка)")
ap.add_argument("--init_adapter", default="", help="начать с уже обученного адаптера (новый график обучения)")
ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="формат весов модели (bf16 вдвое легче по памяти)")
ap.add_argument("--grad_ckpt", action="store_true", help="экономия памяти на активациях (чуть медленнее)")
ap.add_argument("--targets", choices=["attn_mlp", "all-linear"], default="attn_mlp",
                help="куда ставить LoRA; для Qwen3.5 нужен all-linear (иначе слои DeltaNet не обучатся)")
ap.add_argument("--no_think", action="store_true", help="шаблон чата без режима размышлений (Qwen3.5)")
ap.add_argument("--manual_target", action="store_true",
                help="ответ = подсказка шаблона + текст + <|im_end|> (надёжно для шаблонов с <think>)")
ap.add_argument("--val_n", type=int, default=150, help="сколько проверочных примеров считать для val loss")
ap.add_argument("--log_every", type=int, default=5)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--merge", action="store_true", help="в конце слить LoRA в полную модель (out/merged)")
args = ap.parse_args()

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
MAIN = RANK == 0
torch.set_num_threads(args.threads)
os.makedirs(args.out, exist_ok=True)
state_path = os.path.join(args.out, "state.json")
adapter_dir = os.path.join(args.out, "adapter")
json.dump({"model": args.model, "system": args.system, "only_command": args.only_command,
           "max_len": args.max_len, "dtype": args.dtype, "no_think": args.no_think},
          open(os.path.join(args.out, "meta.json"), "w"), ensure_ascii=False)
CHAT_KW = {"enable_thinking": False} if args.no_think else {}

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)
model.config.use_cache = False
if args.grad_ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
if args.resume and os.path.isdir(adapter_dir):
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    print("Адаптер загружен, продолжаю")
elif args.init_adapter and os.path.isdir(args.init_adapter):
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    print("Стартую с адаптера:", args.init_adapter)
else:
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=("all-linear" if args.targets == "all-linear" else
                                     ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = get_peft_model(model, cfg)
model.print_trainable_parameters()


def to_messages(row):
    if "messages" in row:
        return row["messages"]
    if "instruction" in row:
        user = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": row.get("output") or row.get("response") or ""}]
    if "prompt" in row:
        return [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row.get("completion") or row.get("response") or ""}]
    return None


def prep(row):
    cat = (row.get("metadata") or {}).get("category", "?")
    msgs = to_messages(row)
    if msgs is None:
        return {"text": row["text"], "category": cat, "split": row.get("split")}
    msgs = [dict(m) for m in msgs]
    if args.only_command:
        msgs[-1]["content"] = msgs[-1]["content"].strip().split("\n")[0].strip()
    if args.system and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": args.system}] + msgs
    return {"messages": msgs, "category": cat, "split": row.get("split")}


def encode(row):
    if "text" in row:
        ids = tok(row["text"] + tok.eos_token, add_special_tokens=False,
                  truncation=True, max_length=args.max_len)["input_ids"]
        return ids, ids[:]
    msgs = row["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **CHAT_KW)
    if args.manual_target:
        full = prompt + msgs[-1]["content"] + "<|im_end|>\n"
    else:
        full = tok.apply_chat_template(msgs, tokenize=False, **CHAT_KW)
    ids = tok(full, add_special_tokens=False, truncation=True, max_length=args.max_len)["input_ids"]
    p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
    return ids, [-100] * p_len + ids[p_len:]  # учим только ответ ассистента


# ---- данные: стратифицированный отбор валидации, дублирование редких категорий ----
rows = [prep(json.loads(l)) for l in open(args.data, encoding="utf-8") if l.strip()]
rng = random.Random(1)
if any(r.get("split") for r in rows):  # готовая разметка из prepare_data.py
    train = [r for r in rows if r.get("split") != "val"]
    val = [r for r in rows if r.get("split") == "val"]
else:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    train, val = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        k = math.ceil(len(items) * args.val_frac) if args.val_frac > 0 and len(items) >= 20 else 0
        val += items[:k]
        own = items[k:]
        train += own
        if args.min_per_category and own and len(own) < args.min_per_category:
            train += [rng.choice(own) for _ in range(args.min_per_category - len(own))]
rng.shuffle(train)
if args.limit:
    train = train[: args.limit]
with open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8") as f:
    for r in val:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def encode_all(rs):
    out = []
    for r in rs:
        ids, labels = encode(r)
        if any(l != -100 for l in labels):
            out.append((ids, labels))
    return out


examples, val_ex = encode_all(train), encode_all(val)
print(f"обучение: {len(examples)} примеров ({sum(len(e[0]) for e in examples)} токенов за проход), "
      f"валидация: {len(val_ex)}")
if not examples:
    raise SystemExit("Пустой датасет: проверь формат")


@torch.no_grad()
def val_loss(n=150):
    if not val_ex:
        return None
    model.eval()
    tot = 0.0
    sub = val_ex[:n]
    for ids, labels in sub:
        x = torch.tensor([ids])
        tot += model(input_ids=x, attention_mask=torch.ones_like(x), labels=torch.tensor([labels])).loss.item()
    model.train()
    return tot / len(sub)


order = []
for e in range(math.ceil(args.epochs)):
    idx = list(range(len(examples)))
    random.Random(e).shuffle(idx)
    order += idx
order = order[: int(len(examples) * args.epochs)]
total_steps = math.ceil(len(order) / args.accum)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = get_cosine_schedule_with_warmup(opt, min(10, total_steps // 10), total_steps)

if WORLD > 1:
    import datetime
    import torch.distributed as dist
    dist.init_process_group("gloo", timeout=datetime.timedelta(hours=3))  # MASTER_ADDR/PORT, RANK, WORLD_SIZE из окружения
    for p in params:                      # у всех машин одинаковый старт
        dist.broadcast(p.data, 0)

step = 0
if args.resume and os.path.exists(state_path):
    step = json.load(open(state_path))["step"]
if WORLD > 1:
    t = torch.tensor([step])
    dist.broadcast(t, 0)
    step = int(t.item())
for _ in range(step):
    sched.step()
if step:
    print(f"Продолжаю с шага {step}/{total_steps}")


def save():
    """Безопасное сохранение: сначала во временную папку, старый чекпоинт цел до конца. Ошибка не останавливает обучение."""
    try:
        if shutil.disk_usage(args.out).free < 400 * 1024 * 1024:
            print("⚠ Мало места на диске, чекпоинт пропущен. Освободи место!", flush=True)
            return
        tmp = adapter_dir + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        shutil.rmtree(adapter_dir + ".old", ignore_errors=True)
        if os.path.isdir(adapter_dir):
            os.replace(adapter_dir, adapter_dir + ".old")
        os.replace(tmp, adapter_dir)
        shutil.rmtree(adapter_dir + ".old", ignore_errors=True)
        json.dump({"step": step}, open(state_path, "w"))
    except Exception as e:
        print("⚠ Не удалось сохранить чекпоинт:", e, flush=True)


model.train()
if step == 0 and MAIN:
    v = val_loss(args.val_n)
    if v is not None:
        print(f"val loss ДО обучения: {v:.3f}", flush=True)

t0, tokens, run_loss, n_loss = time.time(), 0, 0.0, 0
start_step = step
for s_ in range(step, total_steps):
    batch = order[s_ * args.accum:(s_ + 1) * args.accum]   # глобальный шаг; машины делят его по очереди
    for idx in batch[RANK::WORLD]:
        ids, labels = examples[idx]
        x = torch.tensor([ids])
        loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=torch.tensor([labels])).loss
        (loss / len(batch)).backward()
        run_loss += loss.item()
        n_loss += 1
        tokens += len(ids)
    if WORLD > 1:                          # сумма градиентов со всех машин
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
        dist.all_reduce(flat)
        off = 0
        for p in params:
            p.grad = flat[off:off + p.numel()].view_as(p).clone()
            off += p.numel()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    if MAIN and (step % args.log_every == 0 or step == start_step + 1):
        el = time.time() - t0
        left = (total_steps - step) * el / max(step - start_step, 1) / 60
        print(f"шаг {step}/{total_steps} loss {run_loss / max(n_loss, 1):.3f} {tokens * WORLD / el:.1f} ток/с, "
              f"осталось ~{left:.0f} мин, память до {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} МБ",
              flush=True)
        run_loss, n_loss = 0.0, 0
    if MAIN and step % args.save_every == 0:
        save()

if MAIN:
    save()
    v = val_loss(args.val_n)
    if v is not None:
        print(f"val loss ПОСЛЕ обучения: {v:.3f}")
    print("Готово. Адаптер:", adapter_dir)
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()

if args.merge and MAIN:
    merged = model.merge_and_unload()
    merged.save_pretrained(os.path.join(args.out, "merged"))
    tok.save_pretrained(os.path.join(args.out, "merged"))
    print("Слитая модель:", os.path.join(args.out, "merged"))
