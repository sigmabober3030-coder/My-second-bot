import os, sys, time
import instabox

NAME = os.environ["INSTABOX_MACHINE"]
c = instabox.Client(api_key=os.environ["INSTABOX_API_KEY"])

def alive():
    try:
        return c.machines.exec(NAME, "true", timeout=60).exit_code == 0
    except Exception as e:
        print("exec не прошёл:", repr(e))
        return False

if alive():
    print("ok")
    sys.exit(0)

for action in ("wake", "start", "resume"):
    try:
        getattr(c.machines, action)(NAME)
        print("выполнено:", action)
    except Exception as e:
        print(action, "->", repr(e))
        continue
    if alive():
        sys.exit(0)
    time.sleep(5)

print("не удалось поднять машину")
sys.exit(1)
