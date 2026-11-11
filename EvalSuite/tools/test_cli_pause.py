"""CPU test of the shared-lock path in evalsuite.cli.gpu(): a PAUSE before asking or right at the grant (release and wait).
A fake `peer` and a temp PAUSE file (EVALSUITE_PAUSE_FILE), never the real one. Run: .venv/bin/python -B tools/test_cli_pause.py"""
import os, sys, tempfile, types
d = tempfile.mkdtemp()
os.environ["EVALSUITE_PAUSE_FILE"] = os.path.join(d, "PAUSE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalsuite import cli
assert cli.PAUSE == os.environ["EVALSUITE_PAUSE_FILE"], cli.PAUSE
cli.LOG = os.path.join(d, "gpu.log")
cli.SHARE, cli.DEADLINE = True, 0
cli.held_until = lambda: 0
class Done(Exception): pass
def stop(): raise Done()
cli.free_gb = stop

def run_case(name, pause_at_start, pause_on_grant):
    calls, sleeps = [], []
    st = {"grants": 0}
    if pause_at_start: open(cli.PAUSE, "w").close()
    def fake_run(argv, **kw):
        calls.append(" ".join(argv[:3]))
        if argv[:3] == ["peer", "lock", "share"]:
            assert not os.path.exists(cli.PAUSE) or st["grants"] == 0 and pause_on_grant, "asked during a PAUSE"
            st["grants"] += 1
            if pause_on_grant and st["grants"] == 1: open(cli.PAUSE, "w").close()   # the PAUSE begins as we are granted
            return types.SimpleNamespace(returncode=0, stdout="granted", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 3 and os.path.exists(cli.PAUSE): os.remove(cli.PAUSE)   # the PAUSE ends after 3 waits
    cli.subprocess.run, cli.time.sleep = fake_run, fake_sleep
    try:
        cli.gpu(["x"], "test " + name, 1, 600)
    except Done:
        pass
    print(name, "| calls:", calls, "| sleeps:", len(sleeps), "| PAUSE left:", os.path.exists(cli.PAUSE))
    return calls, sleeps

c, s = run_case("no PAUSE", False, False)
assert c == ["peer lock share", "peer unlock"] and not s, c
c, s = run_case("PAUSE before asking", True, False)
assert c == ["peer lock share", "peer unlock"] and len(s) == 3, (c, s)
c, s = run_case("PAUSE at the grant", False, True)
assert c == ["peer lock share", "peer unlock", "peer lock share", "peer unlock"] and len(s) == 3, (c, s)
print("ALL OK")
