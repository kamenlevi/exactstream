"""Stub job for head-testing scripts/l1cf_monday.sh (L1CF_DRY=1): prints lever1cf.py's progress lines, saves a tiny
state on SIGTERM (exit 75) and resumes from it, and ends with a DONE line. STUB_STEPS, STUB_DT (seconds per step)."""
import json, os, signal, sys, time

N = int(os.environ.get("STUB_STEPS", "120"))
DT = float(os.environ.get("STUB_DT", "0.1"))
ST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", ".l1cf_stub_state.json")
stop = {"f": False}
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
step = json.load(open(ST))["step"] if os.path.exists(ST) else 0
print(f"[1cf] stub pid {os.getpid()} pgid {os.getpgid(0)}: {'resumed at step %d' % step if step else 'fresh'}", flush=True)
t0, s0 = time.time(), step
while step < N:
    time.sleep(DT)
    step += 1
    if step % 10 == 0:
        print(f"[1cf] step {step}/{N} loss 0.4000 trunk 0.050s fit 0.050s {(time.time() - t0) / (step - s0):.3f}s/step",
              flush=True)
    if stop["f"]:
        time.sleep(1.0)                                    # a state save takes a moment
        json.dump({"step": step}, open(ST, "w"))
        print(f"[1cf] SIGTERM: state saved at step {step}, exiting to resume later", flush=True)
        sys.exit(75)
if os.path.exists(ST):
    os.remove(ST)
print(f"[1cf] DONE verdict STUB: {N} steps", flush=True)
