"""D-16 retry screen (MTP Head, Fri 10-30): paired per-step training loss of the H arms vs D-13's U. Same seeds, so
step s of every arm trains on the same batch, and its loss is taken BEFORE that step's update: an online held-out loss
on a batch the arm has not seen. delta = H - U per step; negative = the widened-stream input helps.
Usage: python3 sticky/d16_loss.py [arm ...] -> data/d16_loss.txt"""
import json, os, sys
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
U = json.load(open(f"{TD}/d13_U.state.json"))["log"]["loss"]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
say("D-16 retry: per-step training loss, H arm - U (same batches; loss before the step's update), 90 % bootstrap CI over steps")
rng = np.random.default_rng(0)
for arm in sys.argv[1:] or ("d16_H", "d16_Hb", "d16_Hc"):
    f = f"{TD}/{arm}.state.json"
    if not os.path.exists(f):
        say(f"  {arm}: no state file"); continue
    st = json.load(open(f)); L = st["log"]["loss"]; lr = st["log"]["args"].get("hc_lr")
    n = min(len(L), len(U)); d = np.array(L[:n]) - np.array(U[:n])
    out = [f"  {arm} (hc-lr {lr}, {st['status']}, {n} steps):"]
    for lo, hi in ((1, n), (n // 2, n)):
        x = d[lo:hi]; b = rng.choice(x, (4000, len(x))).mean(1)
        out.append(f"steps {lo + 1}-{hi}: mean {x.mean():+.3f} (CI {np.quantile(b, 0.05):+.3f} {np.quantile(b, 0.95):+.3f})")
    say(" ".join(out)); say("    per step: " + " ".join(f"{v:+.2f}" for v in d))
say("Read: the arm whose late-step delta is below 0 with the CI under 0 gets the walk (turn36-style); all >= 0 -> D-16 KILL at the rapid protocol.")
open(f"{TD}/d16_loss.txt", "w").write("\n".join(lines) + "\n")
