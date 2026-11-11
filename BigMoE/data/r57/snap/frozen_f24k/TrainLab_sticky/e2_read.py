"""E2 / D2 read-out (MTP Head, Mon 11-02; REPORT 'E2 / D2 DATASLOPE-MBPP'): code / math kept per check from the walks
(data/d02b_walk*.txt: D-09's 1/4 and 1/2, L1-20, and E2's x2 / x4 when present) against log2 of the code data share.
Line: the code slope per doubling fitted on L1-20, x2, x4 (or the one step L1-20 -> x2 if x4 is missing): KILL < +0.05,
PASS >= +0.10, BETWEEN otherwise. Beside it: the 5-point fit, math, and the doublings code >= 13 would need at the slope.
Output: data/e2_read.txt (appended)."""
import os, re
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
def kept(tag):
    f = os.path.join(TD, f"d02b_walk{tag}.txt")
    if not os.path.exists(f):
        return None
    s = open(f).read()
    c = re.search(r"code \(HumanEval\+\):.*?kept per check ([0-9.]+)", s)
    m = re.search(r"math \(GSM8K CoT\):.*?kept per check ([0-9.]+)", s)
    return (float(c.group(1)), float(m.group(1))) if c and m else None
pts = [("D-09 1/4", np.log2(0.257), kept("_d09q")), ("D-09 1/2", np.log2(0.506), kept("_d09h")), ("L1-20", 0.0, kept("")),
       ("E2 x2", 1.0, kept("_e2x2")), ("E2 x4", 2.0, kept("_e2x4"))]
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say("E2 / D2 read-out: kept per check (block 16, held-out: code 24 HumanEval+, math 60 GSM8K) against log2(code data share)")
for nm, x, k in pts:
    say(f"  {nm:9s} x {x:+.2f}: " + (f"code {k[0]:.2f}, math {k[1]:.2f}" if k else "not measured"))
new = [(x, k) for nm, x, k in pts[2:] if k]
if len(new) >= 2:
    xs, cs = np.array([x for x, _ in new]), np.array([k[0] for _, k in new])
    sl = float(np.polyfit(xs, cs, 1)[0]) if len(new) > 2 else float(cs[1] - cs[0])
    v = "PASS" if sl >= 0.10 else "KILL" if sl < 0.05 else "BETWEEN"
    say(f"  LINE: code slope per doubling on {'L1-20, x2, x4' if len(new) > 2 else 'L1-20 -> x2 (one step)'}: {sl:+.3f} "
        f"-> {v} (KILL < +0.05, PASS >= +0.10)")
    al = [(x, k) for nm, x, k in pts if k]
    s5 = float(np.polyfit([x for x, _ in al], [k[0] for _, k in al], 1)[0])
    m5 = float(np.polyfit([x for x, _ in al], [k[1] for _, k in al], 1)[0])
    say(f"  all {len(al)} points: code {s5:+.3f}, math {m5:+.3f} per doubling (math data is unchanged past L1-20)")
    if sl > 0:
        say(f"  code >= 13 from {cs[-1]:.2f} at {sl:+.3f} per doubling: {(13 - cs[-1]) / sl:.0f} more doublings "
            f"(x{2 ** ((13 - cs[-1]) / sl):.3g} the code data)")
else:
    say("  not enough points yet")
open(os.path.join(TD, "e2_read.txt"), "a").write("\n".join(lines) + "\n\n")
