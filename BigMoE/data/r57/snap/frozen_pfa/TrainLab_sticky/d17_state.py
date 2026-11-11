"""D-17 follow-up (MTP Head, Fri 10-30): can a cheap STATEFUL router get part of the code / math oracle? Same data and
walk as sticky/d17_route.py (tokens per pass, both drafters at every position). Routers, decided before drafting:
- after-miss: MTP K15 when the previous pass accepted <= t drafts (whichever drafter made it), else DFlash2;
- sticky: keep the drafter of the previous pass while it accepted >= s, else switch.
Also where MTP wins on the oracle path: the previous pass's accepted count, binned.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d17_state.py -> data/d17_state.txt"""
import collections, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d18_hybrid import texts, TD

def kM(out, mtp, p, K=15):
    n = 0
    while n < K and p + n < len(out) and mtp[p][n] >= 0 and mtp[p][n] == out[p + n]:
        n += 1
    return n

def walk(T, rule):
    tok = passes = used = 0
    for out, mtp, kD in T:
        p, L, prev, who = 1, len(out), 15, 0
        while p < L:
            who = rule(prev, who)
            k = min(kM(out, mtp, p) if who else kD.get(p, 0), L - p)
            tok += k + 1; passes += 1; used += who; prev = k; p += k + 1
    return tok / passes, used / passes

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
say("D-17 stateful routers (tokens per pass; MTP share of passes)")
for dom in ("code", "gsm", "ifeval", "multi", "agent"):
    T = texts(dom)
    d0, _ = walk(T, lambda prev, who: 0)
    row = [f"DFlash2 {d0:.2f}"]
    for t in (0, 1, 3):
        r, u = walk(T, lambda prev, who, t=t: int(prev <= t))
        row.append(f"after-miss t{t} {r:.2f} ({r - d0:+.2f}, MTP {u:.0%})")
    for s in (2, 4):
        r, u = walk(T, lambda prev, who, s=s: who if prev >= s else 1 - who)
        row.append(f"sticky s{s} {r:.2f} ({r - d0:+.2f}, MTP {u:.0%})")
    say(f"  {dom}: " + "; ".join(row))
    win = collections.Counter(); tot = collections.Counter()   # on the oracle's own path
    for out, mtp, kD in T:
        p, L, prev = 1, len(out), 15
        while p < L:
            a, b = kD.get(p, 0), kM(out, mtp, p)
            g = "0" if prev == 0 else "1-3" if prev <= 3 else "4-9" if prev <= 9 else "10+"
            tot[g] += 1; win[g] += int(b > a)
            k = min(max(a, b), L - p); prev = k; p += k + 1
    say("    oracle path, MTP longer by the previous pass's accepted count: "
        + ", ".join(f"{g}: {win[g] / max(tot[g], 1):.2f} of {tot[g]}" for g in ("0", "1-3", "4-9", "10+")))
open(os.path.join(TD, "d17_state.txt"), "w").write("\n".join(lines) + "\n")
