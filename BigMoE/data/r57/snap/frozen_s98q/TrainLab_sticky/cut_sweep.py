"""CUTSWEEP (MTP Head, Tue 11-03; information, CPU): the shipped DFlash2 running-confidence cut (verify the depths whose running
product of the selector's cf >= t; L1-14 / S45c, t 0.3) swept per type in the exact official loop. Tables: DFlash2 L1-20 at
EVERY position (data/d02b_walk_d23h_dfl_<set>.npz: own picks and own cf, so the cut rows are exact, past the first miss too).
Cost: D-24's model on the 45 GB default (F(type) + 7.82 ms x rows + 12 ms block pass). TRIED: S45d priced t 0.5 / 0.8 on
D-23's measured checks (-1.1 % / -7.7 % whole set); lower t and a per-type t were not priced. Output: data/cut_sweep.txt."""
import os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
SETS = [("code", "code"), ("math", "math"), ("multi", "chat"), ("ifeval", "chat"), ("mbpp", "code")]
F = {"chat": 50.1, "math": 66.1, "code": 84.1}
ROW, DFL = 7.82, 12.0
TS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say("CUTSWEEP: DFlash2's own running-confidence cut per type, exact official loop on every-position tables (est. tok/s, D-24's model)")
for st, typ in SETS:
    f = os.path.join(TD, f"d02b_walk_d23h_dfl_{st}.npz")
    if not os.path.exists(f):
        say(f"  {st}: table missing"); continue
    Z = np.load(f)
    res = {}
    for t in TS:
        kept = ms = n = rws = 0.0
        for w in range(len(Z["names"])):
            ids, off, ck = Z[f"{w}_ids"], int(Z[f"{w}_off"]), Z[f"{w}_checks"]
            at = {int(A): c for c, (k, m, A) in enumerate(ck)}
            dt, dc = Z[f"{w}_dtok"], Z[f"{w}_dcf"]
            A = int(ck[0][2])
            while A in at:
                c = at[A]; m = min(int(ck[c][1]), 15)
                if m <= 0:
                    break
                true = ids[A - off + 1:A - off + 1 + m].astype(np.int64)
                ok = dt[c, :m].astype(np.int64) == true
                k = int(np.argmin(ok)) if not ok.all() else m
                assert k == int(ck[c][0]) or m < 15, (st, w, A, k, int(ck[c][0]))   # the walk's own accepted count
                rows = 1 + int((np.cumprod(dc[c, :m].astype(np.float64)) >= t).sum())
                kk = min(k, rows - 1)
                kept += kk + 1; ms += F[typ] + ROW * rows + DFL; n += 1; rws += rows
                A += kk + 1
        res[t] = (kept / n, rws / n, 1e3 * kept / ms)
    b = res[0.3][2]
    best = max(TS, key=lambda t: res[t][2])
    say(f"  {st:6s}: " + "; ".join(f"t {t:.2f} {res[t][0]:.2f}/{res[t][1]:.1f}r {res[t][2]:.1f}" for t in TS))
    say(f"  {'':6s}  best t {best:.2f}: est. tok/s {res[best][2]:.1f} vs t 0.3 {b:.1f} (x{res[best][2] / b:.3f})")
open(os.path.join(TD, "cut_sweep.txt"), "a").write("\n".join(lines) + "\n\n")
