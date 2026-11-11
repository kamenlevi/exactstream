#!/usr/bin/env python3
"""a3_lag2_either.py (drafterwall, DESCRIPTIVE, after T2; no line). The lag-2 ceiling with BOTH readers on the unseen sets:
DFlash2 L1-20 right at depth 0 at every anchor (data/d02b_walk_d23h_dfl_<set>.npz: a check at every position, k >= 1)
and the MTP head b5 right at depth 0 (data/d23h/<set>_b5.npz, rt[:, 0]); a drafter that is right at a position exactly
when the named reader(s) are right there at depth 0; kept per check at 16 / 32 / 64 rows."""
import numpy as np, os
TD = os.path.expanduser("~/TrainLab/data"); out = []
def kept(ok, cap):
    r = []; p = 0; n = len(ok)
    while p < n:
        k = 0
        while k < cap and p + k < n and ok[p + k]: k += 1
        r.append(k + 1); p += k + 1
    return r
for st in ("code", "mbpp"):
    Wz = np.load(f"{TD}/d02b_walk_d23h_dfl_{st}.npz"); Mz = np.load(f"{TD}/d23h/{st}_b5.npz")
    mk = {str(k): i for i, k in enumerate(Mz["keys"])}
    K = {(r, c): [] for r in ("dfl", "mtp", "either") for c in (15, 31, 63)}; acc = {"dfl": [0, 0], "mtp": [0, 0], "either": [0, 0]}; nw = 0
    for w, nm in enumerate(str(x) for x in Wz["names"]):
        mi = mk.get(nm[:-4]) if nm.endswith(".npz") else mk.get(nm)
        if mi is None: mi = mk.get(nm)
        if mi is None: continue
        ch = Wz[f"{w}_checks"]; gen = int(Wz[f"{w}_gen"]); ids = Wz[f"{w}_ids"]
        d0 = {int(A): (int(k) >= 1) for k, m, A in ch if int(m) >= 1}
        rt = Mz[f"rt_{mi}"]; a0 = int(Mz["a0"][mi])
        dl, ml = [], []
        A = gen
        while A < len(ids) - 1:
            r = A - 1 - a0
            if A not in d0 or r < 0 or r >= rt.shape[0] or rt[r, 0] < 0: break
            dl.append(d0[A]); ml.append(bool(rt[r, 0] == 1)); A += 1
        if not dl: continue
        nw += 1; dl = np.array(dl); ml = np.array(ml)
        for r, ok in (("dfl", dl), ("mtp", ml), ("either", dl | ml)):
            acc[r][0] += int(ok.sum()); acc[r][1] += len(ok)
            for c in (15, 31, 63): K[(r, c)] += kept(ok, c)
    s = f"{st} ({nw} windows): right at depth 0: " + ", ".join(f"{r} {100 * a / b:.1f} %" for r, (a, b) in acc.items())
    s += "\n    lag-2 everywhere, kept per check at 16 / 32 / 64 rows: " + "; ".join(f"{r} {np.mean(K[(r, 15)]):.2f} / {np.mean(K[(r, 31)]):.2f} / {np.mean(K[(r, 63)]):.2f}" for r in ("dfl", "mtp", "either"))
    print(s); out.append(s)
open(os.path.join(os.path.dirname(__file__), "..", "out", "a3_lag2_either.txt"), "w").write("\n".join(out) + "\n")
