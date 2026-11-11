"""Is a verify row nearly free in CHS5's CPU-bound chat check? Fit ck ~ rows + misses on BigMoE's per-check logs."""
import json, sys, statistics as st
import numpy as np
arms = sys.argv[1:] or ["zsgcb", "zsgc", "zs", "z0"]
for arm in arms:
    d = json.load(open(f"/Users/kamen/BigMoE/data/r57/{arm}_256_checks.json"))
    R, M, CK, BLD, WT, SVC, IO, U = [], [], [], [], [], [], [], []
    keys = set()
    for r in d:
        if r.get("paused"): continue
        keys.add(r["key"].split("/")[0])
        n = len(r["rows"])
        for i in range(1, n):            # skip each key's first check (prefill-adjacent warmup)
            R.append(r["rows"][i]); M.append(r["miss"][i]); CK.append(r["ck"][i]); BLD.append(r["bld"][i]); WT.append(r["wt"][i])
            SVC.append(r["svc"][i]); IO.append(r["io"][i]); U.append(r["uses"][i])
    R, M, CK, BLD, WT, SVC, U = map(np.array, (R, M, CK, BLD, WT, SVC, U))
    print(f"== {arm}: sets {sorted(keys)}, checks {len(R)}")
    for lo in (1, 2, 3, 4, 5, 6):
        s = (R == lo) if lo < 6 else (R >= 6)
        if s.sum() < 5: continue
        lowm = s & (M <= 10)
        print(f"  rows {lo if lo < 6 else '6+':>2}: n {s.sum():4d} ck med {np.median(CK[s]):5.1f} mean {CK[s].mean():5.1f} | bld {np.median(BLD[s]):5.1f} wt {np.median(WT[s]):4.1f} svc {np.median(SVC[s]):4.1f} | miss med {np.median(M[s]):5.0f} uses {np.median(U[s]):5.0f}"
              + (f" | miss<=10: n {lowm.sum():3d} ck med {np.median(CK[lowm]):5.1f}" if lowm.sum() >= 5 else ""))
    X = np.column_stack([np.ones_like(R, dtype=float), R, M])
    b, *_ = np.linalg.lstsq(X, CK, rcond=None)
    res = CK - X @ b
    print(f"  fit ck = {b[0]:.1f} + {b[1]:.2f} x rows + {b[2]:.3f} x miss  (resid sd {res.std():.1f} ms)")
    X2 = np.column_stack([np.ones_like(R, dtype=float), R, SVC])
    b2, *_ = np.linalg.lstsq(X2, CK, rcond=None)
    print(f"  fit ck = {b2[0]:.1f} + {b2[1]:.2f} x rows + {b2[2]:.2f} x svc   (resid sd {(CK - X2 @ b2).std():.1f} ms)")
