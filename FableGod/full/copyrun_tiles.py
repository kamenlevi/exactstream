"""copyrun_tiles.py (Fable FULL, LOSSY350): the 16-row TILES a wide strict copy pass computes (a group with more than 16
rows needs more than one padded tile). Strict row = ranks 0-1 + its ranks 2-9 experts that are members (in the block's
union of ranks 0-1). Same data and copy runs as copyrun_groups.py (12 code map windows, recorded routing).
The npz files are the project's own captures (allow_pickle only for the walk's names array). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I copyrun_tiles.py > copyrun_tiles.out"""
import json, math, os
import numpy as np
H = os.path.expanduser("~")
W = np.load(f"{H}/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)
B = json.load(open(f"{H}/FableGod/decision_map/out/baseline.json"))
TRIG = 8
res = {w: dict(n=0, rows=0, groups=0, tiles=0, pairs=0, pairs2=0) for w in (16, 32, 64, 128)}
for wi, nm in enumerate(str(x) for x in W["names"]):
    f = f"{H}/FableGod/decision_map/out/routing/{nm}"
    if nm not in B or not os.path.exists(f): continue
    R = np.load(f)["ids"].astype(np.int32)
    ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]
    g0 = int(ck[0, 2]) + 1; P = ids[:g0]; A = ids[g0:][:max(0, len(R) - g0)]
    pos = {}
    for i, t in enumerate(P): pos.setdefault(t, []).append(i)
    q = 0; runs = []
    while q < len(A):
        best = 0
        for i in pos.get(A[q], ()):
            L = 0
            while q + L < len(A) and i + L < len(P) and A[q + L] == P[i + L]: L += 1
            best = max(best, L)
        if best >= 8: runs.append((q, best)); q += best
        else: q += 1
    for q, L in runs:
        for Wd, r in res.items():
            s = g0 + q + TRIG; e = min(g0 + q + L, len(R))
            for b in range(s, e, Wd):
                E = R[b:min(b + Wd, e)]
                if len(E) < min(Wd, 16) // 2: continue
                r["n"] += 1; r["rows"] += len(E)
                for l in range(48):
                    U = np.unique(E[:, l, :2])
                    mem = np.isin(E[:, l, :], U)                 # [rows, 10]: the row's experts that are in the union
                    mem[:, :2] = True
                    ex, cnt = np.unique(E[:, l, :][mem], return_counts=True)
                    r["groups"] += len(U); r["tiles"] += int(np.ceil(cnt / 16).sum()); r["pairs"] += int(cnt.sum()); r["pairs2"] += 2 * len(E)
for Wd, r in res.items():
    if not r["n"]: continue
    print(f"W {Wd:3d}: {r['n']:3d} blocks, {r['rows'] / r['n']:.1f} rows; per block: groups {r['groups'] / r['n']:.0f}, 16-row tiles {r['tiles'] / r['n']:.0f} (x{r['tiles'] / r['groups']:.2f}), "
          f"pairs {r['pairs'] / r['n']:.0f} ({r['pairs'] / r['rows'] / 48:.2f} of 10 ranks per row-layer; ranks 0-1 alone {r['pairs2'] / r['n']:.0f}); rows per group {r['pairs'] / r['groups']:.1f}")
