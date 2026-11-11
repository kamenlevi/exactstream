"""Predicted error energy of missing-expert estimators under frugal routing (Breakthrough D1b, 2026-10-21 night).
125B traces with router probabilities (data/expA/traces125/*_800_g.npz); frugal wait-top-2 with an LRU cache of C
experts per layer (only waited experts are loaded); for every missing rank r >= 2 (expert u, top-10 weight w):
  drop      : error energy w^2 N_u^2                                  (N_u = ||Y_u|| on the calibration rows)
  sere-all  : w^2 min(N_u^2, min_v D[u,v]^2) over ALL resident v not used by the token (best case of SERE)
  sere-r20  : the same over resident v among the token's router top-20 (what a routing row can carry)
  shared    : w^2 (r_sh N_u)^2 with the optimally scaled shared-expert output (P<l>.npy row 1)
  meanw     : the same with a resident mean-weights expert (row 3); meanout: the mean of all outputs (row 5, a bound)
Prints the error energy relative to 'drop' per estimator (pairwise distances are averaged over calibration tokens, so
this is a per-expert-pair prediction, not a per-token one). usage: sere_sim.py [C ...]"""
import glob, os, sys
import numpy as np
from collections import OrderedDict

NL = 48
T = os.path.expanduser("~/Breakthrough/data/simtab")
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz")))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}
layers = [l for l in range(NL) if os.path.exists(f"{T}/D{l}.npy") and os.path.exists(f"{T}/P{l}.npy")]
Dt = {l: np.load(f"{T}/D{l}.npy").astype(np.float32) for l in layers}
Nt = {l: np.load(f"{T}/N{l}.npy").astype(np.float32) for l in layers}
Pt = {l: np.load(f"{T}/P{l}.npy").astype(np.float32) for l in layers}
print(f"tables for {len(layers)} layers")


def run(C):
    E = {k: 0.0 for k in ("drop", "sere-all", "sere-r20", "shared", "meanw", "meanout")}
    for k in ids:
        tr, wt = ids[k], w[k]
        for l in layers:
            D, N, P = Dt[l], Nt[l], Pt[l]
            c = OrderedDict()
            for t in range(tr.shape[0]):
                row = [int(e) for e in tr[t, l]]
                used = set(row[:2])
                for e in row[:2]:
                    if e not in c: c[e] = 1
                res_now = [e for e in c if e not in used]
                for r in range(2, 10):
                    u, ww = row[r], float(wt[t, l, r])
                    if u in c:
                        used.add(u); continue
                    n2 = float(N[u]) ** 2 * ww * ww
                    E["drop"] += n2
                    cand = [v for v in res_now if v not in used]
                    best = min([float(D[u, v]) ** 2 for v in cand], default=np.inf)
                    E["sere-all"] += min(n2, best * ww * ww)
                    c20 = [v for v in row[10:20] if v in c and v not in used]
                    b20 = min([float(D[u, v]) ** 2 for v in c20], default=np.inf)
                    E["sere-r20"] += min(n2, b20 * ww * ww)
                    E["shared"] += n2 * float(P[1, u]) ** 2
                    E["meanw"] += n2 * float(P[3, u]) ** 2
                    E["meanout"] += n2 * float(P[5, u]) ** 2
                for e in row[:10]:
                    if e in c: c.move_to_end(e)
                while len(c) > C:
                    c.popitem(last=False)
    return {k: v / E["drop"] for k, v in E.items()}


for C in [int(a) for a in sys.argv[1:]] or [51, 26]:
    r = run(C)
    print(f"C={C}: error energy relative to dropping: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
