"""Wait-by-gate-mass under speculative verify passes (Breakthrough, 2026-10-20 22:45). 125B traces with router probs.
A pass = K consecutive tokens (verify rows; ~3 accepted per pass at MTP 3:3). Per layer and pass, each row needs the
ranks covering theta of its top-10 mass (at least rank 0); the pass reads the union of needed experts not resident
(LRU capacity C, whole 4-bit experts, 2.77 MB). Rows' other ranks are computed if resident (frugal), else skipped.
Prints per token: loads, MB, missed mass; per layer and pass: mean / P90 loads (the parallel read depth of one wait).
usage: pass_mass_sim.py [C ...]"""
import glob, sys
import numpy as np
from collections import OrderedDict

NL = 48
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}


def run(C, theta, K):
    loads = ntok = 0
    per, missed = [], []
    for k in ids:
        tr, wt = ids[k], w[k]
        T = tr.shape[0]; ntok += T
        cw = np.cumsum(wt, -1)
        for l in range(NL):
            c = OrderedDict()
            for a in range(0, T, K):
                need, rows = [], range(a, min(a + K, T))
                for t in rows:
                    nw = int(np.argmax(cw[t, l] >= theta - 1e-6)) + 1
                    for r in range(nw):
                        e = int(tr[t, l, r])
                        if e not in need: need.append(e)
                miss = [e for e in need if e not in c]
                loads += len(miss); per.append(len(miss))
                for e in miss: c[e] = 1
                for t in rows:
                    m = 0.0
                    for r in range(10):
                        e = int(tr[t, l, r])
                        if e in c: c.move_to_end(e)
                        else: m += float(wt[t, l, r])
                    missed.append(m)
                while len(c) > C:
                    c.popitem(last=False)
    p = np.array(per)
    return loads / ntok, np.mean(missed), p.mean(), np.percentile(p, 90)


for C in [int(a) for a in sys.argv[1:]] or [51, 102]:
    print(f"C={C} ({100 * C / 512:.0f} %)", flush=True)
    for theta in (0.36, 0.70, 0.85):
        for K in (1, 4):
            ld, mm, pm, p90 = run(C, theta, K)
            print(f"  theta {theta:.2f} K={K}: loads {ld:6.1f}/token ({ld * 2.77:4.0f} MB)  missed {mm:.3f}  per layer-pass {pm:4.1f} (P90 {p90:.0f})", flush=True)
