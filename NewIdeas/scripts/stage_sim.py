"""Staging-ring prefetch for frugal's waited ranks (NewIdeas 2026-10-22, CPU, 125B traces).
Frugal: ranks 0-1 waited if cold (read + insert into the layer's LRU of C slots), ranks 2-9 computed only if resident.
Prefetch: at layer l-1, the d=1 look-ahead top-k for layer l (MoEPrefetch *_400_pf P1, router of l on l-1's MoE input)
is read if not resident. 'stage': into a small GLOBAL ring (no eviction from the cache; promoted only if routed at
ranks 0-1). 'cache': inserted into layer l's LRU (Breakthrough's equal-RAM version: pollutes). Reports per token: waited
reads, hidden reads, wasted prefetch reads, all SSD reads, missed gate energy of ranks 2-9 (top-10 renormalised)."""
import glob, sys
import numpy as np
from collections import OrderedDict

NL = 48
PF = sorted(glob.glob("/Users/kamen/MoEPrefetch/data/traces125/*_400_pf.npz"))
GG = {f.split("/")[-1].split("_")[0]: f for f in glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz")}
DATA = []
for f in PF:
    name = f.split("/")[-1].split("_")[0]
    p, g = np.load(f), np.load(GG[name])
    L = np.stack([p[f"L{l}"][:, :10] for l in range(NL)], 1).astype(np.int64)          # [T, NL, 10] rank order
    G = np.stack([g[f"G{l}"][:400, :10] for l in range(NL)], 1).astype(np.float64)
    W = G / G.sum(-1, keepdims=True)
    P1 = np.stack([p[f"P1_{l}"] for l in range(NL - 1)], 1).astype(np.int64)           # P1[t, l] predicts layer l+1
    DATA.append((name, L, W, P1))


def recall():
    print("d=1 recall of the true ranks 0-1 inside the predicted top-k (125B, 4 prompts x 400 tokens)")
    for k in (2, 3, 4, 6, 8, 16):
        tot = hit = 0; band = {}
        for name, L, W, P1 in DATA:
            for l in range(1, NL):
                pr = P1[:, l - 1, :k]
                h = (L[:, l, :2, None] == pr[:, None, :]).any(-1)                    # [T, 2]
                hit += h.sum(); tot += h.size
                b = "1-7" if l < 8 else ("8-39" if l < 40 else "40-47")
                bh, bt = band.get(b, (0, 0)); band[b] = (bh + h.sum(), bt + h.size)
        print(f"  top-{k:2d}: {hit / tot:.3f}  " + "  ".join(f"L{b} {v[0] / v[1]:.3f}" for b, v in sorted(band.items())))


def sim(C, k, mode):
    wait = hidden = waste = reads = ntok = 0; miss_e = 0.0
    for name, L, W, P1 in DATA:
        T = L.shape[0]; ntok += T
        caches = [OrderedDict() for _ in range(NL)]
        for t in range(T):
            staged = set()
            for l in range(NL):
                c = caches[l]
                # prefetch issued for layer l at layer l-1 (staged set built before layer l runs)
                cur = staged; staged = set()
                if k and l + 1 < NL:
                    for e in P1[t, l, :k]:
                        e = int(e)
                        if e in caches[l + 1]: continue
                        reads += 1; staged.add(e)
                        if mode == "cache":
                            cc = caches[l + 1]; cc[e] = 1
                            if len(cc) > C: cc.popitem(last=False)
                used = set()
                for r in range(10):
                    e = int(L[t, l, r])
                    if e in c:
                        c.move_to_end(e); used.add(e)
                    elif r < 2:
                        if e in cur:
                            hidden += 1; used.add(e)
                        else:
                            wait += 1; reads += 1
                        c[e] = 1
                        if len(c) > C: c.popitem(last=False)
                    elif e in cur:                                    # staged lower rank: free coverage
                        used.add(e)
                    else:
                        miss_e += W[t, l, r] ** 2
                waste += len(cur - used)
    return wait / ntok, hidden / ntok, waste / ntok, reads / ntok, miss_e / ntok


if __name__ == "__main__":
    recall()
    for C in (51, 26):
        for mode, ks in (("none", (0,)), ("stage", (1, 2, 3, 4)), ("cache", (2, 4))):
            for k in ks:
                w, h, ws, rd, me = sim(C, k, mode)
                print(f"C={C} {mode:5s} k={k}: waited {w:5.1f} hidden {h:5.1f} wasted {ws:5.1f} reads {rd:5.1f}/token  "
                      f"missed E(ranks 2-9) {me:.4f}/token", flush=True)
