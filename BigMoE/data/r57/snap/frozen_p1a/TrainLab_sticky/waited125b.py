"""TrainLab (10-22): a pass-level WAITED budget for B12 on the 125B traces. Per 4-row pass and layer, instead of every
row's ranks 0-1 (the waited reads), wait only for the Uw experts with the largest summed w^2 over the rows' ranks 0-1;
a row's rank-0/1 expert outside that set is dropped (frugal fills from residents; counted at its w^2, Chat1000's energy).
Waited distinct experts per pass and the added energy per token-layer. CPU only."""
import glob, os
import numpy as np
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
NL, K, B = 48, 10, 4
for f in fs:
    name = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
    w = g / g[:, :, :K].sum(-1, keepdims=True)
    T = ids.shape[0] // B * B
    out = []
    for Uw in (None, 5, 4, 3):
        wait, en = [], []
        for b in range(0, T, B):
            for l in range(NL):
                blk, wb = ids[b:b + B, l, :2], w[b:b + B, l, :2]
                if Uw is None:
                    wait.append(len(np.unique(blk))); en.append(0.0)
                    continue
                c = {}
                for r in range(B):
                    for e, x in zip(blk[r], wb[r]):
                        c[int(e)] = c.get(int(e), 0.0) + x * x
                A = set(sorted(c, key=c.get, reverse=True)[:Uw])
                wait.append(len(A & set(c)))
                en.append(sum(x * x for r in range(B) for e, x in zip(blk[r], wb[r]) if int(e) not in A) / B)
        out.append(f"{'natural' if Uw is None else f'Uw {Uw}'}: waited {np.mean(wait):.2f}, added energy {np.mean(en):.4f}")
    print(f"{name:9s} " + " | ".join(out), flush=True)
