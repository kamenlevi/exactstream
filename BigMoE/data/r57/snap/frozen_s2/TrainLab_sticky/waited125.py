"""TrainLab (10-22): does a per-pass expert budget shrink B12's WAITED set? B12 (frugal) waits for each row's ranks 0-1
(read if cold) and uses ranks 2-9 only if resident. Per 4-row pass and layer on the 125B traces: distinct experts among
the rows' ranks 0-1 (the waited set), natural vs with the budget (pass set A = the U experts with the largest summed
w^2 over the rows' top-10; a row's ranks 0-1 are then its best two within A). Plus the full union. CPU only."""
import glob, os
import numpy as np
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
NL, K, B = 48, 10, 4
for f in fs:
    name = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)          # [T, NL, 20] by weight
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
    w = g / g[:, :, :K].sum(-1, keepdims=True)
    T = ids.shape[0] // B * B
    out = []
    for U in (None, 18, 16, 14):
        wait, uni = [], []
        for b in range(0, T, B):
            for l in range(NL):
                blk, wb = ids[b:b + B, l], w[b:b + B, l]
                if U is None:
                    wait.append(len(np.unique(blk[:, :2]))); uni.append(len(np.unique(blk[:, :K])))
                    continue
                c = {}
                for r in range(B):
                    for e, x in zip(blk[r, :K], wb[r, :K]):
                        c[int(e)] = c.get(int(e), 0.0) + x * x
                A = set(sorted(c, key=c.get, reverse=True)[:U])
                w2, u2 = set(), set()
                for r in range(B):
                    inA = [int(e) for e in blk[r] if int(e) in A]                # its top-20 members of A, by weight
                    w2 |= set(inA[:2]); u2 |= set(inA[:K])
                wait.append(len(w2)); uni.append(min(U, len(A)))
        out.append(f"{'natural' if U is None else f'U {U}'}: waited (ranks 0-1) {np.mean(wait):.2f}, union {np.mean(uni):.1f}")
    print(f"{name:9s} " + " | ".join(out), flush=True)
