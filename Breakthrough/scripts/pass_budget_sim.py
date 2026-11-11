"""Pass-level expert budget (Breakthrough, 2026-10-19): a pass of K consecutive tokens may use at most U experts per
layer, chosen by rank-weighted votes of the rows' router top-10 (weight 1/(rank+1)). Reports the share of each row's
rank-0, rank-0/1 and top-10 choices inside the budget, and RAM bytes per token for that pass (U experts x 48 layers x
expert MB, plus dense) if all K tokens were accepted. 125B traces (MoEPrefetch traces125). Quality is NOT modelled.
"""
import numpy as np
T = "/Users/kamen/MoEPrefetch/data/traces125/"; SETS = ["code", "chat", "bulgarian", "math"]; NL = 48
trs = [np.stack([np.load(T + f"{s}_800.npz")[f"L{l}"] for l in range(NL)], 1).astype(np.int32)[:, :, :10] for s in SETS]
w = 1.0 / (np.arange(10) + 1)
def run(K, U):
    r0 = r01 = r10 = n = 0; uni = []
    for tr in trs:
        for a in range(0, tr.shape[0] - K + 1, K):
            blk = tr[a:a + K]
            for l in range(NL):
                ids = blk[:, l]; uni.append(len(np.unique(ids)))
                votes = {}
                for row in ids:
                    for r, e in enumerate(row): votes[e] = votes.get(e, 0) + w[r]
                keep = set(sorted(votes, key=votes.get, reverse=True)[:U])
                for row in ids:
                    r0 += row[0] in keep; r01 += (row[0] in keep) + (row[1] in keep); r10 += sum(e in keep for e in row); n += 1
    return r0 / n, r01 / (2 * n), r10 / (10 * n), np.mean(uni)
print("K rows, U budget per layer: kept share of rank-0 / ranks 0-1 / top-10 uses; natural union; GB/token at 2.5 b (1.54 MB/expert, dense 2.0 GB) if all K accepted")
for K in (4, 16, 32):
    for U in (10, 16, 26, 40, 64):
        a, b, c, u = run(K, U)
        gb = (2.0 + min(U, u) * 48 * 1.536e-3) / K
        print(f"K={K:<3d} U={U:<3d} kept {a:.3f} / {b:.3f} / {c:.3f}   union {u:5.1f}   {gb:5.2f} GB/token -> {250/gb:5.0f} tok/s at 250 GB/s")
