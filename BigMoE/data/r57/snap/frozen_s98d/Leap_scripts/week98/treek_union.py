"""treek_union.py (Leap, Sun 15 Nov 2026 23:5x; bigger trees, NewIdeas' R(24) ask): the expert union per layer at K rows,
from the 84 route files (~/datasets/deepdraft/routes: routes [T, 48 layers, top-10], generated tokens from 'gen' on),
non-overlapping windows of K consecutive generated tokens. Prints the mean union per layer and the busiest layer per check
(p50 / max). The first cut (treek_price.py) used 16 / 32 / 48 / 64 / 96 / 128; this adds 24 and re-checks the rest.
usage: [BigMoE venv] python -I treek_union.py   (CPU, < 1 GB, seconds)"""
import glob, os
import numpy as np

KS = (16, 24, 32, 48, 64, 96, 128)
files = sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/routes/*.npz")))
U = {K: [] for K in KS}
B = {K: [] for K in KS}
for f in files:
    z = np.load(f, allow_pickle=False)
    r = z["routes"][int(z["gen"]):].astype(np.int32)          # [T, L, k], generated tokens only
    T, L, k = r.shape
    for K in KS:
        for s in range(0, T - K + 1, K):
            w = r[s:s + K]                                    # [K, L, k]
            n = np.array([np.unique(w[:, l, :]).size for l in range(L)])
            U[K].extend(n.tolist())
            B[K].append(int(n.max()))
lines = [f"treek_union: {len(files)} route files, generated tokens, non-overlapping windows"]
for K in KS:
    lines.append(f"  K {K:4d}: union per layer mean {np.mean(U[K]):6.1f} (windows {len(B[K])}); busiest layer per check "
                 f"p50 {np.median(B[K]):.0f}, max {max(B[K])}")
out = "\n".join(lines)
print(out)
open(os.path.expanduser("~/Leap/data/week98/treek_union.txt"), "w").write(out + "\n")
