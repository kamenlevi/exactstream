"""Distinct experts per verify pass from MoEPrefetch's 125B decode traces (top-20 per token per layer, rank order):
natural union of S rows' top-10, per layer, mean over layers and passes; the budget sets U are fixed by construction.
usage: union_sizes.py   (CPU)"""
import os
import numpy as np
for tr in ("code", "chat", "math", "bulgarian"):
    z = np.load(os.path.expanduser(f"~/MoEPrefetch/data/traces125/{tr}_400_pf.npz"))
    line = []
    for S in (1, 4, 8, 16, 32):
        u = []
        for L in range(48):
            t = z[f"L{L}"][:, :10]
            for a in range(0, t.shape[0] - S + 1, S):
                u.append(len(np.unique(t[a:a + S])))
        line.append(f"S={S}: {np.mean(u):5.1f} ({np.mean(u) / (10 * S):.2f} of S x 10)")
    print(f"{tr:9s} " + " | ".join(line))
