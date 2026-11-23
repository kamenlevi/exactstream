"""CPU: what per-pass budget U at R-row passes matches b16 / b20 at 4 rows? On Breakthrough's 125B traces (per layer
rank-ordered top-20 ids + gate probs; bulgarian / code 800 tokens, chat 559, math 767), for consecutive R-token passes: the natural union of the
rows' top-10 per layer, and for budget U the share of the rows' w^2 energy outside the budget set (w = top-10 weights
normalised over the top-10; the set = the U experts with the largest summed w^2, as budget_hook.BudgetGate). The
dropped share is a proxy for the budget's damage (every row re-fills its top-10 inside the set).
usage: budget_rows.py [R ...]"""
import glob, os, sys
import numpy as np

RS = [int(r) for r in sys.argv[1:]] or [4, 8, 16]
T = os.path.expanduser("~/Breakthrough/data/expA/traces125/")
out = ["budget_rows: natural distinct experts per layer-pass and the w^2 share outside budget U (mean over layers)"]
for f in sorted(glob.glob(T + "*_800_g.npz")):
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(48)], 1)[:, :, :10].astype(np.int64)
    g = np.stack([d[f"G{l}"] for l in range(48)], 1)[:, :, :10].astype(np.float64)
    sm = g.sum(-1, keepdims=True)
    w = np.where(sm > 0, g / np.where(sm > 0, sm, 1), 0.0)    # (tokens with no gate record: weight 0, skipped below)
    name = os.path.basename(f).split("_")[0]
    for R in RS:
        nat, drop = [], {}
        Us = [U for U in (12, 14, 16, 18, 20, 22, 24, 28, 32, 40, 48) if U < 10 * R]
        for t in range(0, ids.shape[0] - R + 1, R):
            for L in range(48):
                i_, w_ = ids[t:t + R, L].reshape(-1), (w[t:t + R, L] ** 2).reshape(-1)
                u, inv = np.unique(i_, return_inverse=True)
                c = np.bincount(inv, weights=w_)
                nat.append(len(u))
                order = np.sort(c)[::-1]
                tot = c.sum()
                if tot <= 0:
                    continue
                for U in Us:
                    drop.setdefault(U, []).append(order[U:].sum() / tot)
        out.append(f"  {name:9s} R={R:2d}: natural {np.mean(nat):5.1f} | " +
                   " ".join(f"b{U} {np.mean(drop[U]):.3f}" for U in Us if np.mean(nat) > U))
print("\n".join(out))
open(os.path.expanduser("~/TrainLab/data/budget_rows.txt"), "w").write("\n".join(out) + "\n")
