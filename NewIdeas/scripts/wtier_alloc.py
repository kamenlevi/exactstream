"""Kill test (CPU, 125B traces): give the 4-bit tier by router WEIGHT instead of by RANK, at the same average number of
4-bit experts per token-layer (2, as in v3's plan 4433333333). First-order error proxy: the sum of w^2 over the experts left at
3 bits (each 3-bit expert adds an output error ~ w * delta_f with similar ||delta_f||; errors uncorrelated). Published relative:
HOBBIT (2411.01433, precision of missed experts by gating score); untested here.
Pre-registered kill line: weight/rank error ratio > 0.85 -> kill; <= 0.75 -> worth a GPU tf KLD screen.
usage: wtier_alloc.py   (reads ~/Breakthrough/data/expA/traces125/*_800_g.npz, read-only)"""
import glob, os
import numpy as np
NL, K, N4 = 48, 10, 2
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz"))):
    d = np.load(f)
    g = np.stack([d[f"G{l}"][:, :K] for l in range(NL)], 1).astype(np.float64)    # T x L x K, rank order
    w = g / g.sum(-1, keepdims=True)
    w2 = w * w
    e_all3 = w2.sum(-1).mean()
    e_rank = w2[:, :, N4:].sum(-1).mean()
    flat = np.sort(w.ravel())[::-1]
    tau = flat[N4 * w.shape[0] * NL - 1]                     # global threshold: mean count of w >= tau == N4
    e_glob = np.where(w < tau, w2, 0).sum(-1).mean()
    cnt_glob = (w >= tau).sum(-1)
    e_lay = 0.0
    for l in range(NL):                                       # per-layer threshold, same mean count per layer
        fl = np.sort(w[:, l].ravel())[::-1]
        t = fl[N4 * w.shape[0] - 1]
        e_lay += np.where(w[:, l] < t, w2[:, l], 0).sum(-1).mean() / NL
    rows.append((os.path.basename(f).split("_")[0], w.shape[0], e_all3, e_rank, e_glob, e_lay,
                 np.percentile(cnt_glob, [10, 50, 90]), w[:, :, 0].mean(), w[:, :, 1].mean(), w[:, :, 2].mean()))
print("corpus  tokens  sum w^2 at 3 bits: all-3bit / rank plan (ranks 0-1 at 4) / weight, global tau / weight, per-layer tau"
      " | ratio weight/rank (global, per-layer) | 4-bit count per token-layer p10/p50/p90 (global) | mean w rank 0/1/2")
for n, T, a, r, gl, la, pc, w0, w1, w2_ in rows:
    print(f"{n:9s} {T:5d}  {a:.4f} / {r:.4f} / {gl:.4f} / {la:.4f} | {gl / r:.3f} {la / r:.3f} | {pc[0]:.0f}/{pc[1]:.0f}/{pc[2]:.0f}"
          f" | {w0:.3f} {w1:.3f} {w2_:.3f}")
