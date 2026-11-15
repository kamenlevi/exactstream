#!/usr/bin/env python3
"""Masks for the decision map's bt_1c / bt_comb arms (Breakthrough, Sun 15 Nov 2026 01:2x; WEEK98 goal B). From the
exact routing (out/xcap: answer rows' rank-sorted top-10 ids + top-10-normalised weights), per window:
'keep' [rows, 48, 10] = the 4-bit ranks, 'keep3' = the MSB-3 ranks; prompt rows keep all 10 at 4 bits. Answer rows are
cut into 32-row blocks from the first answer row (the last block may be shorter). Per block and layer, each expert's
importance I = sum over the block's rows d of s^d w^2 (s 0.98) picks its tier (lambda 1.6e-2, the 3-tier RD of
lossy_check_price.py): 4-bit if I > 2.775 lambda, MSB-3 if 0.8457 lambda < I <= 2.775 lambda, else dropped.
Self-check: the full blocks' 4-bit-equivalent experts per layer = lossy_check_price.py's 36.4 (+-0.2).
Writes ~/FableGod/decision_map/out/bt/tiers_1c/<window>.npz. Usage: ~/BigMoE/.venv/bin/python -I -B build_tiers_masks.py
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
OUTD = os.path.join(M, "bt", "tiers_1c")
S_, LAM, ROWS, NL = 0.98, 1.6e-2, 32, 48
T4, T3 = 2.775 * LAM, 0.8457 * LAM
os.makedirs(OUTD, exist_ok=True)
base = json.load(open(os.path.join(M, "baseline.json")))
eq_full, n_full = 0.0, 0
tot = np.zeros(3)
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f))
    ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
    m = ids.shape[0]
    n = a + m                                     # window rows (prompt + answer), as the map's keep masks
    assert base[f]["a"] == a and len(base[f]["tk"]) == m, f
    keep = np.ones((n, NL, 10), bool); keep3 = np.zeros((n, NL, 10), bool)
    for b0 in range(0, m, ROWS):
        rows = np.arange(b0, min(b0 + ROWS, m)); rw = S_ ** np.arange(len(rows))
        for l in range(NL):
            I_ids = ids[rows, l]; w2 = w[rows, l] ** 2 * rw[:, None]
            u, inv = np.unique(I_ids.ravel(), return_inverse=True)
            imp = np.bincount(inv, weights=w2.ravel(), minlength=len(u))
            tier = np.where(imp > T4, 0, np.where(imp > T3, 1, 2))           # 0 4-bit, 1 MSB-3, 2 drop
            tr = tier[inv].reshape(len(rows), 10)
            keep[a + rows, l] = tr == 0; keep3[a + rows, l] = tr == 1
            if len(rows) == ROWS:
                eq_full += (tier == 0).sum() + 0.778 * (tier == 1).sum(); n_full += 1
            tot += np.bincount(tier, minlength=3)
    np.savez_compressed(os.path.join(OUTD, f), keep=keep, keep3=keep3)
eq = eq_full / n_full
print(f"masks for {len(os.listdir(OUTD))} windows -> {OUTD}; full blocks' 4-bit-eq experts per layer {eq:.2f} "
      f"(SELF-CHECK vs 36.4 +-0.2: {'PASS' if abs(eq - 36.4) <= 0.2 else 'FAIL'}); tier shares 4-bit / MSB-3 / drop "
      f"{tot[0] / tot.sum():.2f} / {tot[1] / tot.sum():.2f} / {tot[2] / tot.sum():.2f} of union experts")
