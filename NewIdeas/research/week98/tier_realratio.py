#!/usr/bin/env python3
"""TIER RULE ON REAL BYTES (NewIdeas, Mon 16 Nov 17:5x; WEEK98 goal B, number check of the 1c / C6 tier rule).

The 3-tier rule (Breakthrough's lossy_check_price.py RD, used by build_tiers_masks.py / g1g2_map.py) is
  4-bit if I > 2.775 lambda; MSB-3 if 0.8457 lambda < I <= 2.775 lambda; else dropped,
with I = sum over a 32-row block's rows d of s^d w^2 (s 0.98). Its two thresholds are the RD break-evens for errors
(4-bit 0, MSB-3 0.08, drop 1.0) and RELATIVE BYTES 4-bit 1.0 / MSB-3 0.778: (1 - 0.778) / 0.08 = 2.775 and
0.778 / (1 - 0.08) = 0.8457. 0.778 = 1.661 / 2.135 MB, the OLD piece sizes. The real pieces (records3, Leap 73ddc5a,
checked 13:5x) are 2.785 / 2.064 MB, ratio 0.741, so the RD break-evens are 3.2375 and 0.8054.

Question: at EQUAL modelled distortion D = sum I x err (the rule's own objective), how many expert bytes does the
real-ratio rule save over the old one, at lambda 1.6e-2 (1c) and 1.8e-2 (C6)? Pre-run line: >= 2 % of expert bytes
= worth handing to Breakthrough for G2's lambda; < 1 % = a footnote.

Data: ~/FableGod/decision_map/out/xcap (36 map windows; answer rows' top-10 ids + normalised weights), full 32-row
blocks from the first answer row, as build_tiers_masks.py. CPU only. Usage:
  taskpolicy -b ~/BigMoE/.venv/bin/python -I -B tier_realratio.py
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
S_, ROWS, NL = 0.98, 32, 48
B4, B3 = 2.785, 2.064          # MB per expert piece, records3 (real)
E3 = 0.08                      # MSB-3 error per w^2 (lossy_check_price.py calibration)
OLD = (2.775, 0.8457)
NEW = ((1 - B3 / B4) / E3, (B3 / B4) / (1 - E3))

base = json.load(open(os.path.join(M, "baseline.json")))
imps = {"code": [], "all": []}
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f))
    ids, w = z["ids"].astype(np.int64), z["w"].astype(np.float64)
    dom = base[f]["dom"]
    m = ids.shape[0]
    for b0 in range(0, m - ROWS + 1, ROWS):               # full blocks only
        rw = S_ ** np.arange(ROWS)
        blk = []
        for l in range(NL):
            u, inv = np.unique(ids[b0:b0 + ROWS, l].ravel(), return_inverse=True)
            imp = np.bincount(inv, weights=(w[b0:b0 + ROWS, l] ** 2 * rw[:, None]).ravel(), minlength=len(u))
            blk.append(imp)
        v = np.concatenate(blk)
        imps["all"].append(v)
        if dom == "code":
            imps["code"].append(v)


def price(I, lam, th):
    t4, t3 = th[0] * lam, th[1] * lam
    n4 = (I > t4).sum(); n3 = ((I > t3) & (I <= t4)).sum()
    d = (E3 * I[(I > t3) & (I <= t4)]).sum() + I[I <= t3].sum()
    return n4 * B4 + n3 * B3, d, n4, n3


lines = [f"tier rule on real bytes: OLD thresholds {OLD[0]:.4f} / {OLD[1]:.4f} lambda (ratio 0.778), "
         f"REAL-ratio {NEW[0]:.4f} / {NEW[1]:.4f} lambda (ratio {B3 / B4:.3f}); E3 {E3}"]
for key in ("code", "all"):
    I = np.concatenate(imps[key]); nb = len(imps[key])
    lines.append(f"[{key}] {nb} full 32-row blocks")
    for lam in (1.6e-2, 1.8e-2):
        bo, do, n4o, n3o = price(I, lam, OLD)
        # lambda' for the real-ratio rule at the same D (bisection; D rises with lambda)
        lo, hi = lam * 0.5, lam * 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if price(I, mid, NEW)[1] < do:
                lo = mid
            else:
                hi = mid
        lamn = 0.5 * (lo + hi)
        bn, dn, n4n, n3n = price(I, lamn, NEW)
        bs, ds, _, _ = price(I, lam, NEW)
        lines.append(
            f"  lambda {lam:.1e}: OLD {bo / nb / 1e3:.3f} GB/check (4-bit {n4o / nb / NL:.1f}, MSB-3 {n3o / nb / NL:.1f} per "
            f"layer), D {do / nb:.4f} | REAL-ratio same lambda {bs / nb / 1e3:.3f} GB, D {ds / nb:.4f} ({100 * (ds / do - 1):+.1f} %) | "
            f"REAL-ratio at equal D (lambda {lamn:.3e}): {bn / nb / 1e3:.3f} GB ({100 * (bn / bo - 1):+.2f} % bytes; 4-bit "
            f"{n4n / nb / NL:.1f}, MSB-3 {n3n / nb / NL:.1f})")
out = "\n".join(lines)
print(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_realratio.txt"), "w").write(out + "\n")
