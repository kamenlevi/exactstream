#!/usr/bin/env python3 -I
"""budget_c3.py (Fable C, mission C3): one cycle model for ROUTE_50.md, on measured terms [K] and the audit's corrections.
Per 16-row-block check on code: cycle = chain + exposed SSD; chain = dense + experts + draft + commit + CPU + rest.
  experts = 48 x (2 x 0.094 + 0.01015 us-ms x groups x g x tm) x k   [EXPERT_KERNEL fit; OOO's two calls pay the dispatch
           intercept twice; g = the gate's group factor (replay_c2: Q1 0.99 x0.79, Q1 0.9 x0.66); tm = tier merge x0.85;
           k = kernel: 3c 1.17 at 16 rows (1.0 at 9), NAX6 1.0, NAX6-N 0.85 (N7-14 +17.3 %, bitwise), NAX6-NH 0.81 (+23.3 %, FAST)]
  groups(rows) interpolates 56 / 65 / 77 at 9 / 12 / 16 rows [replay_c2, EXPERT_KERNEL]; rows = min(16, kept + 3) [band45: 9.1 rows at kept 6.1]
  SSD transfer = MB per kept token x kept / 11.7 GB/s [c45l_256: 11.7 GB/s in demand]; exposed = max(0, transfer - 0.7 x chain) [G]
  kept lever on bytes: MB per kept token x (kept / 11.0) ** -0.5 [G, SSD_WALL's x0.965 for +7 % kept]
usage: nice -n 15 ~/BigMoE/.venv/bin/python -I budget_c3.py"""
import numpy as np


def groups(rows):
    return float(np.interp(rows, [9, 12, 16], [56, 65, 77]))


def experts(kept, g, tm, k):
    rows = min(16, kept + 3)
    kk = k if k != "3c" else 1.0 + 0.17 * (rows - 9) / 7
    return 48 * (2 * 0.094 + 0.01015 * groups(rows) * g * tm) * kk


def cycle(kept, mb, dense, exp_ms, draft, commit, h, cpu=2.4, rest=1.5):
    """h = the hidden share of the SSD transfer: 0.38 measured today (c45l_256: HumanEval/68 demand 124 ms, check + commit
    170.8, warm 87 + 4.7 -> exposed 79, hidden 45 = 36 %; algebra/694 ~40 %) [K]; 0.6 with better prefetch [G]"""
    chain = dense + exp_ms + draft + commit + cpu + rest
    transfer = mb * (kept / 11.0) ** -0.5 * kept / 11.7
    exposed = transfer * (1 - h)
    return chain, transfer, exposed, chain + exposed, 1000 * kept / (chain + exposed)


# SSD MB per kept token [K replay_c3, t 0.3 checks, six cap45 prompts]: (EXACT, Q1 0.99, Q1 0.9, EXACT no sidecar, Q1 0.9 no sidecar)
SSD = {"M-17 C 256 (42.0 GB)": (54.1, 44.1, 36.9, 41.4, 24.2), "joint C 256 +2.07 (44.1 GB)": (47.6, 38.9, 32.6, 36.1, 21.1),
       "joint C 256 +4.14 (46.2 GB)": (41.4, 34.5, 29.3, 30.8, 18.6), "M-17 C 320 (50.5 GB)": (30.8, 27.3, 24.3, 21.1, 14.7),
       "joint C 320 +2.07 (52.6 GB)": (27.8, 24.8, 22.0, 19.6, 13.8)}
PACK = 0.891                                      # the 11-bit scale / bias pack, exact (W24-sp -10.9 %)

print("== A. the engine at each RAM point; cycle ms (chain / SSD transfer / exposed) and tok/s ==")
print("   expected = dense FAST 17.2 (HC8) / EXACT 23, draft 4.5, commit 2.0, SSD hidden 38 % (measured); targets = dense 15 / 20, draft 2.8, commit 0.5, SSD hidden 60 %")
for ram, (ex, q99, q9, exd, q9d) in SSD.items():
    print(f"-- {ram}")
    for kept in (11.0, 12.2, 12.6):
        for label, mb, dense_e, dense_t, g, tm, k in (
                ("FAST  Q1 0.99 NAX6-NH            ", q99, 17.2, 15.0, 0.79, 1.0, 0.81),
                ("FAST  Q1 0.9  NAX6-NH + tier merge", q9, 17.2, 15.0, 0.66, 0.85, 0.81),
                ("FAST  Q1 0.9  + sidecar drop      ", q9d, 17.2, 15.0, 0.66, 0.85, 0.81),
                ("EXACT 3c (today's reference)      ", ex, 23.0, 20.0, 1.0, 1.0, "3c"),
                ("EXACT NAX6-N (NAX reference)      ", ex, 23.0, 20.0, 1.0, 1.0, 0.85),
                ("EXACT NAX6-N + pack               ", ex * PACK, 23.0, 20.0, 1.0, 1.0, 0.85)):
            e_ms = experts(kept, g, tm, k)
            ce = cycle(kept, mb, dense_e, e_ms, 4.5, 2.0, 0.38); ct = cycle(kept, mb, dense_t, e_ms, 2.8, 0.5, 0.6)
            print(f"   kept {kept:4.1f} {label} experts {e_ms:4.1f} | expected {ce[3]:5.1f} ({ce[0]:4.1f}/{ce[1]:4.1f}/{ce[2]:4.1f}) -> {ce[4]:5.0f} tok/s"
                  f" | targets {ct[3]:5.1f} ({ct[0]:4.1f}/{ct[1]:4.1f}/{ct[2]:4.1f}) -> {ct[4]:5.0f}")

print("\n== B. what 350 needs at kept 12.6 / 13.5: chain <= 36 / 38.6 ms with the SSD hidden (<= 25 / 27 ms of transfer) ==")
for kept in (12.6, 13.5):
    for label, g, tm, k in (("Q1 0.99 NAX6-NH", 0.79, 1.0, 0.81), ("Q1 0.9 NAX6-NH + tier merge", 0.66, 0.85, 0.81), ("Q1 0.9 NAX6-NH + tier merge, OOO single call", 0.66, 0.85, 0.81)):
        e_ms = experts(kept, g, tm, k) - (4.5 if "single" in label else 0)
        print(f"   kept {kept}: {label:46s} experts {e_ms:4.1f}; chain at targets = 15 + {e_ms:4.1f} + 2.8 + 0.5 + 2.4 + 1.5 = {15 + e_ms + 2.8 + 0.5 + 2.4 + 1.5:4.1f} ms -> {1000 * kept / (15 + e_ms + 2.8 + 0.5 + 2.4 + 1.5):4.0f} tok/s if the SSD hides")
