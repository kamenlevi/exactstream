> (a quote from the project owner, kept private)
the exact check's cost R(K) = check(K) / A16 for K = 16 / 32 / 48 / 64 / 96 / 128 rows, and the tree-check overhead c(K).
Desk only, from measured parts:
  - the expert union per layer at K rows: the 84 route files (~/datasets/deepdraft/routes, generated tokens, non-overlapping
    windows), measured here: 73.2 / 109.7 / 135.5 / 156.5 / 187.4 / 211.1;
  - the byte floor: 3.44 GB of dense weights + 48 x union x 2.135 MB of 4-bit experts, at 280 GB/s (pipe_price.py);
  - the engine's A16: 62 ms (G32T 61-64, ISO2 64.1); C32 through v32's split: 1.625 x A16 (G32T);
  - the 16-row dense side by class (Breakthrough's CHECK55 profile, engine-scaled; ms): projections 11.24, HC 10.44,
    MoE dense side 4.16, attention core 3.52, GDN core 2.20, head 1.89, PLE 0.60 (sum 34.05). The experts take the rest
    of A16 (~28 ms, ~1.05 x their 16-row floor), and that ratio is kept at every K;
  - per extra 16-row tile in ONE weight pass (CB1, 16 -> 32 rows, cold standalone): +21-29 % on the dense projections and
    the head (+25 % used), HC +25 % (the profile's 10.44 -> 13.0), the MoE dense side +60 % (split today); the cores
    scale with the rows (+100 % per tile);
  - TREE16-TIME v2: a bushy 16-node tree costs +1.55 ms engine-scaled over its chain (c16 = 1.025).
Paths priced: REF2-T SPLIT (today's code above 32 rows: every op as 16-row calls, so the dense side x K/16);
REF2-T ONE-PASS (one weight pass for every K: kernels to build beyond 32 rows, as v32one / qmm16c2a do at 32);
+BOF (the exact levers that passed on the slice: -2.5 ms at 16, -2 at 32 [est], kept flat beyond);
CONTRACT (track B's dense side: -4 ms [est from C8], kept flat beyond 32; attention by key lists not included).
Every K-row number above 32 rows is an estimate [est]; C7c tonight and E2 Monday measure 32 rows on the engine."""
import numpy as np

A16 = 62.0
BW, DENSE_GB, EXP_GB, NL = 280.0, 3.44, 2.135e-3, 48
UNION = {16: 73.2, 32: 109.7, 48: 135.5, 64: 156.5, 96: 187.4, 128: 211.1}
UMAX = {16: (119, 152), 32: (181, 226), 48: (223, 290), 64: (256, 317), 96: (298, 357), 128: (332, 380)}
D16 = dict(proj=11.24, hc=10.44, moe=4.16, att=3.52, gdn=2.20, head=1.89, ple=0.60)
GROW = dict(proj=0.25, hc=0.25, moe=0.60, att=1.0, gdn=1.0, head=0.26, ple=0.37)   # per extra 16-row tile, one pass
EXP_X = (A16 - sum(D16.values())) / (NL * UNION[16] * EXP_GB / BW * 1e3)


def floor_ms(K):
    return (DENSE_GB + NL * UNION[K] * EXP_GB) / BW * 1e3


def experts_ms(K):
    return EXP_X * NL * UNION[K] * EXP_GB / BW * 1e3


def dense_onepass(K, att_grow=1.0):
    t = K / 16 - 1
    g = dict(GROW, att=att_grow)
    return sum(v * (1 + g[k] * t) for k, v in D16.items())


SPLIT_TILE = (1.625 * A16 - EXP_X * NL * UNION[32] * EXP_GB / BW * 1e3) / sum(D16.values()) - 1   # G32T-calibrated


def dense_split(K):
    return sum(D16.values()) * (1 + SPLIT_TILE * (K / 16 - 1))


def tree_c(K, chain_ms, p):
    """c(K) = 1 + the tree code's overhead / the chain check: TREE16-TIME's +1.55 ms at 16 nodes, scaled by nodes^p"""
    return 1 + 1.55 * (K / 16) ** p / chain_ms


rows = []
print(f"experts at {EXP_X:.2f} x their floor (A16 {A16} = dense {sum(D16.values()):.2f} + experts {A16 - sum(D16.values()):.2f})")
print(f"{'K':>4s} {'union':>6s} {'floor':>7s} {'fl/A16':>6s} | {'split':>7s} {'R':>5s} | {'1-pass (att sub-lin..lin)':>25s} | "
      f"{'+BOF':>7s} {'R':>5s} | {'contr':>7s} {'R':>5s} | c(K) linear .. ^1.5 | union max/layer p50, max")
for K in (16, 32, 48, 64, 96, 128):
    fl = floor_ms(K)
    ex = experts_ms(K)
    sp = ex + dense_split(K)
    op = ex + dense_onepass(K)
    opl = ex + dense_onepass(K, att_grow=0.5)                  # the attention core sub-linear (keys shared across rows)
    bof = op - (2.5 if K == 16 else 2.0)
    con = op - 4.0
    c1, c2 = tree_c(K, op, 1.0), tree_c(K, op, 1.5)
    rows.append((K, fl, sp, op, opl, bof, con, c1, c2))
    print(f"{K:4d} {UNION[K]:6.1f} {fl:7.1f} {fl / A16:6.2f} | {sp:7.1f} {sp / A16:5.2f} | {opl:6.1f}-{op:5.1f} {opl / A16:4.2f}-{op / A16:4.2f} | "
          f"{bof:7.1f} {bof / A16:5.2f} | {con:7.1f} {con / A16:5.2f} | {c1:.3f} .. {c2:.3f} | {UMAX[K][0]}, {UMAX[K][1]}")
print(f"calibration: the split path's dense side grows {SPLIT_TILE:.2f} per extra 16-row tile (G32T: C32 split = 1.625 x A16); "
      f"one-pass at 32 rows {rows[1][3] / A16:.2f} x A16 vs the expected 1.40-1.50 (C7c tonight, E2 Monday)")
