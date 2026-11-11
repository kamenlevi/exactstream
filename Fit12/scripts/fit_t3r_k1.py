"""Fit12 T3R K1 (2026-10-25; NewIdeas' bet T3R 19:58, ~/NewIdeas/QUEUE.md row T3R; pre-registered in PLAN.md 'T3R'): SSD
bytes per token, v3 vs T3R in the same RAM, with item (a)'s engine replay (fit_v3replay.py: ensure / make_room /
lfu_victim of ~/BigMoE/ext/bigmoe_ext.cpp; engine LFU 0.99, 16-row checks accepting 10, the 4 expA trace sets, warm from
token 0, counted from token 400, MB = 1e6 B).
Per layer two tiers, as v3: a base tier for every top-10 use and a residual tier for ranks 0-1. The tiers never interact,
so each is replayed alone over a capacity grid and the split is searched exactly.
Records per expert, 16 KiB-padded like the engine's (records3: MSB 2,162,688 = 132 x 16 KiB, LSB 770,048 = 47 x 16 KiB):
  v3   MSB-3 (3-bit plane + bf16 s3u / b3u per 64) 2,162,688 B; LSB sidecar (1-bit plane + bf16 bu per 64) 770,048 B
  T3R  TRL-3 3INST base: 3 b/w + a 12-bit initial state per 256-weight tile + a bf16 scale per row = 1,879,680 B
       -> 115 x 16 KiB = 1,884,160 B; R1: 1 b/w + 12 / 256 + bf16 row scale = 650,880 B -> 40 x 16 KiB = 655,360 B
       (the RHT signs come from seeds: no bytes)
Budgets per layer = v3's slots x its records: 12 GB = B12LL C_m 51 + C_l 20 (DECISIVE, NewIdeas' B10-12 row), 12 GB at
C_m 59 + C_l 20 (side: < 12 GB to ~2.6K prompts), 24 GB = C_m 147 + C_l 46. T3R: the best residual pool C_r (the base
gets what is left). Side: v3 re-split in the same budget; NewIdeas' B10-12 v3 numbers re-derived as a cross-check.
KILL (NewIdeas): T3R < 20 % fewer bytes / token than v3 at 12 GB.
Usage: ~/BigMoE/.venv/bin/python -B fit_t3r_k1.py   (CPU, 1 core, a few minutes)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_v3replay as V

PAD = 16384
pad = lambda b: -(-b // PAD) * PAD


def trellis_bytes(k):
    """gate / up: 640 rows of 2,560; down: 2,560 rows of 640; tiles of 256 weights, L = 12"""
    bits = 3 * 640 * 2560 * k + 3 * (640 * 2560 // 256) * 12 + (640 + 640 + 2560) * 16
    assert bits % 8 == 0
    return bits // 8


BASE_RAW, RES_RAW = trellis_bytes(3), trellis_bytes(1)
BASE, RES = pad(BASE_RAW), pad(RES_RAW)
assert (BASE_RAW, RES_RAW, BASE, RES) == (1879680, 650880, 1884160, 655360)
MSB, LSB = V.MSB, V.LSB
BUDGETS = (("12 GB, B12LL (C_m 51 + C_l 20)", 51, 20, True), ("12 GB at C_m 59 + C_l 20", 59, 20, False),
           ("24 GB (C_m 147 + C_l 46)", 147, 46, False))
DEC = (16, 10, 0.99)
cache = {}


def misses(tr, C, top, R, A, d):
    """per set: misses per counted token of one tier (C slots, ids[:, :top]), summed over the 48 layers"""
    key = (C, top, R, A, d)
    if key not in cache:
        per = []
        for s in V.SETS:
            T, m = tr[s][0].shape[0], 0
            for L in range(V.NL):
                t_, x = V.Tier(C, d), tr[s][L]
                for t in range(0, T, A):
                    a = t_.call(list(dict.fromkeys(x[t:t + R, :top].reshape(-1).tolist())))
                    if t >= V.T0:
                        m += a
            per.append(m / sum(min(A, T - t) for t in range(0, T, A) if t >= V.T0))
        cache[key] = np.array(per)
    return cache[key]


def mb(tr, Cb, Cr, bb, br, proto):
    """MB / token (mean over the 4 sets, as item (a)) and the per-set values"""
    per = (misses(tr, Cb, 10, *proto) * bb + misses(tr, Cr, 2, *proto) * br) / 1e6
    return float(per.mean()), per


def best(tr, budget, bb, br, crs, proto):
    out = []
    for Cr in crs:
        Cb = (budget - Cr * br) // bb
        if Cb >= 10:
            out.append((mb(tr, Cb, Cr, bb, br, proto)[0], Cb, Cr))
    return min(out), out


def main():
    t0 = time.time()
    tr = {s: [np.load(os.path.join(V.TR, f"{s}_800_g.npz"))[f"L{L}"][:, :10].astype(np.int64) for L in range(V.NL)] for s in V.SETS}
    print(f"[k1] records per expert: v3 MSB {MSB:,} + LSB {LSB:,} B | T3R base {BASE:,} ({BASE_RAW:,} unpadded, "
          f"{BASE / MSB:.3f}x MSB) + R1 {RES:,} ({RES_RAW:,}, {RES / LSB:.3f}x LSB) | rank 0-1 bits / weight: v3 "
          f"{(MSB + LSB) * 8 / 4915200:.3f}, T3R {(BASE + RES) * 8 / 4915200:.3f}", flush=True)
    v, _ = mb(tr, 147, 46, MSB, LSB, DEC)
    print(f"[k1] check vs item (a): v3 24 GB 16-row / accept 10 / LFU 0.99 = {v:.1f} MB/token (item (a) 115.9)", flush=True)
    assert abs(v - 115.9) < 0.1
    for proto, theirs in (((1, 1, 0.0), 366.8), ((1, 1, 0.99), 330.5), ((4, 3, 0.0), 368.6), ((4, 3, 0.99), 314.8)):
        v, _ = mb(tr, 51, 20, MSB, LSB, proto)
        print(f"[k1] cross-check vs NewIdeas b10_12gb.txt, v3 51 / 20, rows {proto[0]} accept {proto[1]} "
              f"{'LRU' if not proto[2] else 'LFU ' + str(proto[2])}: mine {v:.1f} vs theirs {theirs} ({100 * (v / theirs - 1):+.1f} %)", flush=True)
    for nm, cm, cl, decisive in BUDGETS:
        budget = cm * MSB + cl * LSB
        crs = range(max(2, cl // 4), 3 * cl + 1)
        v3, v3per = mb(tr, cm, cl, MSB, LSB, DEC)
        (vb, cmb, clb), _ = best(tr, budget, MSB, LSB, crs, DEC)
        (tb, cb, cr), grid = best(tr, budget, BASE, RES, crs, DEC)
        _, tper = mb(tr, cb, cr, BASE, RES, DEC)
        (tu, cbu, cru), _ = best(tr, budget, BASE_RAW, RES_RAW, crs, DEC)
        cut = 100 * (1 - tb / v3)
        edge = " (EDGE of the C_r grid)" if cr in (crs[0], crs[-1]) else ""
        print(f"\n[k1] {nm}: {budget / 1e6:.1f} MB / layer ({48 * budget / 1e9:.2f} GB of slots), 16-row / accept 10 / LFU 0.99", flush=True)
        print(f"  v3 as configured {cm} / {cl}: {v3:.1f} MB/token | MSB misses / token {misses(tr, cm, 10, *DEC).mean():.1f}, "
              f"LSB {misses(tr, cl, 2, *DEC).mean():.1f} | per set " + " ".join(f"{s} {x:.1f}" for s, x in zip(V.SETS, v3per)), flush=True)
        print(f"  v3 best split in this budget: C_m {cmb} / C_l {clb}: {vb:.1f} MB/token ({100 * (vb / v3 - 1):+.1f} %)", flush=True)
        print(f"  T3R best: C_b {cb} ({cb / cm:.3f}x C_m) / C_r {cr}{edge}: {tb:.1f} MB/token | base misses / token "
              f"{misses(tr, cb, 10, *DEC).mean():.1f}, R1 {misses(tr, cr, 2, *DEC).mean():.1f} | per set "
              + " ".join(f"{s} {x:.1f}" for s, x in zip(V.SETS, tper)), flush=True)
        print(f"  T3R vs v3 as configured: {cut:+.1f} % fewer bytes / token; vs v3's best split {100 * (1 - tb / vb):+.1f} %; "
              f"per set " + " ".join(f"{s} {100 * (1 - a / b):.1f}" for s, a, b in zip(V.SETS, tper, v3per)), flush=True)
        print(f"  T3R with unpadded records (side): C_b {cbu} / C_r {cru}: {tu:.1f} MB/token ({100 * (1 - tu / v3):.1f} % fewer)", flush=True)
        print("  T3R by C_r: " + ", ".join(f"{c} {x:.1f}" for x, _, c in grid[::max(1, len(grid) // 12)]), flush=True)
        if decisive:
            print(f"DECISIVE K1 {nm}: T3R {tb:.1f} vs v3 {v3:.1f} MB/token = {cut:.1f} % fewer (kill < 20 %) -> "
                  f"{'KILL' if cut < 20 else 'PASS'}", flush=True)
            for proto in ((1, 1, 0.0), (1, 1, 0.99), (16, 10, 0.0)):
                a, _ = mb(tr, cm, cl, MSB, LSB, proto)
                (b, cb2, cr2), _ = best(tr, budget, BASE, RES, crs, proto)
                print(f"  side protocol rows {proto[0]} accept {proto[1]} {'LRU' if not proto[2] else 'LFU ' + str(proto[2])}: "
                      f"v3 {a:.1f}, T3R {b:.1f} (C_b {cb2} / C_r {cr2}) = {100 * (1 - b / a):.1f} % fewer", flush=True)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
