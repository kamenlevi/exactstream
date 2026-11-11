"""First-principles time model of a 125B verify pass on the M5 Pro (64 GB, SSD 14.6 GB/s), calibrated to measurements,
to see which stack of levers could reach the goals (100-150 tok/s at 6-12 GB, 50-75 tok/s on a 16 GB / 3 GB/s mini).
Per pass (MTP depth 3: 4 rows, 3.1 committed tokens/pass on the trace prompts):
  T = compute + between + stall_visible
  compute   46 ms at 34 GB (pass_profile125: 48 % of 96 ms, ~130 GB/s); of it routed GLU ~20 ms (48 x 0.42 ms, stock M=4)
  between   17 ms (inter-pass: final mixer, lm_head, sampling, acceptance, MTP drafts, CPU syncs)
  stall     misses/pass x 0.30 ms (31-34 ms at 107 LRU misses/pass, frac 0.5); misses from the trace sim (LRU / OPT)
Calibration check: C=128 (17 GB) predicts 141 ms/pass (22 tok/s) vs 23.7 measured (depth 2); C=96 (12.7 GB) 168 ms
(18.5) vs 19.8 at 13 GB. Levers (multiplicative on their term):
  3c      routed GLU -25 %             (bench_tierglu, bitwise)          -> compute -5 ms
  t3      3-bit tiers: miss bytes x0.78 (MSB part 2.15 + LSB 0.77 for 2 of ~10 ranks vs 2.765 MB)
  lip     LIP-4 insertion: misses x0.96 (insert_policy125)
  ooo     out-of-order experts: hides min(stall/layer, 0.3 ms) per layer (ooo_bound125)
  opt     Belady-class caching: misses x OPT/LRU (opt_bound125; the unreachable floor, ~x0.6)
  eff     compute at the read ceiling + dependent-kernel floor: 17.7 ms bytes + 48 x 12 x 20 us = 29.2 ms
  thin    inter-pass work 17 -> 4 ms (everything on the GPU, no CPU syncs)
  hide    all SSD traffic overlapped: stall_visible = max(0, SSD time - compute - between)
usage: pass_model125.py   (CPU, instant)"""
MISS = {  # misses per pass, 4-bit experts, from data/opt_bound125.txt (LRU, OPT)
    4.2: (698.0, 611.7), 8.5: (483.2, 309.2), 12.7: (351.3, 202.4), 17.0: (261.2, 145.9), 25.5: (158.7, 91.6), 34.0: (107.2, 67.5)}
TOK = 3.1
SSD_MS_PER_MB = 1 / 14.6


def tps(gb, levers, tok=TOK, ssd_gbs=14.6, scale=1.0):
    """scale: expert-union growth vs 4 rows (misses and expert compute both scale with it)."""
    lru, opt = MISS[gb]
    miss = (opt if "opt" in levers else lru) * scale
    if "lip" in levers and "opt" not in levers:
        miss *= 0.96
    mb = 2.765 * (0.78 if "t3" in levers else 1.0)
    routed = 20.0 * scale * (0.75 if "3c" in levers else 1.0)          # routed GLU ms (stock 4-bit ~20 ms at 4 rows)
    # eff: bytes at the ~300 GB/s read ceiling + the dependent-kernel floor (matvec law: ~20 us per dependent big kernel,
    # ~12 per layer -> 11.5 ms/pass; round 29: cutting overlapped small kernels buys nothing)
    compute = (2500 + 2780 * scale) / 300 + 11.5 if "eff" in levers else 26.0 + routed
    between = 4.0 if "thin" in levers else 17.0
    stall_per_miss = 0.30 * (0.85 if "t3" in levers else 1.0)          # latency-dominated: bytes help less than x0.78
    stall = miss * stall_per_miss
    if "ooo" in levers:
        stall -= min(stall, 48 * 0.3 * min(1.0, stall / (48 * 0.65)))   # per layer min(stall_L, 0.3 ms), stall_L ~ p50
    if "hide" in levers:
        ssd = miss * mb / ssd_gbs                                        # ms (MB / GB/s)
        stall = max(0.0, ssd - compute - between)
    T = compute + between + stall
    return 1000 * tok / T, T


STACKS = [("today", set()), ("+3c+t3", {"3c", "t3"}), ("+3c+t3+ooo+lip", {"3c", "t3", "ooo", "lip"}),
          ("+...+eff+thin", {"3c", "t3", "ooo", "lip", "eff", "thin"}),
          ("+...+eff+thin+hide", {"3c", "t3", "lip", "eff", "thin", "hide"}),
          ("+...+eff+thin+hide+opt", {"3c", "t3", "eff", "thin", "hide", "opt"})]
print("tok/s (ms/pass) at 3.1 tokens/pass, M5 Pro SSD 14.6 GB/s; cache budget = experts only (+~2.5 GB dense +KV)")
print("stack".ljust(28) + "".join(f"{g:>14.1f} GB" for g in MISS))
for name, lv in STACKS:
    print(name.ljust(28) + "".join(f"{tps(g, lv)[0]:9.1f} ({tps(g, lv)[1]:5.0f})" for g in MISS))
print("\n16 GB Mac mini (SSD 3 GB/s), full stack incl. hide (+opt in the 2nd row):")
for lv in ({"3c", "t3", "lip", "eff", "thin", "hide"}, {"3c", "t3", "eff", "thin", "hide", "opt"}):
    print("   " + "".join(f"{g:>5.1f} GB: {tps(g, lv, ssd_gbs=3.0)[0]:5.1f} tok/s   " for g in (4.2, 8.5)))
print("\nTokens/pass needed for 100 tok/s with the full stack + opt at 12.7 GB and 34 GB (union grows ~k^0.67, so bytes/pass x(k/4)^0.67):")
for tok in (3.1, 4, 6, 8, 12):
    k = tok * 4 / 3.1
    sc = (k / 4) ** 0.67
    print(f"   {tok:4.1f} tok/pass (~{k:4.1f} rows): {tps(12.7, {'3c', 't3', 'eff', 'thin', 'hide', 'opt'}, tok=tok, scale=sc)[0]:6.1f} tok/s"
          f" | 34 GB {tps(34.0, {'3c', 't3', 'eff', 'thin', 'hide', 'opt'}, tok=tok, scale=sc)[0]:6.1f}")
