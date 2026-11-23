# CHECK25 data (Leap): the lossy 32-row check at <= 25 ms (plan: data/week98/check25_plan.md)

## BITS + ACCURACY (Mon 16 Nov 12:15-12:16, one share lock beside EvalSuite's shared run, no timing; swap +0)
- **qmm32n** (scripts/week98/qmm32n.py check -> qmm32n_check.txt): **110 / 110 bitwise == two 16-row qmm16n_m calls**
  on all 11 FAST dense shapes, at DN-KS v1's ks (M 17 / 20 / 24 / 31 / 32 x 2 seeds). So it is a FAST-safe one-pass
  form, with no new rounding. Error vs fp32 = the 16-row form's own (3.1-4.3e-3 of max |ref|: the bf16 output rounding).
- **dn32 / dn32q** (scripts/week98/dn32.py check -> dn32_check.txt):
  - dn32 (bf16 x uint4 from threadgroup memory) has qmm32n's error on every shape, so the layout and nibble order are
    right.
  - **dn32q (int8 activations x int4) is ~2x it: 6.2-7.7e-3 of max |ref|.** This is the lossy part (EvalSuite's int8
    gate).
- **The FAST-NAX one-call at 32 rows** (scripts/week98/ooo1c_m32_check.py -> ooo1c_m32_check.txt; layers 20 / 44's
  converted pools, BIGMOE_MAXK 320): **24 / 24 scenarios bit for bit == two 16-row calls.**
  - Covered: COH on / off; random routing; an expert at rank 0 of all 32 rows (32 pairs -> two 16-pair groups); two
    popular experts; 2-D nolsb marks; late slots polled from the release region.
  - The base posts cover every used expert.
  - The grouping already starts a new group every 16 pairs of one key (tierglu_nax6._GRP), so there is no D8-class
    drop.

## SPEED, the 32-row dense matmuls (Mon 16 Nov 12:22:20-12:22:52, QUIET lock granted after 1 min while EvalSuite's runner waited; 32 s; swap +0)
scripts/week98/qmm32n_bench.py both (log: q1.log; runner check25_q1.sh). Per FAST dense shape, the per-check set of
distinct random native 4-bit weights (cold), 7 interleaved rounds, median of 6.
- **'ind' (calls independent in one graph) is the throughput reference.**
- **'chain' (each call after the previous one through a link op) reads 1.1-1.3x 'ind' on most shapes but 2.6x on GDN
  qkv|z** (16-row 242 vs 106 us; the engine's own 16-row chain is 113.6, TABLE_350). So chain absolutes are suspect;
  use it only for form ratios.
- The 1-call shapes (PLE key / value, the head) read ~200-300 us: the per-round sync, not the kernel.

**Totals per 32-row check, ms ('ind'):**

| 16-row (A16's cost) | two 16-row calls (today's > 16 FAST route) | qmm32n at DN-KS ks (bitwise) | qmm32n best ks | dn32 best | dn32q + its quantizer | dn32q pre-quantized | floor max(bytes, matrix@32) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11.35 | 19.30 | 15.27 | 14.12 | **13.20** | 12.76 | 11.21 | 6.90 |

**The big shapes, us per call ('ind'):**

| shape | x per check | 16-row | two 16-row | qmm32n | dn32 | dn32q+q | bytes / matrix@32 floors |
|---|---:|---:|---:|---:|---:|---:|---|
| GDN qkv\|z 16384x2560 | 36 | 106.2 | 213.4 | 153.5 | **133.8** | 126.3 | 84.3 / 95.9 |
| out / o_proj 2560x6144 | 48 | | 78 | **57.0** | 58.1 | 51.3 | 31.6 / 36.0 |
| att q 12288x2560 | 12 | | 172 | 121.0 | **118.1** | 104.1 | 63.2 / 71.9 |

**Readings:**
- **The one-pass forms cut today's > 16-row route by 4.0-6.1 ms per check** (19.30 -> 15.27 / 13.20). dn32 (no ALU
  decode) beats qmm32n by ~13 % on GDN qkv|z.
- **Pre-registered line (GDN qkv|z <= 115 us PASS, > 146 KILL):**
  - qmm32n 153.5: **KILL** as the 32-row kernel (it stays as the bitwise reference form);
  - dn32 133.8: BETWEEN;
  - dn32q pre-quantized 119.7: BETWEEN, near PASS.
  - **dn32 is the one to wire (FAST)**, dn32q for the C6 + int8 number.
- **Even the best forms sit at 1.3-1.6x max(bytes, matrix floor) on the big shapes.** The rest is per-kernel latency
  and occupancy, which needs tuning (more simdgroups per A fragment, double-buffered staging).
- **The small shapes are 3-34x their floors** (b|a 16 us, k / v 24, indexer 39-42, shared gate|up 22, shared down 14;
  ~3.9 ms per check in all). That is fixed cost per dispatch: they need folding (a|b into qkv|z, k|v into q, router +
  shared as one stream), not faster kernels.

## INT8 DENSE (G1 = C6 + int8, EvalSuite lossygate amendment 5; Mon 12:30-12:34, share locks, bits only)
scripts/week98/int8d.py (installer) + dn32.py (kernels now 1-32 rows, native staging). int8d.py check -> int8d_check.txt:
- **Error vs fp32** (max |err| / max |ref|): int8 6.6-7.9e-3 vs bf16 2.9-3.9e-3 on all 11 FAST dense shapes, M 1 / 2 / 7
  / 16 / 17 / 24 / 32.
- **Row-invariant:** every row of an M-row int8 call == its 1-row call, bit for bit, 36 / 36 per shape.
- **Native staging:** dn32q / dn32 reading DENSE-N-RAM's native words (staged in the kernel) == the copy-staged forms bit
  for bit, 14 / 14 per shape. So int8 needs **no second weight copy** (MEM unchanged; the +1.7 GB I first quoted is gone).
- **Scope (confirmed by NewIdeas 12:29):** CODE prompts only, via a runtime switch (int8d.ON[0]) that BigMoE's screen sets
  per item (R57_INT8D_SETS). The bf16 path's bits are untouched; IFEval stays token identity.

## FUSED 1c-TIERS KERNEL (Mon 12:45, share lock, bits only)
scripts/week98/t1cf.py (t1cf_check.txt): one dispatch per MoE layer for BigMoE's _t1c (tiers_mx, the per-expert counter,
the 2-D nolsb) + _fast_call's km lines (~28 ops -> 1).
- **Equal to BigMoE's path on 384,000 / 384,000 slots, all five outputs** (km'd scores, km'd ids, nolsb, counts, tier):
  1,600 random checks at M 16 / 32, uint32 and int32 ids.
- I_e is summed in a fixed row-major order. BigMoE's scatter-add order is unfixed, so a tie at an fp32 threshold could in
  principle differ; none did.
- Wired into the c25 copy as arm F32df (scripts/week98/c25/patch_c25_t1cf.py; opt-in C25_T1CF).
- For the real loop it needs BigMoE's harness (scripts/test_t1c_wiring.py, 3.1 M slots), an opt-in flag and a new
  snapshot after EvalSuite's gate (BigMoE's conditions, 12:0x).

## CHUNKED GDN v1 (Mon 12:58, share locks): CORRECT but 3x SLOWER, parked
scripts/week98/gdn32c.py (one threadgroup of 512 per v-head; log-space decays; fp32 on the ALU):
- **Correctness** (gdn32c_check.txt) vs mlx_lm's sequential kernel: y relative error <= 1.8e-3 (below bf16's ~4e-3
  rounding), state ~1e-7..1e-6, all finite, at T 17 / 24 / 32, typical and strong decays (the latter would underflow a
  plain decay product).
- **Speed** (gdn32c_bench.txt; ROUGH, a share lock beside a 20 %-GPU share run): 332.9 us per layer vs mlx_lm's 108.2
  at T 32. So **KILL as built.**
- Likely causes:
  - only 48 threadgroups per layer, against the sequential kernel's 1,536;
  - per-thread arrays (dl / w / s0, 96 floats) indexed in loops, which likely spill;
  - the serial forward substitution.
- A redesign needs more parallelism per head (threads over (dv, t)) and the matrix units for the C x C x D products.
  It is out of the c25 arms.

## ROUGH paired timings (Mon 12:59, share lock beside EvalSuite's ~40 GB process at ~60 % GPU + MoonShots' bit tests; ratios only)
- **hc8r32** (hc8r32_bench.txt), 96 HC reads per check, ms:

  | mode | 16-row | two 16-row | read32 |
  |---|---:|---:|---:|
  | ind | 7.94 | 14.83 | 8.04 |
  | chain | 9.96 | 13.53 | 11.01 |

  The one-pass read costs about one 16-row read: **-2.5..-6.8 ms per 32-row check** vs today's split (rough).
- **Native staging vs copy staging** (GDN qkv|z, us per call):
  - dn32: 146.8 native vs 136.3 copy (+8 %);
  - dn32q + quant: 143.2 native vs 138.9 copy (+3 %).
  - So the no-copy form (int8d, f32one's dn32) costs ~0.4-0.8 ms per check for 1.7 GB less memory. A faster
    staging (lane shuffles into column runs, then uint4 stores) is a later fix.

## SLICE TEST (Mon 13:04-13:05, EvalSuite's SIB pause, share lock; 40 s, peak 8.5 GB, swap +0): slice1.log
f32one_test.py on a 4-layer slice (G G G A) of the 125B, FAST stack as lossy-45 sets it (DNR, HC8 + hc8_fast, att_single,
SVH 98,304, K-73 / K-83 / HCB forms).
- **G1's form (int8d at the real loop's 16 rows) PASS:**
  - int8 vs bf16 logits max |diff| 0.227, argmax 16 / 16;
  - 29 int8 calls + 16 quantizer dispatches per forward;
  - the commit replay with int8 is consistent: keep 1 / 9 / 15, then a 4-row probe, == a k-row int8 check bit for bit.
  - Sent to BigMoE 13:1x (its G1 key test).
- **f32one (qmm32n, bitwise forms) == v32's 2 x 16 split on every logit both heads compute** (max |diff| 0 at M 32 / 24 /
  17). The "rows bitwise 0 / 32" and the argmax 22 / 32 are a test artifact: v32 alone has no 17-32-row SVH, so its 32-row
  head is the FULL head, while f32one keeps SVH. A 4-layer slice's argmax often falls outside SVH's subset.
  - The commit test with f32one: keep 1 / 9 / 16 / 17 / 24 / 31 probes bitwise 4 / 4.
- **dn32 (D):** max |logit diff| 0.14-0.19 vs the bitwise forms.
- **int8 at 17-32 (I vs D):** 0.22, argmax 31 / 32, 24 / 24, 17 / 17.
- **Chunked GDN (C vs D):** 0.80-1.09, argmax 28 / 32. Parked anyway (3x slower).
- **LAUNCH PROXY** (non-view graph primitives of one verify forward on the slice; per-check dispatches differ, a proxy):
  - A16 391;
  - 32-row split (v32) 640 (CustomKernel 142, Concatenate 74);
  - one-pass f32one 580 (CustomKernel 75: about half the custom launches, but MTPLX's eager GDN / attention glue at 32 rows
    adds elementwise ops: Multiply 47, AsType 41). The next diet target: the GDN prework's fused 2-16-row kernel
    (gdn_exact) has no 32-row form.

## CLEAN TIMINGS q3 (Mon 13:11:57-13:12:39, QUIET lock, granted after MoonShots released its idle share lock; swap +0)
- **Dense matmuls per 32-row check, ms ('ind'):**

  | 16-row | split | qmm32n (bitwise) | qmm32n best | dn32 | dn32 native | dn32q + quant | dn32q native + quant | dn32q pre-q | floor |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 11.88 | 19.86 | 15.11 | 13.91 | **12.97** | **12.97** | 12.68 | 12.86 | 11.22 | 6.90 |

  - Native staging is free when quiet; the earlier +3-8 % was load noise.
- **hc8r32**, ms per check (96 reads):

  | mode | 16-row | two 16-row | read32 |
  |---|---:|---:|---:|
  | ind | 5.02 | 10.52 | 6.32 |
  | chain | 6.43 | 9.26 | 8.14 |

  The one-pass HC read saves 1.1 ms (chained) to 4.2 ms (independent) per 32-row check vs today's split.
- **gdn32c:** mlx_lm's step 107.2 us per layer (3.86 ms per check at T 32) vs the chunked v1 334.3. **3.1x slower: KILL
  confirmed.**
- **exp32 (experts) FIRST RUN INVALID:** its stand-in post did np.array(ids) per call (a GPU sync per call; the engine's
  posts are async), so 765-806 us per call (~2.6x the floor) measures the harness, not the kernel. Fixed with
  precomputed sync-free posts (v2); re-run next. The ks 8 variant fails to build (dropped).

## SLICE RE-TEST (Mon 13:14, 25 s): f32one's 17-32-row GDN prework through gdn_exact's fused kernel
- O (f32one, now with the fused prework at 17-32 rows) == v32's split on every shared logit at M 32 / 24 / 17 (max |diff|
  0); the commit test is bitwise at keep 1-31.
- **Launch proxy:** O's verify forward 580 -> **523** graph primitives on the 4-layer slice (-57, ~19 per GDN layer). Per
  check that is ~-680 dispatching ops (proxy).

## ENGINE A/B run2b (Mon 13:24:57-13:25:57, QUIET; REAL engine, lab checks: G32T TIME MODE on 3 code windows with true-state tokens, C 160 with every expert resident, misses 0; 2 warm-up + 12 rounds x 9 arms, order rotated; data/week98/c25/run2b_*)
| arm | what | median ms per check | build+submit / wait / commit | expert GB (upper) | vs F16 |
|---|---|---:|---|---:|---:|
| F16 | today's 16-row check | 38.88 | 34.90 / 3.64 / 0.34 | 1.89 | 1 |
| F32s | 32 rows, today's route (v32 split) | 68.18 | 60.51 / 7.22 / 0.44 | 3.12 | 1.741 |
| F32o | + f32one one pass (qmm32n) | 58.77 | 52.34 / 6.01 / 0.35 | 3.15 | 1.498 |
| F32d | + dn32 native | 58.06 | 51.77 / 5.96 / 0.35 | 3.12 | |
| F32df | + t1cf fused tiers | 58.42 | 52.08 / 6.01 / 0.36 | 3.12 | 1.497 |
| F32if | + int8 dense | 59.62 | 53.08 / 6.12 / 0.35 | 3.12 | |
| F32if6 | + C6 lambda 0.018 | 58.32 | 51.90 / 6.07 / 0.35 | 2.92 | 1.489 |
| F32c5 | F32df + V3 (hc4i) + V1 (16K head) emulation = C5 | 56.65 | 51.16 / 5.09 / 0.34 | 3.19 | 1.448 |
| F32c6i | C5 + C6 + int8 | **56.41** | 50.95 / 5.05 / 0.35 | 2.96 | **1.440** |

Paired diffs (36 rounds, median [IQR]):
- F32o - F32s **-9.46 [-9.67, -9.25] ms**: the one-pass form.
- F32d - F32o -0.75.
- F32df - F32d +0.49: the fused tiers are slower in the engine. Open.
- F32if - F32df +1.13: int8 is slower in the engine; the kernel bench said -0.3. Open.
- F32if6 - F32if -1.20.
- F32c5 - F32df -1.90.
- F32c6i - F32c5 -0.25.

**Floors:**
- C5 arm: 5.32 GB per check on these windows (dense 2.13 + experts 3.19, upper) -> 19.0 ms; measured 56.6 = **2.98x**.
- C6 + int8: 5.09 GB -> 18.2 ms; 56.4 = **3.10x**.

**The 25 ms goal is 2.26x away.**

**Wall split:** build+submit includes GPU waits (the engine syncs inside a check; CHECK55 showed this for the exact check), so it is NOT a CPU share. The CPU-vs-GPU split needs a libmtlprof pass (next slot).

**Bits** (first timed round; (rows bitwise, argmax rows, rows, max shared-logit diff)):
- F32df == F32d bitwise in every row: t1cf is exact.
- F32o vs F32s: argmax 94 / 96, max 2.7-6.7. F32s's first 16 rows vs F16: 47 / 48, 2.4-3.2.
- Why: the 1c tiers give each expert ONE tier per check (min over its slots: `texp ... .minimum(tier)`). Row grouping changes tiers by design, and any tiny upstream difference can flip a tier.
- So these A/Bs can't call f32one bitwise at full depth; an exact-mode (tiers off) A/B can. The 4-layer slice is bitwise on shared logits.
- F32c5 vs F32df argmax 69 / 96: my V1 emulation is FR-Spec's general top-16K list, not the code list. Timing only.

**exp32 v2** (the post, same quiet window; sync-free posts; layer 20's pools; real 1c routing; 24 blocks; 44.4 kept experts per 32-row block; 0.0826 GB per call):
- 32 rows: 571-614 us per call = 135-145 GB/s = **1.94-2.08x the byte floor** across ks 1/2/4, nsg_b 1/2/4 and COH on/off (+-4 %).
- 16 rows: 496-498 us (144 GB/s, also 2x).
- So the 2x is structural, not a split choice. 48 layers x ~0.58 ms = ~28 ms at the map's bytes.

## CORRECTION (Mon 13:4x): the expert bytes were priced 23-30 % low. The 25 ms check is under its own bound at C5 / C6
- **The constant:** every check budget priced an expert at 2.135 MB (4-bit) / 1.661 MB (MSB-3 plane):
  - Leap's cheapcheck.md (11-14), pipe_price, treek_price, check25_bound;
  - c25_engine's "expert GB";
  - exp32_bench;
  - Fit12's lossy32_price (E4, E3);
  - Breakthrough's LOSSY_CHECK_PRICE.
- **What the engine reads** (records3 native+p11+pad pieces):
  - MSB part 2.064 MB (3 x w3 614,400 B + 6 x s3 / b3 36,864 B);
  - LSB part 0.720 MB (3 x l1 204,800 + 3 x b 35,216);
  - so **4-bit 2.784 MB, MSB-3 2.064 MB (x1.304 / x1.243)**.
- **Three checks agree:**
  - records3/index.json (layer 20);
  - the c25 engine's own pools: C_m 175 + C_l 112 per layer = 21.21 GB over 48 layers = 175 x 2.0644 + 112 x 0.7209 MB per layer;
  - config.json: 3 x 640 x 2560 at 4-bit g64 = 2.765 MB in MLX's own format.
- **Re-priced** (scripts/week98/check25_bound_real.py -> data/week98/check25_bound_real.txt): C5's experts 3.99 -> **5.05 GB** per check, C6's 3.63 -> **4.60 GB**.
  - Bound at 100 % of every peak: C5 26.8-27.7 ms, C6 + int8 25.2, C6 + V4 + int8 24.0 (96 % needed).
  - For 25 ms at 85 %: experts <= 3.83 GB per check (with V4 + int8 dense).
- **Measured efficiencies, re-stated:**
  - exp32's one-call: 105 MB per 32-row call (not 83) at 571-614 us = 171-184 GB/s = **1.52-1.64x** the 280 GB/s floor (was "2x").
  - The engine's F32c6i: ~5.9 GB per check -> floor 21.0 ms; 56.4 ms measured = **2.69x**.
- **Exact floors re-priced** (dense 3.44 GB; 2.784 MB per 4-bit expert; 280 GB/s, (295)): 1 row 15.9 -> 17.1 ms;
  16 rows (union 72.5) 38.8 -> 46.9 (44.5); 32 rows pooled (union 108.5) 52.0 -> 64.1 (60.8); 32 rows code (union
  95.7) 47.3 -> 57.9 (55.0).

## THE ONE-CALL READS 27 % EXTRA EXPERT BYTES (Mon 14:1x, CPU count: scripts/week98/groups_count.py)
- **Groups vs experts:** the lossy one-call (ooo1cq) groups pairs by (expert, tier). On the decision map's 32-row blocks
  of real 1c routing (12 code windows, all 48 layers, 4,752 block-layers): 46.5 distinct experts but 60.4 groups; 13.9
  experts are in two groups (4-bit rows + MSB-3 rows), so their 2.06 MB MSB plane is read twice.
- **Bytes:** group reads 134.9 MB vs 106.1 MB unique = **x1.27** (x1.23 from mixed tiers of kept uses; 2.1 groups per
  block-layer exist only for weight-0 drops); x1.22-1.36 by layer.
- **Plus the 16-pair cut:** every group is also cut into segments of <= 16 pairs, each reading its weights again. That
  gives 62.7 segments per block-layer -> 139.6 MB = **x1.315 of the unique bytes**.
- **The merged kernel's target:** one decode per expert for both tiers and up to 32 rows (two 16-row MMA tiles from one
  read, as read32 / qmm32n do), i.e. ~-24 % of expert traffic.
- **Reading exp32's 1.52-1.64x:** ~x1.27 from these re-reads (if they go to DRAM) x ~1.2-1.3 streaming.
  - The exact engine's experts (one group per expert, all 4-bit) stream at ~290 GB/s on real sizes (CHECK55 engine
    pass K: 33.71 / 50.65 ms for 9.78 / 14.66 GB).
- **Next:**
  - Tue slot #1's exp32 nlv measures whether the second reads cost DRAM or hit the SLC: all-MSB-3 = one group per
    expert, vs the real mask.
  - If DRAM: a merged-group kernel (MSB read once, both tiers' rows from it, same arithmetic per row; bitwise) saves
    ~21 % of expert time, about -4 ms per 32-row check.
- **HC read, K-split sweep (ROUGH, beside V1's 60 % GPU; hc_np_bench.py):** read32 per check at NP 20 / 40 / 80 / 160 =
  9.64 / 9.47 / 11.25 / 13.93 ms (ind), 12.31 / 12.10 / 13.79 / 16.23 (chain); NP 40 drifts 1 bf16 ulp (max rel 3.6e-3).
  - More K1C threadgroups don't help, so the read's ~2.8x over its 2.25 ms byte floor is the fixed cost of three
    small dependent kernels per read (~65 us quiet x 97 reads).
  - Option: fold K2B into K1C's last threadgroup (-1 launch per read, ~-0.5 ms per check).
  - Quiet re-check in Tue slot #1's post.
- **ROUGH nlv (Mon 14:16, beside V1's 60 % GPU, +-10 %; exp32_bench.py nlv):**

  | mask | us per call | GB unique |
  |---|---:|---:|
  | real | 767 / 844 | 0.1045 |
  | all MSB-3 (one group per expert, no LSB) | 676 | 0.0945 |
  | all rank 0-1 at 4-bit | 920 | 0.1148 |

  - Fit t = a x unique + b x re-read: a ~ 140 GB/s per unique byte; a re-read costs only ~1/4 of a DRAM byte (mostly SLC
    hits: the two tier groups of an expert are adjacent in the dispatch order).
  - So the merged-group kernel would save ~5-8 % of expert time (not 16-24 %): ON HOLD.
  - The real gap is the one-call's base streaming: ~140 GB/s here even with one group per expert, vs ~290 for the exact
    path (CHECK55, quiet).
  - Tue slot #1 (quiet + engine pass K) re-measures; then the target is that base rate (x / h re-reads, the head and
    grouping stages, kernel A -> B).
- **gdn_fast through the real GDN path: PASS** (Mon 14:23, 4-layer slice, gdnfast_slice_test.py). O + FASTGDN == O on
  every logit and every cache state at M 32 / 24 / 17, and through the commit replay (keep 1 / 9 / 16 / 17 / 24 / 31).
- **hc4 slice test, run 1 FAILED in its reference, not in hc4i:**
  - With hc4i off, ref2 sends 4-bit mixers to leapk.hc_read (built for bf16 mixers): a Metal build error.
  - So "the eager read" needs ref2 / f32one off for that call. The test is fixed (MODE ref / cmp); re-run after pkslice.
  - In the engine, hc4i served every read (F32c5 ran).
- **V3 slice test (Breakthrough's hc4 switch 210d95d over hc4i), Mon 14:3x, 4-layer slice; hc4_slice_run4.log:**
  - **Switch mechanics PASS:**
    - engage(True) swaps every mixer.
    - hc4i serves every read: 8 per verify at 16 rows (hc4r.read) and at 32 rows (read32), over v32 and over f32one.
    - engage(False) restores the originals bit for bit (16 / 16 rows).
    - Never engaged: hc4i passes every 8-bit read through.
  - **Numerics vs MTPLX's eager read of the same 4-bit modules (ref2 / f32one off for that call):**
    - mixed rel 0.64 % (16 rows) / 1.01 % (32, v32) / 0.94 % (32, f32one); written bitwise; inject 1.03 %.
    - NOT the inject's bf16 rounding: the eager read with the bf16-dequantized inject gives the same numbers.
    - **The FAST stack's own HC8 read (ref2's hc8_fast, gated in lossy-45) vs ITS eager 8-bit read: mixed 0.93 %, inject
      0.93 %, written bitwise.** So hc4i matches the FAST HC form's own deviation; the test's 1.8e-3 bar was the
      wrong yardstick (hc4r vs hc8r's form).
  - **Slice logits** (hc4i vs every HC read eager): argmax 16 / 16, 29 / 32 (v32), 32 / 32 (f32one); max |diff| 0.18-0.50.
  - **Caveat for the map:** bt_mix4's numerics are the eager read, which is ~1 % from what hc4i computes, the same gap
    HC8 has today.

## CPU SPLIT, slice preview (Mon 18:05, 4-layer slice, share lock beside Breakthrough's map screen; cpusplit_slice.py -> cpusplit_slice.txt)
> (a quote from the project owner, kept private)
could remove, or the CPU waiting on the GPU? c25_cpusplit.py splits the forward's wall into Python build / MLX encode
(thread CPU inside mx.async_eval / mx.eval) / blocked in eval / other waits.
- **At 32 rows the slice's forward submits NOTHING** (0 async_eval). build + submit = Python build only: 2.17 ms (F32s) /
  1.78 (F32o) per 4 layers. The logits' eval then encodes and runs everything (wait 15.30 / 12.55).
- **F16 (16 rows):** SB_FIRST fires (1 async_eval). Python build 2.41, encode 0.36.
- **Dispatches per 4 layers:** F16 304, F32s 565, F32o 383.
- **GPU busy:** 81-84 % of the check (ROUGH: the map screen held ~60 % of the GPU).
- **Scaled to 48 layers [P]:** ~20 ms of Python build + ~12-14 ms of encode (~3 us per dispatch) = **~30-35 ms of CPU per
  32-row check.**
- **The engine's phase shape is the reverse** (build + submit 51, wait 5). So BigMoE's path submits and waits inside the
  forward.
  - Whether those waits are GPU-paced (GPU busy ~50) or serialize CPU with GPU (~35-40) is what tonight's gpub run
    measures (F32c6i / F16 / F32s / F32o, libmtlprof + c25_cpusplit, quiet, after the g1b key test).
- **Either way, a check under ~35 ms needs the CPU side cut:** compile / specialize, or a replayed path
  (SPLIT_1116.md, modules 1a / 1b).

## Module 3 folds: dn32g, the grouped dense dispatch (Mon 21:0x-21:1x, share lock, <= 1.3 GB; scripts/week98/dn32g.py)
- **What:** several 32-row 4-bit dense calls on ONE input in ONE dispatch (dn32 / dn32q forms). Each segment runs
  dn32.py's own kernel text with its own K split (ks), so every tile's arithmetic and K order are those of its separate
  call. One threadgroup = SG simdgroups; a ks-1 segment packs SG tiles per threadgroup (part-filled at its end).
  - Folds: GDN qkv|z (ks 1) + b|a (ks 10); att q (ks 1) + k + v + indexer qk (ks 10 each).
  - These ks values are the engine's own: c25's env sets REF2_DN_KS=1 (ref2._DNKS).
- **Bits: PASS** (data/week98/check25/dn32g_check.txt). grouped == separate on 224/224 (GDN) and 448/448 (attention):
  - forms: dn32 + dn32q, copy + native layouts;
  - M 1, 2, 7, 16, 17, 24, 32;
  - the split-K sum in rounds of 5 partials (1 KB per simdgroup, as in the separate calls) and in one pass.
- **Speed: ROUGH, inconclusive** (data/week98/check25/dn32g_bench.txt). It ran beside others' GPU work (68 % busy), as a
  dependent chain of 48 folds at M 32, int8 native:

  | fold | bitwise form (engine ks) vs separate | all-ks-1 form (FAST) vs the engine's separate |
  |---|---|---|
  | GDN | +8 / +78 us (two runs) | -7 / +4 us |
  | attention | +7 / +28 us | -29 / -44 us |

  - The bitwise form packs the ks-1 tiles 10 per threadgroup (320 threads). Under contention it trails; a quiet bench
    decides.
  - The FAST form (the small calls at ks 1, their dn_ks-off K order) changes those calls' bits. It is a FAST lever:
    adoption = same scores (Kamen's rule).
  - Per check (ROUGH): attention -0.35..-0.5 ms GPU, GDN ~0. Launches: -72 per check (36 GDN + 36 attention), i.e. ~-1.4
    ms of Python if the check is CPU-paced (tonight's gpubB says which).
- **Next:** int8fold.py (the hook: the groups learned from consecutive calls on the same input, served from one
  dispatch). Unit test tonight; engine slice test after Tue's slot #1. Then a quiet bench in Wed's module slot.
- **int8fold.py, the hook (Mon 21:1x; unit test PASS, data/week98/check25/int8fold_test.txt):**
  - On top of int8d, back-to-back int8d-native calls on the same input object are learned as a group. The group's first
    call then runs every member as one dn32g dispatch on int8d's own quantized input; the others return from a cache.
  - Synthetic test, int8d's wrapper over the real quantized_matmul: a GDN-like pair, an attention-like quad and a lone
    call; forward 1 learns, forwards 2-3 fold.
    - MODE bit: 210/210 outputs == int8d's (58 folds, 116 hits, 2 groups).
    - MODE ks1 (FAST): the ks-1 calls are equal, the small calls within 1.8e-3 relative.
  - Engine slice test (~8 GB) after Tue's slot #1. It needs an F32c6i + fold knob in c25_engine (not touched until
    tue1 ends; tonight's try B imports it live).

## Module 4 feasibility: grid barriers inside one kernel (Mon 21:2x, share lock, < 0.3 GB; scripts/week98/gbar_test.py)
Question: can one persistent kernel run the HC read's K1C -> K2B -> K3C with grid barriers instead of three dependent
launches? This is Fable B's STATS-FORWARD goal of one launch per read. Their writer route is closed in my form: the
pending write already lives in the next read's K1C. Data: data/week98/check25/gbar_test.txt.
- **Barrier:** one device atomic counter; spins are capped, so a group that never arrives shows up as a FAIL flag, not
  a hang.
- **Correctness needs device-scope fences.** atomic_thread_fence(mem_device, seq_cst, thread_scope_device) goes before
  the arrive and after the wait.
  - Without them, plain reads of other groups' values are 50-60 % stale.
  - Relaxed atomic reads were wrong 16 of 102,400 times.
  - With the fences, plain reads were 0 wrong of up to 409,600, at every P tried.
- **Co-residency:** 0 spin-cap fails at P 20 / 40 / 60 x 352 and 80 x 128 threads, also with 28 KB of threadgroup
  memory per group (K3C's need is 28.7 KB).
- **Cost (ROUGH, shared GPU):** a fenced grid barrier is 1.41-1.58 us. A dependent kernel boundary is 10.2 us (a chain
  of tiny kernels, Python per launch included).
- **So:** a one-launch HC read is feasible. 2 barriers per read replace 2 launches: ~-2..-3.5 ms GPU [P] and -192
  launches per check (~-3 ms CPU). The kernel: hc8p32 = hc8r32's K1C32 / K2B32 / K3C32 phase code under grid barriers,
  bitwise == read32 by construction.

## Module 4: hc8p32, the HC read in ONE launch (Mon 21:2x-21:3x, share lock, <= 3 GB; scripts/week98/hc8p32.py)
- **What:** one persistent kernel of 40 threadgroups x 352 threads runs hc8r32's read in three phases, with two fenced
  grid barriers between them (gbar_test.py).
  - Phase 1 = K1C32's own text (groups 0-19).
  - Phase 2 = K2B32's own text (groups 0-31).
  - Phase 3 = K3C32, two 32-column items per group, sharing one mix1 staging. The tail runs in 8-row passes with the
    same per-element arithmetic and stream order.
  - The barrier counter is a persistent 4 KB input that the kernel resets at its end (or a zeroed output per call).
  - 4-bit (V3) mixers: hc4r's own load / decode transforms, applied to the same phase texts.
- **Bits: PASS 72/72** (data/week98/check25/hc8p32_check.txt). read32p == hc8r32.read32 (8-bit) and == hc4r.read32
  (4-bit) for (mixed, written, inject) bit for bit, over 3 seeds, with and without the pending write, M 17 / 24 / 32,
  in both counter modes. Spin-cap fails: 0.
  - So it is exact-safe: it changes no bits anywhere it replaces read32.
- **Speed: ROUGH** (others' GPU work at 52 % + Chrome; data/week98/check25/hc8p32_bench.txt). 96 chained reads, each
  on its own cold mixers, ms per check:

  | mixers | 3 launches | 1 launch | delta |
  |---|---|---|---|
  | 4-bit (V3, the C5 / C6 + int8 arms) | 29.92 | 23.54 | -6.38 |
  | 8-bit | 29.75 / 48.16 | 24.14 / 27.36 | -5.61 / -20.8 (two runs; the second read32 looks inflated) |

  Launches go from 288 to 96 per check. Absolute times are inflated by contention: quiet read32 is ~8-12 ms per check.
- **Next:**
  - Engine wiring after Tue's slot #1 (f32one / hc4r / c25 are imported live by tonight's try B and tue1: untouched
    until ~12:00).
  - Slice bits, then the quiet A/B in Wed's module slot.
  - Module 4's line: chained HC read <= 5.0 ms per check (8.14 today in c25).
