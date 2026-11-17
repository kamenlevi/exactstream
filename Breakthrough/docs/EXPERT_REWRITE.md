# EXPERT REWRITE: the experts' MMA / stream overlap on the 32-row lossy check (Breakthrough, from Tue 17 Nov 2026 09:3x)

Owner: Breakthrough. Leap reviews. Assigned by the orchestrator 09:3x (relayed by NewIdeas).

The rewrite is FAST: it adopts on EvalSuite's same-scores gate (the 11-02 two-track rule) and needs no Kamen call. It lands
only as an OPT-IN knob: a module in ~/Breakthrough whose install() wraps `ooo1cq.tier_glu_nax6n_1cq`, with ON = [False].
Leap adds the c25 arm F32c6x = F32c6i + ON. Nothing goes into Leap's tree from my side.

## Where the time goes today (record: Leap SPLIT_1116.md 'Module 2'; MoonShots LEVERS_A 'MODULE 2', b384412 / d50e76a)
- Setting: layer 20, exp32's 24 real 32-row blocks, quiet, per call. The kernels:
  - grouping 19 us;
  - kernel A (gate | up + swiglu) 388 us;
  - kernel B (down) 198 us;
  - combine 9 us.
- exp32's way: 582.6 us wall, 534.3 GPU busy. Per check that is ~23 ms of expert busy time. The byte floor is ~16 ms
  (4.6 GB at ~290 GB/s).
- MoonShots' ablations (8 blocks, GPU us, A / B):

  | variant | A | B |
  |---|---|---|
  | base | 352 | 181 |
  | no code loads | 331 | 169 |
  | no decode | 319 | 166 |
  | no MMA (x loads die with it) | 247 | 125 |
  | pure read of the same bytes | 234 | 115 |

  The kernels are compute-bound: the MMA (~105 / 56 us) runs serially with the stream instead of under it.
- Every bit-exact restructure MoonShots tried was slower (register prefetch, 2 tiles per threadgroup, grid caps,
  B in tile rounds). Its reading: occupancy. M = 8 tiles run padded on the accelerator.
- The orchestrator's target: ~-5.5 ms per check (23 -> ~17.5 ms busy), i.e. about -128 us per call on A + B.

## Leap's terms (09:34)
- Harness: exp32_bench.py, old vs new INTERLEAVED in one process: same blocks, alternating rounds, COH on, at 32 rows
  and at the first 16 rows. Plus ooo1c_m32_check.py's scenarios on the candidate.
- Knob: one wrap of `ooo1cq.tier_glu_nax6n_1cq`, which covers COH on and off. It keeps the same signature, the same three
  outputs and the same region / offs / status semantics: wait on non-resident slots, post every used expert + sidecar.
  EVERY pool read goes through coherent(system), as in coh1cq. Plain reads gave run-to-run diffs on slot reuse, and c25's
  warm windows will not catch that.
- The FAST report carries:
  - a) install() with ON off == today, bitwise;
  - b) 5 repeats bitwise identical, COH on;
  - c) a 32-row call vs its two 16-row halves;
  - d) max + mean rel vs ooo1cq and vs an fp32 dequant reference, layers 20 + 44, with the worst row;
  - e) the m32_check scenarios: no hang, posts cover every used expert, late release OK;
  - f) us per call median (interleaved) at 32 and 16 rows, GB/s vs 280, mtlprof GPU busy;
  - g) extra memory and compile time.
- Leap wires any candidate >= 5 % faster per 32-row call with a)-e) clean. The c25 bits pass (argmax rows + max |logit
  diff| vs F32c6i) is Leap's; EvalSuite's same-scores gate decides adoption.
- Engine number: Leap's slot #3 (Wed morning) if a)-e) pass before Leap arms it tonight (~21:00); else slot #4 (Thu).

## Screen X0: can the accelerator's MMA overlap the stream at all? (pre-registered Tue 09:4x, before any run)
- **Why:** every lever below assumes the GPU can run the MMA under the weight stream once the dependency chain allows
  it. If even INDEPENDENT loads and MMAs add up, no rewrite can deliver the overlap. Then I stop and say so.
- **Kernels** (synthetic, in scripts/xrw/x0_overlap.py). All four share kernel A's grid: 62 segments x 20 column tiles,
  256 threads (8 simdgroups) per threadgroup, 10 quant groups per simdgroup. They read the same ~70 MB code buffer
  pattern: 6 words per lane per group, the native layout's addresses.
  - S1 STREAM: the code loads only, folded into one word per lane.
  - S2 MMA: the 20 matmul2d(16, 32, 32) per simdgroup on a threadgroup-resident uint4b tile and register x, with no
    device code loads.
  - S3 BOTH-INDEPENDENT: S1's loads and S2's MMAs in one loop, with no data dependency between them.
  - S4 BOTH-DEPENDENT: today's chain (load -> decode -> threadgroup -> MMA), the reference shape.
- **Timing:** GPU time per kernel from 3 interleaved rounds of 20 launches each in one process, median. The run happens
  under a share lock (shared GPU), so ratios only. If the shared run is too noisy (S1 spread > 15 %), it waits for a
  quiet ticket.
- **Lines:**
  - FEASIBLE if S3 <= 1.15 x max(S1, S2) (the smaller term at least ~70 % hidden);
  - KILL if S3 >= 0.90 x (S1 + S2) (independent work does not overlap);
  - else PARTIAL (the hidden share is reported).
  - Info only: S4 vs S1 + S2.

### X0 amendment (Tue 09:5x, after a smoke on the SHARED GPU, before the real run)
- **The smoke** (1 round x 3 launches; MoonShots 60 % + MTP Head 25 % of the GPU beside it; smoke only, no verdict):
  S1 314, S2 327, S3 813, S4 481 us.
- **S3 as built was not the "independent" case.** Its loop folded the loaded words right after the loads, BEFORE the
  MMAs. A simdgroup waits for a load at its first use, so each iteration paid the DRAM latency and then the MMAs.
- **Amended:** S3 issues the loads at the top of the iteration and folds them after the MMAs, so the loads are in flight
  during the MMAs. The smoke's form stays as S3f (info). Lines unchanged.
- **The smoke's main reading (for the plan, not a verdict):** S2, the MMA path alone with no code loads, already costs
  about as much as the whole kernel A. MoonShots' quiet "no code loads" ablation (331 us) agrees. The bottleneck is the
  accelerator path itself, not missing overlap.
- **New info screen X1** (scripts/xrw/x1_rates.py, no lines): the MMA path's cost by form and operand type, on kernel A's
  grid with threadgroup-resident operands. 3.25 G MAC per launch. Smoke on the shared GPU:

  | form | us |
  |---|---|
  | R1 today (cooperative bf16 left from device + epilogue) | 352 |
  | R2 (no epilogue) | 301 |
  | R3 (int8 threadgroup left x int4b -> int32) | 217 |
  | R4 (bf16 threadgroup left) | 262 |
  | R5 (R3 + per-group epilogue) | 287 |

  MPP's matmul2d header lists int8 x int4b -> int32 and uint8 x uint4b -> int32 among its supported types.

### X0 + X1 run 1 (Tue 09:46, share lock, GPU shared with MTP Head's run + others) = NO VERDICT (noise rule)
X0's S1 spread was 21.7 % (> 15 %), and spreads ran up to 143 % (X0) and 322 % (X1). Under the pre-registered rule the
verdict waits for a quiet ticket (asked of NewIdeas 09:5x). The medians are recorded in
data/xrw/x0_overlap.txt / x1_rates.txt and are not quoted.

## Candidates built (Tue 10:0x); correctness on real routing (share lock; no timing)
- **x2_mma_layout.py PASS** (one-simdgroup unit test vs numpy, exact). For matmul2d(16, 32, 32) with a threadgroup int8
  left and a threadgroup int4b right in the expert kernels' code layout:
  - the int32 dest is exact;
  - capacity is 16 per lane and every (row, column) appears once;
  - get_multidimensional_index(i) = (column, row);
  - a nibble v reads as the signed (v ^ 8) - 8, so an unsigned code XOR 8 reads as q - 8;
  - the second K half accumulates into the same dest.
- **The knob is scripts/xrw/xknob.py:** ON = [False], MODE = w4a8 | tgl16. install() wraps
  ooo1cq.tier_glu_nax6n_1cq once, which covers COH on and off. With ON off, every call goes to the original.
- **Both candidates keep:**
  - ooo1cq's grid and grouping;
  - its head (a_one_call_q: waits on late slots, QOM drops, rs_m / rs_l posts) and B's drop (b_drop), applied by
    Leap's own transforms;
  - coherent(system) pool reads via coh1cq's _body / _p11_hdr / _ren (12 coherent pool pointers per kernel);
  - one threadgroup slice per simdgroup, with layout-agnostic chunk partials.
- **C1 w4a8.py (FAST): int8 activations x int4b codes -> int32.**
  - quant64 (one simdgroup per (row, 64-value group): int8 codes, absmax / 127 scale, float sum) replaces rowsum64(x)
    and rowsum64(h).
  - Codes are XORed with 8 and the bias absorbs the offset: b + 8 s'.
- **C2 tgl16.py: bf16 x uint4b -> float, as today.** The only change is that the left operand (the tile's 16
  activation rows) is staged in threadgroup memory once per quant group, instead of loaded as per-lane fragments from
  device memory.
- **xrw_check.py:** the decision map's code windows, 2 real 32-row blocks per layer, layers 20 + 44, 640 pairs per layer.
  The fp32 reference is a numpy dequant of the pools (p11.unpack + native maps), with h kept in fp32. ooo1cq sits within
  0.44-0.46 % (mean) of it, which checks the reference.

  | item | C2 tgl16 | C1 w4a8 |
  |---|---|---|
  | a) ON off == original, bitwise | PASS | PASS |
  | b) 5 repeats bitwise + the same resolved slots | PASS | PASS |
  | c) 32 rows == two 16-row halves, bitwise | PASS | PASS |
  | pairs bitwise == ooo1cq | **1,280 of 1,280** | 0 of 1,280 |
  | d) rel L2 vs fp32, mean / max (L20) | 0.0044 / 0.0059 (= ooo1cq) | 0.0329 / 0.0686 |
  | d) rel L2 vs fp32, mean / max (L44) | 0.0046 / 0.0078 (= ooo1cq) | 0.0354 / 0.0744 |
  | g) first call incl. compile | 0.8 s | 0.9 s |

- **Reading:**
  - **C2 is an EXACT rewrite on these pairs**: the accelerator gives the same sums whichever memory feeds its left
    operand. If it is faster, it needs no FAST gate.
  - **C1's int8 activations cost ~3.4 % mean output error per expert**, about 8x ooo1cq's own error vs fp32. That is a
    real FAST price, to weigh against its speed.
- **Timing waits for quiet.** The 09:59 shared bench ran at 3x the quiet times, with spreads of 80-130 %: no number.
  The quiet runner (scripts/xrw/xq_after_tue1.sh, re-armed 10:0x) takes a quiet ticket when Leap's tue1 ends (~12:00).
  It runs X0 + X1 + xrw_bench (orig vs w4a8 vs tgl16, interleaved with the order rotated, layer 20, 24 real blocks,
  32 and 16 rows).
- **e) Leap's m32 scenarios on the candidates (scripts/xrw/xrw_m32.py; Tue 10:03, share lock): 24 of 24 PASS for BOTH.**
  The run exec's Leap's ooo1c_m32_check.py unchanged except two edits: its OUT points to data/xrw/m32_<cand>.txt, and
  the knob is installed after its imports.
  - It covers layers 20 / 44 x COH on / off x rand, rand_2d, pop1 (an expert on all 32 rows), pop2, pop2_2d and
    rand_late (the kernel polls the release region; no hang).
  - In every scenario the 32-row call == two 16-row calls bit for bit, and the base posts cover every used expert.
  - All 72 calls per candidate went to the candidate: {'cand': 72, 'orig': 0}.
- **So items a)-e) are clean for both candidates.** f) waits for the quiet bench; g) is reported above.
- **C1 w4a8's error profile (Leap's ask 10:04; 6 real blocks per layer, 1,920 pairs per layer, Tue 10:05):**

  | layer | rel L2 vs fp32: share > 2 / 5 / 10 % | p50 / p90 / p99 | worst pair | ooo1cq: share > 2 % / worst |
  |---|---|---|---|---|
  | 20 | 85.3 / 3.4 / 0.00 % | 0.034 / 0.043 / 0.058 | 0.075 (row 165, rank 4, expert 102, tier 3) | 0.0 % / 0.0075 |
  | 44 | 85.2 / 7.5 / 0.21 % | 0.036 / 0.047 / 0.074 | 0.113 (row 101, rank 5, expert 287, tier 3) | 0.0 % / 0.0087 |

  The error is systematic: most pairs sit at ~3.5 %, with no outlier tail. Per-64-group int8 absmax on x and h is too
  coarse for this model. Finer scales (per 32 / 16) or bf16 h in kernel B would cut it, at a cost in speed. This is
  decided only after the quiet bench shows whether int8 buys speed at all.
- **Variants for the quiet bench (scripts/xrw/tgl16v.py; checks Tue 10:08, 2 real blocks per layer, 1,280 pairs):**
  - **tgl16g:** kernel A with ONE simdgroup per K chunk doing gate AND up on one staged activation tile. That is half
    the x staging, 4 simdgroups / 128 threads per threadgroup, and 18.7 KB of slices. Kernel B is tgl16's. Result:
    **bitwise == ooo1cq on 1,280 of 1,280 pairs**, and a)-c) PASS. It is a second exact candidate.
  - **a8b16:** int8 kernel A (w4a8's) + bf16 kernel B (tgl16's). Mean rel L2 vs fp32 is 2.0 % (L20) / 2.5 % (L44),
    against w4a8's 3.3 / 3.5 %, and 59-75 % of pairs are > 2 %. So kernel A's int8 x carries most of the error; h adds
    the rest. FAST only.
  - The knob takes MODE = tgl16g | a8b16 too (bench variants).
- **The quiet runner was re-armed at 10:0x** with five arms: orig, tgl16, tgl16g, w4a8, a8b16. A 1-round shared smoke
  confirmed that all five build and run (its timings are not quoted).
- **Steering bench (Tue 10:09, shared GPU; xrw_bench --minmode 15; NOT a quotable number).** Each call ran alone in its
  command buffer, 15 interleaved repeats per block and arm. Per arm: the median over 24 blocks of the min GPU interval.
  orig was 844 / 750 us at 32 / 16 rows (quiet: ~580 at 32 rows), so contention remains.

  | arm | / orig, 32 rows | / orig, 16 rows |
  |---|---|---|
  | tgl16 | 0.957 | 0.942 |
  | tgl16g | 1.113 | 1.096 |
  | w4a8 | 0.947 | 0.915 |
  | a8b16 | 0.956 | 0.914 |

  Reading (to be checked in quiet): int8 barely beats bf16 in the REAL kernel, though X1's synthetic MMA path was
  ~25 % faster in int8. MoonShots' quiet ablation says the same thing: without code loads the kernel still takes 331 of
  352 us. The kernel is compute-bound across decode + staging + epilogue + MMA, not MMA-bound and not waiting on an
  overlap. Fusing gate + up per simdgroup (fewer, heavier simdgroups) is slower.
- **Timing ablations of tgl16 added to the quiet run** (tgl16v.py; ABL_C bits: 1 no code loads (fake words still
  decoded), 2 no decode, 4 no MMA, 8 no epilogue, 16 no activation loads). They show where tgl16's time goes and pick the
  next lever.
- **The quiet runner uses frozen copies** (data/xrw/frozen_q1, md5s in data/xrw/frozen_q1.md5; outputs via XRW_OUTD to
  data/xrw). Re-armed 10:1x: X0 + X1, then the five-arm bench (32 + 16 rows, 9 rounds), then the ablation bench (32 rows,
  7 rounds).
- **x2b_dest_formula.py PASS (Tue 10:1x):** with threadgroup tensor operands, the cooperative dest (bf16 x uint4b ->
  float and int8 x int4b -> int32) puts element i of lane l exactly at Leap's analytic position (col = 16 (i >> 3) +
  fn0 + (i & 3), row = fm0 + 8 ((i >> 2) & 1)), on all 32 lanes.
- **tgl16e = tgl16 with the epilogue on that formula.** Per group and lane, the scales / biases of its 8 columns load
  as 4 float4 and its 2 row sums as 2 floats, instead of ~48 scalar threadgroup loads through index arrays. Result:
  **bitwise == ooo1cq on 1,280 of 1,280 pairs**, and a)-c) PASS.
- **The quiet runner was re-armed at 10:1x on frozen copies data/xrw/frozen_q2** (md5s in frozen_q2.md5). The main bench
  has six arms: orig, tgl16, tgl16e, tgl16g, w4a8, a8b16. Then the ablation bench.
- **X3 shapes (info; scripts/xrw/x3_shapes.py; Tue 10:17).** Shared GPU, but each figure is the MIN over 50 interleaved
  launches. Static threadgroup operands, no epilogue, kernel A's grid; each form does the same useful work.

  | form | shape (M, N, K) x ops | types | min us | / W16 |
  |---|---|---|---|---|
  | W16 (today) | (16 tokens, 32 features, 32) x 2 | bf16 x uint4b | 220.3 | 1.000 |
  | F16 | (32 features, 16 tokens, 32) x 2 | int8 weights x bf16 x^T | 209.8 | 0.952 |
  | F8 | (32, 8 tokens, 32) x 2 | int8 x bf16 | 209.8 | 0.952 |
  | F16b | (32, 16, 64) x 1 | int8 x bf16 | 217.7 | 0.988 |
  | H16 | (32, 16, 32) x 2 | half x bf16 | 211.0 | 0.958 |

  - **N = 8 costs exactly what N = 16 costs.** Tokens pad to 16 on N just as on M (MoonShots' M = 8 screen), so no
    operand placement removes the 16-row padding. Exact weights-left forms (int8 codes x bf16) gain only ~5 %; one
    K = 64 op gains nothing.
  - **The accelerator floor of kernel A's padded work is ~210-220 us**, just under the code stream (~234 us; MoonShots'
    pure read). A + B today are ~388 + 198 us; the floor is about max(MMA ~220 + ~110, stream ~234 + ~115) ~ 350 us. So
    the orchestrator's ~-5.5 ms per check is the FULL-overlap prize: the accelerator, the decode ALU and the stream
    would all have to run concurrently.
  - Per simdgroup-group on one core, the rough budget is MMA ~44 ns, ALU ~40 ns (~200 instructions over 4 schedulers)
    and stream ~60 ns, against ~146 ns if fully serial. The candidates attack the ALU part (tgl16e: ~40 fewer loads per
    group) and the MMA part (int8). The quiet ablations measure each.
- **tgl16eh (Tue 10:21):** tgl16e with the activation tile staged one K half at a time. Its threadgroup holds ~18.6 KB
  (tgl16: ~27 KB, ooo1cq: ~20 KB). **Bitwise == ooo1cq on 1,280 of 1,280 pairs.** Steering (shared, min-of-15) cannot
  separate the exact variants at 32 rows: tgl16 0.966, tgl16e 0.961, tgl16eh 0.954 x orig. At 16 rows: 0.939 / 0.941 /
  0.953. So neither the epilogue's loads nor threadgroup occupancy looks binding; quiet will say.
- **Context for the FAST error (x4_tier_budget.py, Tue 10:23; fp32 dequant, real rows, 300 samples per layer).** The
  lossy check already accepts the gap between an expert's tier-3 output (MSB-3 + b3, ranks 2-9) and its tier-4 output:
  mean rel L2 **0.244 (L20) / 0.265 (L44)**, p90 0.274 / 0.297. w4a8's ~3.4 % from int8 activations is about 7x smaller.
  Added in quadrature it moves a tier-3 pair's error from ~0.25 to ~0.252. That is the scale EvalSuite's same-scores gate
  judges, should int8 earn its place on speed.
- **tgl16ep (Tue 10:24):** tgl16e + a distance-1 register prefetch (group g + 1's codes, LSB, activations, scales are
  loaded before group g's decode). **Bitwise == ooo1cq on 1,280 of 1,280 pairs**, but slower in steering:

  | arm | 32 rows (orig 883 us) | 16 rows |
  |---|---|---|
  | tgl16ep | 0.997 | 1.019 |
  | tgl16e | 0.969 | 0.952 |
  | w4a8 | 0.907 | 0.942 |

  The registers cost more than the extra loads in flight, as MoonShots found for the old kernel (C1). DROPPED.
- **Where the exact track stands before quiet:** every bitwise restructure steers at ~0.95-0.97 x orig. The resources
  look balanced: MMA ~210-220 us (X3), stream ~234 us, ALU ~150-200 us for kernel A. Today's ~352 us already hides about
  half of the serial sum, so big exact gains would need near-perfect three-way overlap.

## The TIER-4 UPGRADE (Tue 10:3x): fewer segments, not better overlap (FAST; batch-dependent)
- **Why:** the resources per segment are roughly balanced (MMA, decode ALU, stream), so a large exact gain needs
  near-perfect overlap. Cutting SEGMENTS cuts MMA, decode and threadgroups together.
- **The rule (scripts/xrw/xgrp.py, a new grouping kernel; kernels A / B unchanged):**
  - A tier-3 pair (rank >= 2, or a 4-bit rank whose sidecar is -2) joins its expert's tier-4 key when the expert has an
    eligible 4-bit pair in the call (sidecar resident or late, not a QOM drop).
  - That expert's LSB plane is read for that pair anyway, so there are no extra bytes, and the upgraded pairs run on the
    4-bit weights, which are more accurate.
  - Real pairs sort first within a key. A 16-pair group headed by an upgraded pair (only when > 16 pairs share a key)
    keeps tier 3.
  - The candidates' head keeps rs_l's meaning: a pair whose own sidecar entry is -2 posts -2.
- **Count (x5_seg_merge.py, CPU, 4,752 real 32-row blocks, every captured layer):** segments per call 62.7 -> 50.5
  (-19.4 %). Distinct experts 46.5. 137 of 275 tier-3 pairs upgrade. Today's 60.9 at layer 20 matches MoonShots' GPU
  profile, which checks the count.
- **xgrp self-test: 80 of 80 PASS.** On 40 random calls (late slots, QOM drops, -2 sidecars, an expert on > 16 rows) the
  GPU grouping == its numpy replica both with and without the upgrade, and up = 0 == ooo1cq.gpu_grp_1cq.
- **Checks (2 real blocks per layer; layers 20 / 44):**

  | item | tgl16eu (tgl16e + upgrade) | w4a8u (w4a8 + upgrade) |
  |---|---|---|
  | a) ON off == original | PASS | PASS |
  | b) 5 repeats bitwise; rs_m / rs_l == ooo1cq's | PASS; PASS / PASS | PASS; PASS / PASS |
  | c) 32 == two 16-row halves | **NO, by design** | **NO, by design** |
  | pairs whose tier changed | 183 / 176 of 640 (~28 %) | same |
  | unchanged pairs bitwise == ooo1cq | 457 of 457 / 464 of 464 | 0 (int8) |
  | d) rel L2 vs fp32 at each pair's tier, mean / max | 0.0044 / 0.0059; 0.0046 / 0.0078 (= ooo1cq's level) | 0.027 / 0.069; 0.029 / 0.074 |
  | e) Leap's m32 scenarios | posts cover all used experts in 24 of 24; late release no hang; 32 == 16+16: 0 of 24 | not run |

- **c) is the price:** a pair's tier now depends on the OTHER rows of the call (does its expert have a 4-bit pair
  there?). The same context can get different expert outputs in different check batches. On real routing the upgraded
  pairs move toward the more accurate tier (the tier-3 / tier-4 gap is ~0.25 rel L2, x4_tier_budget).
- **Steering (Tue 10:30; the GPU was quieter: orig 658 us at 32 rows, tight per-block spreads; NOT a quotable number):**

  | arm | / orig, 32 rows | / orig, 16 rows |
  |---|---|---|
  | tgl16e | 0.970 | 0.967 |
  | tgl16eu | **0.902** | **0.881** |
  | w4a8u | **0.867** | **0.867** |

- Decision asked of Leap (reviewer) 10:3x: is batch dependence acceptable on the FAST track (EvalSuite's same-scores
  gate judges quality)? The exact candidates (tgl16 / tgl16e) stay separate.
- **Leap's decision (10:33): the upgrade is ACCEPTABLE on the FAST track.** Today's 1c tiers are already
  batch-dependent: an expert's tier comes from I_e summed over ALL rows of the call (t1cf.py), and c25's F32s vs F16 give
  0 / 16 rows bitwise. The upgrade adds no new class of dependence and only moves pairs to MORE precise weights.
  Conditions:
  - (1) deterministic for the same call content: b) PASS;
  - (2) no extra bytes: c25's expert GB per check must not rise (Leap reads it in the arm);
  - (3) it changes the gated config's numerics, so adoption needs its OWN EvalSuite pass. NewIdeas / the orchestrator
    decide whether it joins Saturday's stack before Thursday's tier 1.

  tgl16e stays the exact candidate. Leap's arms after tue1: F32c6xk / F16gk per MODE (tgl16e, tgl16eu).
- **The FROZEN knob for those arms: data/xrw/knob_v1** (README + md5s in data/xrw/knob_v1.md5). Release check
  (scripts/xrw/knob_check.py, which imports every module from knob_v1 and asserts it by __file__) PASS on 2 real blocks
  each of layers 20 / 44:
  - tgl16e == ooo1cq bitwise in y, rs_m and rs_l;
  - tgl16eu's unchanged pairs == ooo1cq bitwise (71-105 of 320 pairs per block upgraded);
  - 3 repeats are identical.
- **The upgrade at 16 rows (Saturday's check size; x5_seg_merge.py X5_ROWS=16; 9,840 real 16-row blocks, all layers):**
  segments per call 48.0 -> 38.9 (-19.0 %); distinct experts 37.9; 61.4 of 137.6 tier-3 pairs upgrade. The same
  fraction as at 32 rows.

## QUIET RESULTS (Tue 11:05:37-11:05:45; my ticket right after Leap's tue1 released at 11:05:35; NO-SWAP 46 GB)
Frozen copies data/xrw/frozen_q3 (md5s in frozen_q3.md5). Outputs: data/xrw/{x0_overlap,x1_rates,xrw_bench}_quiet.*,
xrw_bench_quiet_abl.*; the runner's stdout is in data/xrw/xq_run.txt.

### The kernel bench (exp32's way: 24 real blocks of layer 20 per round in one graph; 9 rounds; arm order rotated)

| arm | 32 rows us / call (GPU busy) | / orig per round, median [min, max] | 16 rows us / call (GPU busy) | / orig |
|---|---|---|---|---|
| orig (coh1cq, live) | 572.0 (542.2) | 1 | 467.9 (438.2) | 1 |
| tgl16 (exact) | 553.2 (523.2) | 0.966 [0.957, 0.975] | 447.8 (419.2) | 0.955 [0.950, 0.967] |
| tgl16e (exact) | 556.2 (526.1) | 0.973 [0.965, 0.984] | 452.0 (421.8) | 0.965 [0.957, 0.981] |
| **tgl16eu (exact kernels + upgrade; FAST)** | **482.3 (451.0)** | **0.843 [0.834, 0.846]** | **387.5 (357.7)** | **0.830 [0.820, 0.840]** |
| w4a8 (int8; FAST) | 533.4 (502.9) | 0.932 [0.928, 0.942] | 436.0 (405.6) | 0.934 [0.923, 0.938] |
| **w4a8u (int8 + upgrade; FAST)** | **468.4 (437.3)** | **0.816 [0.812, 0.829]** | **377.6 (346.4)** | **0.810 [0.796, 0.813]** |

Spreads were 1-3 %. orig matches Leap's own exp32 quiet numbers (586.9 / 496.5 us in tue1's post bench, 11:05; 584-605
on Mon). Against Leap's 5 % wiring bar, tgl16 / tgl16e (exact) fall short at 32 rows (3.4 % / 2.7 %). The upgrade
arms clear it by 3x.

### Where tgl16's time goes (timing ablations, 32 rows, 7 rounds, same window)

| removed | us / call | saved vs tgl16 (553.8) |
|---|---|---|
| the MMA (abl4) | 425.3 | **128.5** |
| the decode (abl2) | 488.8 | 65.0 |
| the code loads (abl1; fake words still decoded) | 493.1 | 60.7 |
| the activation loads (abl16) | 534.4 | 19.4 |
| the epilogue (abl8) | 542.7 | 11.1 |

The accelerator's padded MMA is the largest exposed cost; the decode comes next. Both scale with SEGMENTS, which is
why the upgrade (-19 % segments) cuts ~13 % of the call. Even without the MMA the call sits ~60 us over its byte floor
(105.6 MB at 290 GB/s ~ 364 us).

### X0 (pre-registered): FEASIBLE by the medians, with a stated caveat
- Medians of the 5 round medians: S1 269.0, S2 243.3, S3 289.9 us. S3 <= 1.15 x max(S1, S2) = 309.3 -> **FEASIBLE**
  (91 % of the smaller term hidden). S3f 293.1. S4 (dependent chain) 303.8 = 1.13 x max.
- Caveat: the noise line (S1 spread <= 15 %) is exceeded (109 %) by round 1 alone. Every kernel's first round ran ~2x
  slow (clocks ramping after an idle GPU: S1 562 / 269 / 268 / 269 / 268). Rounds 2-5 agree within 1 %, so the median
  is robust.
- Reading: the accelerator CAN run under the stream when the work is independent. The real kernel's exposed MMA (128 us)
  comes from its per-segment dependency chain and padded work, not from hardware serialization.

### X1 (info) in quiet: the within-screen ratios are the finding
R1 485.1, R2 426.7, R3 333.6, R4 426.4, R5 446.7 us; spreads <= 1.2 %.
- int8 x int4b (R3) runs at 0.78 x bf16 x uint4b (R4).
- A threadgroup-staged left operand (R4) costs the same as a cooperative one (R2).
- The epilogue costs 14 % (R1 vs R2).
- The absolute values are ~2x X3's shared-GPU minimum for the same shapes (W16 220 us). This likely reflects isolated
  launches at low clocks and R-kernels' 26 KB threadgroups, so they are not quoted as rates.

### Priced on Leap's quiet engine numbers (tue1, 11:06) [P: kernel ratios x engine class time; c25's A/B decides]
The EXPERT class is 17.44 ms of the 16-row check (F16) and 26.31 ms of the 32-row C6 + int8 check; 45-47 % of GPU.

| arm | 16-row check | 32-row check |
|---|---|---|
| tgl16eu | ~-3.0 ms (17.44 x 0.170) | ~-4.1 ms (26.31 x 0.157) |
| w4a8u | ~-3.3 ms | ~-4.8 ms |
| tgl16 (exact) | ~-0.8 ms | ~-0.9 ms |

Saturday's 16-row check (Leap: 38.9 ms) would go to ~35.9 with tgl16eu.

## DECISION (orchestrator 11:1x): tgl16eu INTO Saturday's stack; w4a8u NO
- tgl16eu joins BigMoE's r57 stack as a flag BEFORE Thursday's tier 1, if Leap's c25 A/B confirms by Wed evening ~-2.5..-3
  ms at 16 rows, no extra expert GB, and determinism.
- Quality is judged inside the stack's gate (tier 1 Thu + card gate Fri night). If the stack fails, the first bisect turns
  tgl16eu off.
- w4a8u: NO. It adds 3 % expert-output error for only ~0.6 ms more.
- Handoff to BigMoE: docs/EXPERT_REWRITE_HANDOFF.md (wiring lines, scope, smoke check, the frozen knob_v1). EvalSuite is
  told that tgl16eu is in the stack.
- **Robustness before BigMoE wires it (scripts/xrw/xrw_robust.py through the FROZEN knob_v1; Tue 11:1x): PASS, 18 cases.**
  - Coverage: row counts M = 1, 2, 3, 5, 8, 16, 17, 24, 32 x layers 20 / 44; random routing with a popular expert,
    2-D nolsb marks and QOM-dropped experts.
  - tgl16e == ooo1cq in y, rs_m and rs_l, bitwise, in every case.
  - tgl16eu's unchanged pairs == ooo1cq bitwise, with 2 repeats identical; 0-66 pairs upgraded per case.
  - QOM-dropped pairs are exactly 0 in every arm.
- **Leap's arm names (11:12): the deciding number is F16gu - F16g.**
  - F16gk / F32c6k = tgl16e (exact); F16gu / F32c6u = tgl16eu; F16gx = G1 + GDN16 + tgl16eu (the flags BigMoE would get).
  - Each arm's line carries its expert GB per check. Determinism: pass K re-runs F16gu / F16gx and compares rows
    bitwise.
  - Schedule: a 13-arm smoke ~12:00-12:30 (after EvalSuite's V2); the full A/B in slot #2 tonight; the repeat in slot #3
    Wed morning. Results by Wed ~07:00.

## knob_v2 (Tue 11:2x): the same outputs as knob_v1, a faster upgrade grouping
- **Why:** Leap's in-engine smoke (11:18, ROUGH, 2 checks per arm) showed:
  - tgl16e BITWISE in the real check (F16gk == F16g 16 / 16 rows; F32c6k == F32c6i 32 / 32).
  - tgl16eu: argmax 16 / 16 and 30 / 32; expert GB 2.47 -> 2.48 (16 rows; tokens diverge) and 3.91 = 3.91 (32 rows).
  - Walls F16g 38.95 -> F16gu 36.83 (-2.1 ms) and F32c6i 56.35 -> F32c6u 53.73 (-2.6 ms). That is short of the
    -2.5..-3 decision band.
  - x6_grp_cost.py found why: the upgrade's grouping kernel (one threadgroup, on every call's critical path) ran
    ~2.2x today's, 9.0 -> 19.9 us at P = 160 and 16.5 -> 39.9 us at P = 320 (min over real blocks). Each tier-3
    candidate scanned all P pairs.
- **v2 (xgrp.gpu_grp_up2):** eligible 4-bit pairs atomic-min their index into a threadgroup table keyed by identity; a
  candidate reads its entry in O(1). That is the same choice as v1 (the first eligible pair of its identity), and an
  identity outside the tables falls back to v1's scan.
- **Grouping cost (contended GPU; ratios):** orig / v1 / v2 = 28.7 / 64.9 / 34.5 us at P = 160 and 54.4 / 126.5 / 63.7
  at P = 320. The upgrade's grouping overhead falls from ~+125 % to ~+17-20 %, worth ~0.4 / ~0.9 ms per 16 / 32-row
  check at quiet scale [P].
- **Equivalence:**
  - xgrp.self_test2: v2 == v1 in 240 / 240 random calls, including inconsistent sidecar slots, slots beyond the table,
    late slots and QOM drops. The first run of this test compared seg past nseg (never written) and failed falsely; the
    test was fixed.
  - xrw_v1v2.py: tgl16eu with v2 == with v1, BITWISE in y, rs_m and rs_l, on 24 real calls (L20 / L44, 32 and 16 rows).
- **Release checks of the FROZEN data/xrw/knob_v2 (md5s in knob_v2.md5):** knob_check PASS; xrw_robust PASS (18 cases).
- **Steering (shared, NOT quotable; orig 820 us):** tgl16eu with v1 grouping 0.904, with v2 0.878 x orig at 32 rows.

## GENERAL methods vs MODEL-SPECIFIC tuning (orchestrator 11:29, from Kamen: build templates, not one-model layers)
**General: these carry to any MoE model on this GPU (and mostly to any GPU with a matrix unit).**
- **The MMA's left operand staged in threadgroup memory** (tgl16): bitwise-preserving. The accelerator's sums don't
  depend on which memory feeds the operand.
- **The cooperative dest's layout is analytic** (x2b): the epilogue can load scales / biases as vectors.
- **SEGMENT REDUCTION before kernel tuning.** On this accelerator the MMA pads both M and N to 16 (X3), so the work
  scales with segments, not tokens. Any merge of an expert's segments pays directly: the tier upgrade merges precision
  tiers whose higher plane is read anyway, at no extra bytes and with more accurate weights.
- **Single-threadgroup glue on the critical path must be O(P),** not O(P^2) (xgrp v2: table lookups).
- **Fold independent work under the expert stream** (router, shared expert, combine): the MoE-block proof
  (docs/MOEBLOCK.md).
- **The measurement kit:**
  - X0 (can the MMA run under the stream at all?);
  - X3 (operand shapes and their padding);
  - removal ablations per component;
  - interleaved, order-rotated benches with frozen copies;
  - fp32 dequant references;
  - knob releases with md5s.
- **Facts found:** int8 x int4b runs at ~0.78 x bf16 x uint4b's MMA time; register prefetch and fewer-but-heavier
  simdgroups were slower here; 3-bit -> nibble decode is ~12 ALU ops per 8 codes.

**Model-specific: config, not code.** For the 125B:
- hidden 2560, expert intermediate 640, 512 experts, top-10; group size 64;
- the tier plan "4433333333" (ranks 0-1 at MSB + LSB);
- the converted store's records (MSB-3 + LSB planes in the native 32-row tile layout; p11 11-bit scale / bias pieces);
- the K-split ks = 4 / ks_b = 2 / nsg_b = 2 (Leap's NAX6 choices);
- router 8-bit g64, shared expert 4-bit g64 (640);
- the 1c-tier thresholds (s 0.98, lam 0.016).

The MoE-block code takes these from a per-model config file (moeblk: config/<model>.json); a new MoE model needs a new
config, plus a store conversion if its experts use the tiered format.
