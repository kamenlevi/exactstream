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
