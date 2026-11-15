# CHECK55 lever 2: the MoE block's dense side as fewer exact dispatches

**The ask** (the orchestrator, Sun 15 Nov 17:2x; restated in its partial lift at 20:5x):
- "the MoE dense side (8-bit router GEMV + top-k + shared gate / up / down + tail) as ~2 exact dispatches";
- "Kill-test: your kernel alone vs K-73's chain on 3 real layers at 16 and 32 rows, y bitwise, line <= -0.05 ms per layer at A16";
- "Leap integrates; don't touch Leap's files". The transfer rule applies: a paired engine A/B decides adoption.
- Under the partial lift (20:5x): the kernel and its bit test at desk scale now; the timing line waits for a quiet window.

## 1. The chain today (K-73, as the exact-45 engine runs it at 16 rows)
From my CHECK55 profile's kernel lists (data/m1/check55/c55_1105_read.txt, A16k, per layer), 7 dispatches:

| # | step | kernel |
|---|---|---|
| 1 | router logits, 8-bit, N 512, K 2560 (ref2: MLX's own 16-row call) | affine_qmv_wide (nv 4, k_lanes 8) |
| 2 | softmax / top-10 / normalize / sort (R-70) | router_fused |
| 3 | shared gate / up / gate, 4-bit, N 1281 = 640 + 640 + 1 (R-74's concat; MLX's 16-row call) | affine_qmv_wide (nv 4, k_lanes 8) |
| 4 | swiglu(gate, up) (mlx_lm's compiled swiglu) | one compiled kernel |
| 5 | down_proj, 4-bit, N 2560, K 640 (ref2: N >= 1024) | Leap's qmm16c (code MMA, ks 1) |
| 6 | g = y[..., 1280:1281] made contiguous for the tail kernel | copy |
| 7 | tail y + sigmoid(g) * s (R-73), after the experts | moe_tail (exp precise) |

At 17-32 rows (V32_MOE1's _tier_block) the same pieces run, with v32 splitting each matmul into 2 x 16 rows.
- The profile's class time for these pieces is ~1.04 ms per 12-layer slice at A16; that figure also holds the
  slice's own expert combine.
- Leap's QMVW16 (isolated calls) puts steps 1 and 3 at 27.3 and 34.0 us: the two GEMVs dominate.

## 2. The design (3 dispatches: 2 before the experts + the tail)
- **A: both GEMVs in one dispatch**, in qmv_wide's per-output arithmetic.
  - Each output's bits depend only on k_lanes (8): each k-lane sums its groups g = j, j + 8, ... in 8-value sub-chunks.
    Each sub-chunk is an fp32 chain from 0, added in order, then a shuffle ladder (4, 2, 1).
  - The work per thread is regrouped: each lane holds several output rows x several vectors (register blocking), so each
    x value is converted once per lane rather than once per output row.
  - The shared expert's rows k and 640 + k sit in the same lane, so that lane rounds gate and up to bf16 and writes
    swiglu(gate, up) as MLX's compiled op does (Sigmoid functor, then two bf16 multiplies). It also writes g contiguous.
- **B: router top-k + down_proj in one dispatch.**
  - Router threadgroups run R-70's router_fused source verbatim, one per row.
  - down_proj threadgroups run qmm16c's source verbatim (ks 1; nt only groups tiles) on A's swiglu rows, with 16-row
    tiles (2 at 32 rows).
- **The tail** stays moe_tail. It no longer needs the g copy.
- Steps 1-6 become A + B. 7 -> 3 dispatches per layer, and s still exists before the experts, so OOO's spins still wait
  behind the shared expert.
- Code: scripts/m1/check55/moe_dense.py. It uses Leap's sources read-only (router_fused._SRC, qmm16._SRC3, leapk's MLX
  quantized header).

## 3. PRE-REGISTERED bit test (before any run; desk scale, share lock, MEM < 2 GB)
- **Weights:** layers 0, 24 and 47 of the 125B (the dense shard model-00016: mlp.gate 8-bit; shared_expert.gate_proj /
  up_proj / down_proj and shared_expert_gate 4-bit; group 64).
- **Rows:** 16 and 32 (the verdict). Extras, reported: 1, 8, 17, 24.
- **Inputs per (layer, rows), 6 draws:**
  - normal(0, 1) x 0.5 / 1 / 2;
  - heavy-tailed (normal plus 2 % of elements x 30);
  - all rows equal;
  - a zero row and a constant row (router ties: equal logits, top-k broken by index), the rest normal.
- **Reference = K-73's chain as the engine runs it:**
  - Leap's `router_fused(blk.gate(x), 10)` and `shared_cat(blk, x)`, imported read-only;
  - under ref2.apply() (8-bit router and the N 1281 concat on MLX's 16-row call; down_proj on qmm16c);
  - at 17-32 rows also v32.install(ref2) under attention_phase("decode_verify"), its 2 x 16 split as in _tier_block.
- **Compared bitwise, every element:** inds, scores, g, s. Also moe_tail(y, g, s) for a random y (it follows from g
  and s).
- **PASS:** every compared element equal on every layer x rows in {16, 32} x draw.
- **Unit checks, which must pass too:**
  - A's logits == MLX's 16-row router call;
  - A's swiglu rows == mlx_lm's swiglu of MLX's gate / up;
  - my swiglu == mlx_lm's compiled swiglu on all 65,536 bf16 gate values x 8 up values.
- **Sensitivity (shows the test can fail):**
  - (a) kernel A with the ladder order reversed (1, 2, 4) must differ from the reference on some logits;
  - (b) flipping the lowest mantissa bit of one x element must change some output.

## 4. PRE-REGISTERED timing line (NOT tonight: a quiet window only, then the engine A/B decides)
- **Setup:** per layer, the chain's 7 dispatches as the engine runs them vs A + B + moe_tail, on the 3 layers as a
  dependent chain, many reps per timed graph, median of 7 rotated rounds, GPU time per layer.
- **Kill line (the orchestrator):** fused - chain <= -0.05 ms per layer at 16 rows. 32 rows are reported.
- **Adoption:** a paired engine A/B (Leap integrates; NewIdeas' transfer rule).
- Any desk timing before then is labelled rough (a shared Mac) and is not a verdict.

## 5. RESULT: BIT TEST PASS (Sun 15 Nov 21:15; share lock, MLX peak 0.03 GB, 1.6 s)
Data: data/m1/check55/lever2/l2_bit.txt / .json; dispatch lists in l2_paths.txt. Code: moe_dense.py + test_moe_dense.py
(+ paths_moe_dense.py). Leap's sources used read-only: router_fused._SRC sha256 28ce6b100f2b, qmm16._SRC3 ed0691cfd8f3.
- **Verdict cases** (layers 0 / 24 / 47 x rows 16 / 32 x 6 draws): inds, scores, g, s and the tail all bitwise equal,
  0 differing elements.
- **Extras** (rows 1 / 8 / 17 / 24): also 0 differing. All 108 cases are clean.
- **Units:**
  - swiglu: 0 of 524,288 (gate, up) pairs differ, all 65,536 bf16 gates incl. inf / nan;
  - A's logits 0 / 8192, swiglu rows 0 / 10240 and g 0 / 16 differ vs MLX's own 16-row calls, on each layer.
- **Sensitivity:** the reversed ladder changes 83 of A's outputs; a 1-ulp flip of one x element changes 66 outputs.
  The comparison is sharp.
- **The reference took the engine's paths:** ref2 counted qmm16c 1 + mlx16 2 per 16-row chain, and v32 split all 3
  matmuls 2 x 16 at 32 rows.
- **Dispatches** (libmtlprof, layer 24, after a warm call), with outputs bitwise equal in that run too:

| rows | K-73's chain | moe_dense + moe_tail |
|---|---|---|
| 16 | **7**: qmv_wide 8-bit, router_fused, qmv_wide 4-bit, swiglu, qmm16c, g copy, moe_tail | **3**: A, B, moe_tail |
| 32 | **16**: the 3 matmuls twice each, 6 concat copies of v32's halves, router_fused, swiglu, g copy, moe_tail | **3** |

- **Next:** desk timing (rough, labelled as such), then the pre-registered timing line in a quiet window. Then Leap's
  engine A/B: the swap is `inds, scores, g, s = moe_dense.dense_pre(self, x)` for the router_fused + shared_cat pair in
  K-73's _exact_block and v32's _tier_block.

## 6. ROUGH desk timing (Sun 21:18-21:39; a shared Mac, GPU 45-78 % busy with MoonShots' run; NOT the verdict)
Data: data/m1/check55/lever2/l2_pieces_rough1-7.txt (piece bench: chained calls via mx.depends, the 3 layers rotating,
96 calls per graph, 9-11 rotated rounds, GPU busy from libmtlprof), l2_bench_rough1-2.txt (the whole chain, noisier),
l2_diag_rough1.txt.

**Per piece, 16 rows** (min / median us per call; the ranges span runs rough2, rough3, rough6 and rough7):

| piece | K-73's chain | mine |
|---|---|---|
| both GEMVs | MLX's 2 qmv_wide calls: 52.8-68.3 / 71.4-82.4 | A (16:4:2:4): 51.2-59.1 / 56.9-73.3 |
| router top-k + swiglu + down_proj + g copy | 4 dispatches: 9.9-23.5 / 28.2-37.2 | B: 3.6-4.1 / 4.8-7.1 |

**Saving per layer before the experts** (same run, mine vs the chain):
- 16 rows: -7.0 / -10.7 / -11.9 / -28.6 us by the mins; -36 to -42 us by the medians.
- 32 rows: -4 to -26 us by the mins; -63 to -116 us by the medians. That is without v32's 6 concat copies, which the
  real 32-row chain also runs.
- Most of the 16-row saving is B: one dispatch for 4, and the g copy gone.
- By the mins, the line (<= -50 us per layer at 16 rows) looks out of reach. The quiet window decides.

**What bounds the GEMVs** (diag_moe_dense.py: throwaway, non-exact copies of kernel A, 16 rows):

| variant | min us / call |
|---|---|
| exact | 61.3 |
| no x loads | 43.4 |
| no weight loads | 47.7 |
| neither | 7.4 |
| twice the FMAs | 59.2 |

- They are bound by loads, not arithmetic.
- **Variants tried**, all with the same per-output arithmetic and all BITWISE on the full bit test (l2_bit_*.txt). None
  beat kernel A, which is at parity with MLX's qmv_wide (as Leap's QMVW16 found for MLX's launch):

| variant | min us per call, 16 rows | vs A in the same run |
|---|---|---|
| A2, x tiles in threadgroup memory | 81-145 | 59 |
| A2p, the tiles padded (no bank conflicts) | 78-85 | 54-58 |
| A3, wide uint2 / uint4 loads | 62-75 | 57 |
| A4, one k-lane per simdgroup with rows across lanes (x loads become broadcasts) | 134 | 75 |
| A5, the group loop unrolled | 55-59 | 56 |

- **Production path:** dense_pre's default, A at 16 vectors x 4 rows per lane (the best medians), plus B.

## 7. AMENDMENT before any quiet-window timing (Sun 21:4x)
- The quiet-window verdict times dense_pre's default (moe_dense.DEFAULT_A = rv 16, rr 4, rp 2, sgn 4) + B + moe_tail
  against the chain, as in section 4 (bench_moe_dense.py "chain,mine").
- No other config enters the verdict. The variants of section 6 are reported as information only.

## 8. RESULT: the quiet-window verdict MISSED (Mon 16 Nov 00:26, quiet ticket, power mode 2, NO-SWAP 43 GB)
Run by run_l2_quiet.sh right after the engine profile's attempt (the orchestrator's slot). Data:
data/m1/check55/lever2/l2_bench_quiet.txt (THE verdict: bench_moe_dense.py, 7 rounds, 16 passes per graph, forms
"chain,mine" as section 7 fixed them), l2_pieces_quiet.txt (information), run_l2_quiet.log. Both ran under the ticket,
10 s in all. The files' first line still says "ROUGH, shared Mac": that is a fixed label in the scripts, not the conditions.

| rows | K-73's chain, GPU us / layer | mine (dense_pre's default + B), GPU us / layer | mine - chain | per round (mine - chain, us) |
|---|---|---|---|---|
| 16 | 64.0 | 94.4 | **+30.3** | +33 +44 +33 +30 +27 +28 +29 |
| 32 | 106.1 | 116.0 | +9.8 | +7 +7 +10 +6 +10 +10 +10 |

- **Verdict: MISSED.** The line was mine - chain <= -50 us per layer at 16 rows; mine is slower in every round.
- **Bits held:** the end states are bitwise equal across the forms at both lengths.
- **The pieces** (information; min us per call, 96 chained calls per graph, 9 rounds):

| piece | 16 rows | 32 rows |
|---|---|---|
| MLX's two calls (8-bit router + R-74's 4-bit concat); at 32 rows, two halves | 44.0 | 85.2 |
| A, the default (16:4:2:4) | 49.6 | 95.0 (median 124.6) |
| A 4:4:2:4 | 44.5 | 126.9 |
| K-73's B side (router_fused + swiglu + qmm16c + the g copy) | 7.7 | 8.9 |
| B | 2.8 | 5.5 |

**Reading.**
- Even the pieces, the most favourable reading, give no path to -50 us. At 16 rows, the default A + B is 52.4 us against
  the chain's 51.7; with A 4:4:2:4 it would be 47.3 (-4.4 us).
- The whole-chain bench charges mine ~30 us per layer more than its pieces add up to. The chain matches its pieces
  (64.0 = 51.7 + a ~12 us tail); mine does not (94.4 vs ~64). This is not explained, and it is not needed for the verdict.
- Both forms' per-round GPU times fall through the run (chain 92 -> 58, mine 125 -> 87): the GPU clock was still rising
  on the short graphs. The rotation spreads that over both forms, and the gap holds in the last rounds (+28, +29).
- Section 6's rough gain came from the piece bench on a shared GPU. The rough whole-chain benches (l2_bench_rough1-2)
  already had mine slower at 16 rows (+16; +8..+49 by config), and I set them aside as noise. On a quiet GPU the sign
  is clear.

**Consequence.**
- Lever 2 is killed by its pre-registered line.
- Told the orchestrator and Leap (00:3x). My recommendation to Leap is to drop E14's A16m / C32onem arms: at ~+30 us x
  48 layers they would cost ~+1.5 ms per 16-row check. Leap's 7d93cab stays inert while E14_L2 is unset.
- What stays: the exact construction (108 / 108 bitwise) and kernel B, -4.9 us per layer in the pieces. B alone is worth
  ~0.2 ms per check at best, so it is not proposed.
