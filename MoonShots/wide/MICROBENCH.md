# Grouped expert kernel: does a wide verify pass get to read each expert once? (MoonShots, 2026-10-22; the 350 road)

**RESULT (13:50, one real 125B layer of v3 pools in RAM; 3 layers x 3 routing traces):**
- **Reading each expert once buys NOTHING by itself.** Today's 3c kernel already gets most of the DRAM reuse from the
  cache, and its cost is per-pair work: ~6.0 µs per (row, rank) pair at S >= 8, the same whether the pass touches 16 or
  150 distinct experts. Expert-major kernels that keep that per-pair work are 10-20 % SLOWER. That holds for MLX's
  sorted gather_qmm and for my bit-identical grouped 3c.
- **The lever is a kernel that dequantizes each expert ONCE and applies it to all of its rows.** Its per-MoE-layer
  bound is the measured read-once floor:
  - natural union: 1.4-3.1x at S=16 and 1.6-4.3x at S=32, depending on the layer;
  - budget set U=32 at S=16: 3.5-3.7x;
  - U=48 at S=32: 4.8-4.9x.
- **The budget routing pays only with such a kernel.** With today's 3c it changes nothing (0.958 vs 0.971 ms at S=16).
- At the pass level (MoE is about half of a wide all-hits pass, S=16), that kernel would give ~1.2-1.3x with the natural
  union and ~1.4-1.5x with U=32 (upper bound ~1.55x). So path350's ~1.4x needs BOTH the new kernel and the budget routing.
- **14:49 update (section 5):** the bit-identical dequant-once form of 3c is built and exact, but 1.6-2x SLOWER. Holding
  unpacked terms costs more registers than re-unpacking saves on this GPU. The remaining route is an MMA-tile kernel; it
  is not bit-identical and was not built today. Until then the "one read per expert" lever is unrealized.


NewIdeas' question: path350 assumes ~1.4x from wide verify passes (S = 8-32 rows) that read each distinct expert once.
How much expert-weight time does that save per MoE layer? Two cases: the natural union of the rows' top-10, and a
per-pass budget set (TrainLab's batch-aware routing, U = 32 at 16 rows).

## 1. How many distinct experts a pass touches (CPU; wide/union_sizes.py)
Source: MoEPrefetch's 125B decode traces (exact model, greedy, router top-20 per token per layer, 400 tokens). A pass is
S consecutive tokens; figures are the mean over 48 layers and all passes.

| trace | S = 4 | S = 8 | S = 16 | S = 32 |
|---|---|---|---|---|
| code | 26.9 (0.67 of S x 10) | 43.1 (0.54) | 67.3 (0.42) | 101.4 (0.32) |
| chat | 25.8 (0.64) | 41.3 (0.52) | 64.7 (0.40) | 95.2 (0.30) |
| math | 27.8 (0.70) | 46.3 (0.58) | 72.9 (0.46) | 111.1 (0.35) |
| bulgarian | 22.8 (0.57) | 35.3 (0.44) | 56.1 (0.35) | 86.5 (0.27) |

These match TrainLab's natural U(16): 69.9 / 64.6 / 72.8 / 56.7.

**Upper bound on the expert-byte cut**, comparing one read per (row, rank) pair with one read per distinct expert:
- natural union: 1.5x at S=4, 1.9x at S=8, 2.4x at S=16, 3.1x at S=32 (code);
- budget set U=32 at S=16: 5.0x; U=48 / 64 at S=32: 6.7x / 5.0x.

## 2. Measured per MoE layer (wide/microbench.py; data/wide/microbench_L*_*.json; 13:48-13:50)
Setup: one real layer's v3 pools (records3 MSB-3 + LSB-1, all 512 experts in RAM), routed experts only, chained calls.
Each cell is the mean over 6 passes of the trace. ms per layer:

| layer / trace | S | set | distinct | 3c (today) | MLX sorted (expert-major) | 3c_grp (mine, expert-major, == 3c bitwise) | read once (floor) | 3c / read-once |
|---|---|---|---|---|---|---|---|---|
| L20 code | 4 | natural | 26.2 | 0.265 | 0.325 | 0.309 | 0.205 | 1.30 |
| L20 code | 8 | natural | 41.2 | 0.501 | 0.603 | 0.572 | 0.337 | 1.49 |
| L20 code | 8 | U=16 | 16 | 0.488 | 0.555 | 0.590 | 0.141 | 3.46 |
| L20 code | 16 | natural | 64.4 | 0.971 | 1.139 | 1.092 | 0.519 | 1.87 |
| L20 code | 16 | U=32 | 32 | 0.958 | 1.077 | 1.112 | 0.272 | 3.51 |
| L20 code | 32 | natural | 96.4 | 1.915 | 2.179 | 2.138 | 0.724 | 2.64 |
| L20 code | 32 | U=48 | 48 | 1.897 | 2.091 | 2.164 | 0.395 | 4.80 |
| L40 chat | 16 | natural | 39.0 | 0.959 | 1.094 | 1.119 | 0.308 | 3.11 |
| L40 chat | 16 | U=32 | 31.3 | 0.957 | 1.072 | 1.121 | 0.258 | 3.71 |
| L40 chat | 32 | natural | 54.2 | 1.900 | 2.126 | 2.179 | 0.442 | 4.30 |
| L6 math | 16 | natural | 96.4 | 1.026 | 1.183 | 1.106 | 0.748 | 1.37 |
| L6 math | 16 | U=32 | 32 | 0.958 | 1.083 | 1.118 | 0.272 | 3.53 |
| L6 math | 32 | natural | 151.8 | 2.012 | 2.235 | 2.112 | 1.223 | 1.65 |

- 3c on MSB-3 only (plan 3333333333) runs 6-7 % below 3c. MLX unsorted gather_qmm runs 23-28 % above 3c.
- Checks: 3c_3 == MLX unsorted == MLX sorted (max diff 0); 3c_grp == 3c bit for bit (all 3 runs).
- "read once" = the 3c kernel once per distinct expert, with one row. It matches distinct x 2.15 MB at ~270 GB/s to
  within 10 % in every cell, so it is the DRAM floor. My separate bandwidth probe (an integer reduction) was unreliable
  (52-219 GB/s) and is not used.
- Why the floor is far below 3c at S >= 8: today's time is about distinct x 8 µs (DRAM) + repeated pairs x ~4 µs. The
  ~4 µs is dequantizing and dotting a cached expert again for each further row.

## 3. The dense half of a wide pass (wide/dense_scaling.py; MLX 4-bit / bf16 matmuls, one GDN layer's dense shapes)

| rows | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| plain MLX matmul, sum of 5 shapes | 200 µs | 335 µs | 426 µs | 443 µs |
| today's 5-row chunks (BIGMOE_DUPROW_CHUNK=5) | 200 µs | 358 µs | 695 µs | 1,278 µs |

- With plain matmuls the dense work grows only ~2.1-2.2x from 4 to 16-32 rows. MLX switches to its tiled kernel at
  M >= 16; M = 8 is its worst spot.
- Today's chunking would cost 3.5-6.4x. Wide passes need plain matmuls; whether that is exact against narrower passes
  is a numerics question, not a speed one.

## 4. What it means for path350 (all-hits, per 48-layer pass; GPU only, no SSD)
Today at S=4 the pass is 37.9 ms: ~13 ms routed MoE (48 x ~0.27) + ~25 ms everything else.

| S = 16 rows | routed MoE | rest of the pass (dense x2.1, a rough scale-up) | pass |
|---|---|---|---|
| today's kernels | 48 x 0.97 = 46.6 ms | ~50 ms | ~97 ms |
| dequant-once kernel, natural union (L20 code: floor 0.52 ms; ~0.62 with ~1 µs per extra row, my estimate) | 25-30 ms | ~50 ms | ~75-80 ms (1.2-1.3x) |
| dequant-once kernel + budget U=32 (floor 0.27 ms; ~0.40 estimated) | 13-19 ms | ~50 ms | ~63-69 ms (1.4-1.5x) |

So the "one read per expert" piece is real only as a **multi-row dequant-once kernel** (GEMM-like: unpack each 3-bit
weight once into registers, FMA it into every row of that expert). That kernel does not exist yet; MLX's sorted qmm is
not it at these row counts. Natural routing gives ~1.2-1.3x per wide pass; with TrainLab's budget set ~1.4-1.5x. The
rest of the pass (dense, attention, GDN at S rows) then becomes the larger half.

## 5. The dequant-once multi-row kernel, built and measured (14:02-14:49; wide/mr3c.py, wide/run_mr*.sh)
NewIdeas asked for it (14:00), bit-identical to 3c per row so the exact main track could use it at S=4 too.

Design:
- one threadgroup per (output tile, chunk of <= RM pairs of one expert and tier), grouped by a GPU csr kernel;
- per lane and block, the code bytes are loaded once, then masked and converted once per weight row (MLX's qdot terms
  float(w & mask));
- then EXACTLY MLX's qdot arithmetic per pair: the same pre-scaled x, term order, scale * accum + sum * bias, block order
  and simd_sum;
- the same LY=8 combine.

**Correctness: BIT-IDENTICAL to 3c** at S = 4 / 8 / 16 / 32 on L20 code, L40 chat and L6 math (max |diff| 0 in every
check).

**Speed: 1.6-2x SLOWER than 3c everywhere** (data/wide/mr_*.txt, mr2_*.txt):

| variant (RA, RB, RM) | S=4 (3c 0.266 ms) | S=16 natural (3c 0.969) | S=16 U=32 (3c 0.958) |
|---|---|---|---|
| 2, 2, 1 (no amortization) | 0.426 | 1.553 | 1.538 |
| 2, 2, 8 | 0.527 | 1.780 | 1.554 |
| 4, 4, 1 | 0.544 | 1.943 | 1.869 |
| 4, 4, 4 | 0.589 | 1.902 | 1.689 |
| 4, 4, 8 | 0.662 | 2.101 | 1.859 |

- Even with nothing amortized (RM=1) the restructured kernel costs 1.6x. Three things add up there: two output rows per
  threadgroup instead of 3c's four (twice the x preparation), an unfused combine, and the pair indirection.
- MORE rows per chunk make it SLOWER, not faster. Holding the unpacked terms in registers (20 floats per weight row per
  16 values) costs more than re-unpacking them per pair; RA=4 is worse still.
- 3c's per-pair loop (unpack inline, a few registers, high occupancy) is close to the best this formulation gets on this
  GPU.

**What is left:**
- A different arithmetic: dequantize a weight tile once into threadgroup memory, then simdgroup-matrix MMA over the
  expert's rows. That is how MLX's qmm works, and MLX's sorted qmm was also slower at these row counts (section 2).
- Such a kernel changes the summation order, so it cannot be bit-identical to 3c. It would need its own quality check, or
  a new exact reference.
- Not built today.

**S=4 (today's 4-row MTP verify):**
- The bound per MoE layer is the read-once floor, 0.205-0.258 vs 3c 0.265-0.292 ms (1.13-1.30x by layer).
- Per in-RAM pass that is at most ~1.06-1.08x (MoE ~13 of 37.9 ms). At 24 GB, where the pass is mostly SSD-bound, it is
  ~2-3 %.
- No kernel reaches it yet: the bit-identical dequant-once form is slower.

Files:
- wide/microbench.py, wide/grouped3c.py (the bitwise expert-major 3c), wide/union_sizes.py, wide/dense_scaling.py
- data/wide/*.json / *.txt
