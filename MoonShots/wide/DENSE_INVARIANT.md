# L1-01: batch-invariant dense kernel for 8-16-row passes (design note, MoonShots, Sat 24 Oct)

NewIdeas' queue item L1-01: exact wide verify passes (spec == plain bit for bit, the v3 sense) without 5-row chunking.
Kill line (queue): the batch-invariant kernel is > 1.5x slower than MLX's best at M = 8.

## What exactness needs, and where it breaks today

Spec == plain holds when every op gives each row the same bits at any row count M (batch invariance), with the same
kernels on the decode path (M = 1). v3 gets there with BigMoE's fixes (fuse.py duprow, VERIFY_ROWS_SDPA, GDN_EXACT). For
the dense half:

- **4-bit dense (most dense bytes: GDN in_proj 23.7 MB, out_proj 8.8 MB, router + shared 2.6 MB per layer; lm_head).**
  MLX runs `qmv_fast` up to a shape-dependent batch limit, then `qmm` (tiled, different arithmetic). In `qmv_fast`
  (mlx/backend/metal/kernels/quantized.h) a threadgroup takes ONE input row (tid.x) and 8 output rows (2 simdgroups x 4).
  It accumulates 512-value K blocks per row with `qdot`, then `simd_sum`. So each row's bits depend only on that row:
  batch-invariant at every M below the limit. But the weights are read once per input row, so time grows with M:
  in_proj 93 us at M = 4 -> 181 us at M = 8, then 143 us at M = 16 once MLX switches to qmm (data/wide/dense_scaling.txt).
- **bf16 dense (HC mixers, block inject, PLE projections when bf16).** MLX's bf16 matmul switches kernels at 1 | 2-5 |
  6-12 rows (2560 -> 2560; SpecDecode B6). BigMoE runs M = 1 as [x; x] and verify widths 6..64 as <= 5-row chunks
  (BIGMOE_DUPROW_CHUNK=5). In v3 the HC mixers go through BigMoE's compiled HC read kernels, not MLX's matmul.
- **Price of chunking (measured earlier, GDN-layer dense shapes, us per call).** At M = 8 the 5-row chunks cost only
  ~+7 % over plain MLX (335 -> 358 us): the 4-bit layers are still in qmv's band, so only the bf16 mixers chunk. At
  M = 16, +63 % (426 -> 695 us). At M = 32, 2.6x. The "3.5-6.4x" quoted on the queue is the M = 32 per-shape figure.

## Design: multi-row qmv, bitwise equal to MLX's qmv per row

One threadgroup = 8 output rows x up to 16 input rows. Per 512-value K block, each thread loads its 4 output rows'
weight bytes (8 x uint32), scales and biases into registers ONCE. Then, for each input row m, it loads x_thread[m]
(16 pre-scaled values; the load_vector expressions) and runs MLX's own qdot arithmetic into result[m][row]. The
simd_sum and the bf16 store are unchanged. Every (m, row) sees exactly qmv's operations in qmv's order, so the bits
equal MLX's qmv at any M. That makes it exact against today's reference (decode and S <= limit verify already run
qmv): no reference change, unlike an MMA-tile kernel.

- Registers: result[16][4] = 64 floats + 16 x values + 8 weight words + 8 scale/bias, about 100. If occupancy suffers,
  process M in halves of 8 (weights read twice at M = 16, still 8x fewer reads than qmv).
- Expected cost: the weight read once (in_proj 23.7 MB at ~200-250 GB/s = ~95-120 us), with M x 42 M MACs well under
  it. Expected ~100 us at M = 8 (MLX qmv 181) and ~110 us at M = 16 (MLX qmm 143): exact AND faster than MLX's best.
  If that holds, it is a pass-time lever for line 1, not just an exactness fix.
- The same idea for MLX's bf16 band kernel is possible once that kernel is identified (gemv.h). The bf16 mixers are
  small (6.6 MB each); 5-row chunks cost ~+16 us per layer at M = 8.
- lm_head: if MLX keeps it in qmv's band at M = 8, each pass reads the 4-bit head (~318 MB) 8 times. Then the
  multi-row qmv saves the most there. The band map below says which.

## Cheapest decisive measurement (no build): b8mr/run.py denseinv

The model's real dense layers (B12_HC=0: bf16 mixers as in v3), every distinct (kind, N, K) shape:
1. For M = 1..16, is every row bitwise equal to its M = 1 result and its dup-row M = 2 result? This gives the band map:
   where qmv stops and qmm starts, per shape.
2. us per call at M = 1, 2, 4, 5, 8, 12, 16, plain vs chunked to the invariant band (48 real instances round-robin, so
   DRAM rather than cache).

Go / kill:
- GO (build the multi-row qmv, ~1 day) if the 4-bit layers are qmv-invariant across the band and qmv's time grows ~M.
  The kernel's target is then the weight-read time.
- KILL (or bf16-only) if MLX already keeps the 4-bit layers exact and flat at 8-16 rows, so nothing is left to gain.

## Results (Sat 10:06-10:10; data/b8mr/denseinv.txt, data/b8mr/qmv_mr.txt)
- Band map, MLX 0.32.2, 17 real dense shapes, bitwise per row vs the M=1 and dup-row M=2 results:
  - 4-bit big layers (in_proj_qkv 10240x2560, in_proj_z 6144x2560, o_proj 2560x6144, q_proj 12288x2560, lm_head
    248320x2560): bands {1} {2-12} {13+}. Small 4-bit layers (shared expert, v_proj, in_proj_a, gate): 2-16.
    Dup-row decode sits in 2-12, so these are exact up to 12 rows with no chunking.
  - bf16: PLE key/value proj band 2-5 (5-row chunks: +60 / +25 us at 8 rows); HC mixers on MLX's eager path: up
    1-5, down 1-15; block inject 1-16.
- MLX us per call in the exact band vs qmm at 16 rows (M = 2 / 8 / 12 | 16):
  lm_head 1008 / 4280 / 7172 | 2201; in_proj_qkv 54 / 116 / 199 | 90; q_proj 63 / 148 / 260 | 112;
  in_proj_z 34 / 73 / 114 | 161; o_proj 35 / 76 / 114 | 154.
- Prototype (multi-row qmv): bitwise equal to MLX's qmv per row at M = 1..16 (mr 16 and 8), but compute-bound:
  in_proj M = 8 302 us (mr8) vs MLX's best 188 (1.6x); M = 16 608 vs 145 (4x); lm_head M = 8 4612 vs 3038.
  KILLED by the queue's line (> 1.5x MLX's best at M = 8). It also matches MLX's M = 1 kernel, while today's
  reference is the 2-12 band.
- Lever left (L1-01b): per-op kernel choice, fixed for every M including decode. lm_head, in_proj_qkv and q_proj run
  as 16-row padded qmm; the rest stay in the band. Exact under its own numerics: ~-3.4 ms per 8-row pass (lm_head
  -2.1 ms), ~+2.7 ms per decode step. Next check: qmm rows independent of the padding rows' values, then a quality
  check of the new reference.
