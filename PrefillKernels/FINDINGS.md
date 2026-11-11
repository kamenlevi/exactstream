# PrefillKernels: 125B prefill toward ~1,000 tok/s, NAX, sync overhead, verify kernels

Taken over from MoonShots' engineering queue (2026-10-18 23:40). Model: Qwen3.8-Flash-Next 125B 4-bit (MTPLX qwen4_exp,
48 layers = 36 GDN + 12 QSA, 512 experts top-10, hidden 2560, expert 640) streamed by BigMoE on the 64 GB M5 Pro,
MLX 0.32.2 (BigMoE .venv). Exact = bit-identical to BigMoE's reference (LEDGER). Other repos are read-only; patches are
handed over as one-command A/Bs.

## Status (kept current)

| # | item | result | exact? | next |
|---|---|---|---|---|
| N1 | Does MLX use the M5 neural accelerators (NAX) at prefill? | **Yes, already.** Dense 4-bit qmm 25-27 TFLOPS at M >= 2048; routed experts 10.8-12.6 TF at 2K-token chunks, 18-20 at 8K, 22-23 at 16K; bf16 GEMM 28-30 TF. No NAX at decode/verify widths (S <= 8) | deterministic | - |
| N2 | Routed experts: same rows at other groupings | **bit-identical** (sorted gather_qmm, bm 32 vs 64, 2-4 chunks grouped, shifted boundaries) | yes | PK2 |
| N3 | Which op makes PF3 non-identical? | (1) **MLX 0.32.2 bug**: gather_qmm_rhs_nax at bm 64 returns wrong rows when the routed row count is not a multiple of 64; (2) `shared_expert_gate` ([1, hidden], N=1) through qmm_splitk with split_k = 512/ceil(M/32) (8 at 2K rows, 2 at 8K) | - | told MoonShots, BigMoE |
| PK1 | sync-free layer-major prefill (FULLSCR scratch, no CPU id reads, async eval) | 35B 8K: bit-identical; 5.4-5.75 s vs lmp 5.0 (no gain alone on the 35B) | **yes** (cache + tokens) | 125B night |
| PK2 | PK1 + routed experts over 8K-token groups (block forward per chunk, switch_mlp stubbed), groups padded to 64-row multiples | 35B: bit-identical | **yes** | 125B night |
| PK3 | PK2 + one-pass Metal scratch gather (one indexed copy instead of concatenate + take) | 35B 8K: **4.24-4.38 s vs lmp 5.0 (-13-15%)**, ref 9.8 | **yes** | 125B night |
| PK4 | PK3 + PLE n-gram stage of all chunks precomputed on a thread pool (MTPLX's own worker method; no GPU drains) | MTPLX census: the stage = 2.3 s of host stalls per 16K prefill | by construction (+ per-chunk row check) | 125B night (pk4 mode) |
| PK5 | PK3 + expert-aligned SwitchGLU (each expert padded to tile multiples, MLX's NAX gather body as a custom kernel) | custom gather clone bit-identical at BM 64/128/256; BM 64 = MLX speed, taller tiles slower | by construction | unit test tonight (aligned_glu.py), 125B after |
| QK1 | exact masked softmax for the QSA dense-mask fallback (where pass folded; mirrors MLX's block/looped softmax) | est. -0.7 s at 16K (more at 32K) | by construction, lsize to confirm | unit test tonight (qk1.py) |
| GDN-R | gated-delta prefill recurrence with R value rows per simdgroup (same per-row arithmetic as mlx-lm's kernel) | est. -0.3-0.6 s at 16K | by construction, to confirm | unit test tonight (gdn_fast.py) |
| PKD1 | decode/verify: GDN in_proj x4, QSA q/k/v, router + shared gate/up + shared-expert gate as one dispatch each, via MLX's own qmv_wide_impl | est. -1.5-2.5 ms/verify pass | by construction | unit test (fused_qmv.py) + 125B decode A/B tonight |

## N1-N2: NAX probe (`scripts/nax_probe.py`, `data/nax_probe.txt`, 2026-10-18 23:53)

MLX 0.32.2 dispatch (mlx/backend/metal/quantized.cpp @ v0.32.2):
- dense QuantizedMatmul, M >= get_qmv_batch_limit (gen 17: 33 / 25 / 13 by K, N): `qmm_splitk` (non-NAX simdgroup MMA, split over K
  to ~512 threadgroups) which falls back to `qmm` -> `qmm_t_nax` (bm 32 if M <= 32 else 64, bn 64, bk 64) when split_k <= 1. At
  prefill widths with N >= 512 the split is 1 -> NAX. **Narrow outputs (N = 1: shared_expert_gate) stay on split-K with an
  M-dependent split -> M-dependent bits.**
- GatherQMM with M == 1, B >= 16, sorted rhs and B/E >= 4 (SwitchGLU's sorted prefill path, >= 64 ids): `gather_qmm_rhs_nax`,
  bm = 32 if rows/E < 64 else 64 (E = experts in the weight tensor), bn 64, bk 64.
- below those (decode / verify S <= 8): qmv / qmv_wide / gather_qmv. **No NAX on verify passes at S = 6-8**; MTPLX has an opt-in
  NAX verify qmm (MTPLX_NAX_VERIFY, m16 tile, rows padded to 16, K-split partial sums: deterministic, not bit-identical -> v2).
- MLX 0.32.2 has no NAX attention kernel (steel_attention only); MTPLX's QSA layers at < 32K context run MLX's unfused SDPA
  fallback (head dim 256 + bool mask), materializing the score plane.

Real layer-0 expert weights, skewed synthetic routing (top-10 of 512), median of 7:

| tokens/call | rows/expert | bm | gate (2560->640) | down (640->2560) | SwitchGLU | us/token | x48 layers x16K |
|---|---|---|---|---|---|---|---|
| 2,048 | 40 | 32 | 6.23 ms 10.8 TF | 5.34 ms 12.6 TF | 19.9 ms 10.1 TF | 9.73 | 7.65 s |
| 4,096 | 80 | 64 | 9.38 ms 14.3 TF | 7.97 ms 16.8 TF | 31.4 ms 12.8 TF | 7.66 | 6.03 s |
| 8,192 | 160 | 64 | 14.5 ms 18.5 TF | 13.2 ms 20.4 TF | 50.8 ms 15.9 TF | 6.20 | 4.88 s |
| 16,384 | 320 | 64 | 24.5 ms 21.9 TF | 23.6 ms 22.7 TF | 90.5 ms 17.8 TF | 5.53 | 4.35 s |

All deterministic (repeat bit-identical). Row invariance (part B): a 2,048-token chunk's rows computed inside a 2- or 4-chunk
group (bm 64) are **bit-identical** to the chunk alone (bm 32), also with the weights padded to 3E so the group runs at bm 32,
and with shifted chunk boundaries. So MLX's NAX gather is row-invariant across M, tile size and tile composition.

Dense 4-bit qmm (part C, NAX): GDN in_proj 2560->16480 27.6 TF at M=2048; out_proj 6144->2560 26.2; attention q+gate 27.1;
o_proj 25.9; shared expert 16-17 TF at 2048 rows (22-24 at 8192). Rows at M = 512 / 2048 / 8192 bit-identical.
bf16 GEMM (part D): HC up 320->10240 identical across M; **HC down 10240->320 differs between M=2048 and 8192 (350 elements,
max 0.0156)**; router-shaped bf16 2560->512 identical. NAX bf16 peak 28-30 TFLOPS (4096^3 / 8192^3).

Caveat: part E (verify widths) timed single small ops with an eval each, so it measures launch+sync overhead (~250 us), not
the kernels; redo with in-graph repetition if needed.

## 16K prefill budget (what 1,000 tok/s = 16.4 s needs)

Per-chunk costs at current kernels (2,048-token chunks; SpecDecode data/prefill_paths125.txt for GDN/QSA/HC, this repo for MoE):
routed MoE 19.9 ms (8K groups: 12.7), GDN 19.5 (projections 9.3 at 25.5 TF, rest 10.6), QSA 16.5 -> 106 ms as context grows
2K -> 16K (dense-mask fallback), HC read + write ~5.5, shared expert + router ~1.5, FULLSCR scratch ~25 ms per layer.
Sum at 16K with perfect overlap and no host overhead: MoE 7.65 (PK2: 4.9) + GDN 5.6 + QSA 5.8 + HC 2.1 + shared/router 0.6 +
scratch 1.2 + misc 0.5 = **23.5 s (700 tok/s), PK2 20.7 s (790 tok/s)**, vs 42.3 s measured (PF2, PF16 row) and 37.1 (PF3).
Beyond that: QSA attention (exact block-skip of fully masked KV tiles, or v2 NAX flash), GDN recurrence (v2), HC (MTPLX lanes).
GDN/QSA prefill paths belong to SpecDecode.

## N3: the MLX 0.32.2 NAX gather bug and the PF3 bisect (2026-10-19 00:05-00:09)

`scripts/align_probe.py` (`data/align_probe.txt`, `data/align_probe_fp32.txt`), real 125B layer-0 experts, SwitchGLU sorted path,
rows compared with an aligned 8,192-token call:

| tokens | routed rows | bm | rows % bm | vs aligned run |
|---|---|---|---|---|
| 8192, 8160, 6144, 4096, 2048 | x10 | 64/32 | 0 | identical |
| 8191 / 8190 / 6143 / 4095 / 4097 | 81910 / 81900 / 61430 / 40950 / 40970 | 64 | 54 / 44 / 54 / 54 / 10 | **~all tokens differ, max \|d\| 1.9-3.3** |
| 2047 / 2040 / 1000 | 20470 / 20400 / 10000 | 32 | 22 / 16 / 16 | identical |

fp32 dequantized reference (6 tokens of the 8,191 call): unaligned bm-64 call errs 0.19-0.65 (= max |ref| 0.23-0.65), aligned
call 0.001-0.003. Deterministic (repeat runs equal), so a repeat-sha test cannot catch it. Reference prefill (2,048-token chunks)
is bm 32 on the 125B (40 rows/expert) and aligned bm 64 on the 35B (2,048 x 8 / 256 = 64), its tails bm 32: safe.
Hit: PF3 (8K groups: a 16K prompt's second group is 8,191 tokens), any prefill chunk > ~3,276 tokens on the 125B (> 2,048 on the
35B) whose token count is not a multiple of 32 (125B, top-10) / 8 (35B, top-8). To check: MLX 0.32.4 (.venv324).

`scripts/moe_block_bisect.py` (`data/moe_block_bisect*.txt`): mlx-lm Qwen3NextSparseMoeBlock with real layer-0 weights, 4 chunks
of 2,048 vs one group of 8,192 rows: router logits (8-bit qmm) 0 differing, shared expert 0, routed experts 0, weighted top-k sum 0,
**shared_expert_gate 4,748 of 8,192 differ** (whole block 405,589). With a 2,047-token last chunk the grouped experts also hit the
bm-64 bug (35B and 125B). PK2's split (routed experts per group, rest of the block per chunk) is bit-identical and deterministic.

## PK1-PK3 on the streamed 35B (`patches/nosync_prefill/bench_pk35.py`, `data/pk35*.txt`)

Qwen3.6-35B-A3B 4-bit, BigMoE ExpertStreamer at 55 % (141/256), PF1 staging + FULLSCR, 8,192-token prompt, all modes in one
process from fresh caches, busy Mac (another 400 % CPU job):

| mode | prefill s | cache hash | tokens | peak GB |
|---|---|---|---|---|
| ref (chunk-major) | 9.82-10.06 | 4e2375fbf7 | 9897e36d3c | 14.3 |
| lmp (PF2 + FULLSCR) | 4.99-5.34 | = | = | 13.4 |
| pf3 (MoonShots) | 7.15 | 9258d44a29 (differs) | f1e99a9fc6 (differs) | 14.1 |
| pk1 | 4.24-5.75 (noisy; 5.40/5.75 in the clean repeat) | = | = | 14.6-14.7 |
| pk2 (before the pad fix: differs, = pf3's tokens) | 4.97 | = | = | 15.2 |
| pk3 | 4.24-4.38 | = | = | 15.0 |

## QSA exact-sparse design notes (for the QSA prefill kernel; decide after `qsa_tiles_16k`)

Stock lane below 32K history (MTPLX qwen4_exp, phase "unknown"): indexer -> dense bool mask [S, T] (512 selected 4-token
blocks + the 0-3-token tail) -> MLX SDPA fallback at head dim 256: `scores = (bf16(scale)*q) @ k^T` (bf16 plane
[24, S, T]), `where(mask, scores, bf16_min)`, `softmax(precise)`, `probs @ v`. At T = 16K one chunk writes and reads a
1.6 GB plane several times (~98 ms of the 106 ms QSA chunk).
- Softmax (mlx/backend/metal/kernels/softmax.h @ 0.32.2): T > 4096 -> `softmax_looped`, threadgroup = kernel max threads,
  N_READS-element chunks at offset r*lsize*N_READS + lid*N_READS, per-thread online (maxval, normalizer), then simd and
  threadgroup reductions. A fully masked chunk leaves the per-thread state bit-identical (bf16_min - max underflows
  fast::exp to exactly 0; the rescale factor is exp(0) = 1.0; a masked-only thread is zeroed in the reductions either
  way). Blocks are 4-token aligned, so with N_READS = 4 every chunk is fully selected or fully masked. So **a softmax that
  visits only selected chunks, in MLX's thread/chunk order, is bit-identical** and costs ~T_selected/T of the plane.
- Scores: each element is MLX's bf16 GEMM dot over D = 256; computing only selected (row, block) scores is exact if the
  NAX per-element result does not depend on the tile around it (true for qmm/gather in nax_probe B; to verify for the
  bf16 steel GEMM config MLX picks at K = 256).
- PV (`probs @ v`, K = T): exactness needs MLX's own K-tiling and accumulation (`acc = A_tile x B_tile + acc` per tile,
  NAX internal order inside a tile). Skipping is exact only for K-tiles whose probs are all zero for every row of a NAX
  fragment, so the gain is bounded by the (16-row x bk) tile coverage measured tonight; a per-row gathered NAX PV that
  replays the same per-tile accumulation is possible but is a multi-day kernel.
- v2 alternative: MTPLX's sparse flash lane (QSA_PREFILL, engages at >= 32K) ~50 ms/chunk at 16K vs 102 dense, max |d|
  0.009-0.016 (SpecDecode qsa_lane125.py), itself ~10x its floor.

### Bug source notes (quantized_nax.h @ 0.32.2, `affine_gather_qmm_rhs_nax`)
With `align_M` false only the last threadgroup's bounds change (`tgp_bm`, `sgp_sm`, `is_unaligned_sm`); every other tile
runs the same aligned branch. Yet ~90 % of tokens are wrong (T=4097: 3,681 of 4,097), so the failure is not the boundary
handling itself; it looks like code generation of the align_M=false specialization around #4352's per-simdgroup
`if (sg_active)` skip of `tile_matmad_nax` (the same change ml-explore/mlx#4403 blames for lost JACCL completions).
bm 32 (SM 16, TM 1) unaligned is correct. PK2 pads to 64-row multiples, so it only ever runs the align_M=true variant.
Not reported upstream (public action: Kamen's call); standalone repro `scripts/mlx_bug_repro.py` (random weights).

## Next exact MoE lever (tomorrow): taller NAX gather tiles
Per output element the NAX gather accumulates over K in SK = 32 steps (`tile_matmad_nax` on NAXTile<float, TM, TN>);
bm 32 (TM 1) and bm 64 (TM 2) give identical rows, so the per-element result does not depend on TM. A BM = 128 clone
(same SK/BK/K order) should also be bit-identical and cuts each expert's weight re-reads per group by 2x and the
expert-straddling tiles. Expected: 8K-group SwitchGLU 50.8 -> ~44 ms (GEMMs 42 -> ~36), about -0.6 s at 16K; with 16K
groups another ~-0.3 s. Needs the kernel body compiled outside MLX's metallib (mx.fast.metal_kernel + NAX headers).

## Morning GPU queue (after the night runners; my session takes no locks while mine runs)
1. `scripts/mlx_bug_repro.py` on .venv (0.32.2) and .venv324 (0.32.4): is the bm-64 gather bug fixed in 0.32.4? (its
   night job was dropped: my 00:30 `peer unlock` removed the runner's queue entry, which shares my session's identity)
2. `scripts/aligned_glu.py 64` / `32`: expert-aligned SwitchGLU, bit identity + speed vs mlx-lm SwitchGLU (2K/8K/16K)
3. `scripts/gdn_fast.py 4` / `2` / `8`: R-rows-per-simdgroup gated-delta recurrence, bitwise vs mlx-lm + speed
4. `scripts/qsa_exact_probe.py 16384`: key/row-subset score bit-identity, PV over visible keys, dense-band parity
5. `scripts/fused_router_shared.py`: router + shared gate/up + shared-expert gate in one dispatch via MLX's own
   qmv_wide_impl (bit-exact by construction), bitwise vs mx.quantized_matmul at M = 2..8, per-layer time
6. night results -> `python3 scripts/night_summary.py` -> RESULT lines to the orchestrator

## Method for exact fusion (item 4 and beyond)
MLX's wheel ships its kernel headers; `quantized.h` / `quantized_nax.h` hold the matvec/matmul bodies as METAL_FUNCs
(`qmv_fast_impl`, `qmv_impl`, `qmv_wide_impl`, the NAX tile loops). A custom `mx.fast.metal_kernel` that inlines those
headers (minus `utils.h`, which every custom kernel already has) and calls MLX's own implementation per output block
computes every element with MLX's code and lane layout -> bit-identical, while fusing launches, sharing input reads, or
changing only the tiling between independent rows (shown tonight: the NAX gather clone is bit-identical at BM 64/128/256).
For verify widths, affine `qmv_wide` uses k_lanes = 8 at every M and per-vector arithmetic independent of the vector tile
(vecs_per_tg = ceil(M / ceil(M/5))), which is why dup-row S=1 -> M=2 joins the S <= 8 band.

## Night results 1 (125B, 0.50 resident, BigMoE ref env + PF_PREFILL + PF_FULLSCR; data/night/)
**16K prefill (pk_16k.txt, 01:33, one run per mode, same process): every mode bit-identical** (cache c9896098e8,
32-token sha 185f02405a), incl. MTPLX's "prefill"-phase lanes (pk3p) and its dense-band SDPA (pk3pd) on the whole model.
| mode | s | tok/s | PLE stage s |
|---|---|---|---|
| lmp (PF2 + FULLSCR, synced) | 25.85 | 634 | 3.79 |
| pk1 (sync-free) | 29.67 | 552 | 3.23 |
| pk2 (+ 8K expert groups) | 30.38 | 539 | 3.34 |
| pk3 (+ scratch kernel) | 27.16 | 603 | 3.35 |
| pk3s (pk3, synced per chunk) | 29.07 | 564 | 3.34 |
| pk3p (pk3 + phase "prefill") | 26.23 | 625 | 3.34 |
| pk3pd (pk3p + dense-band SDPA) | 25.57 | 641 | 3.34 |
| pk4 (pk3 + PLE precompute) | **25.25** | **649** | 0 |
Reading: the PK MoE path is slower than lmp on the 125B (pk1 +3.8 s; pk3s vs pk3: async helps 1.9 s; cause open:
the extra live scratch (+1.4 GB, peak 47 vs 45.5 GB) or the per-layer flow). Exact wins on top of pk3: phase lanes
-0.9 s, dense band -0.7 s, **PLE precompute -1.9 s**. Next: lmp + PLE precompute + phase + dense band (all exact,
est. ~22.4 s = 730 tok/s), and find the PK path's extra cost.
**PKD1 (fused decode projections): exact** at decode (pkd1_decode: base/PKD1 sha 3934295b8c, 96 groups) and at verify
widths 2..8 (pkd1_spec: ar2 + spec k3 md7 all sha 457526d6ae); speed: no measurable change (spec t_verify 3.56 vs
3.60 s; ar2 26.5-30.5 noise) at 0.50 residency.
**MLX gather bug, random-weight repro (micro): not reproduced** (all rows ~2.4e-3 relative, aligned or not, bm 32/64,
E=128 K=1024); the real-weight SwitchGLU probe (align_probe) does show it -> the trigger needs more than M % 64.
**4K prefill (pk_4k.txt, 02:11): all exact** (cache 75e3dc03ae): lmp 7.73 s, pk1 11.38, pk3 7.75, pk3s 8.45, pk3p 7.25,
pk4 7.28. PK1's penalty is ~constant (+3.6 s at 4K and 16K) -> a per-layer cost of PK1's GPU scratch build
(concatenate + take with the previous scratch still alive), removed by PK3's one-pass scratch kernel. The 8K expert
grouping (PK2) gains nothing measurable on the 125B. Real exact wins: PLE precompute, MTPLX phase lanes, dense band.
**QSA tile coverage (qsa_tiles_16k.txt, 16K prompt, 96 masked SDPA calls, work-weighted): share of score tiles with a
visible key: 1x4 0.209, 4x4 0.327, 16x4 0.495, 16x16 0.696, 32x16 0.767, 32x32 0.828, 64x64 0.883, 128x128 0.896.**
An exact skip at NAX-tile granularity (16x16) keeps 70 % of the fallback's work (-30 %, ~-1.5 s at 16K); per-row
gathering (1x4) would keep 21 % but needs a row-gathered exact PV. Decision: QK1 (where-pass fold) + the v2 sparse lane
question go to the orchestrator; no tile-skip kernel for now.

## Night results 2 (runner 2, 03:06-03:17; all modes bit-identical to lmp at each length)
| prompt | lmp (2 runs) | lmp4 (+PLE precompute) | lmp4p (+phase "prefill") | lmp4pd (+dense band) (2 runs) |
|---|---|---|---|---|
| 4K | 7.89, 7.03 | 6.68 | 6.57 | 6.36, 6.55 (**-13 %, 634 tok/s**) |
| 16K | 25.78, 26.31 | 25.41 | 24.37 | 23.69, 24.23 (**-8 %, 684 tok/s**) |
| 32K | 59.08, 60.69 | - | - | 54.44, 55.09 (**-8.5 %, 598 tok/s**) |
Hashes: 4K 75e3dc03ae/98f73d01d5, 16K c9896098e8/185f02405a, 32K b9657beb7b/8379caa8cb (= the night-1 lmp runs).
**Why PK was slower (pkwhy_4k): PK1's GPU scratch build** (concatenate + take): lmp 7.95/6.79, pk1 11.54/9.06,
**pk1k = pk1 + one-pass scratch kernel 6.69/6.30 (fastest at 4K)**, pk3 (pk1k + 8K expert groups) 8.43 -> the grouping
does not pay on the 125B. pk3 with 16K groups at 16K: 25.23, 27.04 (noise ~2 s). 32K: lmp 58.96 vs pk3 59.41.
Next: pk1k + PLE precompute + phase + dense band (sync-free, no grouping) at 4K/16K.
PKD1 at S=4 / S=8 (pkd1_s4/s8): exact (sha 457526d6ae everywhere); speed not separable from run order (base runs first).
MTPLX's prefill profiler cannot run with BIGMOE_COMPILE_HC=1 (eval inside a compile trace).
**Sync-free stack vs lmp stack (data/nsk4_4k_16k.txt, 07:19; all exact within the run):** 4K: lmp 7.94 (cold), lmp4pd
6.77/6.43, nsk4pd 6.75/6.55, nsk4 6.78. 16K: lmp 25.95, **lmp4pd 24.09/23.78 (680-689 tok/s)**, nsk4pd 26.95/26.65.
The sync-free path matches at 4K and loses ~2.9 s at 16K (peak 46.9 vs 45.6 GB, first decode token 1.4 vs 0.6 s).
**Hand-over: lmp4pd** (patches/nosync_prefill/nsp_lmp4.py prefill_qwen4_lmp4 under attention_phase("prefill") with
MTPLX_QSA_DENSE_BAND_SDPA=1). PK1-PK5 stay research code.
**Micro tests (data/night/micro.txt, 01:16):**
- **GDN-R (scripts/gdn_fast.py): bit-identical y and state, R=4 1.42x at T=2048 (6.15 -> 4.32 ms per layer-chunk),
  1.36x at 512, 1.02x at 6; R=2 1.33x.** ~-0.5 s at 16K once wired in (next: lmp4pd + GDN-R).
- PK5 aligned SwitchGLU: exact, no gain (+8 % at 2K, -3 % at 8K, 0 at 16K) -> dropped.
- PKD1 fused_qmv: exact; per layer mixed (router+shared -15 us at M=2-4; GDN in_proj and QSA groups +9..+15 us) -> dropped.
- QSA probe: score subsets (keys or queries) bit-identical to the full GEMM's entries; PV over visible keys only: 2 of
  6,144 differ (K-tiling); MTPLX dense band == stock fallback (0 differing).
- MLX gather bug: random-weight direct gather_qmm shows no error on 0.32.2 or 0.32.4; only the real-weight SwitchGLU path
  (align_probe) reproduces it so far.
- QK1 did not compile (placeholder "AX" also replaced inside FLT_MAX) -> fix and rerun.
**Day runner 3 (07:55-):** QK1 bit-identical with MLX's looped-softmax threadgroup of 1024 (768/512/896 differ: the
layout matters) and in the full SDPA at T=16K (46.1 vs 53.7 ms stock; MTPLX's exact dense band removes the same where
pass, so QK1 adds ~nothing on top). GDN-R inside lmp4pd at 16K: 23.79/23.64 vs lmp4pd 23.97/23.52 s, exact, no gain
(engagement not yet verified; profiles pending). lmp 25.78.
4K (lmp4g_4k): lmp 7.93, lmp4pd 6.67/6.43, lmp4gpd 6.55/6.47 -> GDN-R no gain at 4K either.
09:14: prof_lmp_16k failed with `AttributeError: module 'bigmoe_ext' has no attribute 'adopt_full'` (BigMoE's
stream.py moved ahead of its built extension while integrating lmp4pd); the remaining day-3 jobs hit the same error.
Rerun the MTPLX profile (BIGMOE_COMPILE_HC=0) once BigMoE's build settles.

## Per-stage split at 16K (2026-10-20 07:5x, MTPLX_QWEN4_PREFILL_PROFILE=1, BIGMOE_COMPILE_HC=0, barriers inflate totals)
| stage | lmp (31.5 s) | lmp4pd (28.1 s) |
|---|---|---|
| MoE block (routed + shared + router, streamed) | 10.0 | 9.9 |
| QSA attention (12 layers x 8 chunks) | 8.1 | 7.8 |
| GDN (36 x 8) | 6.0 | 4.6 |
| HC read / write | 2.3 / 0.9 | 2.3 / 0.8 |
| QSA indexer | 0.8 | 0.8 |
| PLE layer | 0.2 | 0.2 |
GDN-R engages (72 fast calls, 0 stock at 4K) yet gives no end-to-end gain -> the recurrence is off the critical path.
**Road to 16.4 s (1,000 tok/s) from lmp4pd ~23.7 s needs ~-7.3 s; exact options are nearly spent.** v2 candidates by
size: QSA sparse lane (MTPLX flash/NAX, ~50 ms/chunk flat vs dense 100 ms at 16K: ~-3..-4 s), MoE prefill GEMM over
bigger groups in the lmp flow (isolated SwitchGLU 7.65 -> 4.9 s at 8K groups: up to ~-2.5 s, exact once padded but not
yet realized end to end), GDN chunked/NAX (MLX 0.32.4 mx.fast.gated_delta_update NAX: ~-1.5..-2 s), fused HC
read/write (~-1..-1.5 s). Together ~-8..-10 s -> ~14-16 s (1,000-1,170 tok/s), i.e. reachable only with the v2 batch.
