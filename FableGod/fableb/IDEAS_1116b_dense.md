# IDEAS_1116b: the dense pass of the 32-row lossy check (Fable B, Mon 16 Nov, 17:5x-18:3x)

Lens: PUSH800 section 3, "the dense pass: 8.5 ms of dense compute at 32 rows: cut it". Labels: [K] measured, [P] from
measured parts, [G] estimate (gains halved). Sources: Leap's check25_plan.md + check25/README.md (run2b, the dense
table, hc8r32, int8d), PUSH800 (NewIdeas' 17:5x correction: bounds 27.7 / 25.2 / 24.0 ms), DENSE_REBUILD.md,
IDEAS_1116_check25.md. Screen script: think1116b/hc_statsfwd_check.py (seconds, < 100 MB).

## 0. Where the dense pass really is (the frame's "8.5 ms compute" is the wrong wall)

| part of the 32-row check [K, run2b / bench] | ms | its floor | x |
|---|---:|---:|---:|
| dense matmuls, dn32 one-pass | 12.97 | 6.90 = sum of max(bytes, matrix) | 1.9 |
| of which the three big shapes | ~9.1 | 36 x 95.9 + 48 x 36.0 + 12 x 71.9 = 6.04 | 1.5 |
| of which the small shapes (b\|a, k\|v, indexer, shared) | ~3.9 | ~0.3 | 3-34 |
| HC reads, 96 per check, chained | 8.14 | 2.25 (bytes) | 3.6 |
| GDN recurrence (mlx_lm step at T 32) | 3.86 | 8.6 us + 3.08 us per row per layer [P fit] | row-sequential |
| router ~1.7, attention ~1, glue ~3 | ~5.7 | | |
| **dense side** | **~31** | | |

- **Compute is NOT the dense wall.** At 32 rows bf16 the matrix floor is only 1.14x the byte floor (GDN qkv|z 95.9 vs
  84.3 us; out_proj 36.0 vs 31.6; att q 71.9 vs 63.2). The whole "compute-bound" excess is 8.5 - 7.6 = **0.9 ms** at
  100 % of both peaks. Below ~28 rows the matmuls are byte-bound at bf16; with int8 math they are byte-bound at every
  row count.
- **So int8 / int4 activations, row pruning and sibling sharing have a ceiling of ~0.9 ms on the matmuls.** Measured:
  dn32q saves 0.29 ms (bench, with its quantizer) and costs **+1.13 ms in the engine** (F32if - F32df; the quantizer is
  one more dependent dispatch per matmul on a CPU-bound engine).
- **The dense excess is 12.97 - 6.90 = 6.1 ms of latency and fill, plus 5.9 ms of HC-read fixed cost, plus ~4 ms of
  router / glue launches.** That is 16 ms of quality-free room on the dense side; nothing in it needs a gate.
- **The engine is CPU-bound today:** run2b's F32c6i is build + submit 50.95 ms, GPU wait 5.05, commit 0.35. The GPU
  finishes 5 ms after the CPU stops encoding. Every launch removed pays ~7 us of CPU and ~1.6 us + its fill on the GPU.
  Until the CPU is under the GPU, a dense ms is worth less than a dense launch. (Fable C audits section 1; this is the
  one number I would add to it: the 56.4 is a CPU number, so "42 % of the bound" compares a CPU time with a GPU bound.)

- **CORRECTION (Leap 18:2x):** "build + submit" is the wall of the forward call, not CPU time; the slice preview puts
  the CPU at ~30-35 ms per check [P] (Python build ~0.45 ms per layer, encode ~3 us per dispatch). Whether the check is
  CPU-paced is tonight's GPU-busy measurement (Leap bets ~50 ms GPU, the orchestrator ~35-40). The launch-count
  argument stands either way; the "CPU-bound by 45 ms" reading above does not.

## 1. STATS-FORWARD: the HC read in one launch (best idea) [P] -2.5..-4.5 ms, no quality risk

- **Mechanism.** The HC read is three chained launches per layer: K1C (per-K-tile partial projections of the 4 streams
  through the lowrank mixer), K2B (sum the 64 tile partials, gate coefficients), K3C (the up mix, 3.8 MB of weights). Leap
  measured the read at 2.8x its byte floor and named the cause: the fixed cost of three chained launches
  (README "More K1C threadgroups don't help"). The streams K1C reads were WRITTEN by the previous layer's HC writer,
  which holds the same rows and the same K tiles in hand. So the writer computes K1C's partials as it writes (same tiles,
  same order, same fp32 math: bitwise the same partials; screen below), and its last threadgroup (atomic ticket, as in
  Leap's "fold K2B into K1C's last threadgroup" option) runs K2B. The read of the next layer is K3C alone: **one launch
  per read instead of three**. The writer is the pending write, so it stays in the layer chain, but the K1C -> K2B ->
  K3C chain of three dependent launches becomes one launch inside a kernel that already runs.
- **Bytes.** Unchanged: the writer reads the next layer's mixer weights (3.3 MB per read, +1.1 ms at 280 GB/s inside a
  kernel that already streams); the read keeps K3C's 3.8 MB. Net bytes 0.
- **Price.** Chained read today 8.14 ms = 96 x 85 us. One-launch K3C at 1.3x its own bytes: 96 x (1.6 + 1.3 x 13.6) =
  1.9 ms; the writer grows by ~1.3 ms (its K1C tiles at 280 GB/s plus one reduction). **~3.2 ms vs 8.14: -4.9 [P];
  halved for the chain-latency part I cannot measure: -2.5..-4.5.** Launches -192 per check (CPU -1.3 ms).
- **Screen (done, seconds, CPU):** think1116b/hc_statsfwd_check.py: the writer's per-tile partials equal K1C's bitwise
  and sum bitwise in K2B's order (True / True on random data at the real shapes 32 x 4 x 2560 -> 320). The real test is
  Leap's chained bench: PASS <= 5.0 ms per check for the 96 reads (writer + read), KILL > 7.0.
- **Quality risk:** none (same arithmetic, same order; for the exact track bitwise by construction if the tile order is
  kept; the lossy track needs only determinism).
- **What kills it:** (a) the HC writer is itself tiled across rows, not K, so it does not hold whole K tiles of a row
  (then the partials need a [tiles][rows][4][320] fp32 buffer, 10 MB per layer: still one launch, +0.5 ms of bytes);
  (b) the atomic-ticket last threadgroup serializes 64 partial sums per row (K2B's 9.4-10.5 us chain, my hc_k2c lesson:
  price it chained only); (c) the writer is not the one kernel I think it is (hc8r32 has its own write path; Leap knows).

## 2. The dense matmuls to 1.15x bytes: -4.2 ms [P], quality-free; half of it lands by Friday

- **Small shapes first (-2.4 ms [P]):** ~3.9 ms per check at 3-34x their floors is per-dispatch fixed cost, not kernel
  speed. The folds exist: a|b into qkv|z (tiles 512-514, bitwise per tile), k|v into q (416 tiles), router + shared
  gate|up + shared down as one stream, the indexer into the attention q call. ksplan.py prices each; Leap's README says
  the same ("they need folding, not faster kernels"). Drafted, not run. **By Friday: yes** (bitwise, no gate; -1.5..-2.4
  ms GPU and -300..-400 launches = -2..-3 ms CPU).
- **Big shapes (-1.8 ms [P]):** 1.3-1.6x max(bytes, matrix). The fill model (DENSE_REBUILD section 1: t = 1.6 + 1.15 us
  x max(KG/ks, tiles x KG/C)) says the GDN qkv|z call is 512 tiles over C ~230 simdgroups = 2.2 waves with a 20 %-empty
  tail wave; out_proj 80 tiles at ks 6 is fill-bound. Double-buffered staging (two K groups in flight per simdgroup) and
  a K split that makes the wave count integral (ks 2 at 512 tiles -> 1,024 units = 4.5 waves; ks 3 -> 6.7; pick ks so
  that units / C is near an integer) are the two known levers. **By Friday: maybe** (Leap's tuning line; -0.9 halved).
- **Screen:** the fill model on each shape (ksplan.py, seconds) vs Leap's measured us per call; PASS if the model is
  within 15 % on the three big shapes, which makes its ks / staging prediction trustworthy. GPU test is Leap's bench.
- **What kills it:** the matrix-unit issue rate at M 32 (dn32's bf16 x uint4 from threadgroup memory) is the bound, not
  fill. Then the big shapes stay at ~1.3x and only the folds pay.

## 3. Take int8 dense OFF the G1 gate (an audit, 0 ms, protects a quality budget)

- G1 = C6 + int8. C6 (lambda 1.8e-2) is bytes: F32if6 - F32if = -1.20 ms [K]. int8 activations are compute: bench -0.29,
  engine **+1.13** [K], ceiling 0.9 ms at 100 % of both peaks (section 0), and they carry a quality cost (int8 error
  6.6-7.9e-3 vs bf16 2.9-3.9e-3 on every FAST dense shape [K]).
- **Ask:** gate C6 alone (the lambda is what pays); keep int8 as a timing-only arm. The freed gate slot goes to a lever
  with kept in it (#8 span-conditional close-call, #13), which is the binding wall (kept x1.43-1.62 for 775 at the
  corrected bounds).
- Int4 activations: the matrix units have no int4 path; nothing to gain even in principle. Dead at the desk.

## 4. Row pruning and sibling sharing: what they are worth ON THE DENSE SIDE (for Wednesday's early-death pricing)

- **Matmuls:** 0 below 28 rows (byte-bound), <= 0.9 ms at 32 -> 0 rows. A probe that drops 13 of 32 rows after layer 16
  saves ~0.1 ms of matmul compute.
- **GDN step:** 8.6 us + 3.08 us per row per layer [P fit on 57.9 / 107.2 us at 16 / 32 rows]. Drop 16 rows after
  layer 8: -1.48 ms; 13 after 16: -0.96; 10 after 24: -0.55 (script). Attention and HC per-row parts: ~-0.3 more.
- **So early death's dense value is ~1-1.5 ms; its expert value is ~4-5 ms** (ED-CHAIN: -26 % expert bytes at p 0.95
  after layer 17, on 18 ms of experts at real sizes). Price Wednesday's probe on the experts; the dense side is a rounding
  term. This also means a dense-only early death (drop rows from the dense pass but keep their experts) is worthless.
- **Sibling sharing in the GDN step (tree-level-parallel recurrence, sequential length = tree depth ~10 instead of 32):
  KILL at the desk.** The per-node state is 48 heads x 128 x 128 fp32 = 3 MB; a level-parallel form passes parent states
  through memory: 32 nodes x 6 MB x 36 layers = 7 GB per check. The DFS walk with fork states in registers (Leap's
  TREE16 L1) is the right form; it is sequential over the 32 nodes by nature (3.08 us each). Two siblings per thread in
  registers needs 3 x 128 floats per thread: over the register budget. Recorded so nobody re-proposes it.
- **Sibling sharing in the matmuls:** none exists; sibling rows have different inputs at every layer. Dead.

## 5. The dispatch diet: what lands by Friday, and what each is worth (CPU 51 ms is the wall)

| item | launches | GPU ms | CPU ms (7 us each) | state | Friday? |
|---|---:|---:|---:|---|---|
| att_single / fs4 (single-call SDPA) | -672 | -4.1 [K FLAT] | -4.7 | in the stack (R-62) | **yes, wired in f32one** |
| the folds a\|b, k\|v, router + shared, indexer | -300..-400 | -1.5..-2.4 [P] | -2..-3 | drafted, bitwise per tile | **yes** |
| t1cf fused 1c tiers (replaces eprep under 2-D nolsb) | -1,350 | +0.49 [K, open] | -9.5 | built; slower on GPU, open | yes if the +0.49 is found |
| STATS-FORWARD HC read (section 1) | -192 | -2.5..-4.5 [P] | -1.3 | this note | Thu, if Leap takes it |
| GDNB (fused GDN body) | -144 | -0.5 [G] | -1.0 | Leap's | Wed-Thu |
| HC-read glue, GDN / attention glue at 32 rows (eager MTPLX ops) | -100..-200 | -0.5 | -1..-1.4 | not started | partial |
| **sum** | **-2,750..-2,950 of 4,022** | **-9..-12** | **-19..-21** | | |

- With all of it the check is ~CPU 30 / GPU ~40-44 ms: still CPU-bound by a hair, at ~44 ms wall [P]. That is the
  Friday number the dense side can defend: 56.4 -> ~42-46 (Leap's own 40-46 agrees).
- **What it is worth for 775:** x1.25-1.3 on tok/s at fixed kept. The target needs x3.4-3.6 over today's 215-227. So
  the dense side buys a third of the log-distance; kept x1.43-1.62 and the 3 ms draft are the other two thirds, and
  nothing in my lens touches them.

## 6. Beyond Friday, the one dense lever that changes the frame: a replayed dense pass (ICB) [G]

- At 32 rows every dense shape is fixed (rows padded to 32); only the expert call's list changes per check. So the dense
  DAG per layer (~60 of the ~85 launches per layer) can be pre-encoded once into an indirect command buffer with
  fixed activation buffers and replayed every check; only the expert one-call and the 1c glue are encoded live. CPU
  51 -> ~12-15 ms [G]; the check becomes GPU-bound at whatever the GPU sum is (~35-40 by Friday's parts, ~30 with the
  fused-layer rebuild). Needs the dense pass outside MLX (my DENSE_REBUILD kernels are already raw Metal; the glue is
  the work). 2-3 weeks; the same class Leap lists as "replayed command path". It is the only route I see to a check
  under 30 ms wall, and it is quality-free.

## 7. Bottom line (<= 15 lines)

1. The dense "8.5 ms compute" is not a wall: at 32 rows the matrix floor is 1.14x the byte floor; int8 / int4 / row
   pruning / sibling sharing have a 0.9 ms ceiling on the matmuls, and int8 measured +1.13 ms in the engine.
2. The dense excess is latency and launches: matmuls 1.9x floor, HC reads 3.6x, ~4 ms of router / glue; ~16 ms of
   quality-free room, on a check that is CPU-bound (build 51 ms, GPU wait 5).
3. Best idea: STATS-FORWARD, the HC read in one launch (the previous layer's writer emits K1C's partials and K2B's
   coefficients): -2.5..-4.5 ms [P], -192 launches, no quality risk; CPU screen PASSed (bitwise partials); Leap's
   chained bench decides (PASS <= 5.0 ms per check for the 96 reads).
4. By Friday on the dense side: SDPA single-call (in), the folds (-1.5..-2.4), t1cf if its +0.49 is found, STATS-FORWARD
   if taken: 56.4 -> ~42-46 ms wall, x1.25-1.3 tok/s. Not more.
5. Take int8 off the G1 gate: it does not pay and it spends a quality gate; gate C6 alone.
6. Early death's dense value is ~1-1.5 ms (the GDN step's 3.08 us per row-layer); price it on the experts (~4-5 ms).
7. Under 30 ms wall needs the replayed dense pass (ICB) or the fused-layer rebuild: weeks, quality-free, the only
   dense route that changes the frame.
