> (a quote from the project owner, kept private)

**Starting point.**
- Today: ~440 tok/s (PF-A at 32 GB) and 555 (fixed PF-B, research copy).
- 1.5K tok/s = **0.667 ms per prompt token** in all; a 30K prompt in 20 s.
- PF-A's first token is about SSD bytes / 11.3 GB/s plus 1.5 ms per token. The expert reads (about 6 s at 7.9K) are
  serialized with compute (TABLE_350, Sun 08:13).

## Prior art (read first, 11-11 12:1x)

**PrefillKernels (FINDINGS.md, 10-18..10-20)** measured the stages of a 16K layer-major prefill:

| stage at 16K (lmp4pd, 28.1 s with barriers) | seconds |
|---|---|
| MoE block (streamed) | 9.9 |
| QSA attention | 7.8 |
| GDN | 4.6 |
| HC read / write | 3.1 |
| QSA indexer | 0.8 |

Its conclusion: the exact options were "nearly spent" near ~700 tok/s, and 1,000-1,170 needed the non-exact v2 batch
(QSA sparse lane, GDN chunked / NAX, MoE bigger groups, fused HC). Since then PF-B's padded one-call-per-layer MoE became
exact.

**Exact building blocks known:**
- **Score subsets are bit-identical.** Computing only some key columns or query rows gives the full GEMM's entries.
- **The softmax can visit only the selected chunks.** MLX's looped softmax over selected chunks only, in MLX's thread and
  chunk order, is bit-identical.
- **PV skips are limited.** PV can skip only whole all-zero tiles of MLX's own tiling. A PV over a gathered key subset
  differs (2 of 6,144 elements).
- **Tile coverage at 16K.** A query row sees 12.5 % of the keys. Tiles holding any visible key are 63 % at 16x16, 82 % at
  32x32 and 92 % at 64x64 (qsa_tiles_16384.json), so skipping inside MLX's tiling saves only 8-18 %. The gain has to
  come from never writing the score plane (~98 of 106 ms per 2K chunk at 16K).
- **GDN-R** (an exact faster recurrence) is 1.42x in isolation but gave no end-to-end gain: it is off the critical path.

## First kernel numbers (11-11 12:19-12:21, share locks, nothing else on the GPU; MLX 0.32.2 = the engine's)

**QSA attention, one layer, per 1,024-row chunk** (scripts/prefill15/qsa_bench.py, real shapes, QSA-like mask;
data/prefill15/qsa_bench_1111.txt):
- *stock*: MLX's fallback.
- *band*: MTPLX's exact dense band, which the engine uses.
- *sol*: unmasked SDPA over only the 2,051 visible keys. This is a timing-only proxy for a fused exact kernel that never
  writes the score plane.

| context T | stock ms | band ms | sol ms | per prompt token over 12 QSA layers: stock / band / sol (ms) |
|---|---|---|---|---|
| 4,096 | 6.5 | 5.1 | 2.7 | 0.076 / 0.060 / 0.032 |
| 10,240 | 16.3 | 13.0 | 2.7 | 0.191 / 0.152 / 0.032 |
| 20,480 | 34.0 | 27.1 | 2.7 | 0.398 / 0.317 / 0.032 |
| 30,720 | 53.2 | 43.3 | 2.8 | 0.623 / 0.507 / 0.032 |

The band grows linearly with context, so its whole-prompt cost grows quadratically. Over a 30K prompt (mean context
~15K) the band costs ~0.235 ms per token, ~7 s, against ~1 s at the proxy. That is a third of the 1.5K budget at 30K,
so **a fused exact QSA kernel is needed for 30K**. At 10K the band is ~0.75 s of a 6.7 s budget.

**Routed experts, one layer** (scripts/prefill15/moe_bench.py; real layer-0 weights, skewed top-10 routing, mlx-lm's
sorted SwitchGLU with the rows padded to a multiple of 64 (BUG1, exact); data/prefill15/moe_bench_1111.txt):

| tokens per call | rows per expert | SwitchGLU ms | TFLOPS | ms per prompt token x 48 layers |
|---|---|---|---|---|
| 1,024 (PF-A's chunk) | 20 | 14.5 | 7.0 | 0.678 |
| 4,096 | 80 | 33.5 | 12.0 | 0.393 |
| 10,240 | 200 | 67.5 | 14.9 | 0.316 |
| 30,720 (PF-B: one call per layer) | 600 | 182.3 | 16.6 | 0.285 |

- **PF-A's 1,024-token calls alone cost 0.678 ms per token, more than the whole 1.5K budget.** PF-B's one call per
  layer is required.
- At PF-B widths SwitchGLU runs at 16.6 TF against dense 4-bit ~25-27 TF (PrefillKernels N1). That is the expert-kernel
  lever: up to ~1.6x on 0.285 ms per token.
- (The script's dense column picked the wrong tensor, the small in_proj_a, so it is not a dense reference. Peak MLX
  memory was 10.6 GB, more than the ~6 GB estimated; it was the only job running.)

## Budget for 1.5K at 30K (0.667 ms per token), from these numbers and PrefillKernels' stage split

| part | now (ms per token) | exact target |
|---|---|---|
| routed MoE | 0.285 (PF-B) | ~0.18 (expert kernels toward the dense rate, fused sort / gather) |
| QSA attention | ~0.235 (band) | ~0.03-0.05 (fused exact kernel, no score plane) |
| GDN | ~0.29 | ~0.2 |
| HC read / write | ~0.19 | ~0.1 (fused) |
| the rest | ~0.1 (indexer, PLE, shared expert, router) | ~0.1 |
| SSD reads (32 GB) | ~0.2, serialized | 0, overlapped |
| host overhead | ~0.5 (fixed PF-B measures ~1.8 ms per token) | ~0.05 |

Sum of the targets: ~0.66-0.68 ms per token, i.e. 1.5K only if every part reaches its target. This matches the
orchestrator's estimate (1.3-1.5K near the exact ceiling).

**Order, cheapest first:**
1. PF-B into the engine, with BigMoE's SEGALL fix, plus the SSD overlap.
2. Measure the real stage split of PF-B at 10K / 30K (MTPLX_QWEN4_PREFILL_PROFILE, at night).
3. The fused exact QSA kernel.
4. Expert kernels.
5. HC / GDN.

## Step 2: the stage split of PF-B at 32 GB (11-11 12:27-12:31, the cache clone + an instrumented pflm copy)

**How it was measured.**
- scripts/prefill15/prof/: pflm_prof.py is the engine's pflm.py with timers in PF-B's `_split_layers` and around the
  whole prefill. A sitecustomize.py import hook loads it as `pflm`, so the pinned clone is untouched.
- Run as `run.py --config chat --mem 32 --env R57_PFLM=3 --max-tokens 1` on ~10K and ~30K-token prompts (cut from
  long_ctx's papers prompt): 8,953 and 28,966 tokens, 1,024-token chunks, MoE groups of <= 4,096 tokens (9 / 3 and 29 / 8).
- **sync=0** is the real run, timestamps only. **sync=1** adds a barrier after every sub-step, which gives GPU and
  build time per stage. Both report wall and main-thread CPU (thread_time).
- Files: data/prefill15/stage_split_1111/.

| | 10K: sync=0 | 10K: sync=1 | 30K: sync=0 | 30K: sync=1 |
|---|---|---|---|---|
| first token | 18.31 s, **489 tok/s** (2.045 ms/token) | 20.47 s | 55.83 s, **519 tok/s** (1.927 ms/token) | 64.53 s |
| main-thread CPU | 5.68 s | 4.59 s | 17.48 s | 15.18 s |
| lifetime peak (32 GB line) | **35.58 GB** | 34.76 GB | **40.68 GB** | 38.93 GB |

**Per stage, sync=1 wall in ms per prompt token** (CPU part in brackets):

| stage | 10K | 30K |
|---|---|---|
| QSA attention (12 layers) | 0.231 (0.113) | **0.643 (0.318)** |
| GDN (36 layers) | 0.450 (0.117) | 0.444 (0.046) |
| MoE: router, shared expert, scratch rebuild, padded SwitchGLU (build) | 0.460 (0.080) | 0.389 (0.027) |
| MoE: the group's eval | 0.078 | 0.075 |
| HC read + write | 0.216 | 0.223 |
| SSD expert read (`_read_all`, main thread) | **0.597 (5.34 s)** | 0.185 (5.35 s) |
| PLE | 0.025 | 0.025 |
| untimed (the PLE n-gram stage before my timers, embedding, logits, clear_cache) | 0.230 (2.06 s) | 0.244 (7.07 s) |

**What it says** (the orchestrator's host-overhead item first):
- **The expert read is a flat ~5.35 s per prefill.** It runs on the main thread and is barely hidden: the sync=0 read time
  equals the sync=1 one. That is 29 % of a 10K prefill.
  - **Lever:** read layer L+1's experts while layer L computes (a read thread, or an `ahead` of a whole layer). It saves
    up to ~5 s, e.g. 10K 18.3 s -> ~13 s (~690 tok/s).
- **QSA's host work grows with context:** 9.2 s of main-thread CPU inside the QSA call at 30K, 0.32 ms per token, likely
  the dense-mask build. Its GPU part (~0.33 ms per token) matches the band bench (~0.235 average plus the indexer).
  - **Lever:** the mask on the GPU, or no mask (the fused kernel).
- **The untimed ~7 s at 30K** is to be timed next, most likely the PLE n-gram stage, whose table is streamed from the SSD.
- **The GPU floor at current kernels.** sync=1 wall minus main-thread CPU minus the read's I/O wait is ~11.7 s at 10K and
  ~45 s at 30K. Against the real 18.3 / 55.8 s, about 6.6 s and 10.4 s are lost to serialization (host and I/O).
- **MoE in 4K groups costs ~0.46 ms per token.** The bench gives 0.393 at 4K and 0.285 at 30K, so one call per layer
  (R57_PFLM_TOK >= the prompt, with the SEGALL fix) is worth ~-0.1 to -0.15.
- **Reading at 30K, in ms per token:**
  - QSA 0.64, about -0.6 with the mask off the CPU plus the fused kernel;
  - MoE 0.46, about -0.25 with one call and better kernels;
  - GDN 0.44, about -0.15 with GDN-R wired and the gates fused;
  - untimed 0.24, perhaps -0.2;
  - HC 0.22, about -0.1 fused;
  - read 0.19, about -0.15 overlapped.

  The sum, ~-1.4 ms per token from 2.23 (sync) / 1.93 (real), gives ~0.6-0.8 ms per token, i.e. ~1.3-1.6K tok/s, if all
  of it lands.
- **Memory.** PF-B at 32 GB peaks at 35.6 GB at 10K and 40.7 GB at 30K, so the 32 GB setting runs over its line here. Any
  32 GB speed is quoted with these peaks.

## Step 1, exact host work: PF-B pipelined across layers (written 11-11 14:2x; the A/B runs tonight at 32 GB)

**A correction to my 12:3x note.** It said overlapping the expert read "saves up to ~5 s" at 10K. That was too simple.
PF-B already runs each layer's read while the GPU executes that layer's queued attention, and at 10K the two take about
the same time (~112 vs ~122 ms per layer). The ~6.6 s lost at 10K goes elsewhere:
- **The attention build:** the GPU idles while the CPU builds the next layer's attention graph (~65 ms per layer at 10K).
- **The MoE path's two syncs:** pflm's prefill_call turns the router's indices into numpy (np.array(indices) waits for
  the GPU up to the router) and evals its output, and each MoE group is evaluated. Those syncs are there because the
  single full-E scratch is overwritten by the next layer's read.

The extension's read_rebuilt also holds the GIL for its whole ~110 ms per layer: its nanobind binding has no
gil_scoped_release, and its dispatch_apply workers do the work while the calling thread waits.

**The change, R57_PFLM_PIPE=1** (scripts/prefill15/prof/pflm_prof.py, `_split_layers_pipe`; with R57_PFLM=3):
- **Two full-E scratches** (+1.43 GB).
- **The MoE builds lazily.** In pipe mode prefill_call has no numpy and no eval, since all 512 experts of the layer are
  already in the active scratch, and it keeps the original ids as mode 3 does. Each layer's MoE outputs are submitted with
  async_eval.
- **Then, on the main thread:** wait for layer L-1's outputs, which frees the other buffer while layer L stays queued, then
  read layer L+1's experts into it while the GPU runs layer L. Layer 0 is read first.
- **Exactness:** same ops on the same inputs in the same stream order, so bitwise equal by construction. It falls back to
  the plain loop if a layer's piece shapes differ.
- **Expected, if GPU-bound:** ~690 tok/s at 10K (now 489) and ~610 at 30K (now 519).
- **Next on top:** a GIL-free read binding (a copy of the ext with gil_scoped_release and pre-evaluated destinations), so
  the main thread can also build during the read.

**Tonight's runner: prof_pipe.sh.** Armed 14:2x for 00:30-06:00, behind the gates (Kamen idle, no other lock or engine,
NO-SWAP >= 46 GB).
- PF-B at 32 GB, PIPE=0 vs 1, at 10K and 30K, with the order alternated.
- --max-tokens 32 with R57_LOGHASH=1 for the bitwise check: the tokens and every decode check's verify-logit hashes must
  be equal.
- The PLE-stage timer is now in both loops (the untimed ~7 s at 30K).

Why not today: two full-E scratches put even the 24 GB setting past the ~25 GB cap while Kamen's apps are open.

**QSA, 11-11 14:3x (code reading).**
- **The engine runs MTPLX's stock QSA prefill lane.** No MTPLX_* option is set anywhere in the configs, the screen or
  run.py, so the stock lane is the dense bool mask plus MLX's unfused SDPA fallback. My bench's "stock" column (16.3 /
  34.0 / 53.2 ms per 1K chunk at 10K / 20K / 30K) is what the engine pays, not the band column.
- **MTPLX's exact dense band (MTPLX_QSA_DENSE_BAND_SDPA=1)** was measured bit-identical to the stock fallback by
  PrefillKernels (0 differing) and is ~19 % faster at 30K (43.3 ms). It is a free exact lever, if tonight's bitwise check
  agrees.
- **The 9.2 s of QSA main-thread CPU at 30K is not the tiled indexer.** MTPLX_QSA_SCORE_TILE_ROWS, with an mx.eval per
  tile, is opt-in and off. It is also not spin-waiting, since GDN's barrier evals show little CPU. Reading did not find the
  source.
- **Tonight, a second runner** (scripts/prefill15/prof2/prof_qsa.sh, armed 14:3x, after prof_pipe.sh) runs 30K, PIPE=1,
  with the dense band. It wraps cProfile around every QSA self_attn call (R57_PFLM_QSA_PROFILE, in a separate copy so
  tonight's A/B module is untouched) and checks bitwise against prof_pipe.sh's 30K PIPE=1 run (tokens + LOGHASH).

## The GIL-free read binding (11-11 16:2x, CPU + SSD only, ~1.1 GB)

scripts/prefill15/ext_ov/ holds a copy of BigMoE's engine/ext/bigmoe_ext.cpp as module `bigmoe_ext_ov`. BigMoE's source is
not edited. The only changes:
- read_rebuilt and read_rebuilt_layout carry `nb::call_guard<nb::gil_scoped_release>()`;
- `a.eval()` on each destination is replaced by an `is_available()` check, so no MLX scheduling happens without the GIL.
  The destinations must be evaluated beforehand, as the scratches are.

It builds with the clone's own nanobind (third_party/nanobind) and MLX 0.32.2 in ~5 s, into the scratchpad; the .so is not
committed.

**Test** (gil_read_test.py; layer 0, experts 0..191, 0.53 GB per read; data/prefill15/gil_read_test_1111.txt):
- **Bitwise:** the scratches filled by the original and by the copy are EQUAL.
- **The GIL:** during a 48 ms read in a background thread, a pure-Python loop on the main thread ran at **12.9 %** of its
  normal speed with the original binding and at **85.6 %** with the copy.
- **Read speed:** ~11 GB/s.

**So a background reader using the copy lets the main thread build the next graphs while the read runs**, and it touches
no MLX state. Next: R57_PFLM_PIPE=2, background reads through bigmoe_ext_ov, queued tonight after the PIPE 0 / 1 A/B and the
QSA run.

## PIPE=2 and the FAST QSA option, queued for tonight (11-11 16:3x)

- **R57_PFLM_PIPE=2** (scripts/prefill15/prof3/: a third module copy, so tonight's PIPE 0 / 1 module stays as it was):
  PIPE 1 with layer L+1's expert read in a one-worker background thread through bigmoe_ext_ov.
  - The main thread builds layer L+1's attention while the read runs. The worker touches no MLX state, and the GIL-free
    test above showed the main thread keeps ~86 % of its speed.
  - Runner prof3/prof_pipe3.sh: 10K and 30K, bitwise vs tonight's PIPE=0 runs (tokens + LOGHASH).
- **The FAST QSA option = MTPLX's own sparse QSA prefill pipeline**, NAX flash plus the direct consumer. It is on by
  default on M5, but engages only past a 32,768-token context: MTPLX_QSA_PREFILL_MIN_CONTEXT, _FLASH_MIN_CONTEXT and
  _DIRECT_MIN_CONTEXT all default to 32768. Setting the three to 4096 engages it from 4K.
  - Not exact: PrefillKernels measured max |d| 0.009-0.016 for this lane, ~2x faster per chunk at 16K.
  - It needs no new code; the gathered-key PV the orchestrator described is what this lane already does, with a NAX flash
    kernel.
  - Runner prof4/prof_fast4.sh: PIPE=1 at 10K and 30K. It reports the time and how many of the 32 decoded tokens equal
    the exact PIPE=1 run's.
  - Adoption goes through EvalSuite's same-scores gate (the two-track rule), and the exact fused kernel stays the exact
    path.
- **Tonight's chain**, each step behind gates and a share lock retried until granted:
  1. prof_pipe2.sh, from 00:33;
  2. prof_qsa2.sh, the dense band + the QSA cProfile;
  3. prof_pipe3.sh, PIPE=2;
  4. prof_fast4.sh, the FAST lane.

**Night 11-12: the 4 runners SKIPPED (no slot; the orchestrator foresaw it).** No run started.
- They need NO-SWAP of 46 GB or more. They had only 17-19 GB free (swap 4.7 GB): Kamen's sessions held about 15 GB, and
  BigMoE's DEEPDRAFT capture held the GPU.
- Each one exited at its last start: prof_pipe2 at 06:00 ("no free window", another lock held), prof_qsa2 at 07:00,
  prof_pipe3 at 07:30 and prof_fast4 at 08:00.
- No numbers. The chain is to be re-armed for the next night with room, after clearing scratchpad prof15/*.done.
