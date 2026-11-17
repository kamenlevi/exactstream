# CHECK SPLIT (Leap, Mon 16 Nov 2026 18:0x-18:2x): the lossy 32-row check as modules other sessions can own

For: the orchestrator (asks 17:5x, 18:0x and 18:1x) and NewIdeas (orders the GPU). Leap integrates and owns the measured
check.

How it works:
- Owners work in their own folders and read ~/Leap read-only.
- Each owner delivers a RESULT line (number, bits, conditions) and a patch file.
- Leap copies the patch into ~/Leap/third_party/<owner>/ and wires it as an opt-in knob in the c25 engine copy.

Labels: [K] measured; [P] priced from measured parts; [est] estimate.

## 0. Today, the target, and what paces the check

### Today and the target
- **Today:** F32c6i 56.41 ms [K].
  - REAL engine, lab checks: G32T TIME MODE, 3 code windows, true-state tokens, C 160 resident, misses 0.
  - run2b, Mon 13:25, QUIET.
  - Wall phases: build + submit 50.95 / wait 5.05 / commit 0.35.
- **Target:** 85-90 % of the real bound = **28.0-29.6 ms** at C6 + int8.
  - The bound is 25.2 ms: dense 7.6 + experts 16.4 (4.60 GB) + cores 1.2 (check25_bound_real.txt).
  - The PUSH800 frame's 21.7 used the old expert size. It was corrected at 17:5x.
  - The lab windows carry ~5.9 GB per check, so their own floor is 21.0 ms. Efficiency is always quoted on the measured
    windows' own bytes.

### CPU or GPU: OPEN until tonight's measurement (Kamen's question)
"build + submit" (50.95 ms) is the wall of the Python forward call. It holds Python's graph build, MLX's encode and any
waiting inside the call. The phase split alone cannot say which of these it is.

**Slice preview** (Mon 18:05, 4-layer slice, share lock; data/week98/check25/cpusplit_slice.txt). The CPU numbers are
thread CPU, so they hold beside other jobs:
- At 32 rows the slice's forward submits nothing to the GPU (0 async_eval calls).
- Python builds the whole lazy graph at ~0.45 ms per layer (F32o; F32s 0.54, F16 0.60). MLX then encodes ~380 dispatches
  per 4 layers inside the logits' eval.
- Scaled to 48 layers: **~20 ms of Python build + ~12-14 ms of MLX encode (~3 us per dispatch) = ~30-35 ms of CPU per
  32-row check** [P].
- The engine's phase shape is the reverse (build + submit 51, wait 5). So BigMoE's path submits and waits inside the
  forward.

**The two readings** (the measurement decides between them):

| reading | GPU busy per check | what a perfect replay wins today |
|---|---|---|
| the in-forward waits are GPU-paced (Leap's bet) | ~50 ms | ~2-6 ms |
| they serialize CPU with GPU (the orchestrator's read) | ~35-40 ms | ~15-20 ms |

**For GPU-pacing,** three run2b A/Bs that separate CPU from GPU:
- **C6:** cuts expert bytes only (-0.25 GB real, same kernels) and moves the wall -1.20 ms, those bytes' time at
  ~200 GB/s.
- **V3 + V1:** bytes only. -1.90 ms.
- **t1cf:** ~28 graph ops per MoE layer become one kernel. +0.49 ms.

CHECK55's exact engine in the same harness is GPU-busy 97-99 % of the wall. There, +2 ms of thread CPU moved the wall
+0.13-0.17 ms.

**Against it:** ~30-35 ms of CPU against a 56 ms check is not small. If the forward syncs per layer, CPU and GPU time add
up instead of overlapping.

**The measurement: armed for TONIGHT** (pid 94402, data/week98/c25/frozen/gpub_arm.sh).
- **When:** NewIdeas' first slot after BigMoE's g1b key test (G1A_KEYTEST_DONE). It holds while any other process is over
  16 GB, and needs NO-SWAP 39 and a quiet ticket. ~15 min; latest start 00:10.
- **What:** arms F32c6i / F16 / F32s / F32o on the 3 G32T windows. Per check, with libmtlprof and c25_cpusplit.py:
  - GPU busy (whole check and during the forward);
  - Python build / MLX encode / blocked in eval / other waits;
  - dispatches and command buffers.
- **Then:** the number goes to the orchestrator at once.
- **Fallback:** Tue 09:30 slot #1 now records the same split (c25_engine turns it on under C25_PROF).

**Pre-registered rule** (CHECK55's R1, on F32c6i):

| GPU idle | the check is | what it means |
|---|---|---|
| >= 30 % of the wall | CPU-paced | Module 1 leads; kernel modules count only below the CPU line. MoonShots moves from module 2 to module 1a. |
| <= 10 % | GPU-paced | Rank the modules by GPU ms. Module 1 binds once the GPU side is under ~30-35 ms. |
| between | both | Rank by CPU + GPU. |

**Either way, module 1 is required:** a check under ~35 ms cannot carry ~30-35 ms of CPU.

**RESULT (try B, Tue 00:51; quiet lock; 3 G32T windows x 8 rounds = 24 checks per arm) [lab, c25 harness]:
GPU-PACED.** File: data/week98/c25/gpubB_run.txt.

| arm | wall | GPU busy | GPU idle | CPU thread | dispatches | command buffers |
|---|---:|---:|---:|---:|---:|---:|
| F32c6i | 56.14 | 54.16 (96 %) | 1.98 | 24.36 | 3,172 | 266 |
| F16 | 38.64 | 37.33 (97 %) | 1.32 | 24.50 | 3,242 | 242 |
| F32s | 67.74 | 65.35 (96 %) | 2.39 | 37.95 | 5,690 | 340 |
| F32o | 58.48 | 56.47 (97 %) | 2.01 | 26.03 | 3,801 | 266 |

- **The CPU split for F32c6i:** build + submit 50.72 = Python build 12.00 + MLX encode 11.17 + blocked in eval 27.38.
  - Thread CPU is 23.20 ms over 23 async_evals. The GPU is 97 % busy during build + submit.
  - Expert misses 0, SSD 0 ms.
- **Verdict by the rule:** idle 3.5 % <= 10 %, so rank the modules by GPU ms.
  - A perfect CPU replay wins <= ~2 ms of wall today.
  - Module 1 binds once the GPU is under ~23 ms: today's thread CPU, lower than the ~30-35 estimate.
  - MoonShots stays on the GPU side: the FAST small-segment expert path first, then module 5.
- **Caveat:** spin_wait (the loader-release poll) counts as GPU busy. 1a's SLOTS kernel removes it on the GPU side.
- **Sent** 00:5x to the orchestrator, Breakthrough and MoonShots.

## 1. The modules

| # | module | owner | today | line (PASS / BETWEEN / KILL) | check effect if PASS | days |
|---|---|---|---|---|---|---|
| 1a | replayed check: record once, replay per check (fixed buffers, a native encoder or Metal indirect command buffers; GPU-driven expert routing) | **Breakthrough** (libmtlprof's author); design: Fable B | ~0.7 ms of CPU per layer (slice [P]: build 0.45 + encode ~0.25) | one real layer replayed, outputs bitwise == MLX: <= 50 / 50-200 / > 300 us of CPU per layer replay, or not bitwise = KILL | CPU per check ~30-35 -> <= ~3 ms | one layer by Wed 20:00; full check Thu-Fri (stretch) |
| 1b | compile / specialize the fixed 32-row check + the launch diet | **Fit12** | ~30-35 ms of CPU per check [P]; dispatches TBD tonight | thread CPU (build + encode) <= 15 / 15-22 / > 22 ms per 32-row check; bitwise vs today's route | the CPU line under the GPU's | plan Tue 13:00 (after the numbers), build Wed, engine Wed night |
| 2 | expert one-call streaming | **MoonShots** (offered 14:49; confirmed 18:1x) | 571-614 us per 105 MB call = 171-184 GB/s [K quiet] | <= 400 / 400-480 / > 480 us per call | -7..-9 ms at C6 bytes | candidate Wed 12:00; quiet bench by Wed 18:00 |
| 3 | dense matmuls at 32 rows (folds, big shapes, int8 quantizer in the norm) | **Leap** (Fable B's IDEAS_1116b section 2) | 12.97 ms dn32 / 12.86 dn32q + quantizer (11 FAST shapes, 'ind') [K quiet] | <= 9.5 / 9.5-11.5 / > 11.5 ms; int8's engine +1.13 must reach <= 0 | -3..-5 ms GPU, -300..-400 dispatches | Tue-Wed |
| 4 | HC read in one launch (STATS-FORWARD + hc4i) | **Leap** (Fable B's design, IDEAS_1116b section 1) | 8.14 ms chained, 96 reads at 8 bit [K quiet] | <= 5.0 / 5.0-7.0 / > 7.0 ms chained (writer + read) | -2.5..-4.5 ms, -192 dispatches | build Wed, engine Wed night |
| 5 | attention core by per-row key lists | **after Tue's numbers** (MoonShots after module 2, or Leap) | exact route 2.70 ms, 2,244 dispatches (CHECK55 C32 [K]); lossy route (att_single) TBD | <= 0.8 / 0.8-1.5 / > 1.5 ms per check; <= 1 dispatch per layer | -1..-2 ms; dropped if the lossy core is already <= 1 ms | Wed-Thu |
| 6 | GDN core | **Leap**: gdn_fast (built, bitwise) | 3.86 ms (mlx_lm, T 32) [K quiet] | gdn_fast bench in Tue's post; chunked v2 PARKED | ~-1.5 ms | wire Wed |
| 0 | integration and the measured check | **Leap** | 56.41 | Friday's report | | tonight's measurement; Tue #1, Wed night #3, Thu #4 |

### Module 1a: the replayed check (Breakthrough; Fable B designs the GPU-driven routing)
- **Mechanism:** record the check's GPU work once and replay it per check. The record holds per dispatch: the pipeline,
  the bound buffers and offsets, the grid and the barriers. libmtlprof already hooks these classes in-process.
- **The replay writes only the new inputs:**
  - the tokens' embeddings;
  - the tree mask;
  - the KV and GDN state positions.
- **What must change to make it replayable:**
  - Fixed buffers: allocation per check is MLX's, so the replay needs its own arena.
  - **GPU-driven expert routing** (Fable B's design task):
    - Today the 1c one-call's posts come from BigMoE's loader on the CPU.
    - With all experts resident (the lab and C 160 code windows), the posts can be computed on the GPU from the router's
      ids.
    - A miss needs a fallback: leave the replay, or a GPU-side flag plus a re-run.
  - No value-dependent Python branches inside the check.
- **Milestone 1 (Wed 20:00):** one real layer replayed: a GDN layer plus its MoE at 32 rows, real weights.
  - Its outputs and states must be bitwise == MLX's.
  - The CPU microseconds per replay are measured.
  - Line: PASS <= 50 us per layer, KILL > 300 us or not bitwise.
- **Milestone 2 (Thu-Fri, stretch):** the full check replayed, timed in Leap's c25 harness.
- **What Breakthrough needs:**
  - read-only copies of scripts/week98/c25/ and third_party/mtlprof;
  - the layer's weights via the loop's loader;
  - Leap's slice harness (cpusplit_slice.py / pk_slice_test.py) for a first replay on the slice;
  - share locks;
  - one A/B arm in Leap's engine slots.
- **The old KILL** of a replayed path rested on the exact engine being GPU-bound (CHECK55). For the lossy check it
  re-opens on tonight's number; any check under ~35 ms needs it.
- **Fable B's design is in (Mon 18:3x):** ~/FableGod/fableb/REPLAYED_CHECK.md.
  - A SLOTS kernel per MoE layer resolves expert -> slot from host-owned tables and still writes the routing row the
    loader reads. Misses use kernel A's existing poll.
  - The other CPU inputs become 6 fixed buffers. The SDPA's KV length is the one per-check grid.
  - BigMoE must not reuse slots during a check.
  - Price: < 1 us CPU per layer with indirect command buffers; ~45 us per layer with a native re-encode after the diet.
  - **Leap's note:** spin_wait is a GPU-side poll, so the loader-release latency counts as GPU busy. SLOTS therefore also
    cuts GPU time (~48 x the release latency) and ~480 dispatches per check. It pays even if tonight says GPU-paced.
- **Breakthrough: YES** (18:12). Start Tue ~09:00, after its CHECK BYTES read-outs.

### Module 1b: compile / specialize the fixed 32-row check + the launch diet (Fit12)
- **mx.compile** of the per-layer glue at a fixed 32 rows: norms, residual adds, gates, casts, the router's softmax /
  top-10 / renorm, the tail.
  - It needs: no CPU read-backs inside the compiled region; static shapes (the check is always 32 rows); custom kernels
    callable inside compile.
  - It also needs a bit test: compile's fused kernels must keep bf16 rounding between ops. If they don't, it becomes a
    FAST lever under EvalSuite's gate.
- **The diet from built, bitwise pieces** (Leap supplies them):
  - t1cf (the 1c glue as one kernel per MoE layer; its +0.49 to be found);
  - module 4 (-192 dispatches) and module 3's folds (-300..-400);
  - the int8 quantizer in the RMSNorm (-193);
  - router + shared as one stream.
- **Line:** thread CPU (build + encode, as c25_cpusplit measures it) <= 15 ms per 32-row check, bitwise vs F32c6i's route
  on the lab windows and the 4-layer slice.
- **What Fit12 needs:**
  - tonight's split (Python build vs encode per check) and Tue's per-class dispatch counts (Leap sends by Tue 13:00);
  - read-only copies of the c25 engine and the slice harness;
  - share locks.

### Module 2: expert one-call streaming (MoonShots)
- **Tue 07:4x: the FAST small-segment FMA path = KILL on paper (MoonShots af29c50, LEVERS_A 'MODULE 2'); module 2 is
  CLOSED.** A CPU replica of _GRP1CQ on exp32's 99 blocks: 60.9 segments / call, 5.25 pairs each; <= 2 pairs = 23 % of
  segments but 7.2 % of pairs. The MMA costs ~2.43 us per segment, fixed per 16-row tile; SIMD FP32 FMAs ~1.2 us per pair
  + ~1.3 pair-equivalents per segment to convert nibbles. So FMA wins only on 1-2-pair segments: <= ~0.3 ms per check
  even with free conversion. The experts stay ~23 ms busy per check; their byte floor is ~16 ms (4.6 GB at ~290 GB/s), so
  the gap is the MMA / stream non-overlap. MoonShots moved to BREAK98 (the orchestrator 07:3x); module 5 is open.
- **Kernel:** ~/Leap/scripts/ooo1cq.py, `tier_glu_nax6n_1cq` (FAST-NAX on NAX6-N, the converted records3 pools).
  - The exact control is ooo1c.py: one group per expert, all 4-bit.
  - It streams ~290 GB/s in the engine (CHECK55 pass K: 50.65 ms for 14.66 GB).
- **Entry point:** ~/Leap/scripts/week98/exp32_bench.py, in the BigMoE venv (MEM ~3 GB). It loads layer 20's pools
  (all 512 resident) on real 1c routing: 24 blocks of 32 rows from the decision map's code windows.
  - Modes: default (ks / nsg_b / COH variants), `prof` (kernels and GPU busy per call; mtlprof loads before mlx),
    `nlv` (real / all-MSB-3 / all-4-bit masks).
- **First question: fixed cost or rate?**
  - A lossy call reads 105 MB; the exact path reads ~305 MB per layer.
  - A ~150-200 us fixed part would explain most of the gap: the grouping stage `gpu_grp_1cq`, kernel A, kernel B and
    the combine are dependent launches.
  - So fit t = fixed + bytes / rate over the 24 blocks with an intercept. My rough nlv fit had no intercept.
- **Gotchas:**
  - Posts must be sync-free and precomputed. The first exp32 run did np.array(ids) per call and measured the harness.
  - **The D8 class:** groups are cut every 16 pairs of one key (tierglu_nax6._GRP), so a 32-pair expert is two groups.
    A kernel that takes more than 16 pairs per group must handle that explicitly.
  - The 2-D nolsb (b43efc4).
  - The km path: a dropped use reads its row's rank-0 id at weight 0. Weight-0 pairs still form groups.
  - Tier 4 = MSB + LSB (spread8, smul 0.5, bias b); tier 3 = MSB-3 (spread3, smul 1.0, bias b3).
  - x is re-read from device memory per simdgroup.
  - Use the real piece sizes: MSB 2.064 MB, LSB 0.720 MB per expert.
  - If gather_qmm appears: MLX's sorted-gather bug above 32,767 rows (pad to 64).
- **Line** (exp32_bench default, per 32-row call on 105 MB):
  - PASS <= 400 us (>= 262 GB/s);
  - BETWEEN 400-480;
  - KILL > 480.
- **Bits:** == ooo1cq bit for bit on:
  - ~/Leap/scripts/week98/ooo1c_m32_check.py's 24 scenarios (layers 20 / 44; COH on / off, 32-pair experts, nolsb,
    late slots);
  - exp32's 24 real blocks.

  A form that reorders the K sum is a separate FAST candidate. It must report its max rel vs ooo1cq and vs fp32;
  EvalSuite gates it and Kamen adopts it.
- **Baseline:** Leap sends Tue 13:00 numbers: the quiet exp32 prof / nlv from slot #1.
- **If tonight says CPU-paced,** MoonShots joins 1a instead (its offer, 17:5x).
- **MoonShots' diagnostic** (Mon 22:4x, quiet; layer 20, exp32's 24 blocks; GPU us per call; ~/MoonShots/week98/m2/,
  commit b384412):
  - Per kernel: A 388, B 198, grouping 19, combine 9, rowsum64(h) 4, xs 3.
  - exp32's way: 582.6 wall / 534.3 GPU busy; 8 dispatches, 2.7 barriers, 2.88 command buffers per call.
  - Ablations (A / B): base 352 / 181; no code loads 331 / 169; no decode 319 / 166; no MMA (the x loads go with it)
    247 / 125; a pure read of the same bytes 234 / 115 (~290-300 GB/s).
  - **So the call is COMPUTE-bound, not fixed cost or rate:** memory adds ~20 us over the compute. The excess is the
    MMA (16-row matmul2d tiles on ~5.2 pairs per segment: ~2/3 padding) and the decode.
  - Five bit-exact forms (register prefetch, grid y capped at 96, two tiles per threadgroup, B in tile rounds, the
    combine folded into B) are all == ooo1cq, and none is faster. Best: 585.6 vs 584.0.
  - **Floor at today's bytes:** 364 (105.6 MB at ~290 GB/s) + 32 (grouping / rowsum64 / combine) + 48 (wall over busy)
    = **~444 us wall, ~396 us busy**, even with the compute fully hidden. Kernel work alone tops out at BETWEEN. PASS
    also needs the wall over busy gone (module 1) or fewer bytes.
  - **Leap's call (22:5x):**
    - Screen 8-row tiles. Kernel A is matmul2d 16 x 32 x 32, so time the M=8 form first; run the bit sweep only if it
      is faster. Moved row bits make it a FAST candidate.
    - Then a cheaper decode, only if A stays > ~300 us.
    - Skip the grouping kernel (<= ~10 us to gain).
    - The 1a move follows the pre-registered rule on tonight's number.
  - **The 8-row screen** (MoonShots 22:51, quiet; LEVERS_A 'MODULE 2', d50e76a) **is NEGATIVE.**
    - M=8 tiles run padded on the accelerator: A 396.2 vs 388.8 GPU us; exp32's way 599.8 vs 584.3 us wall.
    - Their row bits do match M=16 (0 of 4,096 rows differ on 256 random tiles), so speed alone kills the lever.
  - **Verdict (Mon 22:5x): module 2's bit-exact track is KILL** (583 us > 480).
    - The excess is the MMA not overlapping the stream (memory floor 234 + MMA ~105 us in A).
    - Overlap needs more occupancy, which means a threadgroup-memory rewrite (the epilogue reuses the double code buffer
      for the chunk partials): days of work, and not sure to pay.
  - **What is left:**
    - A FAST small-segment path: <= 8 pairs on SIMD FMAs. It is not bit-exact, so it needs max rel vs ooo1cq and vs
      fp32, plus EvalSuite's gate. It is now first in line, since tonight said GPU-paced. It is adopted on EvalSuite's
      same-scores gate (Kamen's two-track rule; orchestrator 01:0x); NewIdeas orders when.
    - The cheaper decode, worth ~-1 ms per check: parked.

### Module 3: dense matmuls at 32 rows (Leap)
- **The folds** (bitwise per tile):
  - a|b into qkv|z;
  - k|v into q;
  - shared gate|up + down;
  - the indexer into att q.
- **The big shapes at <= 1.2x their floor** (double-buffered staging, ks chosen for integral waves; Fable B's fill
  model).
- **The int8 quantizer inside the RMSNorm,** so int8's engine +1.13 ms (one dependent dispatch per matmul) turns into
  dn32q's bench gain: -0.1..-0.3 ms with its quantizer, -1.75 pre-quantized ('ind', quiet).
- **Bits:**
  - bf16 forms == qmm32n / dn32;
  - int8 row-invariant and == int8d's dn32q per shape.

### Module 4: the HC read in one launch (Leap, from Fable B's STATS-FORWARD)
- The previous layer's HC writer emits K1C's partials, and its last threadgroup runs K2B, so the read is K3C alone.
- Fable B's CPU screen: the partials are bitwise in K2B's order.
- **Line** (Fable B's): chained bench, 96 reads, writer + read: PASS <= 5.0 ms, KILL > 7.0.
- It also takes the K2B fold, and hc4i for V3.

### Module 5: attention core by per-row key lists (owner after Tue's numbers)
- **Scope:** the 12 attention layers at 32 rows, one dispatch per layer (all heads).
- **Keys:** the prefix plus each row's tree ancestors.
- **Reference:** Leap's 90dae21 (ATT32-INV) and ref2's att_single (R-62).
- **Bits:**
  - row-invariant: a row's output == its own 1-row call, bit for bit;
  - vs MTPLX's masked SDPA, max rel <= bf16 rounding (~4e-3);
  - slice logits argmax-equal.

  It changes exact bits, so the exact track needs a re-certification. For lossy it is a FAST lever.
- **Line:** <= 0.8 ms per check at the G32T windows' key lengths.
- If Tue's profile shows the lossy attention core already <= 1 ms, this module drops.

### Module 6: GDN core (Leap)
- gdn_fast is bitwise through the real path (slice, 14:23) and benches quietly in Tue's post.
- Chunked GDN v2 is PARKED. v1 was 3.1x slower. The state's own traffic (~0.8 ms per check) caps any gain beyond
  gdn_fast at ~1 ms. It is revived only if Tue shows the core > 2.5 ms after gdn_fast.

### Module 0: integration and the measured check (Leap)
- **Tonight (armed, after g1b):** GPU busy vs the CPU split. Sent at once.
- **Tue 09:30-12:00, slot #1** (armed):
  - pass T / K per class, launch count, GPU busy, CPU split on 8 lossy arms;
  - exp32 prof / nlv, gdn_fast, HC NP.
  - By 13:00 every owner gets its class's engine ms, dispatches and a re-priced line.
- **Tue, free and bitwise:** the 32-row seams.
  - REF2_SEAM_HI=32 (SB_FIRST + PLE_LATE; Breakthrough pass S 2757be6: -2.0 ms per exact 32-row check).
  - The c25 arms run without them today: c25_env unsets REF2_SEAM_HI. At 32 rows the forward submits nothing until
    late (slice), so this seam also starts the GPU earlier.
- **Wed night, slot #3:** every module that PASSed, as a knob, in one engine A/B with bits:
  - exact-form modules bitwise vs F32c6i;
  - FAST ones: argmax and max diff on the lab windows.
- **Thu, slot #4: Friday's number.** C5 and C6 + int8, with efficiency vs peak, the launch count, the CPU split, and per
  class.

## 2. What the lines add up to (ms per 32-row check at C6 + int8 map bytes) [P / est]

**GPU side:**

| class | today [est from parts] | every module at PASS | every module at stretch | 85 % of the bound |
|---|---:|---:|---:|---:|
| routed experts (4.60 GB) | ~26 | 17.5 | 16.4 (290 GB/s) | |
| dense matmuls | 12.9 | 9.5 | 8.0 | |
| HC read + write | ~8.5 | 5.0 | 3.2 | |
| GDN core | 3.9 | 2.4 | 1.5 | |
| attention core | 1-2.7 | 0.8 | 0.8 | |
| router, 1c glue, tail | ~4-6 | 3.0 | 1.5 | |
| PLE, seams, read, commit, other | ~3-4 | 2.0 | 1.0 | |
| **GPU sum** | ~59-63 (lab windows: 56.4 [K]) | **~40** | **~32** | **28.0-29.6** |

**CPU side:** today ~30-35 [P slice]. Module 1b's line is <= 15; module 1a's is <= ~3.

- **The check is max(GPU, CPU)** if they overlap, and more if they serialize. **Tonight's number says which.**
- **Every module at PASS gives ~40 ms GPU (63 % of the bound). Every module at stretch gives ~32 (78 %).** 85 % needs
  every module beyond its stretch.
- **The experts' 17.5 ms assumes ~400 us of GPU busy per call** (Mon 22:5x, from MoonShots' diagnostic). Each +50 us of
  busy per call adds ~2.2 ms per check (17.5 / 400 x 50). Today's 534 us busy is ~23 ms. The busy floor (~396, compute
  fully hidden) is still inside the term, but no bit-exact form has hidden the compute yet.
- **With module 2's bit-exact track KILLed (22:5x), the experts stay at ~23 ms of busy per check.** That holds unless
  the FAST small-segment path or the occupancy rewrite lands. "Every other module at PASS" then sums to ~46 ms GPU, not
  ~40.
- **The dense side is the gap.** The experts at the exact path's rate are 95-100 % of their term. Dense, HC, glue and
  cores total 7.6 + 1.2 = 8.8 ms in the bound, but are ~16 at stretch: per-layer latency and fill, ~25 dependent
  launches per layer [est].
- **For 775 at those checks** (draft 3, overhead 1.5):
  - a ~40 ms check needs kept 34.6;
  - a ~32 ms check needs kept 28.6;

  against today's 16.91.
- **Dependency for a measured DECODE** (not the check):
  - The c25 route is Leap's copy, in G32T time mode. BigMoE's loop runs > 16-row checks through the slot path today
    (R57_ROWS 16).
  - A measured 32-row decode needs BigMoE to carry the route and the PASSed modules as opt-in flags, then a snapshot,
    a key test and EvalSuite's gates.
  - Leap can hand over the route and the modules on Thu. Timing is BigMoE's and EvalSuite's call.

## 2b. Shape (NewIdeas 18:1x)
- A 32-row CHAIN check may become Saturday's shape: lossy rules help chains more than trees (HE+ chain-32 INERT + R1(1)
  21.76 optimistic vs tree-32 18.11).
- So every module bench stays shape-neutral: 32 rows either way.
- For a chain, attention is the plain causal mask (module 5 gets simpler), and the GDN needs no state forks.
- A chain's draft is 31 levels, 11.2 ms at the QAT head's f, against ~3 ms for the tree. The 775 sums above assume
  draft 3.

## 3. GPU asks for NewIdeas
- **Tonight:** the GPU-busy measurement. Armed after g1b; ~15 min; latest start 00:10.
- **Standalone quiet benches, <= 10 min each, Wed:** module 2 (two), modules 3 and 4 (one together), module 1b (one CPU
  profile), module 1a (one layer replay).
- **Bit tests** run under share locks, <= 3 GB each, any time. Module 1a's first replay is on the slice (~9 GB).
- **Engine slots** stay as armed: Tue 09:30 #1, Wed night #3, Thu #4.
