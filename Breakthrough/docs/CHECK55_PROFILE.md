# CHECK55 step 0: where the exact 16- and 32-row check spends its time (Breakthrough)

Breakthrough, Sun 15 Nov 2026 11:0x EET.
- The ask: the orchestrator, 10:5x. This is step 0 of Leap's CHECK55 plan (~/Leap/data/week98/check55_plan.md, 76817d3).
- What it is for: one attribution table, at 16 and 32 rows, set against the floors. The table orders Leap's steps 1-5.
- Pre-registered here, before any profile run. Results are appended below the line.

## Setup
**Harness.** scripts/m1/check55/c55_profile.py, my own copy of Leap's C6 setup.
- The slice: 12 layers resident (GGGA x 3), about 23 GB.
- Imports, all read-only: ~/Leap/scripts ref2 + week98 v32, and MTPLX.
- Every arm runs on its own cache from a 1,024-token prefill, advancing 32 tokens per round (as C6).
- SB submits every 4 layers on 2-32-row checks.

**Configuration: the exact-45 engine's own 32-row path** (BigMoE G32T / C32m, 03:1x).
- v32 + V32_MOE1.
- No mr2, no v32mr2, no hc32, no DENSE-N copies.
- ref2 forms: att_rope, idx_prep and hc_write (K-73), plus hc_gate and gdn_inproj_x (K-83).
- gdn_body off, as in the 45 preset.

**Arms.**
- A16: a 16-row check. C32: a 32-row check.
- A16k / C32k: the same with K-73's MoE block (router_fused, shared_cat, moe_tail), ported to the resident experts.
  The engine's own K-73 block falls back to MTPLX's stock block on the slice, because the slice has no tier switch.

**Instrument: no xctrace.** libmtlprof (scripts/m1/check55/mtlprof.m) is an in-process hook on the concrete Metal classes
MLX uses (AGXG17X compute context, command buffer and device).
- It counts every dispatch, barrier, fence, encoder and command buffer.
- It names each dispatch's pipeline.
- It reads each command buffer's own GPUStartTime / GPUEndTime from a completed handler.
- Tested 10:5x on a toy run (200 x matmul + mul + add): exactly 600 dispatches, pipelines named, GPU busy 36.9 of 40.5 ms.
  A 351 ms GPU wait cost 0.6 ms of thread CPU, so thread CPU measures work, not waiting.

**Pass T: timing, production scheduling.** 2 warm-up rounds + 12 measured, with the arm order rotated.
- Wall per check, in three phases:
  - forward call: Python build with SB's submits inside;
  - read: argmax().tolist();
  - commit: commit_verified_window + eval + synchronize.
- Thread CPU per phase. The encode CPU inside SB's mx.async_eval calls is measured separately; Python build = forward
  CPU minus that.
- GPU busy per phase (the union of command-buffer GPU intervals), GPU idle = wall - busy, and the GPU's first start.
- Dispatches, barriers, encoders and command buffers per check, and the kernels by pipeline name.

**Pass K: attribution, in separate rounds.**
- Every op-class boundary flushes (mx.async_eval), so each command buffer holds one class.
- Per class: dispatches, GPU ms (its command buffers' GPU time, gross and net of an empty buffer's time), Python build
  CPU and encode CPU. Also dispatches per layer.
- The classes:
  - other (embed, tile, glue), ple;
  - hc_read, hc_write;
  - gdn_proj, gdn_core, att_proj, att_core;
  - moe_router, moe_experts, moe_shared, moe_tail;
  - head_mixer, head_lm;
  - read, commit.

**Floors.** Distinct bytes per check at 280 GB/s, as Leap's pipe_price: the dense weights once, plus the routed union per
layer (pass K's routing) x bytes per expert.

## Predictions (from the record)
The record:
- C6: graph build 0.54 / 0.62 of a check.
- B1: ~1,600 kernels + ~480 glue flushes per 48-layer check, at ~7 us of encode each.
- P2: a ~40-kernel MTP step costs ~0.54 ms fixed, ~13 us per kernel all-in.

| # | prediction |
|---|---|
| P1 | CPU (build + encode) >= 0.5 x wall at both lengths; GPU busy <= 0.65 x wall |
| P2 | dispatches per check on the 12-layer slice: A16 ~350-450, C32 ~600-800 |
| P3 | CPU per dispatch (build + encode) ~10-15 us; encode alone ~4-8 us |
| P4 | the 17-row step (C32 - A16) is >= 60 % CPU-side, in proportion to dispatches |
| P5 | GPU busy at A16 ~1.2-1.6 x its floor |

## Decision rules: how the table orders steps 1-5
- **R1, ranking.** The check's critical resource decides how classes are ranked:
  - GPU idle >= 30 % of wall at C32: the check is CPU-bound, so rank classes by CPU (build + encode). Steps 1-2 (fewer
    launches) lead.
  - Idle <= 10 %: GPU-bound, so rank classes by GPU ms.
  - Between: rank by CPU + GPU.
- **R2, step 1's first target.** The step (C32 - A16) is split into delta CPU and delta GPU, by class. The class with
  the largest delta is step 1's first target.
- **R3, piece 2.** The MoE dense side (router + shared + tail, K-73 arms) becomes optional piece 2 if it ranks in the
  top 3 by R1's cost at both lengths, or costs >= 15 % of the check.
- **R4, bytes.** Step 5 (bytes at the floor) ranks last unless GPU busy <= 1.3 x the floor. Above that, GPU time is not
  bound by bytes.
- **R5, the tail.** If read + commit >= 10 % of wall, step 4's tail part ranks up. The gate's round trips cannot be seen
  on the resident slice.

## Validity
- **V1.** The hook's own cost is <= 1 % of wall. A microbench in separate processes, with the dylib loaded vs not,
  gives the per-dispatch cost x dispatches per check.
- **V2.** Pass K's dispatches per check are within +-3 % of pass T's: the flushes add no kernels.
- **V3, info only.** A16 wall is within +-25 % of C6's 22.67 ms. The configuration differs (gdn_body off), and absolute
  ms across runs are not comparable (BigMoE ISO2).
- **V4.** C32 / A16 is about 1.5-1.65 (C6 1.58 with mr2 + hc32; engine 1.625).
- **Pass K's GPU sum vs pass T's GPU busy** is reported as a ratio (expected 1.0-1.4, because flushing loses overlap).
  Classes are given raw and scaled to pass T.

## Not covered by the slice
The engine's 61-64 / 99-102 ms (48 layers) also contain three things the slice lacks:
- the gate's per-layer CPU service;
- the tier experts (NAX6, OOO);
- the 9-line gate at 32 rows.

A night run on the engine closes that gap: my own copy of r57_screen_fast's G32T time mode with the same hook. It is
proposed, not booked.

---

## Pass X: causal knock-outs (pre-registered Sun 11:1x, after pass T / K of c55_1105, before running them)
Pass T found the check GPU-bound on the slice. The GPU is busy 94 % / 91 % of the wall at A16 / C32, and thread CPU is
0.35 / 0.33 of the wall. That is the opposite of P1 and of the plan's premise, so it gets a causal test before it is sent.

The run: C55_KO=1, the same harness. Arms A16, A16cpu, C32, C32cpu, C32h1, rotated within each round; 12 rounds; quiet
ticket. Each knock-out is scored by its paired per-round delta against its base arm.

| knock-out | what it does | if GPU-bound (pass T's reading) | if CPU-bound |
|---|---|---|---|
| X1: A16cpu / C32cpu | + 2.0 ms of Python thread CPU per check (a busy loop, 1/12 in each layer call) | wall + <= 0.7 ms (<= 35 % of the added CPU) | wall about + 2 ms |
| X2: C32h1 | the LM head as ONE 32-row MLX call instead of v32's 2 x 16: one read of the 358 MB head (not exact; timing only) | wall - 1.3 to - 2.0 ms, following GPU busy | wall about 0 (it drops only 2 dispatches) |

**Lines.**
- Both knock-outs as predicted: the GPU-bound reading is confirmed, and the ordering below stands.
- X1 at >= + 1.4 ms, or X2 at |delta wall| < 0.5 ms: the reading is wrong somewhere. Report that, and do not send the
  ordering as settled.

## RESULT (Sun 11:04-11:10, quiet tickets, power mode 2): on the slice the exact check is GPU-bound at both lengths
Runs: c55_1105 (pass T + K) and c55_ko_1110 (pass X). Data: data/m1/check55/. Read-outs: c55_1105_read.txt and
c55_1105_class_floors.txt.

**Headline.** On the slice the check is GPU-bound.
- The GPU is busy 94 % (A16) / 91 % (C32) of the wall. Thread CPU (build + encode + read + commit) is only 0.35 / 0.33
  of the wall, and it runs while the GPU works.
- C6's "build share 0.54 / 0.62" was the forward call's wall time, and that includes waiting on the GPU. At A16 the
  forward call takes 12.7 ms of wall but only 6.5 ms of thread CPU.
- Pass X confirms it causally:
  - +2.0 ms of CPU moves the wall +0.32 ms (A16) / +0.80 ms (C32);
  - one read of the LM head instead of two moves the wall -1.55 ms, with GPU busy -1.60.

### 1. Per check (pass T, medians, ms)
| | A16 | C32 | A16k | C32k |
|---|---|---|---|---|
| wall | 23.37 | 38.16 | 23.13 | 36.97 |
| CPU: build / encode (SB + read phase) / commit | 3.66 / 4.29 / 0.14 | 5.79 / 6.57 / 0.16 | 3.34 / 3.52 / 0.12 | 4.91 / 5.88 / 0.16 |
| CPU total (share of wall) | 8.09 (0.35) | 12.51 (0.33) | 6.98 (0.30) | 10.95 (0.30) |
| GPU busy (share of wall) | 21.98 (0.94) | 34.61 (0.91) | 21.64 (0.94) | 34.34 (0.93) |
| GPU idle | 0.90 | 3.43 | 0.83 | 3.37 |
| GPU first start after the check begins | 0.44 | 3.13 | 0.38 | 3.01 |
| dispatches / barriers / command buffers | 1105 / 572 / 85 | 2060 / 760 / 108 | 1033 / 554 / 77 | 1916 / 736 / 108 |
| floor (dense 0.97 GB + routed union) | 12.36 | 16.32 | 12.35 | 16.32 |
| GPU busy / floor | 1.78 | 2.12 | 1.75 | 2.10 |

- CPU per dispatch, all in: 7.3 us (A16) / 6.1 us (C32). Encode alone: 3.9 / 3.2 us.
- C32 / A16 = 1.633 (the engine's G32T: 1.625).
- K-73's MoE block beats MTPLX's stock block by 0.24 / 1.19 ms at A16 / C32, with 72 / 144 fewer dispatches.

### 2. By class (pass K, GPU ms scaled to pass T; class floors from the weights at 280 GB/s)
| class | A16 GPU (floor; GPU / floor) | C32 GPU (GPU / floor) | the 17-row step: + GPU | dispatches A16 / C32 |
|---|---|---|---|---|
| MoE experts | 11.27 (8.88; 1.27x) | 15.79 (1.23x) | +4.52 | 312 / 312 |
| dense projections (GDN + attention) | 2.81 (1.37; 2.1x) | 5.95 (4.4x) | +3.14 | 42 / 84 |
| HC read + write (bf16 mixers, 26 MB a layer) | 2.61 (1.13; 2.3x) | 3.25 (2.9x) | +0.64 | 132 / 408 |
| head (mixer + lm_head) | 1.89 (1.32; 1.4x) | 3.79 (2.9x) | +1.91 | 12 / 28 |
| MoE dense side (router + shared + tail): stock / K-73 | 1.23 / 1.04 (0.18; 6.9x / 5.8x) | 1.89 / 1.68 | +0.66 / +0.64 | 180 / 360 (K-73 108 / 216) |
| attention core (no weights) | 0.88 | 1.83 | +0.95 | 338 / 692 |
| PLE | 0.60 (0.24; 2.6x) | 0.82 | +0.22 | 22 / 28 |
| GDN core (no weights) | 0.55 | 1.11 | +0.56 | 63 / 144 |
| glue, read, commit | 0.14 | 0.19 | +0.05 | 9 / 9 |

**Dispatches per layer at A16.**
- GDN layer ~62: HC read 10, GDN core 7, projections 3, experts 26, router 3, shared 5, tail 7, HC write 1.
- Attention layer ~170: the attention core alone is 112.5 (MTPLX's 12 head chunks).
- At C32: ~111 and ~305.

### 3. The 17-row step (C32 - A16 = +14.80 ms)
| component | ms |
|---|---|
| GPU busy | +12.63 |
| GPU idle | +2.53 (the GPU starts 2.7 ms later) |
| CPU | +4.42 (hidden) |

- **Experts +4.52 ms.** The union grows 75 -> 108 per layer: +1.11 GB, +3.97 ms at the floor. An exact 32-row check
  must read these bytes.
- **Dense projections +3.14 and head +1.91.** v32's 2 x 16 reads each weight twice.
- **The rest:** the second chunks of the latency-bound pieces, +2.8 in all (attention core +0.95, MoE dense side +0.66,
  HC +0.64, GDN core +0.56).
- **The late GPU start: ref2's SB_FIRST (seam b) and PLE_LATE (seam d) fire only at 5-16 rows** (_dec_call's
  `5 <= rows <= 16`). On a 32-row check the GPU therefore waits for the first 4-layer submit: first start 3.13 ms vs
  0.44.
  - This is also where pass X's C32 CPU leak sits: the spin in the first 4 layers, 0.65 ms of its 1.94, delays the
    start one for one.

### 4. Against the pre-registration
- **Predictions.**
  - P1 FAIL: the check is GPU-bound, not CPU-bound.
  - P2 FAIL: 1,105 / 2,060 dispatches, not 350-450 / 600-800.
  - P3 FAIL, low: 7.3 / 6.1 us per dispatch all in.
  - P4 FAIL: the step is GPU-side.
  - P5 FAIL, high: 1.78x.
- **Validity.**
  - V1 PASS: the hook costs <= 0.01 us per dispatch.
  - V2 PASS: pass K's dispatches are +0.4 % / +0.2 % of pass T's.
  - V3 PASS: A16 23.37 vs C6's 22.67.
  - V4 PASS: C32 / A16 1.633. Pass K's GPU sum / pass T's GPU busy is 1.02 / 1.03.
- **Pass X.**
  - X2 PASS.
  - X1 PASS at A16 (+0.32).
  - X1 at C32 is +0.80: over the 0.7 prediction, under the 1.4 "wrong" line. Its cause is the late start above.
- **Rules.**
  - R1: idle 9 % at C32, so the check is GPU-bound: rank classes by GPU ms.
  - R2: step 1's first avoidable target is one weight pass for the dense projections + LM head at 17-32 rows (+5.0 of
    the +14.8). The experts' +4.5 is bytes.
  - R3: the MoE dense side is 5-6 % of GPU, rank 5, so **optional piece 2 does not qualify**.
  - R4: the experts run at 1.23-1.27x their floor. The dense side is at 2-10x its floor, a kernel-efficiency problem
    rather than bytes.
  - R5: commit + read < 1 % of wall.

### 5. Reading for the engine (an extrapolation, not measured)
Method: 48 layers = 4 x the slice's per-layer classes, the head counted once, the experts at about their floor.
- **A16:** dense side ~34 ms + experts ~26 ms = ~60 ms. The engine measures 61-64.
- **C32:** ~60 + ~40 = ~100 ms. The engine measures 99-102.
- **The excess over the floor.** The engine's excess (~24 / ~48 ms, check55_plan) equals the dense side's GPU time
  over its weight floor (~22 / ~48 ms).
- **Caveat.** The engine's gate and its NAX6 / OOO experts are not on the slice. An engine run with the same hook (my
  copy of G32T's time mode) would confirm this.

### 6. The order this gives CHECK55
1. **Step 1, re-aimed at GPU time.**
   - First: one weight pass for the dense projections + LM head at 17-32 rows. That is ~10-14 ms of the engine's
     32-row step, engine-scaled.
   - BigMoE's C32m "mr2 saves nothing" ran inside the 35-40 % drift that ISO1 / ISO2 later found, and was never re-run
     clean. C6 measured mr2 at -1.4 ms on the slice. So: re-run C32m clean.
   - Then: one-call cores at 32 rows (attention, already bitwise in Leap's 90dae21; HC, MoE dense side, GDN). That is
     ~2.8 ms per 12 layers, ~10 ms engine-scaled.
   - Free and exact: widen SB_FIRST + PLE_LATE to 17-32 rows, for up to ~2.7 ms per 32-row check (testing now).
2. **Dense-side kernel efficiency at 16 rows** (it helps both lengths).
   - Projections run at 2.1x their floor and HC at 2.3x: ~11 + ~10 ms at A16, engine-scaled. Bringing them toward
     ~1.2x saves ~9 ms.
   - The built exact levers missing from the 45 preset go first: hc_b (-0.9..-1.1 ms) and GDNB (-0.7 ms).
3. **Step 3, the attention core as one dispatch.** It costs 0.29 ms per attention layer at A16, at 112 dispatches a
   layer: engine ~3.5 ms at A16, ~7.3 at C32.
4. **Step 2, the dispatch diet, re-priced** at ~-2 ms. Its CPU side is hidden; what remains is ~2 us of GPU per
   dispatch removed.
5. **Step 5, bytes:** this applies to the experts only (1.23-1.27x on the slice's resident kernel). The engine's NAX6 is
   not measured.
6. **Step 4:** the commit / read tail is < 1 %. The gate cannot be seen on the slice.

## Pass S: SB_FIRST + PLE_LATE widened to 17-32 rows (pre-registered Sun 11:1x, before running)
**The arm: C32sb.** My wrapper replays ref2 _dec_call's seams (b) / (d) and _ple_stage's deferral at 17-32 rows, using
ref2's own module state. Leap's files are untouched.
- Seam (b): layer 0's output goes to the GPU before layer 1.
- Seam (d): the PLE rows are staged at the PLE layer, not before layer 0.
- It is scheduling only: the logits must be bitwise vs C32.

**The run:** arms A16, C32, C32sb, rotated within each round; 20 rounds; quiet.

**Prediction.**
- C32sb's GPU first start drops from ~3.1 ms to <= 1.0 ms.
- Wall -1.5 to -2.7 ms.
- GPU busy unchanged (+-0.3).

**Line.** Wall <= -1.0 ms and logits bitwise: recommend it for the 17-32-row path, a one-line change of `<= 16` to
`<= 32` in ref2 (Leap's call). Otherwise drop it.

## Pass S RESULT (Sun 11:15, quiet, 20 rounds; c55_sb_1116): PASS, -2.0 ms per 32-row check, bitwise
- **Bits:** C32sb vs C32, 32 / 32 rows bitwise. The seams fired once each per forward.
- **The GPU's first start:** 2.54 -> 0.48 ms (paired -2.03).
- **Wall on warm pairs** (rounds where A16 ran first): **-1.99 ms**, range [-2.28, -1.47] over 7 rounds. Cold vs cold:
  -2.13 ms. GPU busy -0.14 (unchanged).
- The pre-registered line (<= -1.0 ms, bitwise) is met: **recommended to Leap** (ref2 `_dec_call` / `_ple_stage`:
  `<= 16` -> `<= 32`).
- **Why the all-pairs IQR was wide** ([-9.9, +2.4]): the cold PLE effect below.

## Found on the way: cold PLE n-gram staging costs +5-8 ms (A16) / +8-12 ms (C32) of wall per check (a LEAD)
Data: data/m1/check55/c55_ple_cold.txt.

**How it showed up.** Every arm checks the same new tokens each round, so the round's FIRST arm reads those tokens'
n-gram rows cold from the SSD-backed 29.8 GiB table (file-backed pages); the later arms find them warm. First-in-round
vs later, the median wall:

| arm | c55_1105 | c55_ko_1110 | c55_sb_1116 | the GPU's first start when cold |
|---|---|---|---|---|
| A16 | +8.09 ms | +4.98 | +4.75 | 0.4 ms: seam (d) stages after layer 0's submit, but the read blocks the thread and the GPU runs dry |
| C32 | +11.97 | +8.03 | +8.06 | 11-14 ms: no seam (d) at 32 rows, so the read comes before any GPU work |
| C32sb | | | +7.92 | 0.47 ms |

**Open question for BigMoE / Leap: does the real decode loop pay this every check?**
- G32T re-checks the same tokens every round, so it is always warm and never sees it.
- In real decoding each check's tokens are new: their n-gram rows are cold unless MTPLX's hot cache (1023 MB) or its
  prefetch already holds them.
- If the loop does pay it, two fixes:
  - prefetch the drafted tokens' rows while the drafter runs;
  - make sure the stage overlaps GPU work (seam d at every check length).
