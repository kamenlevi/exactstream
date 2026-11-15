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
