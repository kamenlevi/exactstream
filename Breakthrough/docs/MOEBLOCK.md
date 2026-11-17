# MOE BLOCK PROOF: one layer's MoE block as a fused kernel chain at >= 85 % of its byte floor (Breakthrough, from Tue 17 Nov 2026 11:2x)

The orchestrator's GO (11:25): the proof of the 550 path's engine half, by Thursday.
- Build the MoE BLOCK of ONE layer (router -> experts -> shared -> combine) as a compiled, fused kernel chain at >= 85 % of
  its byte floor, measured quiet.
- Base it on the expert-rewrite knob. Overlap the MMA with the stream (X0: FEASIBLE), with router / shared / combine folded
  around it. Target ~1.1x max(stream, MMA).
- Bitwise or FAST-gated; Saturday's path does not change. Leap reviews, wires it as a c25 arm, and measures it in slot #4
  (Thu); a candidate by Thu ~12:00 makes it.
- Mind account-1's weekly pace: spend on this, not side work.

## The block today (one tier layer, decode / verify rows; BigMoE bigmoe/tier.py install_block + ref2._ooo_sel)
1. **shared** = attention_gate(shared_expert(x), shared_expert_gate(x)). That is MLX 4-bit g64 gate | up | down
   (640 x 2560 each) and a 2560 -> 1 gate.
2. **router:** softmax(gate(x)) (MLX 8-bit g64, 512 x 2560), argpartition top-10, normalize, sort by score.
3. **the 1c tier glue + posts** (t1cf / ref2), then the expert one-call (grouping, rowsum64, kernel A, rowsum64, kernel
   B) and combine_phase.
4. **y + shared.**

## Pre-registration (Tue 11:2x, BEFORE any timing of a block candidate); Leap's five review terms (11:25), verbatim in substance
1. **FLOOR** = the bytes the block must read per call, at 280 GB/s. It is also quoted against the 305 GB/s streaming
   ceiling (expread).
   - The kept routed experts read their MSB-3 plane, plus their LSB plane where a 4-bit use reads it. These are the real
     piece sizes from the store's index, per block, as exp32 computes them: MSB 2.064 MB, LSB 0.720 MB per expert.
   - The shared expert (MLX 4-bit g64, 3 x 640 x 2560: 2,457,600 B of codes + 307,200 B of scales / biases) is
     2.765 MB.
   - The router (MLX 8-bit g64, 512 x 2560: 1,310,720 + 81,920 B) is 1.393 MB. The shared_expert_gate is 1.4 KB.
   - Activations are not counted.
   - On exp32's 24 layer-20 blocks the routed part averages 0.1045 GB (32 rows) and 0.0907 GB (16 rows), so the floor is
     ~388 us (32 rows) / ~339 us (16 rows). The line is computed per block from that block's own bytes; these are the
     means.
2. **Bench:** exp32's real 1c blocks (the decision map's code windows), layers 20 + 44, 32 AND 16 rows. Quiet,
   interleaved against TODAY's chain (router + glue + one-call + shared + combine) in one process, with the order
   rotated per round. All slots resident (the lab windows' state, as c25's warm windows); the loader's misses are out
   of scope here.
3. **PASS:** block time <= floor / 0.85 (>= 85 % of the floor), as the median over blocks of the per-block ratio, at BOTH
   32 and 16 rows, on both layers.
   - BETWEEN: 75-85 %.
   - FAIL: < 75 %.
   - Info: x vs 305 GB/s, x vs today's chain, and MMA / stream / ALU from ablations.
4. **Numerics:** bitwise vs today's chain, or FAST with Leap's a)-g):
   - error vs an fp32 reference (xrw_check-style, extended to the whole block);
   - determinism (repeats bitwise);
   - 32 vs 16+16 stated.
   tgl16eu's tier-4 upgrade is FAST and already accepted (Leap 10:33). Anything new is listed with its own check.
5. **Semantics:** the streaming semantics are kept (wait on non-resident slots, posts, coherent pool reads) and there are
   no extra bytes. The wire form for c25 is ON / MODE switches in a frozen folder with md5s, named moeblk_v1 (the knob_v*
   names belong to the expert rewrite).

## Plan (to be filled in)

## Findings while planning (Tue 11:3x)
- **The harness** (scripts/moeblk/: moeblk.py + config/q125b.json + bench.py) runs today's chain from the engine's real
  components:
  - MLX 8-bit router logits, Leap's router_fused and t1cf;
  - device slot lookups (the posts' result when every expert is resident; the post kernels are out of both arms);
  - the live COH one-call (knob-switchable);
  - combine_phase;
  - the MLX 4-bit shared expert + its sigmoid gate;
  - the add.

  Its routing equals the decision map's captured routing on 57 % of rows' top-10 SETS. The rest differ by 1 expert
  (48 / 128 rows) or 2-3 (7 / 128), with the 10th / 11th expert EXACTLY tied in bf16 (median margin 0.0). That is
  boundary tie-breaking, not a routing bug, and both arms use the same routing.
- **The component split of today's block** (32 rows, layer 20, 8 blocks; CONTENDED GPU, proportions only): routed
  experts 0.78, shared expert 0.17, router + glue + slots 0.15, combine + add 0.08 of the block. They sum to 1.18, so
  today's chain already overlaps ~18 % of them.
- **What 85 % needs:** with the experts at ~77 % of their own floor (tgl16eu) and the rest added, the block sits near
  ~70 %. 85 % needs the experts at ~90 % AND router / shared / combine folded or hidden.
- **X7 (x7_async.py; 1 simdgroup per core).** A pair of 16x32x32 MMAs takes ~362 ns: ~180 ns of LATENCY per op.
  The simdgroup's own ALU work placed between the issue and the use hides ~50 % of the smaller cost; the accelerator is
  partly asynchronous within a simdgroup. So the expert kernels are latency-bound per simdgroup and need many in flight.
- **o1p (pipelined ooo1cq: decode g + 1 between g's MMA and its epilogue; scripts/xrw/o1p.py):** BITWISE == ooo1cq
  (1,280 of 1,280 pairs, a)-c) PASS), but SLOWER in steering: 1.039 / 1.049 x orig at 32 / 16 rows. With the upgrade,
  o1pu 0.934 vs tgl16eu 0.879. The extra live registers (cg across the decode) cost more occupancy than the latency
  they hide, as MoonShots' register prefetch did. DROPPED.

## Candidate F1 (scripts/moeblk/msh.py; Tue 11:4x)
- **What it is:** the routed expert kernels (tgl16eu) with the SHARED EXPERT folded in as extra segments of the same two
  kernels (grid y >= the routed capacity).
  - The shared weights' MLX 4-bit codes are copied as nibbles (no spread decode) with bf16 scales / biases, and run
    through the same MMA / epilogue / K chunks / swiglu.
  - The router, glue and slots stay exact (MLX qmm + router_fused + t1cf + lookups). Then combine_phase (exact) and one
    compiled gated add (the shared gate stays MLX's tiny qmm).
- **check_f1.py (2 real blocks per layer, L20 / L44, 32 and 16 rows):**
  - the routed rows == tgl16eu BITWISE (y, rs_m, rs_l) in all 4 cases; 3 repeats bitwise;
  - the shared rows vs an fp32 dequant reference: F1 0.46-0.50 % mean (max 0.76 %). MLX's own shared expert is
    0.46-0.49 % at 16 rows (F1 vs MLX 0.01 %) and 0.81-0.85 % at 32 rows. F1's shared path is as accurate as MLX's
    at 16 rows and MORE accurate at 32;
  - the whole block vs today's chain: 5.4-6.9 % mean rel L2 per row (max 13-16 %). That is the accepted upgrade's
    tier-4 pairs (FAST) plus the shared path.
- **Steering (Tue 11:42) is UNUSABLE:** today's chain measured 1,400 us at 32 rows and 3,000 us at 16 under heavy
  contention. f1 / today 0.80 (32) and 0.92 (16) carry no weight. A quiet ticket was asked of the orchestrator.
- **Ready for quiet (Tue 11:4x):** scripts/moeblk/mq_run.sh on frozen copies data/moeblk/frozen_q1 (md5s in
  frozen_q1.md5).
  - Step 1 is the pre-registered block bench: today / knob / F1, L20 + L44, 32 + 16 rows, 12 blocks x 6 calls x 5 rounds.
  - Step 2 is tgl16's dependency ablations: abl4 = no MMA; abl32 = the MMA reads another simdgroup's tile, so it does
    not depend on its own decode while the decode still runs; abl36 = both. abl32 separates dependency latency from
    resource contention.
  - msh now imports the routed kernels from the FROZEN knob_v2 (MOEBLK_KNOB), the code Leap's arms run. F1 re-checked
    there: the routed rows are bitwise, 4 / 4.
  - The run launches on the orchestrator's window, or in NewIdeas' evening slot (18:00-20:30).

## Candidate F2 = F1 + the FRONT fold (scripts/moeblk/front.py; Tue 11:5x)
- **What it is:** one single-threadgroup kernel after router_fused replaces ~6 critical-path dispatches:
  - Leap's t1cf (replicated expression for expression);
  - the slot lookups;
  - the upgrade grouping (xgrp v2, reading each pair's slots from its own thread's registers);
  - rowsum64(x) (Leap's sequential sum).
- **front_check: BITWISE == the separate kernels** in sc', ids', nolsb, pre_m, pre_l, pidx, seg (up to nseg), nseg and
  xs. Real blocks, L20 / L44, at 32, 16, 8 and 1 rows: PASS.
- **F2 == F1 BITWISE** on 12 real blocks (L20 / L44, 32 / 16 rows).
- **The quiet run's frozen copy is now data/moeblk/frozen_q2** (front + f2 added; md5s in frozen_q2.md5); its arms are
  today / knob / f1 / f2.

## QUIET RESULT (Tue 11:55:32-11:55:43; the orchestrator's placed ticket after Leap's smoke; NO-SWAP 48 GB): F1 / F2 = FAIL vs the pre-registered 85 %
The frozen copies are data/moeblk/frozen_q2. Output: data/moeblk/bench_q1.*, the runner's stdout in data/moeblk/mq_run.txt,
and the ablations in data/xrw/xrw_bench_q2_abl.*. Per block: 6 back-to-back calls in one graph, 5 rounds rotated,
12 real blocks; the median over blocks.

| layer / rows | floor | today | knob (tgl16eu) | F1 (shared folded) | F2 (F1 + front) | F2 efficiency -> verdict |
|---|---|---|---|---|---|---|
| L20 / 32 | 372.1 us | 859.6 us | 734.3 (0.843) | 737.9 (0.853) | **685.6 (0.797)** | 0.522 -> FAIL |
| L20 / 16 | 239.4 | 699.6 | 602.7 (0.855) | 622.1 (0.875) | 603.2 (0.873) | 0.400 -> FAIL |
| L44 / 32 | 345.8 | 1058.0 | 916.3 (0.859) | 923.4 (0.859) | 889.0 (0.835) | 0.400 -> FAIL |
| L44 / 16 | 233.7 | 677.5 | 570.3 (0.848) | 597.3 (0.900) | 573.3 (0.855) | 0.404 -> FAIL |

(x / today in parentheses, per block medians.)

**Readings:**
- **The block's non-expert chain is the dominant gap, not the expert kernels.**
  - At L20 / 32 rows today's block costs 860 us. Its expert one-call alone costs 583 us in the same window (xrw bench,
    orig). So the router logits (MLX qmm), router_fused, t1cf, the slot ops, the MLX shared expert, combine_phase and
    the add cost ~280 us, for ~4 MB of bytes (~15 us at the floor).
  - That is ~20 DEPENDENT dispatches, several of them single-threadgroup kernels running on ONE core of 20.
  - In F2 the remaining non-expert part still costs ~200 us.
- **The front fold works:** F2 vs F1 is -52 us at L20 / 32 and -19 us at L20 / 16, and it is EXACT. Per 48-layer check
  that is ~-2.5 ms at 32 rows / ~-0.9 ms at 16 rows [P], an engine lever on its own.
- **Folding the shared expert into the expert kernels does NOT pay:** F1 vs knob is 0.853 vs 0.843 and 0.875 vs 0.855.
  Its extra segments cost about what MLX's three qmm cost. Drop it; keep MLX's shared expert (or run it concurrently).
- **The expert kernels' exposed MMA is accelerator THROUGHPUT, not dependency latency** (ablations, 32 rows):

  | arm | us |
  |---|---|
  | tgl16 | 562.5 |
  | no MMA (abl4) | 438.5 (-124) |
  | MMA on ANOTHER simdgroup's tile, its own decode still running (abl32) | 561.9 |

  Removing the decode -> MMA dependency changes nothing. Together with o1p's failure, in-simdgroup pipelining cannot
  hide it. The padded accelerator work (16-row tiles at ~4-6 real rows per segment) is a hard floor at these row counts;
  only fewer segments (the upgrade) or a faster MMA type (int8, rejected for quality) cut it. (abl36 ran as abl32: the
  bit-32 branch takes precedence over bit 4.)
- **The X0 premise vs the real block:** independent MMA and stream DO overlap (X0 FEASIBLE). But the real block is
  limited by (1) a long chain of small dependent kernels and (2) the accelerator's padded throughput. The 85 % line
  would need the WHOLE block in ~2 dispatches (a persistent megakernel with device-wide phase barriers: router -> front
  -> A -> B -> combine). That is a large, risky build, and the expert part alone would still be at ~80 % at best.

## DIAG (pre-registered Tue 12:0x, before running; INFO, not a verdict): where the block's non-expert time goes
The quiet result leaves ~200-280 us per block outside the expert kernels. Two causes predict different next steps:
- CPU-side: Python graph build + MLX encode per call, with the GPU waiting. Remedy: fewer ops per block, or mx.compile.
  A GPU megakernel would also help, but only by cutting the op count.
- GPU-side: dependent single-threadgroup dispatches, with the GPU busy but under-filled. Remedy: a persistent megakernel
  with device-wide phase barriers.
Measure: bench.py now splits each timed group of REPS calls into CPU build (graph construction), CPU encode
(mx.async_eval returning), wall (after synchronize) and GPU busy (libmtlprof: the union of the command buffers' GPU
intervals). It also counts command buffers committed during the build; must be 0, otherwise an arm syncs internally.
Arm f2c (added before the run): F2's block under mx.compile, traced once per layer and row count. Same kernels and
order, so it must equal f2 bitwise; the bench checks this and drops the arm if not. Its CPU per call next to f2's
shows how much of the CPU part a compiled block removes without any new kernel.
Run: layer 20, 32 and 16 rows, arms today / knob / f2 / f2c, 6 blocks x 6 reps x 3 rounds, under a SHARE lock (EvalSuite holds
~60 % of the GPU). GPU busy is therefore an UPPER bound and idle a LOWER bound; the CPU numbers do not depend on the GPU.
Reading, per arm at 32 rows:
- CPU (build + encode) per call >= 0.8 x GPU busy per call -> CPU-BOUND. The pre-registered wall mostly measures the
  op count. Next: an mx.compile'd block and a fused glue arm. The megakernel case weakens.
- CPU <= 0.5 x GPU busy and GPU idle <= 15 % of wall -> GPU-BOUND. The megakernel case stands, and only it.
- Between -> MIXED. Report both shares.
Engine context, added before the run (Leap 00:55, ~/Leap data/week98/c25/gpubB_run.txt, recorded in my HANDOVER at TUE
01:0x): the lossy 32-row check is GPU-PACED, with 54.2 of 56.1 ms GPU-busy and 23.2 ms of CPU hidden by the pipeline.
In the engine a block therefore costs about its GPU-busy time, not the bench's wall (the bench builds REPS calls before
evaluating them, so the build is never hidden). So floor / GPU busy is the engine-relevant efficiency. It is reported
as INFO beside the pre-registered verdict; the verdict (wall-based, FAIL) stands as pre-registered.
Owners if the gap is CPU: module 1a replay (mine) and 1b compile / diet (Fit12). Leap's dispatch-diet price applies
(data/week98/dispatch_diet_price.md): per MLX glue dispatch ~3.0 us to build + ~3.5 us to encode on the CPU vs ~1.1-2.1
us on the GPU.
AMENDED Tue 12:1x, before any run:
- The run moves to a QUIET lock. At 12:11, EvalSuite's lossy gate and Sandbox98's capture held the GPU at 98-100 %, so
  a share read of GPU-busy would mean nothing. At 12:15 the Mac was idle (GPU 0 %) and Kamen away: one short quiet lock.
- The grid is the main bench's: L20 + L44, 32 + 16 rows, 12 blocks x 6 reps x 5 rounds.
- Arms: today / knob / f1 / f2 / f2c. f1 is added because F2 - F1 is exactly the front fold.
- A share-lock smoke (L20, 2 blocks, 1 round) runs first, only to prove the arms run. Its numbers are not used.
- Decision rule for the front fold: it goes to Leap's slot #4 as a knob only if F2's GPU-busy per block is >= 10 us below
  F1's at 32 rows on BOTH layers (~0.5 ms per 32-row check [P]). Otherwise the MoE-block line closes with tgl16eu as its
  result. The GPU side decides because the engine is GPU-paced, and Leap's fused t1cf (one threadgroup) was +0.49 ms
  SLOWER in it (Leap HANDOVER: "t1cf +0.49 ... SLOWER in the engine").
- Megakernel: not built. Its gain is bounded by the glue's GPU-busy, which this run measures (today's busy minus the
  expert kernels' busy).
AMENDED again Tue 12:2x, before the quiet run (the smoke ran at 12:15 under share; its numbers are not used):
- The smoke showed the bench's REPS calls are INDEPENDENT, so MLX can overlap call i+1's kernels with call i's. The
  engine's layers are dependent, so a per-call time from 6 independent calls understates a block's GPU timeline.
- So the quiet run has two passes on frozen_q3:
  - Run A (pre-registered grid): reps 6, rounds 5, arms today / knob / f1 / f2 / f2c. Comparable with the 11:55 run.
  - Run B (serialized, engine-like): reps 1, rounds 9, arms today / knob / f1 / f2 / f2c + the components c_exp /
    c_route / c_shared / c_comb (cand_comp.py, on precomputed inputs).
- The decision rule above uses Run B's GPU-busy: F2 - F1 <= -10 us per block at 32 rows on both layers -> the front
  fold goes to slot #4 as a knob.
- The glue's GPU time per block = knob - c_exp (Run B, 32 rows):
  - >= 150 us on both layers (>= ~7 ms per 32-row check [P]): a glue megakernel / fusion is proposed to the
    orchestrator as a BIG lever, priced against Leap's per-class line (tue1: 10.7 ms per 32-row check outside the five
    classes bounds the engine's whole glue).
  - <= 60 us (<= ~3 ms): not worth it. Between: reported, no proposal.
SMOKES (share lock; NOT used for any decision, as pre-registered; data/moeblk/bench_smoke*_share.txt):
- 12:15, L20 / 32 rows, 2 blocks, reps 2, the GPU idle at the start: every arm runs, f2c == f2 bitwise, and no command
  buffer is committed during any build (no hidden syncs). Peak footprint 1.74 GB (one layer).
- 12:17, the frozen_q3 copy, reps 1, all nine arms incl. the components: all run; Sandbox98's training held ~40 % of
  the GPU, so its GPU-busy numbers are inflated ~2x.
- The CPU side does not depend on GPU load, so one smoke reading stands: mx.compile of F2 (f2c) cuts the CPU graph
  build from 65.1 to 13.4 us per call (12:15) and from 117.6 to 27.7 (12:17, CPU-contended), bitwise identical. The
  encode stays (67.6 -> 57.5). For Fit12's 1b compile / diet: a custom-kernel block compiles cleanly.
- The quiet run is ARMED for NewIdeas' 18:00 slot: scripts/moeblk/md_run.sh, MD_START 18:00, MD_LATEST 20:15. It waits
  (never skips) while Kamen is at the Mac. If he is back when the lock is granted, it releases the lock and waits again.
AMENDED a third time Tue 12:4x, before the quiet run:
- EvalSuite 12:3x: the 1c tiers (t1cf) FAILED V2 and are not adopted, so the engine's MoE glue will not run t1cf.
- Added component c_t1: t1cf alone, on router outputs precomputed outside the timed region. frozen_q4 = q3 + that one
  file (cand_comp.py). Runner data/moeblk/md_run2.frozen.sh, tags q4_r6 / q4_r1; q3's runner was stopped while still
  in its start wait, holding no ticket.
- The megakernel rule now applies to the glue WITHOUT the 1c tiers: knob - c_exp - c_t1 (Run B, 32 rows, both layers)
  >= 150 us -> propose; <= 60 -> close. The glue with t1cf is reported beside it.
- The front fold's rule is unchanged (F2 - F1). Caveat: without the 1c tiers the engine's fold would cover only slots +
  grouping + rowsum64, and its share of F2 - F1 cannot be separated here. Leap's engine A/B would price it.
- c_t1 smoke (12:38, share, 1 block, Sandbox98 at ~40 % of the GPU; not used): runs, no hidden sync, GPU-busy 76 us,
  vs 154 for the whole route path (c_route) in the same pass.
SCHEDULE Tue 17:4x (the pre-registered lines, files and runner are unchanged):
- The orchestrator moved the quiet run to the first quiet gap after TrainLab's S22 training (~20:35-20:39), before
  BigMoE's M4 (23:35).
- Waiter data/moeblk/md_after_s22.frozen.sh (= scripts/moeblk/md_after_s22.sh) holds no lock and no ticket until S22's
  runner exits and no other session holds or queues a lock (two reads 60 s apart). Then it starts md_run2.frozen.sh as is.
  Latest start 22:15, so no grant after 23:15. Dry-run: 4 cases (start after both waits; skip at the latest start; the
  real S22 pid; the real lock list).
- A 17:38 attempt on the orchestrator's 17:35 "run now" was withdrawn a minute later. Its queued ticket lived ~30 s and
  made S22 yield once (saved at step 48, resumed 17:39:05). No measurement ran.
RESULT (QUIET, Tue 19:39; the waiter found the gap at 19:38:52, lock granted 19:39:22, NO-SWAP 37 GB, Kamen away;
data/moeblk/bench_q4_r6.* = Run A, bench_q4_r1.* = Run B, rules_q4.txt = the rules applied by scripts/moeblk/rules_q4.py,
paired per block: the median over the 12 blocks of per-block differences, each a median over rounds):
- Integrity: every arm ran; f2c == f2 bitwise on 12 blocks in all four layer / row cases; the candidate arms committed no
  command buffer during the build. c_exp (3) and c_t1 (1) did, once: their cached inputs were computed inside the timed
  region in round 0 on blocks 3-12 (the warm-up covered blocks 1-2). The 9-round median absorbs it: component / knob
  GPU-busy is the same on blocks 1-2 and 3-12 (c_exp 0.785 vs 0.770 at L20, 0.767 vs 0.771 at L44; c_t1 0.047-0.051).
- FRONT FOLD (the rule: Run B GPU-busy, 32 rows, F2 - F1 <= -10 us on both layers): -18.7 us at L20 [-22.8, -13.3],
  -17.1 at L44 [-19.8, -15.1]; 12/12 blocks <= -10 on both -> PASS -> a slot #4 knob for Leap. Info: 16 rows -9.4 / -8.9;
  Run A (independent calls) -22.6 / -23.3 at 32 rows.
  Fit to Saturday's stack, found after the rule: F2 as built folds t1cf (the 1c tiers: out, V2 FAIL) and tgl16eu's
  grouping (xgrp v2, the upgrade: out since 15:0x). The stack's tgl16e groups with gpu_grp_1cq. A stack form (slots +
  gpu_grp_1cq + rowsum64, no t1cf) is not built and not checked bitwise, and its share of the -18 us is unknown. So this
  run gives no knob that fits Saturday as is; Leap decides on slot #4.
- MEGAKERNEL (the rule, amended 12:4x: Run B GPU-busy, 32 rows, glue without t1cf = knob - c_exp - c_t1): 110.1 us at
  L20 [102.0, 115.7], 109.0 at L44 [104.5, 112.6] -> BETWEEN the 60 and 150 lines -> reported, NO PROPOSAL. With t1cf
  140.1 / 139.3 (also under 150). x 48 layers = 5.3 / 5.2 ms per 32-row check [P], inside tue1's 10.7 ms outside the
  five classes. The glue includes the shared expert (c_shared alone 79.9 / 79.5 us), which is weight traffic, so a
  megakernel's removable part is smaller still. F2 already takes 50.0 / 48.0 us off knob's block.
- DIAG reading (Run A, 32 rows): GPU-BOUND at L44 for every arm (CPU 0.36-0.46 x busy, idle 9.0-14.0 % of wall). At
  L20, today and f2c are GPU-BOUND; knob, f1 and f2 are MIXED (CPU <= 0.49 x busy, but idle 16.1-16.4 % > 15 %).
- f2c (mx.compile of F2): CPU build 51.0 -> 9.1 us per call at L20 / 32 (Run A), bitwise, GPU-busy unchanged
  (522.0 -> 521.8). Its wall gain (617.3 -> 578.3) comes from the bench, which never hides the build; in the GPU-paced
  engine it is CPU headroom.
- Block GPU-busy (Run B, 32 rows, L20 / L44): today 705.1 / 673.8, knob (tgl16eu) 621.3 / 602.9, f1 592.6 / 571.9,
  f2 573.0 / 555.1 us. Floor / GPU-busy: today 0.54 / 0.52 -> f2 0.66 / 0.63 (info).
- The pre-registered efficiency verdict (wall-based, Run A) stays FAIL for every arm (floor / time 0.50-0.66).
