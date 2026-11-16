# CHECK25: a measured 32-row lossy check at <= 25 ms (Leap's plan)

> (a quote from the project owner, kept private)
percent in 30 tokens and the check at 25ms ... no compromise. And all until Saturday"). The ask: the lossy 32-row
check (1c-tiers bytes; C5 = 6.12 GB on code, floor 21.9 ms) MEASURED at <= 25 ms by Friday. A full-check microbenchmark
on real weights counts. Inputs: Fit12's LOSSY32 ladder (~/Fit12/data/lossy32/lossy32_price.txt, v2), Breakthrough's
LOSSY_CHECK_PLAN (corrected 5fc4c34) and engine profile (a082f8c), and my records (TABLE_350, screens README).

Labels: [K] measured, [P] priced from measured parts, [est] estimate.

## DAY 1 PROGRESS (Mon 13:1x)
- **Built and bit-tested** (data/week98/check25/README.md):

  | piece | what it is | check | speed |
  |---|---|---|---|
  | qmm32n / dn32 / dn32q | 32-row dense matmuls | qmm32n bitwise | 19.3 -> 13.2 ms per check (quiet) |
  | FAST-NAX one-call at M 32 | the experts | bitwise vs 2 x 16 | |
  | int8d | G1's lever | slice PASS, no extra memory | |
  | t1cf | fused 1c tiers | == BigMoE on 384K slots | ~1,350 fewer dispatches |
  | hc8r32 | one-pass HC read | bitwise | ROUGH -2.5..-6.8 ms |
  | hc4r / hc4i | V3's 4-bit HC read | PASS | |
  | f32one | the one-pass FAST route | slice: == v32 on every shared logit, commit bitwise | |

- **Parked:** the chunked GDN (correct, 3x slower as built).
- **Engine harness c25 READY:**
  - Arms F16 F32s F32o F32d F32df F32if F32if6 F32c5 F32c6i (C5 = 1c + V1 + V3 bytes; C6 + int8).
  - Head test PASS. Smoke tonight behind Breakthrough's map screen; the full run in Tue's ~09:30 slot.
- **Still open:** the experts at 32 rows (exp32 bench queued); the GDN core at 32 rows (sequential ~4.3 ms; a chunked v2
  needs (dv, t) parallelism); the 32-row GDN / attention glue (the launch proxy shows MTPLX's eager ops at 32 rows).

## UPDATE Mon 12:2x: the Friday estimate CORRECTED to ~40-46 ms GPU at C5 (C6 + int8 ~36-42) [est]
- **The measured 32-row dense matmuls** (data/week98/check25/README.md, quiet 12:22): 19.30 ms per check on today's
  two-16-row route -> 13.20 (dn32) / 12.76 (dn32q). That is ~-6 ms, not the -8..-10 I had assumed.
- **The rest of the dense side does not shrink with that work:**
  - HC reads ~4.7 ms (one-pass, to build) or 7.5 (split);
  - GDN recurrence ~4.3 (sequential at 32 rows);
  - router ~1.7, attention ~1, glue ~3.
  - So the dense side is ~25-28 ms.
- **Experts:** 20.4 (1.43x) + 1c glue 1.8-3.4.
- **So the measured C5 check by Friday is ~40-46 ms GPU (Fit12's ladder steps 2-5: 40.5-45.3), not 33-38.**
- **Below that:** chunked GDN -2.8, fused 1c glue -1.8..-3.4, HC4 -1.2, row-dense expert tiles -1.5..-4
  (unproven), early death's dead-row drop -2.7..-5.6 (not mine).
- Sent to the orchestrator 12:2x.

## UPDATE Mon 12:1x (after the plan was sent)
- **The orchestrator 12:1x: plan ACCEPTED.**
  - Slots are approved in principle. Kamen lifted the day limits until he says otherwise. NewIdeas sets the times
    (early-death dev pass Tue night, capture Wed night, proofs Thu-Fri nights).
  - Lossy is allowed: C6 (lambda 1.8e-2), int8 dense activations and V4 are EvalSuite-gated levers. NewIdeas
    pre-registers the three gates this week; a gate PASS is the go.
  - **Friday's report carries two numbers:** the measured check at C5 bytes, and the measured (or bench) check at
    C6 + int8, each with its efficiency vs peak. Plus **the launch count** next to bytes and ms (Fable B's ask, backed by
    the orchestrator).
- **BigMoE 12:0x:**
  - The t1c set: ~/BigMoE/data/stack/lx45t1c1_env.txt + t1cx45t1c1_env.txt, screen data/r57/snap/t1c1 (frozen;
    EvalSuite pins it: copy, never edit). Env recipe: scripts/t1c_keytest.py run_env().
  - V1 / V3 are NOT built and not on its list this week: I emulate both for timing.
  - **The fused tiers kernel: yes in principle**, if it is opt-in and off by default, bitwise vs tiers_mx / km / 2-D
    nolsb / the counter on every slot (scripts/test_t1c_wiring.py, 3.1 M slots), never in frozen t1c1, with a new
    snapshot + key-test re-run after EvalSuite's gate.
  - **The lossy engine never runs > 16 rows through the one-call:** R57_ROWS defaults to 16, and > 16 rows go to the
    slot path. So no lossy 32-row engine route exists today. R57_ROWS=32 + 1c would trip the slot path's assert.
  - => engine measurement #1 needs my own 32-row lossy route in a COPY of the screen, or a standalone full-check
    bench. Mapping the G32T route now (read-only agent).
- **Fable B (the orchestrator 12:1x, ~/FableGod/fableb/IDEAS_1116_check25.md) agrees with the bound** (~35-40 by
  Friday). Folded in:
  - (1) Dispatch diet from built pieces (eprep, GDNB, att_single / fs4, the a|b + k|v folds, router + shared as one
    stream, HC-read glue): 4,022 -> ~2,500 launches, CPU ~17 ms. These come with the f32one wiring (Tue-Wed). Note
    that eprep is bypassed under 2-D nolsb (b43efc4), so the fused 1c kernel replaces it there.
  - (2) Row-dense expert tiles at 32 rows: ~2.9 rows per expert, so 16-row tiles are ~82 % padding. This is Thursday's
    expert item, moved up as its first experiment (NAX6-N on layer 20's real 32-row routing; PASS <= 0.85x).
  - (3) Dead-row drop and (4) misses -> 0: early death's and the loop's, not this bench's.
- **The 32-row dense kernel is a race of three forms** (all written 12:0x-12:1x; they run when the GPU is free):
  - qmm32n (scripts/week98/qmm32n.py): the native layout's register decode for both row tiles. Bitwise == two 16-row
    FAST calls by construction.
  - dn32 (scripts/week98/dn32.py): bf16 x uint4 from threadgroup memory, NAX6's operand form. No ALU decode.
  - dn32q: int8 x int4 from threadgroup memory (NAX-C's form; M27-C: 1.55-1.73x bf16's rate). It gives the C6 + int8
    number.
  - Bench: scripts/week98/qmm32n_bench.py, cold per-check weight sets, independent and chained, vs byte and matrix
    floors.

## 1. Bottom line

> **CORRECTED Mon 13:4x (marked here 17:5x): this section's expert bytes and bounds are WITHDRAWN.** They priced an
> expert at 2.135 MB (4-bit) / 1.661 MB (MSB-3); the records3 pieces the engine reads are 2.784 / 2.064 MB. Real bound
> at 100 % of every peak (scripts/week98/check25_bound_real.py -> data/week98/check25_bound_real.txt): **C5 27.7 bf16 /
> 26.8 int8; C6 + int8 25.2; C6 + V4 + int8 24.0** (experts C5 5.05 GB = 18.0 ms, C6 4.60 GB = 16.4 ms). The 23.9 /
> 23.0 / 21.7 / 20.6 below are the old numbers; do not quote them. See check25/README.md, CORRECTION.

- **At C5's bytes, 25 ms is out of reach for any kernel work on this Mac.** This is the physics, not the schedule
  (scripts/week98/check25_bound.py -> data/week98/check25_bound.txt).
  - **The check is layer-sequential.** In each layer the routed experts wait for the router, the router waits for the
    attention / GDN output, and the next layer waits for the experts. So the dense side and the experts cannot
    overlap.
  - **At 32 rows the dense side is compute-bound.** 3.73 B dense parameters x 32 rows = 119 G multiply-adds: 8.5 ms at
    the measured matrix-unit peak (14.0 T MAC/s bf16 [K]), above its 7.6 ms byte floor. 4-bit or 3-bit weights do not
    change that count.
  - **The experts are DRAM-bound:** 3.99 GB = 14.2 ms at 280 GB/s.
  - **With the cores (~1.2 ms) that is 23.9 ms at 100 % of every peak.** 25 ms needs 96 % of every peak, or 92 % with
    int8 activations on the dense side. The best single kernels today reach ~74-84 % of streaming. Whole checks reach
    53-60 % of the byte floor (exact-45 engine [K]); C5's projections reach 33-43 % [P], because the lossy levers cut
    bytes, not the compute or the fixed costs. Fit12's own "at best" is 29.8 ms.
- **What can reach 25 ms:** fewer bytes AND int8 dense math, then ~80-85 % whole-check efficiency.

  [WITHDRAWN 13:4x, old expert sizes: real C5 27.7 / 26.8, C6 + int8 25.2, C6 + V4 + int8 24.0; see the note above]

  | bytes (code, 32 rows) | dense math | bound at 100 % [P] | at 90 % | efficiency needed for 25 ms |
  |---|---|---:|---:|---:|
  | C5 = 1c-tiers + V1 + V3, 6.12 GB | bf16 | 23.9 | 26.6 | 96 % |
  | C5 | int8 | 23.0 | 25.6 | 92 % |
  | C6 = C5 at lambda 1.8e-2, 5.77 GB on code [CORRECTED 12:4x, Breakthrough: was the pooled 5.55] | int8 | 21.7 | 24.2 | 87 % (was 84) |
  | C6 + V4 (3-bit GDN / attention), 5.43 GB on code | int8 | 20.6 | 22.8 | 82 % (was 79) |

  - Every row past C5 is a quality call (EvalSuite's gate, then Kamen): lambda 1.8e-2, 3-bit GDN / attention, and int8
    activations.
  - The ~80-85 % whole-check efficiency is the megakernel / replayed-command-path class of work (dispatch-diet levers D
    / E, cf70494): weeks, on top of everything below.
- **What I commit to by Friday:** a MEASURED full check at C5's bytes on the engine.
  - Real weights, real routing, 1c-tiers; warm 32-row re-checks (misses 0, labelled as such); GPU, CPU and wall.
  - The C6 bytes as a timing variant.
  - **My estimate: ~33-38 ms GPU per check [est].** Today's > 16-row route is 62-66 ms [P].

## 2. Day by day

**Mon to 14:30 (desk / CPU; EvalSuite's 45 GB gate holds the GPU):**
- This plan, sent by 14:00.
- **qmm32n source.** A true 32-row native-layout 4-bit GEMM on the matrix units:
  - one weight read and one dequant per fragment, for all 32 rows;
  - split-K for the narrow shapes;
  - its accuracy harness (fp32 reference).
- **The f32one installer skeleton.** Opt-in; no live-file edits. ref2 stays as it is until a change is landed with
  BigMoE.

**Mon 14:30 -> evening (GPU share slots from NewIdeas, <= 4 GB each):**
- **qmm32n on every FAST dense shape:**
  - accuracy vs fp32;
  - speed on cold rotating weights (one32_bench's method), vs today's route (two 16-row qmm16n calls) and vs max(byte
    floor, matrix floor).
  - This doubles as the WEEK98 "32 vs 16 rows" microbench.
- **hc8r at 32 rows** (one HC8 read for all 32 rows).

**Mon night:** E14 as booked (00:30-03:00). Nothing else of mine.

**Tue:**
- **E14 read-out** to the orchestrator, EvalSuite (marker), BigMoE and Breakthrough.
- **f32one wired: every FAST op at 17-32 rows in one pass:**
  - qmm32n for the DNR / DENSE-N / SMALL-N weights;
  - hc8r32;
  - the GDN step at T 32;
  - attention and SVH at 32;
  - the router;
  - the FAST-NAX one-call at M 32 with b43efc4's 2-D nolsb. First check groups of more than 16 pairs per expert (the
    D8 bug class).
- **4-layer slice:** f32one vs today's two-16-row FAST route (logits and argmax).
- **12-layer slice timing** in a <= 10 min quiet slot (NewIdeas): F16x2 vs F32one, per class.
- **V1 + V3 as bench options.** Neither exists in the engine (R57_LMSL / R57_HC4 are not built; BigMoE owns them per
  LOSSY_CHECK_SPEC). For the bench:
  - V1 = a gathered 16K head;
  - V3 = HC mixers / inject / PLE at 4-bit at load + an HC4 read kernel.
  - Timing only; quality stays EvalSuite's.
- **The R(K) table to NewIdeas by Tuesday evening**, as committed, with E14's E2 numbers.

**Tue night (a ~40 GB night slot, ~40 min, quiet): full-check measurement #1 on the engine.**
- BigMoE's lossy-45 + 1c-tiers env; warm 32-row re-checks.
- Arms: today's route vs f32one.
- Per class (Breakthrough's pass K), GPU busy, CPU, wall.

**Wed:**
- **The 1c glue fused into one kernel after router_fused:** tiers + km + counter + the dup pick.
  - ~1,630 dispatches per check: -1.8..-3.4 ms GPU, -10.6 ms CPU [est].
  - The tiers equal BigMoE's tiers_mx on every slot.
  - With BigMoE (its _t1c).
- **Chunked GDN at T 32** (FAST: the WY / UT form on the matrix units).
  - Error vs mlx_lm's step; the commit's per-position state handled.
  - -2.5..-3 ms [est].
- **Wed night: measurement #2.**

**Thu:**
- **The expert one-call at M 32 under 1c** (the biggest excess, 1.43x its floor): profile and fix the largest piece.
  No promise.
- **int8 dense activations as a FAST option in qmm32n** (the dense side's compute bound 8.5 -> 4.9 ms). Its error is
  reported; adopting it is Kamen's call.
- **If the engine's CPU binds:** Python specialization of the 32-row path (lever B, part).
- **Thu night: measurement #3 = Friday's number**, at C5's bytes and at C6's (lambda 1.8e-2, timing only).

**Fri:** the report:
- the measured 32-row check (GPU, CPU, wall) at C5 and C6 bytes;
- per class vs its floor;
- the gap to 25 ms and what closes it.

## 3. What does not fit by Friday (plainly)

1. **25 ms at C5's bytes.** The bound is 23.9 ms at 100 % of every peak (section 1). No kernel plan reaches it.
2. **25 ms at C6 + int8 (87 %) or C6 + V4 + int8 (82 %)** [CORRECTED 12:4x: code bytes].
   - The bytes and the int8 math need EvalSuite's gates and Kamen's call.
   - The ~80-85 % whole-check efficiency needs megakernels or a replayed command path (3-6 weeks [est]).
3. **Experts at ~1.1x their floor.** I know no route: the exact NAX road past NAX6 is closed, and int8 (NAX-C) ran at
   NAX6's speed. Thursday is an attempt.
4. **The engine's CPU under 25 ms at 32 rows.**
   - Today it is 27.9 ms, plus 10.6 for the 1c glue.
   - By Friday ~28 with the glue fused.
   - Under 25 needs lever B in full (3-5 days) and lever A (weeks).
5. **Misses -> 0 in real decoding at 45 GB.** The bench is warm. A resident-only rule is a quality item, unpriced and
   not mine.
6. **Moved down:**
   - C7d (exact R24) is deferred unless NewIdeas wants it for R(K).
   - The command-buffer microbench stays lowest (Tuesday's slot).

## 4. What I need

- **GPU share slots by day, <= 4 GB each:** Mon from 14:30 (~2 x 20 min) and Tue-Thu (~3 x 20 min a day).
- **One <= 10 min quiet slot by day on Tuesday** for the 12-layer slice (~26 GB: one big job, NewIdeas' call), or at
  night.
- **Night slots Tue / Wed / Thu:** ~40 GB, ~40 min each, quiet (the full-engine 32-row checks).
- **From BigMoE:**
  - its 1c env for the engine arms;
  - V1 / V3 status;
  - an OK on the fused 1c glue kernel (its _t1c).

## 5. Lines (pre-registered)

- **qmm32n, GDN qkv|z per call:**
  - PASS <= 115 us (1.2x its 96 us matrix floor);
  - KILL > 146 us (today's exact c2a; two FAST 16-row calls are 227).
  - Accuracy first: the error vs fp32 must be within the 16-row FAST form's own error, else fix before timing.
- **f32one on the slice:** each row's logits within the 16-row FAST route's error vs the exact route at 16 rows, and the
  argmax agreeing on >= 99 % of rows.
- **Engine measurement #1:** f32one - today's route <= -10 ms per warm 32-row check (Fit12's step 1 prices -11).
- **Friday:** report the number. My prediction against 25 ms at C5's bytes is a MISS (section 1).
