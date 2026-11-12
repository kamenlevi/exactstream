> (a quote from the project owner, kept private)
# small tests, <= 5 GB under a share lock, time nothing)

## 0. CORRECTED 17:0x: sections 1 and 2 were wrong. The sparse lane cannot engage in this engine

**Why.** MTPLX's gate `_qsa_large_prefill_enabled(rows, total)` first requires `current_attention_phase() == "prefill"`.
- The phase is a ContextVar whose default is "unknown".
- The shipped loop never sets it. r57_screen_rel1.py imports only mtplx.cache_state, and stream.py and
  serve / chat / run.py never call it. The engine's only `attention_phase("prefill")` is BigMoE's research lmp4pd path
  in bigmoe/decode.py, which no shipped config uses.
- `MTPLX_QSA_PREFILL` is read only by that gate (qwen4_exp.py:2242); profiles.py lists the name.

**CPU check with the real gate function** (the clone's venv):
- as shipped, a 1,024-row chunk ending at 40,000 tokens -> **False**; inside `attention_phase("prefill")` -> True;
- at 10,000 tokens -> False either way;
- with the FAST env (the three MIN_CONTEXT variables = 4096): False at 10K and 40K as shipped, True only inside the
  phase.

**So:**
- (1) `exact` stays dense past 32K and the 1.1.0 claim stands. This was a false alarm; I read the AUTO default as the
  whole story.
- (2) `EXACTSTREAM_QSA` was inert (fast never engages; dense pins what already happens). It is REVERTED from
  release/next/serve.py (back to fa705c6; SHA256SUMS OK).
- The EvalSuite pre-registration is ON HOLD.
- prof4/prof_fast4.sh (FAST via env only) would test nothing: dropped.
- The dense band (MTPLX_QSA_DENSE_BAND_SDPA, prof_qsa2) is not phase-gated (env + shape only): still valid.

**To make the FAST lane real (design, untested).** Run the engine's prefill inside `attention_phase("prefill")`, e.g.
serve.py wrapping `G()["prefill"]` as convcache.install does. That would also give chat / lossy MTPLX's AUTO lane past
32K (MTPLX's receipts: +34.8 % at 98K), which this engine does not use today. First check what else in MTPLX reads the
phase.

**Still done, as the orchestrator asked (17:0x):**
- **The engine-level confirmation**, scripts/prefill15/x32k/x32k_exact.sh: exact-32, a 40,016-token prompt (cut from
  long_ctx's 64K papers prompt), auto vs MTPLX_QSA_PREFILL=0, tokens + LOGHASH, no timings. Expected: EQUAL.
  - Gates: NO-SWAP >= 46, a share lock with MEM 38 GB, no other big lock or engine, Kamen idle >= 600 s.
  - Head-tested (arm, skip, exit). ARMED 17:07, last start Fri 08:00.
- **The defensive pin, staged** (release/next/configs/exact-32.env and exact-45.env; not published):
  MTPLX_QSA_PREFILL=0 with a comment. A no-op in this engine; it keeps exact dense if a later engine sets the phase.
  read_config skips comment lines anywhere.

## 1. (WITHDRAWN, see 0) FINDING: `exact` past 32K tokens of context is probably not bitwise

**The setting.** MTPLX's sparse QSA prefill lane is selected by `MTPLX_QSA_PREFILL`. When the variable is unset it
resolves to AUTO, which is ON on M5 (mtplx/models/qwen4_exp.py `_qsa_prefill_enabled`: "Unset resolves AUTO: on where
the NAX flash kernel is supported").
- It engages for chunks whose earliest query has 32,768 tokens of history or more (`MTPLX_QSA_PREFILL_MIN_CONTEXT`,
  default 32768).
- No shipped config sets it. `grep MTPLX_QSA` over configs/*.env, run.py, serve.py and chat.py finds nothing.

**What follows.** A prompt longer than 32K tokens therefore takes the sparse lane in `exact` too.
- PrefillKernels measured that lane at max |d| 0.009-0.016 against the dense path.
- The package checked `exact` only up to 30K tokens (PF-D at 10K and 30K), and its shipped references are short prompts.

**Status.** Read from the code; not yet measured in the engine.
- The test: one prompt over 32K tokens (e.g. 40K) at 45 GB, `exact`, with and without `MTPLX_QSA_PREFILL=0`, comparing
  tokens and R57_LOGHASH. That needs a night slot.

**Fix options (the orchestrator / Kamen decide):**
- (a) Pin `MTPLX_QSA_PREFILL=0` in the exact configs: bitwise at any length, slower past 32K.
- (b) Say "exact up to 32K tokens" in LIMITS.md.

`EXACTSTREAM_QSA=dense` (below) is option (a) as a switch.

## 2. (REVERTED, see 0: inert in this engine) The FAST QSA lane, wired as an opt-in (release/next/serve.py; not in 1.1.0)

`EXACTSTREAM_QSA` takes three values:
- `auto` (the default): as shipped, MTPLX AUTO.
- `dense`: `MTPLX_QSA_PREFILL=0`, the dense path at any length.
- `fast`: the three MIN_CONTEXT variables = 4096, so the sparse lane runs from 4K. It is NOT exact, and it is refused with
  `--config exact`.

**Test on the stand-in engine** (port 8134, no GPU; the stand-in prints the four MTPLX variables it sees):
- fast -> MIN / FLASH / DIRECT = 4096, PREFILL unset;
- dense -> PREFILL = 0;
- auto -> none set;
- `--config exact` with fast -> exit 1, "EXACTSTREAM_QSA=fast is not exact; use it with --config chat or lossy";
- an unknown value -> exit 1.

The stand-in's 11 logs (written through the scratch tree's `out` link into the cache clone's out/) were removed after
the test. They were found by the stand-in's own "mock tool loop loaded" line.

### Pre-registration for EvalSuite's gate (FAST = `EXACTSTREAM_QSA=fast` vs `auto`)

- **Arms.** A = auto (as shipped), B = fast. Both run chat at 32 GB, greedy, same prompts, one process per arm.
- **Where it can differ.** Only prompts with 4K or more tokens of context. Under 4K both arms take the same path.
- **Task sets.**
  1. Primary: the model's known long-context benchmarks that EvalSuite can run offline, at 8K, 16K and 32K (e.g. RULER
     or needle tasks, or a LongBench subset), whichever EvalSuite has. Without those, the Claude Code agent set (32c2:
     12 tasks plus the probe; its prompts are 15-25K tokens).
  2. Control: one short-prompt set (e.g. the IFEval keys). Tokens must be identical in A and B; this checks the wiring.
- **Decision.** EvalSuite's same-scores rule as in the P2 gate. KLD and token agreement are information only. Adoption is
  for chat / lossy only and is Kamen's call (FAST track).
- **Pre-screen (Leap, a night with room).** prof4/prof_fast4.sh: 10K and 30K prompts, 32 tokens, agreement vs the exact
  PIPE=1 run, plus the time. It skipped on 11-12 with no slot.

## 3. PF-B one-build candidate: SEGALL by default + the 45 GB memory trim (scripts/prefill15/onebuild/)

`pflm.py` is the engine's pflm.py (copy: pflm_engine_1111.py) plus two changes; the diff is onebuild_vs_engine.diff,
105 lines.
- **SEGALL by default.** `R57_PFLM_SEGALL` defaults to 1: over a group of chunks, the router and the shared expert run per
  chunk slice too, as the shared expert's gate already did. This is Tue's fix (f432586: 20 / 20 and 26 / 26 exact at
  ~2 % of PF-B's speed). =0 gives the gate-only behaviour of before.
- **The trim, `R57_PFLM_QD=n`** (default 0 = unbounded, as before).
  - The idea: at most n chunks' attention parts are in flight (`mx.async_eval`), and the CPU reads the layer's experts in
    one slice per chunk between them, so the read still overlaps the GPU.
  - Why it should help: PF-B queues all of a layer's chunks before the MoE. At 32 GB its extra peak over PF-A grows with
    the context (+1.6 GB at 10K, +5.2 GB at 30K), which fits several chunks' score planes (~1.5-3 GB each at 30K) being
    allocated at once.
  - Exactness: the same operations in the same order with extra waits, so exact by construction.
  - Not measured.

**CPU logic test** (test_onebuild.py, results in test_onebuild.out: **46 / 46 PASS**). It drives the real
`_split_layers` with stand-in layers on tiny CPU tensors. Five chunks of 3, 3, 3, 3 and 1 tokens with TOK=7 make the
groups [0, 1] and [2, 3, 4], so the short final chunk sits inside a group. For QD 0-3 and SEGALL 0 / 1:
- the outputs are bitwise equal across QD;
- each expert is read exactly once per layer, before the MoE: 4 slices of 4 with QD, one read of 16 without;
- after each wait at most QD chunks are in flight;
- with SEGALL the router and the shared expert see 3, 3, 3, 3, 1 instead of 6, 7.

### Pre-registration: the 45 GB night test (Leap, a night with room)

- **Setup.** The candidate is loaded by the sitecustomize hook, as in prof/, so the pinned clone stays untouched. Chat at
  45 GB, prompts of 10K and 30K tokens, `--max-tokens 32`, `R57_LOGHASH=1`, one process per arm.
- **Arms.**
  - R0: as shipped (PF-D), the reference.
  - R1: `R57_PFLM=3 R57_PFD=0` (PF-B, SEGALL on).
  - R2: R1 + `R57_PFLM_QD=2`.
  - R3: R1 + `R57_PFLM_QD=1 R57_PFLM_CC=1`.
- **Measure.** Tokens and LOGHASH vs R0; the lifetime peak phys_footprint; the first token and prefill tok/s (quiet lock).
- **Gate.** An arm passes if it is bitwise equal to R0 on both prompts and its 30K lifetime peak is no more than R0's
  (48.6 GB per LIMITS) plus the margin the orchestrator sets. The fastest passing arm is the one-build candidate.
- **Room needed.** About 45 GB plus up to ~9 GB plus the OS. Run it only with NO-SWAP of 56 or more.
- **Next step toward one build (not built).** In PF-B's read, rebuild the pool-resident experts on the GPU (PF-D's
  `rebuild_parts`) instead of reading them from the SSD.
