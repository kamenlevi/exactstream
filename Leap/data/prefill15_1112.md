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

## 4. x32k, the engine-level check: exact stays BITWISE past 32K (Thu 19:40-19:52; data/prefill15_x32k/)

exact-32 on a 40,028-token prompt (prompt_40k.txt + the chat template), --max-tokens 32, R57_LOGHASH=1, cache clone, one
process per arm: auto (as shipped) vs MTPLX_QSA_PREFILL=0 (the lane pinned off). Ran in the 1.2.0 night chain after r120,
NO-SWAP 49 / 46 GB.
- **The 32 tokens are EQUAL and the 10 verify-logit hashes are EQUAL.** The sparse QSA lane does not engage in this engine,
  as the CPU gate check said (section 0). So the 1.1.0 / 1.2.0 exact claim holds past 32K, and the pin shipped in 1.2.0 is
  defensive only.
- Each arm: first token 337.8 / 334.4 s (a cold 40K prefill at 32 GB with the default prefill, about 120 tok/s); lifetime
  peak 36.22 GB in both.
- **Correction to the runner's own verdict line.** It printed "tokens DIFFER" because it compared whole tokens.json files,
  and a prompt-file run's item key carries the run's timestamp (multi/cli_1112_194020 vs multi/cli_1112_194614).
  x32k_exact.sh now compares the token lists only. Rerun on the same files: "tokens EQUAL | verify-logit hashes EQUAL".

## 5. Night chain, Fri 13 Nov 00:4x: the prof runners re-armed (start gate) and token-list compares in four runners

- **Why.** After the 00:25 app close, NO-SWAP sat at 44.9-45.0 GB: the orchestrator's 30-s samples read 45 from 00:28 to
  00:36, and vm_stat read 44.94 GB at 00:38. prof_pipe2 (v2) had been waiting on "NO-SWAP 45 < 46" since 00:26. The
  orchestrator (00:3x) offered the call: start at about 45, which is inside the per-run minimum of 44, or keep waiting.
- **Re-armed 00:42-00:44.**
  - All five runners were stopped in reverse order (onebuild's waiter, the 45 GB stage's waiter, pipe3, qsa2, pipe2), so
    no done file released a waiter. The three done files their exits touched were removed before re-arming.
  - prof_pipe2.sh v3b: start gate 44, the per-run minimum (the integer gate truncates 44.90 to 44).
    - Its first run is 10K PIPE=0, with a peak of 35.58 GB on 11-11, so about 9 GB of room.
    - A 30K PIPE=1 run is about 40.7 + 1.6 GB (the second full-E scratch). It comes after two runs; on 11-11 runs like
      these let the OS compress idle pages (NO-SWAP 46 -> 48 -> 51).
    - Swap +2 GB still stops a run.
  - prof_qsa2.sh and prof_pipe3.sh v3: start gate 45 (they follow pipe2's four runs).
  - Chain: pipe2 (pid 80467) -> qsa2 79087 (waits for pipe2.done) -> pipe3 79178 (qsa2.done) -> the 45 GB clean-clone
    stage (after.sh 79655 on 79178; NO-SWAP 52) -> onebuild (after.sh 79788 on 79655; NO-SWAP 56). Last start Fri
    08:00 for each.
- **Compare fix, before any of them ran.**
  - The verdict blocks of prof_pipe2, prof_qsa2, prof_pipe3 and onebuild_45 compared whole tokens.json files with
    json.dumps(tk, sort_keys=True).
  - A prompt-file run's item key carries its timestamp (multi/cli_<date>_<time>), so they would have printed DIFFER on
    equal tokens: the x32k false alarm (section 4).
  - All four now compare the token lists only, as x32k_exact.sh does.

## 6. prof_pipe2 result (Fri 00:44:57-00:48:50; data/prefill15/pipe2_1113/): PIPE=1 is bitwise EQUAL but not faster -> KILL

PF-B (R57_PFLM=3) at 32 GB in the cache clone, prof15's 10K and 30K prompts, --max-tokens 32, R57_LOGHASH=1, through the
instrumented pflm copy (timestamps only, SYNC=0). One process per run, in the order 10K p0, 10K p1, 30K p1, 30K p0. Share
lock, nothing else running, fans at 5000; timings are information only. NO-SWAP was 44 / 44 / 47 / 50 GB before each run
and 50 at exit; swap stayed at 4594 MB.

| run | first token | whole prefill wall / main-thread CPU | lifetime peak |
|---|---|---|---|
| 10K PIPE=0 | 18.34 s | 18.33 / 6.03 s | 35.67 GB |
| 10K PIPE=1 | 18.91 s | 18.91 / 6.85 s | 37.14 GB |
| 30K PIPE=1 | 58.85 s | 58.84 / 24.44 s | 42.16 GB |
| 30K PIPE=0 | 57.43 s | 57.42 / 19.15 s | 40.73 GB |

- **Bitwise: EQUAL on both prompts** (tokens, and all 11 verify-logit hashes each), as built.
- **Not faster, so KILL.**
  - At 10K PIPE=1 is 0.57 s slower (+3.1 %), even though it ran second, the slot that favours an arm.
  - At 30K it is 1.42 s slower (+2.5 %); there PIPE=0 ran second.
  - The peak is +1.47 / +1.43 GB (the second full-E scratch), and the main thread does more CPU work (+0.8 / +5.3 s).
- **Why the pipelining cannot help.** prev_wait is about 0, so the GPU never makes the main thread wait for the
  previous layer. The expert reads (about 5.2-5.4 s of main-thread wall per prefill at either length) were not the
  limit.
- **Where the 30K prefill's host time goes (PIPE=0; stage_sums.txt):**
  - The 12 full-attention layers' hc_write step (mlp_hyper_connection + the chunk's async_eval) takes 15.23 s wall,
    12.88 s of it main-thread CPU: about 22 % of the 57.4 s prefill.
  - Their self_attn graph build ("qsa") takes only 0.11 s.
  - It grows about 8x from 10K (1.62 s CPU) to 30K (12.88 s) for 3x the tokens: quadratic in the context. That is about
    2.5 ns of CPU per (chunk row x context) element, consistent with a dense chunk-by-context mask or a copy the size of
    the cache made on the CPU per chunk. This is a reading, not yet checked in the code.
  - The 36 linear layers' hc_write is 15.37 s wall but only 2.38 s CPU, i.e. mostly waiting on the GPU.
  - moe_build: 11.83 s wall, 1.01 s CPU. ple_stage: 6.99 / 0.47 s.
- **Next.**
  - qsa2 (dense-band SDPA, next in the chain) shows whether the band path removes this CPU cost.
  - Its cProfile wraps only the lazy self_attn build (0.11 s here). It cannot see the async_eval where the 12.9 s is,
    so a small cProfile is expected and does not clear the attention path.
  - The follow-up is to find which op of MTPLX's full-attention path (the stock dense-mask fallback) or the cache
    update runs on the CPU per chunk.

## 7. prof_qsa2 and prof_pipe3 (Fri 00:50-00:55; data/prefill15/pipe3_qsa2_1113/)

Same setup as section 6, chained right after it. Timings are information only: these runs came later in the night, in
slots that can favour an arm.

**prof_qsa2: 30K PIPE=1 with MTPLX's dense-band SDPA (MTPLX_QSA_DENSE_BAND_SDPA=1) + a cProfile of the QSA self_attn builds.**
- **Bitwise:** EQUAL to the stock 30K PIPE=1 run (tokens and verify-logit hashes).
- **Time:** first token 56.55 s against 58.85 s (-3.9 %). The gain is spread over every stage (attn_submit -1.05 s,
  moe_submit -0.64, ple_stage -0.57), which looks like the later slot, not the band.
- **The band does not remove the full-attention CPU cost:** the 12 full-attention layers' attn_submit is 15.48 / 12.88 s
  against 15.95 / 13.20.
- **cProfile:** all 348 self_attn builds (12 layers x 29 chunks) take 0.209 s, so the build is cheap, as section 6
  expected. The cost is in the eval.

**prof_pipe3: PIPE=2 (layer L+1's expert reads in a background thread through bigmoe_ext_ov, GIL released).**

| run | first token | whole prefill wall / CPU | lifetime peak | vs PIPE=0 (section 6) |
|---|---|---|---|---|
| 10K PIPE=2 | 14.77 s | 14.76 / 7.30 s | 37.20 GB | 18.34 s: -3.57 s (-19.5 %) |
| 30K PIPE=2 | 52.58 s | 52.57 / 23.42 s | 42.26 GB | 57.43 s: -4.85 s (-8.4 %) |

- **Bitwise:** EQUAL to PIPE=0 on both prompts (tokens and verify-logit hashes).
- **Why it can be real:**
  - The main thread's read stage goes to 0, and read_wait is 0, so the background reads always finish in time.
  - The saving is about the main-thread read time under PIPE=0 (5.37 s at 10K, 5.31 s at 30K).
  - PIPE=1 pipelined the same reads but kept them on the main thread, which is the bottleneck, so it gained nothing.
- **Cost:** peak +1.53 GB (the second full-E scratch).
- **Not order-balanced:** the PIPE=2 runs came after PIPE=0's and qsa2's.
- **Next:** prof_ab.sh (Fri 00:58, before the 45 GB stage) runs 10K p0 p2 p2 p0 and 30K p2 p0 p0 p2 under /usr/bin/time
  -l, plus one 30K PIPE=0 run with BIGMOE_MLX_CACHE_MB=4096, to test the zero-fill hypothesis below.

**Hypothesis for the full-attention CPU (sections 6-7), to be tested by prof_ab:**
- Head dim 256 is outside MLX's fused attention kernel. MTPLX's _verify_sdpa note says the unfused path "materializes
  the [n_q_heads, q_len, T] fp32 score plane (O(T))".
- The engine caps MLX's buffer cache at 1 GB (stream.py install: BIGMOE_MLX_CACHE_MB, default 1024) and calls
  mx.clear_cache() often.
- So each chunk of a full-attention layer may get fresh, kernel-zeroed pages for its O(T) planes on the main thread:
  about 2.5 ns per (row x context) element, and quadratic over a prompt.
- It predicts process sys CPU of roughly that size (about 13 s at 30K) that grows about 8x from 10K. A bigger cache
  helps only if the planes survive the clear_cache calls.
