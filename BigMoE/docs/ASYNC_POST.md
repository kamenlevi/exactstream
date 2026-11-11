# Async post: scoping (BigMoE, Fri 6 Nov 21:4x, CPU only under the HOLD; Saturday's S0)

Asked by the orchestrator and Leap: can the per-layer post become fully asynchronous, so the GPU never idles while Python
posts and builds the next layer? Below are the code read, what is still unknown, the measurement that settles it, the fix
options, the hours, and the A/B line (written before any build).

## 1. What the code does today

**The gated path is already asynchronous on the host.** This covers decode and verify calls with at most MAXK = 160 ids,
i.e. every 16-row check.
- `bigmoe/gated.py` `post` / `route_and_gate` call `ext.post_ids` (a GPU kernel writes the routing row) and `ext.spin_wait`
  (a GPU kernel spins on the release region). Both are lazy MLX ops; Python does not wait on either.
- `ensure()` does not run on the Python thread. It runs on the ext's own loader thread (`loader_main`,
  `ext/bigmoe_ext.cpp:1652`), a P-core thread at QOS_CLASS_USER_INTERACTIVE. That thread polls the route rows the GPU
  writes, serves each (seq, L) row (reads misses, prefetch scheduling), and writes the release the expert kernel spins on.
- So Leap's proposal ("Python enqueues post_ids and never waits; the ext's thread picks up the ids when the router lands,
  runs ensure and writes the release; the kernel spins on the release") is today's design for gated calls.

**Paths that do block Python per call:**
- ref2 `_sync_slots` / `_tier_slots` (`~/Leap/scripts/ref2.py:1453-1506`) and BigMoE `stream._sync_call`. Each does
  `np.array(indices)` (waits for the router), calls `ext.ensure` from Python (synchronous SSD reads), then `mx.eval(y)`.
- These are taken when a call does not fit the gate: more than MAXK ids (more than 16 rows), MTP head calls, or calls
  wider than the pool.
- MTPLX's PLE stage needs the token ids on the CPU at layer 1 (`bigmoe/decode.py` header). It gathers n-gram rows from
  a 29.8 GiB file-backed table (hot cache 1 GB).
- `BIGMOE_ASYNC_EVERY` is 0 by default, so the whole check's graph is built before anything is handed to the GPU.

**The unexplained part.**
- The Fri day run splits the 68.8 ms check (medians, checks >= 2, 386 checks) into 59.6 ms from the check's start to
  "graph built" (bld), and only 7.5 ms of eval wait after it.
- Leap measures ~10-12 ms of Python per check, ~44.5 ms inside the forward, and 25 ms GPU-bound for the dense pass alone.
- So Python is blocked for roughly 45-50 ms of every check inside `model(...)`. That cannot be the gate on a 16-row check
  (section above), and the read did not find the blocker. Three candidates:
  - (a) a sync-path call inside each check: the commit / replay, an MTP call, or a call over 160 ids;
  - (b) a host read in the forward: MTPLX attention (head-chunked verify SDPA), GDN, PLE / n-gram;
  - (c) Metal command-queue back-pressure: Python blocked in a commit while the queue is full of spinning kernels.

## 2. Measure first (Saturday, after the HOLD lifts; ~10 min GPU, share lock)

1. **Leap's C 224 GPU-busy run** (after tonight's ticket): GPU busy versus wall clock per check, i.e. the idle that is
   available to win. S0 proceeds only if the idle is >= ~8 ms per check (NewIdeas' bar).
2. **BigMoE's locator**: one 2-prompt, CAP-32 run on the stack env with two probes.
   - A per-layer timestamp in `stream._streamed_class.__call__`, taken as each layer's MoE call is entered on the Python
     thread.
   - `cProfile` over the check's `model(...)` call.
   - Together they give which function Python waits in (`np.array`, `mx.eval`, `.item()`, a Metal commit, the PLE gather),
     how often per check, and at which layers.
   - About 1 h, including the 10 GPU minutes.

## 3. Fix options, by what section 2 finds

| finding | fix | exactness | hours (build + bit checks) |
|---|---|---|---|
| (a) sync-path calls per check | route them through the gate: split a > 160-id call into <= 160-id halves on the spin path, or a MAXK-320 gate build | same kernels, same slots (REF2's expert kernel computes every (row, expert) pair on its own, as the gated path does) | 2-4 |
| (b) a host read in the forward | make it lazy, e.g. gather the PLE / n-gram rows for the check's known tokens (tok + drafts, already on the host) before `model(...)`, and pass them in | same values, different time | 3-6 |
| (c) queue back-pressure | fewer, larger command buffers (one `async_eval` per K layers, BIGMOE_ASYNC_EVERY), tuned by the same probe | none: scheduling only | 2-4 |

A per-layer completion callback in the ext is not needed: the loader thread is already that callback for gated calls.

**Bit checks:**
- Every option keeps the math. The check is token identity with today's engine on the **exact** config (NAX6-N exact
  env), 2 prompts x 2 histories, warm and cold.
- FAST (QOM-KEEP) outputs depend on pool residency and timing (EvalSuite Fri: drafts carry history). So on FAST, token
  equality is not a valid exactness test. The FAST arm reports kept per check and tokens per task beside its speed.

## 4. Hours

- Locate: ~1 h, with 10 GPU min.
- Fix: 2-6 h by option.
- Bit checks: ~1 h, with ~5 GPU min.
- A/B: 15 min quiet.
- If the idle is real and has one cause: about half a day, Saturday morning (Leap + BigMoE).

## 5. The A/B line (pre-registered now, before any build)

**Design:**
- 12 HumanEval, palindrome stack / fix / fix / stack, C 224, quiet lock, freeze windows dropped.
- fix = the stack + the chosen fix.

**PASS (all three):**
- median check time >= 5 ms shorter;
- net pair-mean geo steady tok/s >= +5 % vs the stack;
- the exact-config bit check identical.

**KILL:** the bit check differs, or tok/s < +2 %.

**BETWEEN:** otherwise. Kept per check and tokens per task are reported beside.

## 6. Result (Fri 23:5x, Leap's measurement; this replaces the section 1 guess)

- **Python is building, not waiting.** Measured at C 224 on the stack env (real 16-row loop, 3 passes x 11 steady checks): a
  warm 16-row forward spends 40.2 ms in Python's graph build vs 15.8 ms for the dense part alone, i.e. ~0.5 ms of Python per
  layer in the expert call on the gated path, with no misses. The "~45-50 ms blocked" reading in section 1 was wrong.
- **The GPU idle comes from the mid-loop submit cadence.** Today's SB submits every 4 layers on 2-16-row checks
  (`Q._prefill_midloop_eval_layers` -> 4) and leaves 8.1-11.1 ms of GPU idle per cycle (median 8.4).
- **SB 2 recovers most of it.** Submitting every 2 layers cuts the idle to 2.5-3.6 ms and the cycle by ~2.6 ms
  (71.1 / 70.5 / 72.2 -> 68.5 / 67.9 / 68.6 ms), with kept per check identical in all 4 real passes. At 2 rows: -5.71 ms per check.
- **So S0's rebuild is not needed.** The idle is reached by a scheduling knob, R57_SBN (Leap's S2 diff, applied to
  scripts/r57_screen_fast.py; default 4 = today, bitwise). The locator (scripts/s0_locate.sh) stays available.
- **The A/B moves to S2** (Leap's glue A/B, its own line). On ~2.6 ms of a ~70 ms cycle (~+3.7 %), section 5's PASS
  (>= +5 %) would most likely not be met.
- **Where the Python thread's time goes** (Leap's XPT, Sat 00:3x: per-call wall and thread-CPU on the real 16-row loop, SB 2,
  3 steady passes; ~/Leap/data/stepb/gap_xpt.txt). The thread uses ~40 ms of CPU per ~68 ms cycle:
  - graph build across the 48 layer calls: 15-16 ms, of which the expert call (route / post / one-call) is only 3.75-3.97 ms
    (0.08 ms per layer);
  - mx.async_eval: 35-37.5 ms wall, of which 18-19 ms is CPU encoding and ~16-19 ms is waiting inside async_eval for the GPU
    (back-pressure; the GPU is ~96 % busy at SB 2).
  The thread never waits on the router. GPU time is the lever (S1); the Python floor (encoding + build) is S3's.
