# PREREG TEMPLATE: certifying a cache-state-dependent FAST mode (QOM first; EvalSuite, Tue 2026-11-03 23:22)

A template, not a pre-registration. A mode's PREREG copies it, fills every `<...>`, and is committed before any row.

**Source:** Kamen's decision 2, YES (Tue 23:1x, via the orchestrator; NewIdeas research/PUSH_SHORTLIST.md), verbatim:
"yes, for fast mode im bsolutely fine with all of this". A FAST mode may depend on the cache state (QOM, MISS-CUT).
EXACT stays the default and history-independent.

**The orchestrator's conditions (23:19):**
- Run the same-scores test under at least two prompt orders (histories), and require a PASS in both.
- Report the run-to-run spread: tokens that differ between the two orders, and the score delta.
- The same history must reproduce the same tokens (one repeat check).

## Why the usual design does not carry over
1. **The tokens belong to one engine configuration.** Which experts a row keeps depends on what is resident, and that
   depends on the engine's pools.
   - EXACT modes and batch-invariant FAST modes (F1) give the same tokens in any harness, so EvalSuite certifies them
     in its own AR harness. A cache-dependent mode's tokens do not transfer.
   - So the test runs in the shipping engine's cache configuration: BigMoE's engine path at the shipped C, with its
     per-layer pool sizes, eviction policy, prefetch, warm-up and check structure (t 0.3 cut, 8-row chat checks).
   - Another C or pool-size set is another mode. For example, the by-day C 160 line drops more experts (NewIdeas: it
     needs its own map arm, QOM_C=164).
2. **A history is everything the pools saw since the process started:** the earlier prompts and the row's own earlier
   checks. A new process resets it.
   - So a history runs in ONE process, from the engine's own cold start. EvalSuite's time-based chunks would cut
     histories at random points.
   - If a run must be split, the split points are row indices fixed before any row, and the repeat uses the same ones.
3. **Timing must not change tokens.** The keep / drop decision must read residency as a deterministic function of the
   history, e.g. the pools after the check's committed loads. It must never read in-flight I/O or prefetch completion.
   - The repeat check tests this.
   - The logical-cache variant (PSR, arXiv 2610.34721; MoonShots' screen) makes it hold by construction: residency
     changes only through routing. It removes timing, not the dependence on the session's own history, so the
     two-history rule still applies to it.

## Arms (fill in)
- **Mode:**
  - name, flags, theta, C and pool sizes: `<...>`;
  - engine commit, frozen as `snap/bigmoe-<sha7>` with its ext .so sha256, and the Leap commit if its kernels are used:
    `<...>`.
- **Reference:** REF2-T's stored rows (`results/ref2t`: the 210 core; `results/ref2t.r2`: IFX's 220). For sets without
  stored REF2-T rows (agent, RULER), the engine with the mode's flags off, in H1's order. EXACT is history-independent,
  so one order is enough for it.
- **Rows:** at least the core 210 (math 50, IFEval 100, HumanEval+ 60), plus whatever the mode's PREREG adds (agent 29 /
  RULER 24 for long-context modes): `<...>`.
- **Histories:**
  - **H1:** the canonical order. Sets in a block order (`<e.g. HumanEval+, math, IFEval>`), keys sorted within each
    set, one process.
  - **H2:** `random.Random(<seed>).shuffle` of H1's keys, so the sets interleave; one process.
    - The order file is written before any row (`results/<run>_H2/order.txt`) and its sha256 goes into the PREREG.
    - Every row of H2 must have a different predecessor than in H1; if one doesn't, take the next seed.
  - **R, the repeat:** H1 again in a fresh process, for its first `<N>` rows (default 50, ~25 % of the core).
- **The same initial state for every process:** a fresh process, the same C, the engine's own warm-up as shipped, and
  nothing else on the GPU that could evict.

## Step 0 (VOID gates)
1. **Flags off == EXACT:** the engine with the mode off reproduces REF2-T's stored tokens on the step-0 rows (ifeval/3409,
   ifeval/2439, humaneval/HumanEval/27).
2. **The mode engaged:** per-row counters show sure rows, dropped rank 2-9 experts and misses; the drops are > 0 in both
   histories.
3. **The repeat (the orchestrator's condition):** R's `<N>` rows are token-identical to H1's first `<N>`. One differing
   row is VOID. It means the tokens depend on something other than the history, so the mode is not certifiable as built.

## Lines
- **PASS:** H1 passes AND H2 passes every same-scores line in the mode's PREREG against the reference:
  - decision 4 per set (paired delta > -5 points AND exact McNemar p >= 0.05, on unrounded numbers);
  - RULER (<= 1 net item lost) where it runs;
  - mission 7's bars for live-loop modes with a replay;
  - and step 0 holds.
- **FAIL:** a line fails in either history.
- **VOID:** a step 0 gate fails.
- **Stage rules apply per history.** If the PREREG has F1's IFEval stage 2, IFX's 220 run in both histories (H2's order
  of the 220 by the same seed rule), and IFEval's line is decision 4 on the 220 in each.

## Read-out (required: the run-to-run spread)
- **H1 vs H2, per set:**
  - rows token-identical between the two orders;
  - the first-divergence position of each differing row (median, quartiles);
  - token-sequence similarity (decode_report's measure);
  - tokens that differ: the differing rows' tokens after their first divergence, pooled.
- **Score delta H1 - H2 per set,** with the exact McNemar between the two orders (information), next to each order's
  delta vs the reference.
- **The floor beside it:** the same spread for two same-weights exact runs that differ only in summation order (e.g.
  REF2-T vs REF2b, ref.r2 vs floor; tables in RESULT_ref2ct). This shows whether the history moves answers more than
  numerics noise does.
- **Per history:**
  - share of sure rows' rank 2-9 experts dropped;
  - misses and SSD MB per kept token;
  - tok/s (engine numbers, rough unless the PREREG takes a quiet lock).
- **The repeat:** rows identical out of `<N>` (must be all).

## Cost
- About 2.25x a one-history test: H1 + H2 + a 25 % repeat. The flags-off control is extra only for sets without
  stored REF2-T rows.
- QOM on the core 210 in BigMoE's engine at C 256: `<h per history>`, priced once the engine path exists.

## What it cannot say
- Two histories sample the history space; they do not bound it. A long agent session or a mixed-language day can
  differ more. The H1 vs H2 spread is the evidence for how much history matters on these rows.
- The certificate holds for the stated engine commit, C and pool sizes only.

## Engine requirements (BigMoE, before a mode's PREREG is filled)
- **A decode entry EvalSuite can drive:**
  - a fixed key order in one process (no time-based chunks);
  - rows in the suite's decode format (key, set, completion_ids, text, finish);
  - per-row counters (sure rows, dropped rank 2-9 experts, misses, SSD bytes).
- **Deterministic residency for the keep decision** (Why, point 3).
- **The mode behind a flag,** off by default, with flags off == EXACT.
- **Run at the shipped C and pool sizes.**
