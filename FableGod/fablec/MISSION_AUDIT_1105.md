# AUDIT, Thu 5 Nov 07:00: check the three new sessions' answers before anything is built

> (a quote from the project owner, kept private)

## Inputs
Three fresh Fable sessions answered overnight, each with a different amount of context:
- ~/FableGod/zero/ZERO.md (ZERO: the raw problem only, no project knowledge);
- ~/FableGod/goal/GOAL.md (GOAL: the goals + today's measured facts, no methods);
- ~/FableGod/full/FULL.md (FULL: everything).
Their briefs are the MISSION_*.md files beside each answer.

## For every idea (each session gives up to 3 + its bet)
1. **Numbers:** re-derive its arithmetic against our measured data.
   - Sources: ~/FableGod/KERNEL90.md, ~/NewIdeas/research/SWEEP.md, ~/TrainLab/notes/DRAFTER_PLAN_1104.md,
     ~/FableGod/RELAX_350.md, BigMoE's data, today's ~/FableGod/DAY9_2026-11-02.md.
   - Flag every number that is wrong or unsupported.
2. **Already tried?** Search ~/TRIED/TRIED.md and NOVELTY.md. Mark it tried-and-killed (with why), tried-and-alive, or
   untried.
3. **Prior art:** the closest published work you know, or "none found". MoonShots does the full web check afterwards.
4. **Value:** expected gain on our lines (tok/s at 45 GB code FAST / EXACT; MB per kept token at 24 GB), halved, times
   the odds.

## Deliver (by ~09:00; one file: ~/FableGod/fablec/AUDIT_1105.md)
- A ranked table of every idea: gain (halved), odds, cost, tried / untried, the number errors you found.
- Your top 1-3 picks for a <= 10-min test each, with the PASS / KILL line written now.
- One paragraph comparing the three context levels: did knowing less or more help, and how?
- Then one message to the orchestrator: your top pick in one line.

## Rules
Thinking and writing only: no GPU, no downloads, write only in ~/FableGod/fablec/. Marks [K] / [P] / [G].

## Added Wed 21:0x (orchestrator)
- If they have landed, also read MoonShots' two CPU screens from tonight (verdicts in ~/TRIED/TRIED.md):
  - FAR-AHEAD FORESIGHT: a drafter 8-16 checks ahead feeding Belady-like retention;
  - FULL's section 5: the comment-span relax test.
- Prior art the orchestrator found (standard web search; abstracts and summaries only):
  - Foresight:
    - MoE-SpeQ (arXiv 2511.14102): a quantized draft predicts future experts; a lookahead buffer plus lookahead-aware eviction.
    - SP-MoE (2510.10302): draft-based prefetch for the verify pass; deeper prefetch thrashed.
    - SPICE (2608.21240): confidence-aware adaptive lookahead.
    - Only an 8-16-check horizon may be new.
  - FULL's bet:
    - No paper found that relaxes acceptance only in code comments.
    - Lossy verification can inflate output length: arXiv 2610.23818 ("acceptance is not speed"; reported 1.34x accepted tokens per round but 1.54x longer completions at the most relaxed settings) and 2607.26627.
    - Check whether FULL's tok/s math survives longer comments. Require tokens per task and seconds per task in its engine test, not only tok/s.
- Added 22:1x: FAR's section 7 (~/FableGod/far/FAR.md, far_wait.txt) measured the SSD wait law on 106 logged runs: a ~12 ms floor + ~0.155 ms per miss (~14 GB/s, R2 >= 0.95). Check GOAL.md's "the SSD wait is a latency problem, not a bandwidth problem" against it, and re-price GOAL's lever 1 with both terms.

## Added Thu 06:5x (orchestrator): three more answers arrived overnight; audit them in the same table
- `~/FableGod/full/FULL.md` "## Round 2": the re-registered comment rule (Rule C) and R1. Rule C was since measured by MoonShots (M-41: MBPP x1.125 PASS, HumanEval+ x1.074 BETWEEN, 0 code deviations); R1 was KILLed by M-40's census.
- `~/FableGod/far/FAR.md`: Kamen's far-ahead foresight idea, the near / far hybrid, the SSD wait law. The novelty check is MoonShots' research/FAR_NOVELTY_CHECK.md.
- `~/FableGod/cloud3/live/cloud3/REPORT_P0.md` ... `REPORT_P3.md` + `REQUESTS.md`: Cloud Fable 3, on fewer distinct experts per kept token.
  - Its best candidate, c3_b2d_reach07, is PREDICTED at x0.41 / 0.52 / 0.58 of today's tiles inside every quality line. It goes to the decision map Friday night, so audit it FIRST and hardest:
    - the quality pricing (floor_rows join, the "context factor");
    - the strict vs resident-keep forms;
    - the claim "same bytes".
  - Also audit its chat-bytes lever (-36 % at 24 GB) and its 24 request specs.
- Priority: CLOUD3's candidates, then FAR, then ZERO / GOAL / FULL.
