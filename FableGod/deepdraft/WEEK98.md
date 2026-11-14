# WEEK98: the 98 % deep drafter, Sat 14 Nov -> Sat 21 Nov 2026

> (a quote from the project owner, kept private)
itself or the method for it being certified that it will work in a week, because I don't have more time than that to
spare to show this part of the project to my dad. I mean, so yeah, basically we must be able to find the thing which
unlocks 98% accuracy at 30 tokens deep. By then we must know what it is, and either I don't care if in a week we just
know the principles of how everything will work and it's certain, and we still have a week of building everything. I
don't care about that. I just need to get that."

## The goal
- By Sat 21 Nov, know the method that gets the drafter to >= 98 % right per guess at every depth up to 30. Held-out test
  numbers must back it; a small-scale proof counts. The full build can take the week after.
- The 125B stays untouched. Only the drafter learns.
- Measure first on held-out HumanEval+ windows (the 23-30 band), then on MBPP and everyday text.

## Where we start (t30b, per guess)
| | depth 0 | depths 23-30 |
|---|---|---|
| HumanEval+ | 97.90 | 89.66 |
| MBPP | 94.01 | 72.13 |

- Data: about +1 pt per doubling (t30a -> t30b). 98 % this way needs over 100x the data, so it isn't the path.
- Capacity: t30x2 NO EFFECT (-0.79 pt). t30x4 and TAPSQ (multi-layer taps) land Sat ~15:10 / ~15:25.
- 98 % at depth 30 means the deep guesses lose nothing against the first guess, and the first guess gains a little too.

## Why it matters (Kamen, Sat 14 Nov 15:2x, in short)
- Be the first: one proven recipe that keeps any drafter ~98 % right 30 tokens deep, usable on every model, exact
  output, ideally "a little engine package". It unlocks huge local models: the 125B in 30-45 GB at ~1,000 tok/s on code,
  and much bigger models streamed from SSD on a big Mac.
- 98 % per guess for 30 deep = about 23 tokens per check of the big model (sum of 0.98^k for k = 1..30, plus the bonus
  token), against 13.83 for t30b today. The other half of 1,000 tok/s is a fast check, about 23 ms per 32-row check.
- "No retraining": the big model is never retrained. Each model's drafter likely needs one short training, because
  drafterwall (11-11) showed training-free arrangements don't move kept by 20 %. Say so plainly.

## How we go fast, and how we show it's general
- Fast inner loop: try levers on a small local target model first, where a drafter trains in minutes, then confirm the
  winners on the 125B. Downloads follow the standing 11-07 rule (20 GB stays free for swap; log the name, source and
  size).
- Before calling it a method for any model, show it on a second model family as well as the 125B.

## Rules
- Every lever gets a screen of at most 15 min, with "result -> action" written first (pacer/RULES.md, 11-14).
- Long runs only confirm, with a stop line at 10-25 % of their steps against the baseline at the same steps.
- Results go to GitHub the same turn, with a one-line note to the orchestrator.
- A lever that changes the output (lossy) needs EvalSuite's same-scores gate before it counts, and Kamen's call to adopt.

## Day 1 first: the CEILING test (BigMoE)
- How often is the 125B's own choice a near-tie (the top-1 vs top-2 margin, in bins)?
- What share of t30b's misses at depth 0 and at depths 23-30 sit on near-ties?
- Result -> action:
  - If near-ties alone block exact 98 %, the method must include close-call acceptance (lossy, quality-gated).
  - If the misses are mostly NOT near-ties, they are learnable: focus on what the drafter sees and how fresh it is.

## Owners
- **BigMoE:** the ceiling test, then the GPU screens of depth levers (taps, fresher features).
- **MTP Head (TrainLab, from Sun 01:00):** training levers (D98 owner).
- **NewIdeas (from Sun 01:00):** the lever list with a short prior-art pass (published / untested / new), handing out
  15-min screens.
- **Breakthrough:** close-call acceptance (its delta-0.05 pricing), carried to the 98 % line.
- **MoonShots:** levers from outside the drafting literature.
- **EvalSuite:** the quality gate for any lossy lever.
- **Leap:** keeps 1.2.1 ready for Kamen's word. Engine help when a lever needs it, e.g. a cheap partial trunk pass on
  drafted rows.
- **Orchestrator:** the daily plan, verification, reports to Kamen.

## Day plan
- Sat 14: the t30x4 + TAPSQ verdicts; the ceiling test.
- Sun 15 - Wed 18: breadth, with 15-min screens of every depth lever.
- Thu 19 - Fri 20: the best lever's proof run (with a stop line) and the method write-up.
- Sat 21: the answer for Kamen: the method and its numbers, or exactly what blocks it, with the numbers.

## Log
- Sat 14 15:0x: goal set; charter written (orchestrator).
- Sat 14 15:08: CEILING part 1 (BigMoE e4f9cd3): the 125B's near-ties (m < 0.25 nat) are about 2 % of answer rows,
  < 0.1 nat 0.65-0.82 %; the median margin is 8-9 nats. Ties cost at most ~2 pt, so 98 % is not ruled out. Part 2
  (t30b's misses by margin) decides LEARNABLE vs MIXED.
- Sat 14 15:08: TAPSQ amendment 1 (b44d00f): tap lr 1e-4 -> 1e-5. A zero-init wide input at 10x lr spikes the loss
  (0.35 -> 3.78). The same mechanism likely caused t30x2's / t30x4's early spikes, so the capacity verdicts carry that
  caveat. Candidate: a 15-min capacity retest at 1e-5 against t30b's saved per-step KL.
- Sat 14 15:15: t30x4 verdict HURTS: HumanEval+ 23-30 88.14 % vs t30b 89.66 % (-1.51 pt); MBPP 70.01 vs 72.13; kept
  13.20 vs 13.83. Both capacity arms lose about 0.7 pt per doubling. A clean 15-min retest at branch lr 1e-5 is queued
  after TAPSQ and CEILING part 2.
- Sat 14 15:20: **CEILING = LEARNABLE** (BigMoE 62f5b9e, t30b on MBPP 100, every cell).
  - Tie-limited ceiling: 98.87 % at depth 0, 98.76 % at 23-30, so ties don't block exact 98 %.
  - 75.8 % of the 23-30 misses (21.12 of 27.87 pt) are on rows the 125B was sure of (m >= 1 nat). On very sure rows
    (m >= 4) the first guess misses 0.1 % and the deep guesses miss 16.1 %.
  - 7.9 % of the deep misses fall outside the 125B's top-32: the drafter loses the thread with depth.
  - The tie tax is real: depth 0 misses 54 % of near-ties, about 1.1 pt. Exact 98 % leaves about 0.9 pt for everything
    else.
  - Close-call acceptance adds +0.2 / +0.9 / +1.7 pt at depth (delta 0.1 / 0.25 / 0.5): an end bonus that buys back the
    tie tax, not the week's lever.
  - Caveat: MBPP only (the HumanEval+ windows carry no log-probs).
  - **The week goes to what the drafter sees and how fresh its view is at depth.**
  - Breakthrough: close-call stays a small end piece; its main week job is a depth lever (NewIdeas hands it one).
- Sat 14 15:31: TAPSQ judged KILL (MBPP 23-30: C 58.26, T 57.34; ratio 1.364), but it is held as VOID pending a harness
  check.
  - The control arm itself collapsed: t30b 72.13 -> C 58.26 after 80 steps at lr 1e-5, while depth 0 only went
    94.0 -> 93.1.
  - Suspect: the quick trainer gives the deep depths no signal (the smoke's per-depth KL 24-30 reads 0.0000).
  - Line before any rerun: C within ~1 pt of t30b. The raw-taps test waits for that fix.
  - Note: T's deep-depth KL was 0.43-0.49x C's, which is worth a look once the harness is sound.
- Sat 14 15:32: fans ramp 5900 -> 3750 (-250 / 2 min, fans_down_1114.sh), as told to Kamen.
- Sat 14 15:5x: an outside review (pasted by Kamen) made two fair points:
  - The ceiling is per dataset and decoding mode. Ours is MBPP code at greedy, and it's an estimate from the measured
    near-tie miss rate, not a hard bound.
  - Chat and agent text will be harder than code.
  - So early this week (owner NewIdeas / BigMoE): one quick measurement of the same per-depth curve (conditional
    acceptance, depth 1..30) on chat and agent answers. That means a small capture of ~50 answers plus t30b's eval,
    within the 15-30 min rule.
  - The expert-union cost per check is the other half of the speed goal (BYTES10 / union pricing), not this week's
    target.
- Sat 14 15:5x: the outside review's second round. Its "oracle" test (give the drafter the 125B's true states at the
  deep positions) is drafterwall's Law 2 (RESULT.md, 11-11).
  - With the true state two tokens back, the reader holds 97.7 % on HumanEval+ at every depth (vs 86.6 % at 23-30
    without), and 93.5 % on MBPP (vs 76.2 % by 7-10).
  - So the deep drop is stale information that lives in the 125B's forward pass. The fix must be architectural: a
    learned stand-in carried through the draft, or a cheap refresh mid-draft.
  - On MBPP even fresh states top out at 93.5 %, so the reader must improve there too.
  - Contamination check (orchestrator): all 3,072 n1112 training captures are src dlog-magicoder, split train; none
    come from HumanEval or MBPP. TrainLab's notes list MBPP as held out. Whether the 125B saw the benchmarks in
    pretraining is unknowable: the chat/agent curve is the real test.
- Sat 14 15:53: TAPSQ2 harness valid: C vs t30b at MBPP 23-30 -0.07 pt, band 0 -0.04. T runs; verdict ~16:10.
- Sat 14 16:0x: **TAPSQ2 = KILL with a valid control** (BigMoE 068dc9b).
  - DFlash2's fused multi-layer taps at the context / anchor rows add noise, not information: MBPP 23-30 71.43 vs C
    72.06, KL 1.083x, windows 35 better / 37 worse.
  - A richer view of the verified context doesn't reach the deep rows, which fits Law 2.
- Sat 14 16:1x: the next screens, freshness first (orchestrator's call):
  1. FRESH-H (no re-prefill, after CAPQ): the true fused h fed at every drafted row on the correct chain, tq2_C as the
     control, MBPP 23-30. It goes on if it recovers >= half the gap toward MBPP's lag-2 93.5.
  2. Then a re-prefill with raw taps on every row: per-layer freshness (what a cheap partial trunk pass buys), plus the
     anchor raw-taps item from the same data.
> (a quote from the project owner, kept private)
  generality added (orchestrator).
