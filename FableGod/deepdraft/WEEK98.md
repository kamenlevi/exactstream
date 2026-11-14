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
