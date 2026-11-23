# Fast-screen calibration (orchestrator 21:4x), pre-registered Sat 14 Nov 21:5x, before it runs

Question: how fast does the plain control move from the base per training step, at rollout depth 12 vs 30? The answer
sets the fast screen's length (the budget rule: a training lever needs a budget at which the control measurably moves).

Runs (after the long-screen chain, alone on my side of the GPU, so the per-step times are comparable):
- C12: s30o + its Adam, lr 1e-5 (20-step warm-up), pool slices (seed 1115, 2 passes), rollout depth 12, 1,000 steps;
  copies at 100, 200, 300, 500, 700, 1000. Each copy read out at depth 15 (bands 0-14, kept/16) on MBPP-50 + HE-30.
- C30: the same at rollout depth 30, 500 steps; copies at 100, 200, 300, 500; read out at depth 15 AND at depth 31.
- The base s30o read out the same ways (the reference for every delta).
Measure: per copy, the paired delta vs the base on MBPP-50 band 7-14 (bootstrap 95 % CI over windows), plus 0-0, 11-14,
kept/16 (and 23-30, kept/32 for C30); seconds per step at depth 12 vs 30 (same process layout, same slices).
"Measurably moves" = the MBPP-50 7-14 delta's CI excludes 0 AND |delta| >= 1.0 pt.
Result -> action:
- The fast screen = rollout depth 12, read-out depth 15, STEPS = the first C12 copy that measurably moves (rounded up to
  the next copy); the line band = 7-14. Confirm winners at depth 30 (long-screen format).
- If C12 moves only later than C30 per unit of wall time, the fast format is depth 30 at C30's budget instead.
- If no copy moves by 1,000 steps at depth 12: the fast format needs a longer budget or a higher lr; say so, no default.

## v1 above never ran (cancelled 22:31). v2, pre-registered Sat 14 Nov 22:4x, before it runs
Why: the constant-1e-5 setup drifted the control 7-15 pt below s30 (results/screens/LONG_LINES.md). The likely cause is
a constant lr on an annealed head (every constant-1e-5 continuation drifted, with or without pool repeats), plus the
small pool (orchestrator 22:3x).
Changes, for the calibration, the fast format and the transfer checks:
- Pool: pool2 = the 120 pool answers + 500 fresh Magicoder answers (mc2: the next 500 prompts in the seeded order), one
  pass, no repeats (~2,700 slices of 96).
- Schedule: from s30o with its Adam state, a 20-step warm-up to 1e-5, then a cosine down to 1e-6 by the run's last step:
  each run re-anneals like the base did.
Runs: the plain control alone, each budget its own run with its own schedule (same slice order, so budget B sees the
first B slices): depth 12 at B = 100, 200, 400, 800; depth 30 at B = 200, 400. Read-outs vs s30 on MBPP-50 + HE-30 at
depth 15 (all) and depth 31 (the depth-30 runs).
Lines: a budget "holds" if MBPP-50 7-14 (depth 15) is not down by more than 1.5 pt; it "moves" if that delta's CI
excludes 0 and the delta is >= +1.0 pt.
Result -> action:
- The fast budget = the smallest depth-12 B that holds and moves. Fast format = depth 12 at B, read-out depth 15,
  line band 7-14. If a depth-30 budget moves sooner in wall time, the fast format is depth 30 at that budget.
- If some budget drifts down by more than 1.5 pt: the schedule still harms. No default; report it.
- If none moves (all hold, none up by 1 pt): s30 doesn't measurably improve with more data at this budget. Fast screens
  then can't show training levers by the rule. Report it; no default.
