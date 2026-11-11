# Cloud Fable 5.1, task 10 (CLOUD_TASK_12GB_V3.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~20:2x
(The full ANSWER.md stays in the cloud session: 1,101 words, 3 of 8 Python runs, no web.)

- **What the sweep changed:**
  - v2 guessed pass 2 in 97 / 71 / 37 % of blocks at theta 0.99 / 0.9 / 0.7. The truth is 77 / 60 / 47, which is flatter.
  - Engine bytes are 1.16 GB per check, not 0.96.
  - Compute barely moves with theta: pass 1 is fixed at ~42 ms, and pass 2 x its frequency swings only 17-31 ms.
- **The check to run: top-2 at theta 0.8.**
  - Honest 121 / 131 tok/s at kept 12 / 13 (~99 ms per check).
  - Deviation 0.45 % under the x1.5 placeholder, 0.9 % at x3. Both are inside the 1 % budget.
  - Theta 0.7 survives only a factor <= 1.5.
  - Top-5 in pass 1 is dead on bytes alone (2.05 GB per check).
  - One mix is worth a day on the decision map: an adaptive theta, 0.7 where the drafter is sure and agrees with pass 1, 0.9 elsewhere.
- **Where the stall is:**
  - It is pass 1's per-layer chain (routing -> read -> compute) and the first layers of each pass, not pass 2.
  - With pass 2's reads issued at once, pass 2 is compute-bound. So forecasting pass 2's experts is worth only 1-3 ms.
  - Forecasting pass 1's reads, so the SSD runs from t = 0, would move the honest 99 ms toward the ~70 ms bound.
- **Perfect prediction, priced:**
  - A perfect row oracle: +18 %, zero deviation (143 / 155).
  - A perfect expert forecast: +40 % at the limit (172 / 186). This is what 150 needs.
  - Realizable signals:
    - rows: pass 1's margin + the drafter's confidence
    - experts: the draft tokens routed through the early target layers that the early-tap drafter already computes
  - The previous check's routing is too weak (27 % sharing).
- **Verdict:**
  - 120 is the central estimate at theta 0.8 with kept 12-13, and needs "nothing unmeasured, just the drawn schedule".
  - 150 rests on the pass-1 forecast landing at >= 75 % hit with <= 25 % wasted bytes.
  - If the deviation factor comes back at x3, 120 is unchanged and 150 loses its only shortcut.

## Orchestrator's check (20:2x)
- **Arithmetic holds:**
  - theta 0.8: 10 flips / 3,366 code rows = 0.30 %; x1.5 = 0.45 %; x3 = 0.89 %
  - theta 0.7: 0.68 % x1.5 = 1.02 %
  - 99 ms per check = 12 / 121 = 13 / 131
  - 27 % sharing comes from task 4's inputs
- **Overstated: "120 needs nothing unmeasured."** Three of its inputs are not in hand:
  - Kept 12-13 is the drafter ceiling (copy override + two drafters); today's kept is ~10.5, which gives ~106 at the same cycle.
  - The overlap schedule is not built.
  - The deviation factor is still running.
- **150's lever is already measured below its line** (~/TRIED/TRIED.md):
  - Look-ahead top-2 precision is 0.527 raw and 0.683-0.689 at best (next-layer predictor KILL; PREROUTE2 KILL 0.683).
  - Recall at lead 2 / 3 is 27-40 % / 15-26 % (OR1/FC1).
  - Look-ahead prefetch at 12 GB was slower (PF01: 22.7 vs 26.2 tok/s).
  - The forecast from the MTP head's state got recall@10 0.09-0.24 (Rnd23).
  - A forecast of all 48 layers at t = 0 at >= 75 % hit / <= 25 % waste is far beyond these.
- **Routed:**
  - Breakthrough: the adaptive-theta arm on the map, and theta 0.8 as the 12 GB plan's working point.
  - Cloud task 11 (CLOUD_TASK_12GB_V4.md): 150 by another road, with the measured forecast ceiling.
