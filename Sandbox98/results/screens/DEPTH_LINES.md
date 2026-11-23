# DEPTH screen (T2's pointer), pre-registered Sat 14 Nov 23:1x, before it runs

Theory (results/T2_TRUNCATION.md): the head's deep gap is mostly the DEPTH of composition. A 24-layer pass over only the
drafted tokens beats its one-layer carried state by +14 pt at 23-30, and the head already attends to every true state.
Arms, one process, same slices and order: ctl; deep2 (a second decoder layer, a copy of the native one with zero output
projections, so step 0 = s30, checked; own Adam at 3e-5); tie2 (the native layer applied twice, no new parameters); and
distill (its pending re-test, DISTILL2_LINE.md, now sharing this control). 400 steps of 96 anchors, rollout depth 30,
pool2 (one pass), s30o's Adam, warm-up to 1e-5 then a cosine to 1e-6. Read-out at depth 31 on MBPP-50 + HE-30 beside s30.
The two-layer exact gradient was checked against autograd under true fp32 (4e-6). MEM ~7 GB declared; it starts only
when no other session runs a memory-heavy job.
Lines:
- The control must hold vs s30 (MBPP-50 23-30 not down by more than 1.5 pt); otherwise every arm is provisional.
- deep2, tie2 vs ctl, MBPP-50 23-30: >= +2.0 pt with sign p < 0.05 -> DEPTH WORKS: confirm on the full sets with a
  longer run, then a 125B depth test; within +-2 -> NO EFFECT at 400 steps (report how far the new layer's output
  projections grew); <= -2.0 with p < 0.05 -> HURTS.
- distill: DISTILL2_LINE.md's lines.

## Additions (orchestrator 23:0x), pre-registered 23:1x, before any depth read-out
1. **Timing.** ms per drafted token, and per 30-deep draft, at 1 / 2 / 3 layers: decode-style (one row, a ~400-token
   context in the cache, 30 sequential steps, each synced like real drafting, argmax through the target's 4-bit head),
   median of 5 runs after a warm-up, the head's lm_head time split out. Shared GPU, labelled. The 2B head is dense, the
   125B's is MoE + HC, so the ratio per extra layer is what transfers, not the absolute ms. No pass line: input for
   Leap's break-even tables.
2. **Placement.** Every arm's full band curve and kept/32, plus its 23-30 on T2's ladder: the share of the chain -> full
   gap it closes, and the T2 rungs (K = 0 / 16 / 64 / 256 / full, MBPP-50) it sits between.
3. **Window cap** (only if deep2 or tie2 passes): an eval-only cap of the head's attention to the last 256 true pairs.
   Within 1 pt of the uncapped arm at 23-30 -> the cap is free, which bounds the drafter's attention cost.
4. **deep3** (a third untied layer, the same zero-init residual start), only if deep2 is up by >= +1.0 pt at 23-30:
   400 steps, same slices and schedule, compared against dp_ctl (deterministic, so the same control), with the same lines
   plus its increment over deep2.

## Depth results (Sat 14 Nov 23:43)

Placement (addition 2):

| arm (MBPP-50) | 0-0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept/32 | share of the chain -> full gap at 23-30 | on T2's ladder |
|---|---|---|---|---|---|---|---|---|---|---|
| s30 | 88.04 | 77.68 | 68.17 | 64.01 | 61.74 | 58.76 | 55.47 | 4.32 | +0.0 % | below K0 |
| dp_ctl | 88.24 | 77.09 | 67.42 | 63.37 | 61.10 | 58.15 | 54.97 | 4.26 | -1.6 % | below K0 |
| dp_deep2 | 87.27 | 75.91 | 66.26 | 61.59 | 59.14 | 55.73 | 52.49 | 4.09 | -9.2 % | below K0 |
| dp_tie2 | 85.86 | 74.25 | 64.92 | 60.94 | 58.63 | 55.91 | 52.58 | 3.93 | -8.9 % | below K0 |
| dp_distill | 87.92 | 76.95 | 67.33 | 63.36 | 61.17 | 57.96 | 54.48 | 4.25 | -3.1 % | below K0 |

MBPP-50 full-state oracle at 23-30 (gate, same windows) 87.75; s30's chain 55.47. T2 rungs as shares of T2's own gap: K0 44 %, K16 62 %, K64 78 %, K256 95 %

Timing (addition 1, shared GPU):
[td] 1 layer(s): 2.54 ms per drafted token (1.31 layers + 1.23 lm_head+argmax), 82.9 ms per 30-deep draft
[td] 2 layer(s): 3.96 ms per drafted token (2.73 layers + 1.23 lm_head+argmax), 120.6 ms per 30-deep draft
[td] 3 layer(s): 5.32 ms per drafted token (4.09 layers + 1.23 lm_head+argmax), 159.5 ms per 30-deep draft
deep2 below +1 pt: no deep3 (addition 4's condition)

## Result (Sat 14 Nov 23:42; results/SCREENS.log, results/screens/dp.json)
- **Control holds** vs s30: MBPP-50 23-30 55.47 -> 54.97 (-0.50, CI -1.29..+0.24), kept/32 -0.06; HE-30 -1.32. The
  re-annealed pool2 setup no longer drifts, but the control doesn't move up either at 400 steps.
- **deep2: HURTS.** MBPP-50 23-30 -2.48 (CI -3.15..-1.85, windows 10 / 38), depth 0 -0.97, kept/32 -0.17; HE-30 -2.04,
  kept -0.31. Its new layer's output projections grew only to |o_proj| 0.0024, |down_proj| 0.0028 (from 0).
- **tie2: HURTS.** MBPP-50 23-30 -2.38 (CI -3.16..-1.67), depth 0 -2.37, kept/32 -0.34; HE-30 -2.61, kept -0.60.
- **distill: NO EFFECT** against a holding control: MBPP-50 23-30 -0.48 (CI -1.02..+0.14), HE-30 -0.33. Its +3.1 / +8.1
  against the drifting 1e-5 control was drift protection only (DISTILL2_LINE.md's NO EFFECT branch).
- Conditions: no window-cap run (nothing passed) and no deep3 (deep2 below +1 pt).
**Reading.** T2 measured deep composition: a 24-layer pass trained on trillions of tokens. A zero-start layer trained for
400 steps from a converged one-layer head doesn't supply it; it costs ~2.5 pt while it is still near zero. So "add depth to
the head" as tested is a HURTS at this budget. It isn't a refutation of the theory: the depth that helps in T2 is
PRETRAINED composition. What would test it fairly: (a) a 2-layer head trained from the native start for the base's full
budget (one pass, ~35 min), against s30; (b) extra drafter layers initialized from pretrained trunk layers rather than
from zero (the composition is learned already; the input-space mismatch is the open question).
