# Learned-better-stand-in screens (orchestrator 20:2x: items 1-3), pre-registered Sat 14 Nov 20:33, before any read-out
Common setup: s30 + 300 steps on the screen pool (lr 2e-6 constant after a 20-step warm-up, the validated screen
setting), read out on MBPP 50 + HumanEval 30 against the cached control ctl_s30_300_lr2e-6 (which holds against s30:
MBPP-50 23-30 55.47 -> 55.96). Smoke-tested 20:29-20:31 (20 steps each).
Lines, for each lever (result -> action):
- MBPP 23-30 >= +2.0 pt over the control with sign p < 0.05 -> PASS: confirm on the full sets (100 + 60) with a longer
  run (500 steps, its own control), then propose the 125B screen.
- within +-2 pt -> NO EFFECT at this budget (300 steps at 2e-6): a small significant shift (like tapmlp's +0.6) is noted
  as a lead for a longer run at 01:00, not a pass.
- <= -2.0 pt with p < 0.05 -> HURTS.
1. distill (item 1, ORACLE DISTILLATION): at every depth >= 2, LAM 1.0 x KL(teacher top-32 + rest || student), the teacher
   = the head reading the true W (= its own depth-1 output at row s + j - 1, under stop-gradient), on top of the TTT loss.
2. headfeat (item 2, STATE TARGET IN HEAD SPACE): LAM 0.5 x the stand-in's squared error to the true state after the
   head's input norm and fc's hidden half (BigMoE's r2_head space), over the window's variance; projection frozen in it.
3. noisehead (item 3, stage A, NOISE-ROBUST HEAD): in training only, half the rows at each depth >= 2 read true W + noise
   (F 0.7, R2 ~0.5) instead of the carried stand-in. Stage B (a predictor with noise-like errors) is not in this screen.

## Results (21:03-21:32, results/SCREENS.log; headfeat yielded once to a quiet ticket at 21:15:49 and was rerun from scratch)
| lever | MBPP-50 23-30 | MBPP depth 0 | MBPP kept/32 | HE-30 23-30 | HE depth 0 | HE kept/32 | verdict by the line |
|---|---|---|---|---|---|---|---|
| distill | -0.13 (CI -0.41..+0.14, p 0.37) | -0.11 | +0.021 | -0.15 (p 0.71) | -0.05 | -0.010 | NO EFFECT (no shift at all) |
| headfeat | -1.50 (CI -2.48..-0.62, p 0.013) | -0.04 | -0.078 | -0.45 (p 0.043) | +0.01 | -0.229 (p 0.052) | NO EFFECT by the line; a small significant LOSS |
| noisehead | -0.33 (CI -0.88..+0.18, p 0.31) | +0.29 (CI +0.07..+0.51) | +0.014 | +0.15 (p 0.36) | +0.14 (p 0.035) | +0.034 | NO EFFECT at depth; a small depth-0 gain |
(control ctl_s30_300_lr2e-6: MBPP-50 23-30 55.96, kept 4.263; HE-30 68.34, kept 6.200. Swap 7.42 -> 7.37 GB over the batch.)
Reading: at this budget none of the three moves the deep bands up. Pulling the stand-in toward the true state in head
space (headfeat) makes it slightly WORSE, like wd space (feat) did nothing: matching the true state is not what the head
needs from its carried state. Distillation toward the head's own fresh-state output moved nothing in 300 steps at 2e-6
(the plain control itself moves only +0.5 pt in that budget), so item 1 is untested at a real budget, not refuted.
Training on noisy true states nudged the first guess up (+0.3 / +0.1 pt), not the deep ones.
