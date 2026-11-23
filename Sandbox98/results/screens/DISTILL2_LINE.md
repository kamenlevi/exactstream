# distill re-test against a holding control, pre-registered Sat 14 Nov 22:5x, before it runs
Why: distill was the one lead (+3.13 MBPP-50 23-30, +8.12 HE-30) but against a control that drifted 7-15 pt below s30.
Setup: one process, arms ctl + distill (lambda 1.0), pool2 (the 120 pool + 500 fresh answers, one pass, no repeats),
s30o's Adam state, 20-step warm-up to 1e-5 then a cosine to 1e-6 (re-annealed), rollout depth 30 (where distill acted),
400 steps; read-out at depth 31 on MBPP-50 + HE-30 beside s30.
Lines:
- The control must hold vs s30 (MBPP-50 23-30 not down by more than 1.5 pt); otherwise distill is again provisional.
- distill vs ctl, MBPP-50 23-30 >= +2.0 pt with sign p < 0.05 -> PASS: the fresh-state teacher improves the deep chain
  without drift; confirm on the full sets at a longer budget, then a 125B screen;
- within +-2 pt -> NO EFFECT at 400 steps (the drift-prevention was its whole effect);
- <= -2.0 pt with p < 0.05 -> HURTS.
