# Long screens (orchestrator 21:3x), pre-registered Sat 14 Nov 21:44, before any of these runs

Rule for every training screen (orchestrator 21:3x, from the 300-step results): **a training lever needs a budget at
which the control itself measurably moves; otherwise NO EFFECT means untested.** (At 300 steps x 2e-6 the plain control
moved only +0.49 pt from s30.)

## Base s30o
s30's run again (native head, one pass over the 380 base answers, lr 1e-5 cosine, seed 1114) with Adam's state saved
(moments + step; save/resume checked exact on a toy). Check: its weights equal s30's (training is deterministic) -> the
gate's numbers apply to it. If they differ: read s30o out on MBPP-50 + HE-30, use it anyway, and say so.

## Long-screen setup
From s30o, continuing its Adam state, lr 1e-5 constant after a 20-step warm-up, 1,000 steps of 96 anchors on the
screen pool (~1.9 passes over its 120 answers; pass 2 reshuffled), read out on MBPP 50 + HE 30. One control (same slices,
same order, no lever) serves every lever.
Control line: MBPP-50 23-30 within +-1.5 pt of s30's 55.47 and kept/32 within +-0.15 of 4.319 -> no drift, valid.
Up by > +1.0 pt -> the budget moves the head (the rule above is met); down by > 1.5 pt -> drift: the paired lever-vs-
control comparisons still stand, the drift is reported beside them.

## Lever lines (each vs the control)
MBPP 23-30 >= +2.0 pt with sign p < 0.05 -> PASS: confirm on the full sets (100 + 60), then propose the 125B screen;
within +-2 pt -> NO EFFECT at this budget; <= -2.0 pt with p < 0.05 -> HURTS.
1. distill (lambda 1.0): KL to the head's own true-state output at every depth >= 2, on top of the TTT loss.
2. noisehead (F 0.7, half the rows, training only); its depth-0 +0.29 at 300 steps is the hint.
3. headfeat: NOT run. At 300 steps it was a significant loss (MBPP 23-30 -1.50 pt, p 0.013); more steps of the same
   pull would be expected to hurt more, so there is no reason to expect a different outcome (the orchestrator's condition).

Base check: s30o vs s30, max |weight difference| over 15 tensors = 0 -> EQUAL: the gate's numbers apply

Control (22:15): mbpp 23-30 55.47 -> 48.27 (-7.20), kept/32 4.319 -> 3.798, depth 0 88.04 -> 86.63; he 23-30 68.72 -> 53.94 (-14.78), kept/32 6.239 -> 4.824, depth 0 93.58 -> 92.14 -> outside the no-drift band; DRIFT DOWN: reported beside the comparisons

## Outcome (22:3x): the setup FAILED its control line; distill is a provisional lead, noisehead not run
- Control (s30o + 1,000 steps, continued Adam, constant 1e-5, ~1.9 passes over the 120-answer pool) vs s30 on the same
  windows: MBPP-50 23-30 55.47 -> 48.27 (-7.20), kept/32 4.32 -> 3.80, depth 0 88.04 -> 86.63; HE-30 23-30
  68.72 -> 53.94 (-14.78). Far past the -1.5 drift line.
- Cause: mostly the constant 1e-5 on an ANNEALED head (s30's cosine ended at 1e-6). Every constant-1e-5 continuation
  drifts, fresh Adam or continued: 300 steps with no repeats -8.2 (MBPP-50 23-30), 1,000 steps -7.2; constant 2e-6
  held (+0.49). Pool repeats may add to it (HE -14.8 here vs -9.4 at 300 fresh steps). Orchestrator 22:3x pointed at the pool.
- distill vs this drifting control: MBPP-50 23-30 48.27 -> 51.40 (+3.13 pt, CI +1.71..+4.46, sign p 0.029), kept/32
  +0.09; HE-30 23-30 53.94 -> 62.06 (+8.12, CI +6.66..+9.70, 29 / 1 windows), kept +0.40. It still ends BELOW s30
  (51.40 vs 55.47; 62.06 vs 68.72): the fresh-state teacher held the head against about half of the drift. PROVISIONAL:
  re-test against a control that holds.
- noisehead and the constant-lr calibration / transfer runs were cancelled (22:31) before they started.
