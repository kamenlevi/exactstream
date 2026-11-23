# tapmlp screen, pre-registered Sat 14 Nov 20:18 (before any tapmlp read-out beyond the 3-window smoke)
Lever: WEEK98 item 1. A zero-init MLP (RMSNorm -> 2048 -> SiLU -> 2048) maps the TARGET's true residual after layer 12 of
24 at the drafted row to a correction ADDED to the carried stand-in at every depth >= 2 (a half-trunk pass per drafted
token). 300 steps on the screen pool; the head at the screen lr 2e-6, the MLP on its own Adam at 1e-5 (TAPSQ2's tap lr).
Control: ctl_s30_300_lr2e-6 (cached). Windows: MBPP 50 + HumanEval 30.
Context: ridge maps from tap 12 alone reach R2 0.44 and, as a replacement, MBPP 23-30 59.19 vs the chain's 57.32.
Line (result -> action):
- MBPP 23-30 >= +2.0 pt over the control with sign p < 0.05 -> PASS: confirm on the full sets with more steps, then price
  the half-trunk pass (Leap's partial-trunk pricing) before any 125B run.
- within +-2 pt -> NO EFFECT at this budget (300 steps, module lr 1e-5): note the budget; a longer run only if a cheap
  variant (tap 17 or 22) passes first.
- <= -2.0 pt with p < 0.05 -> HURTS.

## Result (20:25, results/SCREENS.log)
- MBPP-50 23-30 55.96 -> 56.57 (+0.61 pt, CI +0.32..+0.88; windows better 33 / worse 12; sign p 0.002); depth 0 +0.00;
  kept/32 4.263 -> 4.277 (+0.014, n.s.).
- HE-30 23-30 68.34 -> 69.16 (+0.82 pt, CI +0.49..+1.13; p 0.016); depth 0 -0.04; kept/32 +0.049 (p 0.064).
- **Verdict by the line: NO EFFECT at this budget** (+0.61 < +2.0). It is not nothing: a small, significant gain on both
  sets after 300 steps from a zero-init module. Per the line, next is a cheaper-to-learn variant (tap 17, ~4 min)
  before any longer run; a deeper tap costs more trunk layers per drafted token, so price it alongside.
- Swap 7.65 -> 7.62 GB across the run (MLX peak 3.6 GB; BigMoE's bank had finished at 20:21:34).
