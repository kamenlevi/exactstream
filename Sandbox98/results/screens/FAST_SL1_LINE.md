# Fast batch sl1: distill + noisehead, pre-registered Sat 14 Nov 22:5x, before it runs
Format: calibration v2's fast format (its budget and rollout depth from results/screens/calibration_v2.json; 400 steps
at depth 12 if it sets none, then labelled "below / unknown budget"), pool2 one pass, warm-up + cosine 1e-5 -> 1e-6 from
s30o's Adam state, arms ctl + distill (lambda 1.0) + noisehead (F 0.7) in one process, read-out depth 15, MBPP-50 + HE-30.
The in-batch control must hold vs s30 (MBPP-50 7-14 not down by more than 1.5 pt); otherwise the results are provisional.
Context: distill was +3.13 (MBPP 23-30) / +8.12 (HE) against a DRIFTING control at constant 1e-5 (provisional).
Lines (each lever vs the in-batch control, MBPP-50 band 7-14):
- >= +2.0 pt with sign p < 0.05 -> PASS: confirm at depth 30 (long format with the same pool / schedule fix), then 125B;
- within +-2 pt -> NO EFFECT at this budget (if the budget moves the control, a real NO EFFECT);
- <= -2.0 pt with p < 0.05 -> HURTS.
