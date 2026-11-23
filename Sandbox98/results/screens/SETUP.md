# Screen setup check (written Sat 14 Nov 20:02, before the runs below)

Finding (20:00): the first control, ctl_s30_300 = s30 + 300 pool steps at constant lr 1e-5 with a fresh Adam (20-step
warm-up), is far below s30 on the same windows: MBPP-50 23-30 55.47 -> 47.31, kept/32 4.32 -> 3.78; HE-30 23-30
68.72 -> 59.36, kept 6.24 -> 4.91. s30's cosine ended at 1e-6; restarting at 10x that with fresh Adam moments moves the
head off its optimum (cf. the 125B's MLX-Adam-restart note). The feat screen at that setting (MBPP 23-30 -2.33 pt vs
its control, p 0.013) is a valid paired difference but on a disturbed head.

Fix to test: screens at constant lr 2e-6 (2x s30's final lr), same 300 steps, same slices.
result -> action: ctl at 2e-6 within +-1.5 pt of s30 on MBPP-50 23-30 (55.47) and kept/32 within 0.15 (4.32) -> 2e-6
becomes screen.sh's default and feat is re-run at it (same line as before); else screens use 1e-6 and this note says so.
(Long-term fix: save the base's Adam state and continue it, as the 125B's TTT30B runs do.)

Result (20:06): control at 2e-6 vs s30 on MBPP-50: 23-30 55.47 -> 55.96 (+0.49 pt), kept/32 4.319 -> 4.263 (-0.056) -> HOLDS: 2e-6 is the screen default

Rule (orchestrator 21:3x): a training lever needs a budget at which the control itself measurably moves; otherwise
NO EFFECT means untested. Long screens continue the base's saved Adam state (results/screens/LONG_LINES.md).
