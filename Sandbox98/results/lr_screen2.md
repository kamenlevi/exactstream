# lr screen, read-out version (written Sat 14 Nov 18:51, before the read-outs ran)

The first rule (lowest accept-until-fail loss over steps 41-80) is VOID: that loss averages over the rows still alive
at each depth, so an arm that chains better carries more rows into the deep, high-KL depths and can show a HIGHER loss
while being better (lr 3e-5 read 1.02 / 0.90 at steps 40 / 60 vs 1e-5's 0.77 / 0.75). It is not comparable across arms.

New rule: read out native and the three 80-step arms (lrs_1e-5, lrs_3e-5, lrs_1e-4; same 80 base slices, cosine) on the
40 held-out Magicoder answers (split val; MBPP / HumanEval stay untouched for the gate).
result -> action: the arm with the highest kept/32 sets the base run's lr (ties within 0.05: the higher 23-30 band).
Result (18:56): kept/32 native 3.18, lrs_1e-5 3.83, lrs_3e-5 3.56, lrs_1e-4 2.67 (23-30: 1.72 / 29.27 / 32.12 / 26.36) -> base lr 1e-5 (results/lr_screen2.json)
The void loss rule would have picked 1e-5 too (steps 41-80: 0.690 / 0.835 / 1.534), so the base lr is the same either
way. The caveat stands for every comparison of training losses across arms: the accept-until-fail loss is a
survival-weighted average of per-depth KLs (weights ~ the rows still alive at each depth), so better chaining raises it.
Judge arms by read-outs (bands, kept/32), or by per-depth KL on fixed rows, never by this loss.
