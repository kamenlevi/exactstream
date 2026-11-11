MTP Head 11-03: the native MTP head b5's chain (D-22's trained head) per map row t (= position t, predicting t+1) as the anchor, depths d = 1..3:
q [T-1, 3] float32 = the head's top-1 probability; right [T-1, 3] uint8 = 1 if its draft equals the 125B's token at t+d, 0 if not,
255 = not computed (prompt rows before gen - 2, or past the answer). Source: data/d24/code_b5.npz (sticky/d24_mtptop3.py).
The chain is TEACHER-FORCED (depth d fed the true tokens t+1..t+d-1), which equals the engine's own-draft chain wherever the
earlier depths were right: tf_eq_own [T-1, 3] = 1 there (all of depth 1; depth 2 if depth 1 right; depth 3 if 1 and 2 right).
Where tf_eq_own = 0 the engine's chain had already missed, so the check stopped before that depth.
