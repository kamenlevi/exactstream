MTP Head 11-03, v2 (ALIGNED; replaces ../mtp_map_code, which was one position late: its depth d described token t+d+1).
The native MTP head b5's chain (D-22's trained head) per map row t (= position t, predicting t+1), depths d = 1..3: the chain
the engine has at row t starts from the trunk's hidden at t-1 and the token at t, so depth d predicts the 125B's token at t+d.
q [T-1, 3] float32 = the head's top-1 probability; right [T-1, 3] uint8 = 1 if its draft equals the 125B's token at t+d, 0 if
not, 255 = not computed (rows before gen - 1, or past the answer). Source: data/d24/code_b5.npz (sticky/d24_mtptop3.py), row
r = t - 1 - a0 (sticky/mtp_map_export.py). TEACHER-FORCED (depth d fed the true tokens t+1..t+d-1), which equals the engine's
own-draft chain wherever the earlier depths were right: tf_eq_own [T-1, 3] = 1 there (all of depth 1; depth 2 if depth 1
right; depth 3 if 1 and 2 right). Where tf_eq_own = 0 the engine's chain had already missed, so the check stopped before it.
