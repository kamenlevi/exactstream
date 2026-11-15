"""MoonShots export v2 (MTP Head, Tue 11-03): the native MTP head b5's teacher-forced chain per code map row, ALIGNED.
v1 (data/mtp_map_code, written inline 11-03 morning) read D-24's row r = t - a0 for map row t; D-24's row r is the MTP anchor
s = a0 + r, whose depth d predicts ids[s + d + 1] (d22_ttt.chain), so v1's depth d described token t + d + 1, not t + d, and
used the trunk's hidden at t. v2 reads s = t - 1 (r = t - 1 - a0): depth d predicts token t + d, as the README says.
Output: data/mtp_map_code_v2/<key>.npz + README.txt."""
import glob, os
import numpy as np
M = np.load(os.path.expanduser("~/TrainLab/data/d24/code_b5.npz"))
mk = {str(k): i for i, k in enumerate(M["keys"])}
out = os.path.expanduser("~/TrainLab/data/mtp_map_code_v2")
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.expanduser("~/FableGod/decision_map/out/moonshots/tf_code_*.npz"))):
    key = os.path.basename(f)[:-4]
    z = np.load(f); toks = z["tokens"]; T = len(toks)
    mi = mk[key]
    ids = M[f"ids_{mi}"]; assert np.array_equal(ids[:T], toks[:len(ids)][:T]) or np.array_equal(ids, toks), key
    a0 = max(0, int(M["gen"][mi]) - 2); p1, rk = M[f"p1_{mi}"], M[f"rk_{mi}"]
    q = np.full((T - 1, 3), np.nan, np.float32); right = np.full((T - 1, 3), 255, np.uint8); tfeq = np.zeros((T - 1, 3), np.uint8)
    for t in range(T - 1):
        r = t - 1 - a0
        if r < 0 or r >= p1.shape[0]:
            continue
        ok_prev = True
        for d in range(3):
            if rk[r, d] < 0 or np.isnan(p1[r, d]):
                break
            q[t, d] = p1[r, d]; right[t, d] = 1 if rk[r, d] == 0 else 0; tfeq[t, d] = 1 if ok_prev else 0
            ok_prev = ok_prev and rk[r, d] == 0
    np.savez(os.path.join(out, key + ".npz"), q=q, right=right, tf_eq_own=tfeq, a0=np.int64(a0))
    print(key, "T", T, "rows with numbers", int((right[:, 0] != 255).sum()), "depth-1 right",
          round(float((right[:, 0] == 1).sum() / max((right[:, 0] != 255).sum(), 1)), 3))
open(os.path.join(out, "README.txt"), "w").write(
    "MTP Head 11-03, v2 (ALIGNED; replaces ../mtp_map_code, which was one position late: its depth d described token t+d+1).\n"
    "The native MTP head b5's chain (D-22's trained head) per map row t (= position t, predicting t+1), depths d = 1..3: the chain\n"
    "the engine has at row t starts from the trunk's hidden at t-1 and the token at t, so depth d predicts the 125B's token at t+d.\n"
    "q [T-1, 3] float32 = the head's top-1 probability; right [T-1, 3] uint8 = 1 if its draft equals the 125B's token at t+d, 0 if\n"
    "not, 255 = not computed (rows before gen - 1, or past the answer). Source: data/d24/code_b5.npz (sticky/d24_mtptop3.py), row\n"
    "r = t - 1 - a0 (sticky/mtp_map_export.py). TEACHER-FORCED (depth d fed the true tokens t+1..t+d-1), which equals the engine's\n"
    "own-draft chain wherever the earlier depths were right: tf_eq_own [T-1, 3] = 1 there (all of depth 1; depth 2 if depth 1\n"
    "right; depth 3 if 1 and 2 right). Where tf_eq_own = 0 the engine's chain had already missed, so the check stopped before it.\n")
print("done")
