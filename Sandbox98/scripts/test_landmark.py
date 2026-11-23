"""Toy test of sb/landmark.py's pieces (CPU, no model): the landmark tables against a brute-force search with the
screen's indexing (p = s + d, q <= s, answer-only, k 8..2), copy (ii)'s re-match at p + 1, the stand-ins (lm1 replaces
exactly, uncovered rows bitwise untouched; lm05 mixes; stale1 reads W[p - 1]) and cov_counts. usage: python -I
scripts/test_landmark.py (from ~/Sandbox98)"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
from sb import landmark as L

LM, sha = L.load_lm()
rng = np.random.default_rng(1117)
ok = []
# a window with a repetitive answer (small vocabulary), prompt 0..gen-1
S, gen, Dp = 120, 30, 31
ids = rng.integers(0, 5, S).astype(np.int64)
a0 = gen - 2
Qt, Kt, Q2 = L.tables(LM, ids, gen, a0, Dp)


def brute(p, s, lo):
    for k in (8, 6, 4, 3, 2):
        if p - k + 1 < 0:
            continue
        c = [q for q in range(max(k - 1, lo), s + 1) if np.array_equal(ids[q - k + 1:q + 1], ids[p - k + 1:p + 1])]
        if c:
            return max(c), k
    return -1, 0


bad = 0; cov = 0; bad2 = 0
for i in range(Qt.shape[0]):
    s = a0 + i
    for d in range(1, Dp):
        p = s + d
        if p + 2 > S - 1:
            assert Qt[i, d] == -1
            continue
        q, k = brute(p, s, gen)
        bad += (Qt[i, d] != q) or (q >= 0 and Kt[i, d] != k)
        cov += q >= 0
        if q >= 0:
            q2, _ = brute(p + 1, s, gen)
            bad2 += Q2[i, d] != q2
ok.append((f"tables == brute force (p = s + d, q <= s, answer-only; {cov} covered cells)", bad == 0 and cov > 0))
ok.append(("copy (ii) re-match at p + 1 == brute force", bad2 == 0))
ok.append(("every landmark is an answer position <= its anchor", bool(((Qt < 0) | ((Qt >= gen) & (Qt <= a0 + np.arange(Qt.shape[0])[:, None]))).all())))
# stand-ins
d_ = 16
W = mx.array(rng.standard_normal((S, d_)).astype(np.float32))
j = 4; n = 10; anc = mx.array(a0 + np.arange(n)); h = mx.array(rng.standard_normal((1, n, d_)).astype(np.float32))
qrow = Qt[:n, j - 1]; c = qrow >= 0
h1 = np.array(L.standin_for("lm1", Qt, W, a0)(j, h, anc)); h5 = np.array(L.standin_for("lm05", Qt, W, a0)(j, h, anc))
hs = np.array(L.standin_for("stale1", Qt, W, a0)(j, h, anc)); hn = np.array(h); Wn = np.array(W)
ok.append((f"lm1: covered rows := W[q] exactly, others bitwise ({int(c.sum())} of {n} covered)",
           bool(c.any()) and np.array_equal(h1[0, c], Wn[qrow[c]]) and np.array_equal(h1[0, ~c], hn[0, ~c])))
ok.append(("lm05: covered rows = 0.5 W[q] + 0.5 h", np.allclose(h5[0, c], 0.5 * Wn[qrow[c]] + 0.5 * hn[0, c], atol=1e-6)
           and np.array_equal(h5[0, ~c], hn[0, ~c])))
pm1 = a0 + np.arange(n) + j - 2
ok.append(("stale1: covered rows := W[p - 1] (p = s + j - 1)", np.array_equal(hs[0, c], Wn[pm1[c]]) and np.array_equal(hs[0, ~c], hn[0, ~c])))
# cov_counts on a hand example: 3 rows x 3 depths
rt = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], np.int8)
cv = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], bool)
r, v = L.cov_counts(rt, cv)
ok.append(("cov_counts: alive = right at 0..d-1 on the same chain, counted over covered cells", r.tolist() == [1, 1, 0] and v.tolist() == [2, 2, 1]))
for nm, val in ok:
    print(f"{'PASS' if val else 'FAIL'}  {nm}")
print("TOY", "PASS" if all(v for _, v in ok) else "FAIL", f"(landmark_source sha256 {sha[:12]})")
sys.exit(0 if all(v for _, v in ok) else 1)
