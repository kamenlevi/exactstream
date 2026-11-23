"""D-17 part 2 CIs (MTP Head, Fri 10-30): 90 % bootstrap CIs over answers for D-17's router read-out (tokens per pass,
accepted + 1, each answer cut at its first end token): MTP K15 / DFlash2, and the per-round oracle (K15) / DFlash2.
Same data and walk as sticky/d17_route.py; ratios of pooled sums (tokens / passes) per bootstrap draw.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d17_ci.py -> data/d17_ci.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d18_hybrid import texts, TD                     # same loader (D-17 every-position data); prints its own table first

def per_answer(T, choose):
    r = []
    for out, mtp, kD in T:
        p, L, tok, n = 1, len(out), 0, 0
        while p < L:
            k = min(choose(out, mtp, kD, p), L - p)
            tok += k + 1; n += 1; p += k + 1
        r.append((tok, n))
    return np.array(r, float)

def kM(out, mtp, p, K=15):
    n = 0
    while n < K and p + n < len(out) and mtp[p][n] >= 0 and mtp[p][n] == out[p + n]:
        n += 1
    return n

lines = []
rng = np.random.default_rng(0)
lines.append("D-17 part 2, 90 % bootstrap CI over answers (tokens per pass; pooled ratio per draw)")
for dom in ("code", "gsm", "ifeval", "multi", "agent"):
    T = texts(dom)
    D = per_answer(T, lambda o, m, kD, p: kD.get(p, 0))
    M = per_answer(T, lambda o, m, kD, p: kM(o, m, p))
    O = per_answer(T, lambda o, m, kD, p: max(kD.get(p, 0), kM(o, m, p)))
    idx = rng.integers(0, len(T), (4000, len(T)))
    def ci(X):
        b = (X[idx, 0].sum(1) / X[idx, 1].sum(1)) / (D[idx, 0].sum(1) / D[idx, 1].sum(1))
        return X[:, 0].sum() / X[:, 1].sum(), np.quantile(b, 0.05), np.quantile(b, 0.95)
    d = D[:, 0].sum() / D[:, 1].sum()
    (m, ml, mh), (o, ol, oh) = ci(M), ci(O)
    lines.append(f"  {dom} ({len(T)}): DFlash2 {d:.2f}; MTP K15 {m:.2f} = {m / d:.3f}x (CI {ml:.3f}-{mh:.3f}); "
                 f"oracle {o:.2f} = {o / d:.3f}x (CI {ol:.3f}-{oh:.3f})")
print("\n".join(lines))
open(os.path.join(TD, "d17_ci.txt"), "w").write("\n".join(lines) + "\n")
