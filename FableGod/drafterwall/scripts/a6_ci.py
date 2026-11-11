#!/usr/bin/env python3
"""a6_ci.py (drafterwall, DESCRIPTIVE; no line). Window-level bootstrap (2,000 resamples, 90 % interval) for the two
headline numbers on the unseen sets: the MTP chain's kept per 32-row check and the lag-2-everywhere drafter's (MTP alone,
and the perfect pick of the two readers), pooled over checks as everywhere else in this folder."""
import numpy as np, os
TD = os.path.expanduser("~/TrainLab/data"); rng = np.random.default_rng(11); out = []
def run(okf, n, cap=31):
    r = []; p = 0
    while p < n:
        k = 0
        while k < cap and p + k < n and okf(p, k): k += 1
        r.append(k + 1); p += k + 1
    return r
for st in ("code", "mbpp"):
    Wz = np.load(f"{TD}/d02b_walk_d23h_dfl_{st}.npz"); Mz = np.load(f"{TD}/d23h/{st}_b5.npz"); mk = {str(k): i for i, k in enumerate(Mz["keys"])}
    per = []
    for w, nm in enumerate(str(x) for x in Wz["names"]):
        mi = mk.get(nm[:-4] if nm.endswith(".npz") else nm, mk.get(nm))
        if mi is None: continue
        K = {int(A): int(k) for k, m, A in Wz[f"{w}_checks"] if int(m) >= 1}; gen = int(Wz[f"{w}_gen"]); ids = Wz[f"{w}_ids"]
        rt = Mz[f"rt_{mi}"]; a0 = int(Mz["a0"][mi]); A = gen; d, m_, rows = [], [], []
        while A < len(ids) - 1:
            r = A - 1 - a0
            if A not in K or r < 0 or r >= rt.shape[0] or rt[r, 0] < 0: break
            d.append(K[A] >= 1); m_.append(rt[r, 0] == 1); rows.append(r); A += 1
        if not d: continue
        d = np.array(d); m_ = np.array(m_); n = len(d)
        c = run(lambda p, k: rt[rows[p], k] == 1, n); l = run(lambda p, k: m_[p + k], n); e = run(lambda p, k: d[p + k] | m_[p + k], n)
        per.append((sum(c), len(c), sum(l), len(l), sum(e), len(e)))
    P = np.array(per, float); W = len(P)
    def stats(idx):
        s = P[idx].sum(0); return s[0] / s[1], s[2] / s[3], s[4] / s[5]
    base = stats(np.arange(W)); B = np.array([stats(rng.integers(0, W, W)) for _ in range(2000)])
    q = lambda x: f"[{np.quantile(x, 0.05):.2f}, {np.quantile(x, 0.95):.2f}]"
    s = (f"{st} ({W} answers), per 32-row check: MTP chain {base[0]:.2f} {q(B[:, 0])}; lag-2 everywhere MTP {base[1]:.2f} {q(B[:, 1])} "
         f"= x{base[1] / base[0]:.2f} {q(B[:, 1] / B[:, 0])}; perfect pick of two {base[2]:.2f} {q(B[:, 2])} = x{base[2] / base[0]:.2f} {q(B[:, 2] / B[:, 0])}")
    print(s); out.append(s)
open(os.path.join(os.path.dirname(__file__), "..", "out", "a6_ci.txt"), "w").write("\n".join(out) + "\n")
