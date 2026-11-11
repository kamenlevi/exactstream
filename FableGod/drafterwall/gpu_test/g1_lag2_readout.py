#!/usr/bin/env python3
"""g1_lag2_readout.py (drafterwall, the read-out of the pre-registered GPU test LAG2-AGENT; CPU, numpy only).
usage: g1_lag2_readout.py <set name> <every-position DFlash2 walk .npz> <MTP depth-31 table .npz>
  walk  : TrainLab sticky/d02b_walk.py with TL_WALK_EVERY=1 (names, <i>_ids, <i>_gen, <i>_checks [n, 3] = k, m, A)
  table : TrainLab sticky/d23h_mtp31.py (keys, gen, a0, rt_<i> [anchors, 31]; row r = the check anchored at r + 1 + a0)
Prints, per 16 / 32 / 64-row check: each reader's real chain (official loop), and the lag-2-everywhere drafter (right at
a position exactly when the reader(s) are right there at depth 0) for DFlash2, the MTP head, and either. Then the lines."""
import numpy as np, sys
def run(ok_fn, n, cap):
    r = []; p = 0
    while p < n:
        k = 0
        while k < cap and p + k < n and ok_fn(p, k): k += 1
        r.append(k + 1); p += k + 1
    return r
def main(st, fw, fm):
    Wz, Mz = np.load(fw), np.load(fm); mk = {str(k): i for i, k in enumerate(Mz["keys"])}
    C = {(r, c): [] for r in ("dfl chain", "mtp chain", "dfl lag2", "mtp lag2", "either lag2") for c in (15, 31, 63)}
    acc = {"dfl": [0, 0], "mtp": [0, 0], "either": [0, 0]}; nw = 0
    for w, nm in enumerate(str(x) for x in Wz["names"]):
        mi = mk.get(nm[:-4] if nm.endswith(".npz") else nm, mk.get(nm))
        if mi is None: continue
        ch = Wz[f"{w}_checks"]; gen = int(Wz[f"{w}_gen"]); ids = Wz[f"{w}_ids"]
        K = {int(A): int(k) for k, m, A in ch if int(m) >= 1}
        rt = Mz[f"rt_{mi}"]; a0 = int(Mz["a0"][mi])
        A = gen; d, m_, rows = [], [], []
        while A < len(ids) - 1:
            r = A - 1 - a0
            if A not in K or r < 0 or r >= rt.shape[0] or rt[r, 0] < 0: break
            d.append(K[A] >= 1); m_.append(rt[r, 0] == 1); rows.append(r); A += 1
        if not d: continue
        nw += 1; d = np.array(d); m_ = np.array(m_); n = len(d); A0 = gen
        for r, ok in (("dfl", d), ("mtp", m_), ("either", d | m_)):
            acc[r][0] += int(ok.sum()); acc[r][1] += n
        for c in (15, 31, 63):
            C[("dfl chain", c)] += run(lambda p, k: k < min(K[A0 + p], 15), n, c) if c == 15 else []
            C[("mtp chain", c)] += run(lambda p, k: k < 31 and rt[rows[p], k] == 1, n, c) if c <= 31 else []
            C[("dfl lag2", c)] += run(lambda p, k: d[p + k], n, c)
            C[("mtp lag2", c)] += run(lambda p, k: m_[p + k], n, c)
            C[("either lag2", c)] += run(lambda p, k: d[p + k] | m_[p + k], n, c)
    f = lambda k: f"{np.mean(C[k]):.2f}" if C[k] else "-"
    print(f"{st} ({nw} answers): right at depth 0: " + ", ".join(f"{r} {100 * a / max(b, 1):.1f} %" for r, (a, b) in acc.items()))
    for r in ("dfl chain", "mtp chain", "dfl lag2", "mtp lag2", "either lag2"):
        print(f"   {r:12s} kept per check at 16 / 32 / 64 rows: {f((r, 15))} / {f((r, 31))} / {f((r, 63))}")
    best_chain = max(np.mean(C[("mtp chain", 31)]), np.mean(C[("dfl chain", 15)]))
    ratio = np.mean(C[("either lag2", 31)]) / best_chain
    print(f"   LINE input: either-lag-2 at 32 rows / the better real chain = x{ratio:.2f}")
    return ratio
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
