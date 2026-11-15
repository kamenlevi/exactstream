"""Walk-strict coverage per band for tree-K (K = 15 / 31 best-first pops of BigMoE's real dumps), the C1 line at K*.
NewIdeas, Sun 15 Nov 22:0x. CPU only; same walk and convention as tree_audit.py (target-less end cells not counted)."""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA
TA.say = lambda s="": None
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
    W = TA.load_windows(tag, pub32)
    tz = np.load(f"{TA.CE}/tree_t30b_{tag}.npz", allow_pickle=True)
    fi, an, nd, dp, ot = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
    for K in (15, 31):
        hit = ot & (nd < K); acc = {}
        for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
            acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
        ks, num, den = [], np.zeros(31), np.zeros(31)
        for X in W:
            p = 1
            while p < X["n"]:
                a = acc.get((X["fi"], int(X["an"][p])), 0) if p < len(X["an"]) else 0
                a = min(a, K)
                num[:a] += 1; den[:a] += 1
                if a < K and X["rt"][p, a] >= 0:
                    den[a] += 1
                ks.append(a + 1); p += a + 1
        bs = " ".join(f"{100 * num[a:b + 1].sum() / den[a:b + 1].sum():5.1f}" if den[a:b + 1].sum() else "    -" for a, b in BANDS)
        print(f"{s} tree-K{K + 1:<3d} kept {np.mean(ks):6.3f}  walk-strict 0|1-2|3-6|7-10|11-14|15-22|23-30: {bs}")
