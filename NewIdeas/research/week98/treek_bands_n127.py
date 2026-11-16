"""Walk-strict coverage per band for REAL big trees (BigMoE's 127-node dry-run dump, top-8 children), HE+ (MBPP when
its dump lands). NewIdeas, Mon 16 Nov 09:2x. Same walk / convention as tree_audit.py / treek_bands.py (targets only).
K = nodes (rows - 1). Self-check: K 15 / 31 must reproduce the 32-pop dump's kept (11.263 / 16.263)."""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA
TA.say = lambda s="": None
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
sets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["he"]
for s in sets:
    tag, rtag, pub32, pub16 = TA.SETS[s]
    W = TA.load_windows(tag, pub32)
    tz = np.load(f"{TA.CE}/tree_t30b_{tag}_n127k8.npz", allow_pickle=True)
    fi, an, nd, dp, ot = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
    for K in (15, 31, 47, 63, 95, 126):
        hit = ot & (nd < K); acc = {}
        for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
            acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
        ks, num, den = [], np.zeros(31), np.zeros(31)
        for X in W:
            p = 1
            while p < X["n"]:
                a = acc.get((X["fi"], int(X["an"][p])), 0) if p < len(X["an"]) else 0
                a = min(a, K, 31)                      # the cells hold depths 0-30: coverage counted to depth 30
                num[:a] += 1; den[:a] += 1
                if a < min(K, 31) and X["rt"][p, a] >= 0:
                    den[a] += 1
                ks.append(min(acc.get((X["fi"], int(X["an"][p])), 0), K) + 1 if p < len(X["an"]) else 1)
                p += (min(acc.get((X["fi"], int(X["an"][p])), 0), K) if p < len(X["an"]) else 0) + 1
        bs = " ".join(f"{100 * num[a:b + 1].sum() / den[a:b + 1].sum():5.1f}" if den[a:b + 1].sum() else "    -" for a, b in BANDS)
        nb = " ".join(f"{int(den[a:b + 1].sum()):5d}" for a, b in BANDS)
        print(f"{s} tree K{K + 1:<4d} kept {np.mean(ks):7.3f}  walk-strict 0|1-2|3-6|7-10|11-14|15-22|23-30: {bs}   n: {nb}")
