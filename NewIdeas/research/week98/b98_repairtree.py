#!/usr/bin/env python3
"""BREAK98: IN-CHECK REPAIR on TREES (NewIdeas, Tue 17 Nov; time = the commit's). Tree-32 = the first 31 best-first pops
of BigMoE's real 127-node k8 dumps (EAGLE-2-31, self-check 16.26 HE+ / 9.03 MBPP). Where the tree's coverage of the true
path ends (accepted depth e < 31), a repair round re-predicts depth e with the TRUE state (cells: the depth-0 guess of
anchor p + e) and, if right, the dead rows are reused for a re-drafted tail = anchor p + e's own tree with the remaining
row budget (31 - e pops; its depth 0 is the repaired token). UPPER BOUND: perfect location, true-state quality, the dead
rows fully recycled. q = the share of repairs reaching true-state quality (the rest end the block as today).
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_repairtree.py"""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tree_audit as TA  # noqa: E402
import adaptk_draft as AD  # noqa: E402

NP = 127
SETS = {"he": ("heplus24", 13.77), "mbpp": ("mbppbank", 6.92)}

def acc_table(tag):
    z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
    fi, an, nd, dp, ot = (z[k] for k in ("fi", "an", "node", "depth", "on_true"))
    key = fi.astype(np.int64) * 10_000_000 + an.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    DP = np.zeros((len(uk), NP), np.int64); OT = np.zeros((len(uk), NP), bool)
    DP[inv, nd] = dp; OT[inv, nd] = ot
    acc = np.maximum.accumulate(np.where(OT, DP + 1, 0), axis=1)      # accepted depth with j + 1 pops
    return {int(k): i for i, k in enumerate(uk)}, acc

def A(acc, i, npops):
    return 0 if npops <= 0 else int(min(acc[i, npops - 1], npops))

def walk(W, S, acc, R, q, rng, budget=31):
    kept = []
    for X, ids in zip(W, S):
        rt = X["rt"]; n = len(ids); p = 1
        while p < n:
            i = ids[p]
            if i < 0:
                kept.append(1); p += 1; continue
            a = A(acc, i, budget); base = 0; left = R
            while a < budget and left > 0:
                e = base + a                                          # absolute failure depth
                if p + e >= n or e >= budget or rt[p + e, 0] < 0: break
                if rng is not None and rng.random() >= q: break
                left -= 1
                if rt[p + e, 0] != 1: break                           # the true-state re-prediction is wrong too
                j = ids[p + e] if p + e < n else -1
                if j < 0: a = e + 1 - base; break                     # repaired token counted, no continuation data
                rem = budget - e                                      # rows left for the re-drafted tail
                a2 = A(acc, j, rem)                                   # anchor p+e's tree; its depth 0 is the repaired token
                a2 = max(a2, 1)                                       # the repair itself is right
                base, a = e, a2
                if base + a >= budget: break
            tot = min(base + a, budget)
            kept.append(tot + 1); p += tot + 1
    return float(np.mean(kept))

def main():
    for name, (tag, pub) in SETS.items():
        W = TA.load_windows(tag, pub)
        idx, acc = acc_table(tag)
        S = AD.steps(W, idx)
        base = walk(W, S, acc, 0, 1.0, None)
        print(f"== {name} ({tag}): tree-32 (EAGLE-2-31) {base:.2f} (published 16.26 / 9.03)")
        for R in (1, 2, 3):
            row = []
            for q in (1.0, 0.5, 0.3):
                v = walk(W, S, acc, R, q, np.random.default_rng(17) if q < 1 else None)
                row.append(f"q{q} {v:.2f} ({100 * (v / base - 1):+.0f} %)")
            print(f"  {R} repair round(s): " + "; ".join(row))

if __name__ == "__main__":
    main()
