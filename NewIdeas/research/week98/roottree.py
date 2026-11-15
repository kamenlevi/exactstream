"""ROOT-TREE (NewIdeas, Sun 15 Nov 02:5x): kept/32 is lost at the walk's check starts (CERTIFY98 amendment 4). There, the
head's top-1 at depth 0 is ~92 % (HE+) / ~90 % (MBPP), and the miss is often its 2nd choice. Policy at a FIXED 32-row
check: the main chain gets 32 - L rows, and a branch of L rows starts from the head's rank-1 token at depth 0. If the
125B's token is the head's rank-1 there, the branch carries the walk.
The branch continuation is read from the same teacher-forced cells (true tokens fed, which a branch drafting from the
correct 2nd choice also sees). Needs the cells' 'rank' field (rank of the 125B's top-1 in the head's order; 0 = right).
Also reports the walk-strict curve at check starts (P(right at j | right at all < j)). CPU. usage: python -I roottree.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C
def walk(rt, rk, L, cap=31):
    """one window's decode walk at a 32-row check: chain rows = cap - L (+ the bonus), root branch of L rows"""
    sub, rks = rt[C.R0:], rk[C.R0:]; out = []; p = 0
    while p < sub.shape[0]:
        main = cap - L if L else cap
        if sub[p, 0] == 1 or L == 0 or rks[p, 0] != 1:
            k = 0
            while k < main and sub[p, k] == 1: k += 1
        else:                                   # the 125B's token is the head's rank 1: the branch is right at depth 0
            k = 1
            while k < L and sub[p, k] == 1: k += 1
        out.append(k + 1); p += k + 1
    return out
for nm in ("heplus", "mbpp100"):
    d = np.load(os.path.expanduser(f"~/BigMoE/data/deepdraft/ceiling/cells_t30b_{nm}.npz"))
    assert ((d["rank"] == 0) == d["right"].astype(bool)).mean() > 0.999, "rank 0 must mean right"
    W = C.windows(os.path.expanduser(f"~/BigMoE/data/deepdraft/ceiling/cells_t30b_{nm}.npz"))
    # rank arrays with the same layout (re-run the layout on 'rank' by swapping it in as 'm')
    dep, rank = d["depth"].astype(int), d["rank"].astype(float)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    RK = []
    for (s, t), (rt, _) in zip(zip(st, en), W):
        dw, rw = dep[s:t], rank[s:t]; rk = np.full(rt.shape, -1.0)
        for j in range(31):
            k = dw == j; rk[:k.sum(), j] = rw[k]
        RK.append(rk)
    # walk-strict at check starts (L = 0 chain)
    hit = np.zeros(31); elig = np.zeros(31)
    for rt, _ in W:
        sub = rt[C.R0:]; p = 0
        while p < sub.shape[0]:
            alive = True
            for j in range(31):
                if sub[p, j] < 0: break
                if alive: elig[j] += 1; hit[j] += sub[p, j] == 1
                alive = alive and sub[p, j] == 1
            k = 0
            while k < 31 and sub[p, k] == 1: k += 1
            p += k + 1
    ws = 100 * hit / np.maximum(elig, 1)
    print(f"== {nm}: walk-strict at check starts, depth 0-14: " + " ".join(f"{x:.1f}" for x in ws[:15]))
    print(f"   depth 15-30: " + " ".join(f"{x:.1f}" for x in ws[15:31]))
    top2 = []; 
    for (rt, _), rk in zip(W, RK):
        sub, rks = rt[C.R0:], rk[C.R0:]; p = 0
        while p < sub.shape[0]:
            if sub[p, 0] >= 0: top2.append(rks[p, 0] <= 1)
            k = 0
            while k < 31 and sub[p, k] == 1: k += 1
            p += k + 1
    print(f"   head's top-2 contains the 125B's token at check starts: {100 * np.mean(top2):.2f} % (top-1 {ws[0]:.2f} %)")
    for L in (0, 2, 4, 8, 12, 16):
        ks = [k for (rt, _), rk in zip(W, RK) for k in walk(rt, rk, L)]
        print(f"   root branch L = {L:2d} rows (main chain {31 - L if L else 31}): kept/32 {np.mean(ks):.3f}")
