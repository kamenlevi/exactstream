"""BREAK98b limits, screen 1 (Lunatic, Tue 17 Nov 08:3x). CPU only, read-only, BigMoE's real 127-node t30b trees.

(a) "nearly free extra rows": what marginal cost per row m (ms) makes a bigger own-information tree pay, and what is
the best tok/s at each m? (b) the own-information placement limit: under the head's calibration, E[accepted] of a tree
= the sum of its nodes' path_p, so the best-first K pops are the optimal K-node own-information tree (path_p falls
along every path, so the top-K set is ancestor-closed); the K-curve below IS the limit of any own-information policy.
  kept(K) : the exact walk on the first K pops (D_K = on-true pops among the first K; next anchor = + kept), K <= 127.
  hindsight(K) : min(K, D_127), the true path inside the 127 candidates capped at K nodes (ORACLE-K).
  E-sum(K) : 1 + the mean of sum(path_p) over the first K pops at the walk's anchors (the calibrated expectation).
Price per check: a 32-row C6 + int8 check at the bound (dense 7.6 + experts 16.4 + cores 1.2 = 25.2 ms) + (K + 1 - 32)
rows x m + overhead 1.5 + draft (3 ms per 31 levels x the tree's depth levels, ~21 at K 31; grows as levels ~ 21 + 9 *
log2(K/31) [G]). Rows below 32 save the expert union by the code law (as IDEAS_1116c).
usage: ~/BigMoE/.venv/bin/python3 -I limit_rows.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/limit_rows.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
F = {1: 480, 2: 406, 3: 365, 4: 337, 6: 300, 8: 276, 12: 241, 16: 218, 24: 185, 32: 163}
def f_of(R):
    if R >= 32: return 163 * 0.75 ** np.log2(R / 32)
    ks = sorted(F)
    if R in F: return F[R]
    lo = max(k for k in ks if k <= R); hi = min(k for k in ks if k >= R)
    return F[lo] + (F[hi] - F[lo]) * (np.log(R) - np.log(lo)) / (np.log(hi) - np.log(lo))
U32 = 32 * f_of(32)
def cyc(K, m):
    R = K + 1
    if R <= 32:
        base = max(7.6, 4.9 * R / 32) + 16.4 * (R * f_of(R)) / U32 + 1.2 * R / 32
    else:
        base = 25.2 + (R - 32) * m
    levels = 21 + 9 * np.log2(max(K, 31) / 31) if K >= 31 else 21 * K / 31
    return base + 1.5 + 3.0 * levels / 31

def walk(acc, r0=1):
    p = r0; ks = []
    while p < len(acc):
        a = int(acc[p]); ks.append(a + 1); p += a + 1
    return ks

KS = (15, 23, 31, 47, 63, 95, 127)
for tag in ("heplus24", "mbppbank"):
    z = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
    fi, an, node, ont, pp = (z[k] for k in ("fi", "an", "node", "on_true", "path_p"))
    cz = np.load(f"{CE}/cells_t30b_{tag}.npz"); cd = cz["depth"]
    st = np.flatnonzero((cd == 0) & np.append(True, cd[:-1] != 0)); en = np.append(st[1:], len(cd))
    key = fi.astype(np.int64) * 10**7 + an.astype(np.int64)
    order = np.argsort(key, kind="stable"); ks_ = key[order]
    b = np.flatnonzero(np.append(True, ks_[1:] != ks_[:-1])); b = np.append(b, len(ks_))
    DK = {K: {} for K in KS}; ES = {K: {} for K in KS}
    for b0, b1 in zip(b[:-1], b[1:]):
        idx = order[b0:b1]; k = int(key[idx[0]])
        for K in KS:
            sel = node[idx] < K
            DK[K][k] = int((ont[idx] & sel).sum()); ES[K][k] = float(pp[idx][sel].sum())
    res = {K: [] for K in KS}; hind = {K: [] for K in KS}; esum = {K: [] for K in KS}
    for s, t in zip(st, en):
        f0 = int(cz["fi"][s]); rows = cz["row"][s:t][cd[s:t] == 0]; ans = rows - 1; n = len(ans) + 1
        for K in KS:
            a = np.zeros(n, int); h = np.zeros(n, int)
            for i, x in enumerate(ans):
                kk = f0 * 10**7 + int(x)
                if kk in DK[K]: a[i] = DK[K][kk]; h[i] = min(K, DK[127][kk])
            res[K] += walk(a); hind[K] += walk(h)
            # calibrated expectation at the same walk's anchors
            p = 1
            while p < n:
                kk = f0 * 10**7 + int(ans[p - 1]) if p - 1 < len(ans) else None
                if kk is not None and kk in ES[K]: esum[K].append(1 + ES[K][kk])
                p += int(a[p]) + 1 if p < n else 1
    say(f"== {tag}: own-information trees (best-first pops), exact walk")
    say("   K pops (rows)   kept   hindsight-K   E-sum(path_p)+1 | tok/s at the bound for marginal row cost m = 0.30 / 0.20 / 0.10 / 0.05 ms")
    for K in KS:
        kp = np.mean(res[K]); hk = np.mean(hind[K]); es = np.mean(esum[K]) if esum[K] else float("nan")
        tps = " / ".join(f"{1000*kp/cyc(K, m):4.0f}" for m in (0.30, 0.20, 0.10, 0.05))
        say(f"   {K:4d} ({K+1:3d})      {kp:6.2f}   {hk:6.2f}        {es:6.2f}        | {tps}")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
