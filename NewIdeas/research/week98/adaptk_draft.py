#!/usr/bin/env python3
"""ADAPTIVE K, stage 2 (NewIdeas, Tue 17 Nov 01:0x): adaptk_oracle.py with the DRAFT COST counted and an implementable
policy. Stage 1 (drafting hidden) gave HE+ ORACLE +36 % / POLICY +16.5 % over the best fixed K (e14), MBPP +12.4 % /
-0.5 %; but its big trees (96-128 rows, ~40 levels) cost real drafting.

Cost per check: T(K, anchor) = check(K) + L x depth(anchor, K - 1 pops), where depth = the tallest of the drafted pops
(one draft forward per level). L per level (ms): 0 (hidden), 0.40 (8-bit + ~32K-screen head, MTP Head S16-S19 est),
0.66 and 0.92 (today's head, WEEK98's 26.2-36.3 ms for K128's 39.6 levels).

POLICY (implementable, one parameter): best-first drafting pops nodes in non-increasing path_p; keep popping while the
next pop's path_p >= tau (at least 15 pops); round the rows (pops + 1) UP to the next kernel size in ROWS and fill the
spare slots with the next pops; verify. Under calibration the optimal rule has this form (include a node iff its
path_p >= lam x the marginal cost of its row). tau is tuned per set (in-sample) and also CROSS-applied (the other set's
tau), which is the honest reading.
ORACLE: hindsight DP over the per-check K choice with the same costs (the ceiling).
KILL: honest cross-set POLICY gain < 3 % over the best fixed K on both sets at L 0.40.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_draft.py
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_audit as TA  # noqa: E402
from adaptk_oracle import ROWS, MODELS  # noqa: E402

LS = [0.0, 0.40, 0.66, 0.92]
TAUS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]
NP = 127
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def dense(dump):
    """per anchor key: path_p [NP], depth [NP], on_true [NP] in pop order; accepted a_K and tree height h_K per rows"""
    fi, an, nd, dp, ot, pp = (dump[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p"))
    key = fi.astype(np.int64) * 10_000_000 + an.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    PP = np.zeros((n, NP), np.float32); DP = np.zeros((n, NP), np.int16); OT = np.zeros((n, NP), bool)
    PP[inv, nd] = pp; DP[inv, nd] = dp; OT[inv, nd] = ot
    acc_run = np.maximum.accumulate(np.where(OT, DP.astype(np.int64) + 1, 0), axis=1)   # accepted depth with j+1 pops
    h_run = np.maximum.accumulate(DP.astype(np.int64) + 1, axis=1)                      # tree height with j+1 pops
    A = {r: np.minimum(acc_run[:, r - 2], r - 1) for r in ROWS}
    Hh = {r: h_run[:, r - 2] for r in ROWS}
    mono = float(np.mean(np.all(np.diff(PP, axis=1) <= 1e-6, axis=1)))
    return {int(k): i for i, k in enumerate(uk)}, PP, A, Hh, mono


def steps(W, idx):
    """the anchor index at every walk position (or -1), per window"""
    out = []
    for X in W:
        ids = np.full(X["n"], -1, np.int64)
        for p in range(1, min(X["n"], len(X["an"]))):
            ids[p] = idx.get(X["fi"] * 10_000_000 + int(X["an"][p]), -1)
        out.append(ids)
    return out


def walk(S, choose, A, cost):
    kept = tm = 0.0
    use = {r: 0 for r in ROWS}
    for ids in S:
        n = len(ids); p = 1
        while p < n:
            i = ids[p]
            r = choose(i) if i >= 0 else 16
            a = int(A[r][i]) if i >= 0 else 0
            kept += a + 1; tm += cost(r, i); use[r] += 1; p += a + 1
    return 1000 * kept / tm, use


def oracle(S, A, cost, lam0, iters=10):
    lam = lam0 / 1000
    for _ in range(iters):
        kept = tm = 0.0
        for ids in S:
            n = len(ids)
            V = np.zeros(n + 140); ch = np.zeros(n, np.int64)
            for p in range(n - 1, 0, -1):
                i = ids[p]; best = -1e18
                for r in ROWS:
                    a = int(A[r][i]) if i >= 0 else 0
                    v = (a + 1) - lam * cost(r, i) + (V[p + a + 1] if p + a + 1 < n else 0.0)
                    if v > best:
                        best, ch[p] = v, r
                V[p] = best
            p = 1
            while p < n:
                i = ids[p]; r = int(ch[p]); a = int(A[r][i]) if i >= 0 else 0
                kept += a + 1; tm += cost(r, i); p += a + 1
        new = kept / tm
        if abs(new - lam) < 1e-12:
            break
        lam = new
    return 1000 * lam


def main():
    res = {}
    say("ADAPTIVE K stage 2: draft cost L ms per level counted; POLICY = pop while path_p >= tau, rows rounded up")
    data = {}
    for s, (tag, _rt, pub32, _p16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
        idx, PP, A, Hh, mono = dense({k: z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p")})
        S = steps(W, idx)
        npop = {t: np.maximum(15, (PP >= t).sum(1)) for t in TAUS}
        rows_of = {t: np.array([next(r for r in ROWS if r - 1 >= k) for k in npop[t]]) for t in TAUS}
        data[s] = (S, A, Hh, rows_of)
        say(f"=== {s}: {len(idx)} anchors; path_p non-increasing in pop order on {100 * mono:.1f} % of anchors")
    for m, T in MODELS.items():
        for L in LS:
            best_tau = {}
            for s, (S, A, Hh, rows_of) in data.items():
                cost = (lambda r, i, T=T, L=L, Hh=Hh: T[r] + (L * float(Hh[r][i]) if i >= 0 else 0.0))
                fx = {r: walk(S, (lambda i, r=r: r), A, cost)[0] for r in ROWS}
                rb = max(fx, key=fx.get)
                pol = {t: walk(S, (lambda i, t=t, ro=rows_of[t]: int(ro[i])), A, cost) for t in TAUS}
                tb = max(pol, key=lambda t: pol[t][0]); best_tau[s] = tb
                o = oracle(S, A, cost, fx[rb])
                res.setdefault(m, {}).setdefault(str(L), {})[s] = {"fixed": fx, "best_fixed": rb, "oracle": o,
                                                                    "policy": {str(t): v[0] for t, v in pol.items()},
                                                                    "tau_in": tb, "use_in": pol[tb][1]}
                say(f"[{m} L {L:.2f}] {s}: best fixed {rb} rows {fx[rb]:.1f} (16: {fx[16]:.1f}, 32: {fx[32]:.1f}) | ORACLE "
                    f"{o:.1f} ({100 * (o / fx[rb] - 1):+.1f} %) | POLICY tau {tb} {pol[tb][0]:.1f} "
                    f"({100 * (pol[tb][0] / fx[rb] - 1):+.1f} %) use {pol[tb][1]}")
            for s, (S, A, Hh, rows_of) in data.items():   # cross-set tau: the other set's best
                other = [x for x in data if x != s][0]; t = best_tau[other]
                cost = (lambda r, i, T=T, L=L, Hh=Hh: T[r] + (L * float(Hh[r][i]) if i >= 0 else 0.0))
                v = res[m][str(L)][s]["policy"][str(t)]; fb = res[m][str(L)][s]["fixed"][res[m][str(L)][s]["best_fixed"]]
                res[m][str(L)][s]["policy_cross"] = {"tau": t, "tps": v}
                say(f"[{m} L {L:.2f}] {s}: CROSS tau {t} (from {other}) {v:.1f} ({100 * (v / fb - 1):+.1f} % vs best fixed)")
    json.dump(res, open(os.path.join(HERE, "adaptk_draft.json"), "w"), indent=1)
    open(os.path.join(HERE, "adaptk_draft.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
