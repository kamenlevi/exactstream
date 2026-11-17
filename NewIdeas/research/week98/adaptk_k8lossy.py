#!/usr/bin/env python3
"""ADAPTIVE K DOWN on the LOSSY line (NewIdeas, Tue 17 Nov; the time is the commit's): adaptk_k8.py's PEEK rule (8 rows
iff the 8th pop's path_p < tau8, else ADAPTIVE K as frozen) under close-call CC(1) acceptance (tree_cc32's rule, as in
adaptk_lossy.py), on LADDER v4's costs. R8 = the 8-row check / the 16-row check, bracketed [est] as in adaptk_k8.py; on
the byte floor it is the bytes ratio (dense ~66 MB + union x 2.135 MB per layer; union 73.2 -> ~46.5 per layer) = 0.744.
tau8 is cross-fit (picked on the other set). Information for the build; the exact screen (adaptk_k8.txt) is the kill.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_k8lossy.py
"""
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adaptk_lossy as AL  # noqa: E402

TAU8S = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
FLOOR_R8 = 0.744
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def arrays3(tag, W):
    """adaptk_lossy.anchor_arrays with an 8-row (7-pop) column and the 8th pop's path_p"""
    z = np.load(f"{AL.CE}/tree_t30b_{tag}_n127k8.npz")
    fi, an, nd, pa, dp, pp, ot = (z[k] for k in ("fi", "an", "node", "parent", "depth", "path_p", "on_true"))
    order = np.lexsort((nd, an, fi))
    fi, an, nd, pa, dp, pp, ot = (a[order] for a in (fi, an, nd, pa, dp, pp, ot))
    b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    byfi = {X["fi"]: X for X in W}
    out = {X["fi"]: {**{f"{a}{k}": np.full(X["n"], -1) for a in ("ex", "cc") for k in ("8", "16", "32")},
                     **{f"h{k}": np.zeros(X["n"]) for k in ("8", "16", "32")},
                     "p7": np.ones(X["n"]), "n09": np.zeros(X["n"], np.int64)} for X in W}
    for b0, b1 in zip(b[:-1], b[1:]):
        f_, a_ = int(fi[b0]), int(an[b0])
        if f_ not in rowof or a_ not in rowof[f_]:
            continue
        r = rowof[f_][a_]; X = byfi[f_]; o = out[f_]
        n_, p_, d_, ppp, t_ = nd[b0:b1], pa[b0:b1], dp[b0:b1].astype(np.int64), pp[b0:b1].astype(np.float64), ot[b0:b1].astype(bool)
        o["n09"][r] = int((ppp >= 0.9).sum())
        o["p7"][r] = float(ppp[n_ == 7][0]) if (n_ == 7).any() else 0.0
        R_, DL_ = X["R"][r], X["DL"][r]
        for NODES, key in ((7, "8"), (15, "16"), (31, "32")):
            selm = n_ < NODES; hit = t_ & selm
            A_t = int(d_[hit].max()) + 1 if hit.any() else 0
            o["ex" + key][r] = A_t
            o["h" + key][r] = int(d_[selm].max()) + 1 if selm.any() else 0
            cc = A_t
            if A_t < min(AL.DEP, NODES) and R_[A_t] > 0 and not np.isnan(DL_[A_t]) and DL_[A_t] >= -AL.DELTA:
                pid = -1 if A_t == 0 else int(n_[np.flatnonzero(hit & (d_ == A_t - 1))[0]])
                kids = np.flatnonzero(selm & (p_ == pid) & (d_ == A_t))
                if len(kids):
                    cur = int(kids[np.argmax(ppp[kids])]); cont = 0; d2 = A_t
                    while True:
                        ch = np.flatnonzero(selm & (p_ == n_[cur]))
                        if not len(ch):
                            break
                        cur = int(ch[np.argmax(ppp[ch])]); d2 += 1
                        if d2 >= AL.DEP or R_[d2] < 0:
                            break
                        if R_[d2] == 0 or (not np.isnan(DL_[d2]) and DL_[d2] >= -AL.DELTA):
                            cont += 1
                        else:
                            break
                    cc = A_t + 1 + cont
            o["cc" + key][r] = cc
    return out


def walk(W, O, acc, choose, cost):
    res = []
    for X in W:
        o = O[X["fi"]]; n = X["n"]; p = 1; kept = tm = 0.0
        while p < n:
            r = choose(o, p)
            a = int(o[acc + str(r)][p])
            if a < 0:
                r, a, h = 16, 0, 0.0
            else:
                h = o["h" + str(r)][p]
            kept += a + 1; tm += cost(r, h); p += a + 1
        res.append((kept, tm))
    return np.array(res)


def tps(X):
    return 1000 * X[:, 0].sum() / X[:, 1].sum()


def main():
    data = {}
    for name, (tag, pub) in AL.SETS.items():
        W = AL.load_cells(tag, pub)
        data[name] = (W, arrays3(tag, W))
    k2 = (lambda o, p: 32 if o["n09"][p] > 15 else 16)
    peek = (lambda t8: (lambda o, p: 8 if o["p7"][p] < t8 else k2(o, p)))
    for cname, (t16, t32, L) in AL.COSTS.items():
        r8s = (FLOOR_R8,) if cname == "floor25" else (0.71, 0.78, 0.86)
        for r8 in r8s:
            t8 = r8 * (t16 - 4.0) + 4.0
            TT = {8: t8, 16: t16, 32: t32}
            cost = (lambda r, h, TT=TT, L=L: TT[r] + L * h)
            for acc in ("cc", "ex"):
                ev = {}
                for name, (W, O) in data.items():
                    ev[name] = (walk(W, O, acc, k2, cost), {t: walk(W, O, acc, peek(t), cost) for t in TAU8S})
                for name in data:
                    other = "mbpp" if name == "he" else "he"
                    tx = max(TAU8S, key=lambda t: tps(ev[other][1][t]))
                    base, sw = ev[name]
                    g = tps(sw[tx]) / tps(base) - 1
                    say(f"{cname:9s} {acc} R8 {r8:.3f} (T8 {t8:.2f} / T16 {t16:.2f} / T32 {t32:.2f} ms, L {L}): {name:4s} "
                        f"KERN2 {tps(base):.1f} -> PEEK (tau8 {tx}, cross-fit) {tps(sw[tx]):.1f} = {100 * g:+.1f} %")
    open(os.path.join(HERE, "adaptk_k8lossy.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
