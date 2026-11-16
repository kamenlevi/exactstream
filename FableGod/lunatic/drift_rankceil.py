"""DRIFT RANK CEILING (Lunatic, THINK_1116c, Mon 16 Nov 18:3x). CPU only, read-only, one process, < 1 GB.
How much of the own -> true gap is RANKING (the 125B's path is among the head's own candidates, just not selected)
vs INFORMATION (the head never proposes it)? On BigMoE's real t30b trees (tree_t30b_{tag}_n127k8.npz: 127 best-first
pops per anchor, on_true flags) under the exact walk (certify98-style: kept = accepted + 1, next anchor = + kept):
  - EAGLE-2-31: the first 31 pops (must reproduce the real tree-32 kept: HE+ 16.26 / MBPP 9.03);
  - ORACLE-31: the true path's nodes among the 127 pops, capped at 31 (any 31-node selection from the head's own
    127 candidates can't beat this);
  - K127: all 127 pops (128 rows).
  - LAB: the true-state chain (= depth-0 guess of anchor a+d; from the cells, cap 31).
Plus the head's calibration by depth on the 127 pops: mean edge p (path_p / parent path_p) vs P(on_true | parent
on_true), and the share of on-true children that are the head's rank-1 / rank-2 / lower edge at each depth.
usage: ~/BigMoE/.venv/bin/python3 -I drift_rankceil.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/drift_rankceil.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def walk(acc, r0=1):
    p = r0; ks = []
    while p < len(acc):
        a = int(acc[p]); ks.append(a + 1); p += a + 1
    return ks

for tag, pub in (("heplus24", 16.263), ("mbppbank", 9.031)):
    z = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
    fi, an, node, par, dep, ont, pp = (z[k] for k in ("fi", "an", "node", "parent", "depth", "on_true", "path_p"))
    # anchors per window from the cells (the walk's anchor rows), as tree_cc32 does
    cz = np.load(f"{CE}/cells_t30b_{tag}.npz"); cd = cz["depth"]
    st = np.flatnonzero((cd == 0) & np.append(True, cd[:-1] != 0)); en = np.append(st[1:], len(cd))
    key = fi.astype(np.int64) * 10**7 + an.astype(np.int64)
    order = np.argsort(key, kind="stable"); ks_ = key[order]
    b = np.flatnonzero(np.append(True, ks_[1:] != ks_[:-1])); b = np.append(b, len(ks_))
    D31, D127 = {}, {}
    for b0, b1 in zip(b[:-1], b[1:]):
        idx = order[b0:b1]; k = int(key[idx[0]])
        D31[k] = int((ont[idx] & (node[idx] < 31)).sum()); D127[k] = int(ont[idx].sum())
    W = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    res = {"EAGLE-2-31": [], "ORACLE-31": [], "K127": [], "LAB chain-32": [], "own chain-32": []}
    for wi, (s, t) in enumerate(zip(st, en)):
        f0 = int(cz["fi"][s]); rows = cz["row"][s:t][cd[s:t] == 0]; ans = rows - 1; n = len(ans) + 1
        a31 = np.zeros(n, int); o31 = np.zeros(n, int); k127 = np.zeros(n, int)
        for i, a in enumerate(ans):
            k = f0 * 10**7 + int(a)
            if k in D31:
                a31[i] = D31[k]; o31[i] = min(31, D127[k]); k127[i] = D127[k]
        res["EAGLE-2-31"] += walk(a31); res["ORACLE-31"] += walk(o31); res["K127"] += walk(k127)
        rt = W[wi][0]
        res["own chain-32"] += C98.kept_walk(rt, C98.R0)
        # LAB: the true-state guess at depth d from anchor i is the depth-0 cell of anchor i + d
        d0 = rt[:, 0]; lab = np.full(rt.shape, -1, np.int8)
        for j in range(rt.shape[1]):
            c = len(d0) - j
            if c > 0: lab[:c, j] = d0[j:j + c]
        lab[rt < 0] = -1
        res["LAB chain-32"] += C98.kept_walk(lab, C98.R0)
    say(f"== {tag}")
    for k, v in res.items():
        say(f"   {k:14s} kept/check {np.mean(v):6.2f}" + (f"   (published {pub})" if k == "EAGLE-2-31" else ""))
    # calibration by depth: children of on-true parents
    pidx = {}
    for b0, b1 in zip(b[:-1], b[1:]):
        idx = order[b0:b1]
        for i in idx: pidx[(int(key[i]), int(node[i]))] = i
    edge_p, child_ok, d_of, rk = [], [], [], []
    for b0, b1 in zip(b[:-1], b[1:]):
        idx = order[b0:b1]; k = int(key[idx[0]])
        kids = {}
        for i in idx:
            if par[i] >= 0: kids.setdefault(int(par[i]), []).append(i)
        for i in idx:
            pi = int(par[i])
            if pi < 0:
                parent_ok, ppar = True, 1.0
            else:
                j = pidx.get((k, pi))
                if j is None: continue
                parent_ok, ppar = bool(ont[j]), float(pp[j])
            if not parent_ok: continue
            edge_p.append(float(pp[i]) / max(ppar, 1e-12)); child_ok.append(bool(ont[i])); d_of.append(int(dep[i]))
    edge_p, child_ok, d_of = np.array(edge_p), np.array(child_ok), np.array(d_of)
    say("   calibration of popped children of on-true parents (edge p vs on-true rate), by depth band:")
    for lo, hi in ((0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 40)):
        m = (d_of >= lo) & (d_of <= hi)
        top = m & (edge_p >= 0.5)
        say(f"     depth {lo:2d}-{hi:2d}: n {m.sum():6d}  mean edge p {edge_p[m].mean():.3f}  on-true {child_ok[m].mean():.3f}"
            f" | edges p>=0.5: n {top.sum():6d} p {edge_p[top].mean():.3f} on-true {child_ok[top].mean():.3f}")
open(OUT, "w").write("\n".join(lines) + "\n")
