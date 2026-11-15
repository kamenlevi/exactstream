"""CALIBRATED TREE SCORES (NewIdeas, Sun 15 Nov 14:4x). It follows tree_shape_cc: ORACLE-15 leaves x1.160 MBPP / x1.076
HE+ of headroom over EAGLE-2-16, which no shape rule captured. Does the head's edge probability, recalibrated, pick better
15 nodes?
- A node's edge prob e = path_p(node) / path_p(parent). Label: on_true, for nodes whose parent is on the true path
  (or is the root).
- Calibration: isotonic (pool-adjacent-violators over 20 quantile bins) of the label on e, per (depth bucket
  {0, 1-2, 3-6, 7-14, 15+} x walk-start flag).
- CROSS-FIT: fit on the even windows and score the odd, then swap, so no window scores itself.
- The calibrated path score is the product of cal(e) along the path. It is monotone, so its top-15 is a closed
  subtree.
- Kept/check uses the S9c walk (R0 = 1), as in tree_shape_cc.
- LINE (written first): PASS if calibrated EAGLE-2-16 beats raw EAGLE-2-16 kept by >= 2 % on BOTH sets -> a
  calibration table in the tree builder (exact, free). BETWEEN at +1-2 %. KILL < +1 %.
usage: BigMoE venv python -I -B calib_tree.py"""
import os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98

CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
SETS = {"mbpp": "mbppbank", "he": "heplus24"}
NODES = 15
BUCK = [(0, 0), (1, 2), (3, 6), (7, 14), (15, 99)]
OUT = os.path.expanduser("~/NewIdeas/research/week98/calib_tree.txt")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
def bucket(d):
    for i, (a, b) in enumerate(BUCK):
        if a <= d <= b: return i
    return len(BUCK) - 1

def walk(acc, n, r0=C98.R0):
    p = r0; ks = nc = 0
    while p < n:
        a = int(acc[p]); ks += a + 1; nc += 1; p += a + 1
    return ks, nc

def pav(x, y, nb=20):
    """isotonic fit of y on x over quantile bins (pool adjacent violators); returns bin edges and fitted values"""
    if len(x) < 50: return np.array([0.0, 1.0]), np.array([y.mean() if len(y) else 0.5])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    w = np.bincount(idx, minlength=len(edges) - 1).astype(float); v = np.bincount(idx, y, minlength=len(edges) - 1)
    keep = w > 0; w, v = w[keep], v[keep] / w[keep]; e_lo = edges[:-1][keep]
    blocks = [[v[0], w[0], 1]]
    for vi, wi in zip(v[1:], w[1:]):
        blocks.append([vi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop(); v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    fit = np.concatenate([[b[0]] * b[2] for b in blocks])
    return e_lo, fit

for s, tag in SETS.items():
    cz = np.load(f"{CE}/cells_t30b_{tag}.npz"); tz = np.load(f"{CE}/tree_t30b_{tag}.npz")
    Wc = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    dep, row, fi = cz["depth"].astype(int), cz["row"].astype(np.int64), cz["fi"].astype(np.int64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0))
    wins = []                                                  # per window: fi, n, anchor -> layout row
    for wi, s0 in enumerate(st):
        e0 = st[wi + 1] if wi + 1 < len(st) else len(dep)
        dw = dep[s0:e0]; an = row[s0:e0][dw == 0] - 1
        wins.append({"fi": int(fi[s0]), "n": Wc[wi][0].shape[0], "row": {int(a): i for i, a in enumerate(an)}})
    byfi = {w["fi"]: w for w in wins}
    nfi, nan_, nnode, ndep, ontrue, pp, par = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p", "parent"))
    afi, aan, aws = tz["a_fi"], tz["a_an"], tz["a_walk_start"].astype(bool)
    wsflag = {(int(f), int(a)): bool(w) for f, a, w in zip(afi, aan, aws)}
    order = np.lexsort((nnode, nan_, nfi))
    nfi, nan_, nnode, ndep, ontrue, pp, par = (a[order] for a in (nfi, nan_, nnode, ndep, ontrue, pp, par))
    bounds = np.append(np.flatnonzero(np.append(True, (np.diff(nfi) != 0) | (np.diff(nan_) != 0))), len(nfi))
    A = []                                                     # per anchor: arrays + edge probs + labels mask
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        f_, a_ = int(nfi[b0]), int(nan_[b0])
        nd, dp, ot, ppp, pa = nnode[b0:b1], ndep[b0:b1], ontrue[b0:b1].astype(bool), pp[b0:b1].astype(np.float64), par[b0:b1]
        idx = {int(n): i for i, n in enumerate(nd)}
        pidx = np.array([idx.get(int(p_), -1) for p_ in pa])
        e = np.where(pidx >= 0, ppp / np.maximum(ppp[np.maximum(pidx, 0)], 1e-30), ppp)
        ptrue = np.where(pidx >= 0, ot[np.maximum(pidx, 0)], True)    # parent on the true path (root = true)
        A.append({"fi": f_, "r": byfi[f_]["row"][a_], "dp": dp, "ot": ot, "e": np.clip(e, 0, 1), "pidx": pidx,
                  "ptrue": ptrue, "ws": wsflag[(f_, a_)], "nd": nd})
    half = {w["fi"]: (k % 2) for k, w in enumerate(wins)}
    def fit(h):
        tabs = {}
        for b in range(len(BUCK)):
            for wsf in (False, True):
                xs, ys = [], []
                for X in A:
                    if half[X["fi"]] != h or X["ws"] != wsf: continue
                    m = X["ptrue"] & (np.array([bucket(int(d)) for d in X["dp"]]) == b)
                    xs.append(X["e"][m]); ys.append(X["ot"][m].astype(float))
                x = np.concatenate(xs) if xs else np.zeros(0); y = np.concatenate(ys) if ys else np.zeros(0)
                tabs[(b, wsf)] = pav(x, y)
        return tabs
    def cal(tabs, b, wsf, e):
        lo, v = tabs[(b, wsf)]
        return v[np.clip(np.searchsorted(lo, e, side="right") - 1, 0, len(v) - 1)]
    T = {0: fit(0), 1: fit(1)}
    acc_raw = {w["fi"]: np.zeros(w["n"], np.int64) for w in wins}; acc_cal = {w["fi"]: np.zeros(w["n"], np.int64) for w in wins}
    for X in A:
        tabs = T[1 - half[X["fi"]]]                            # cross-fit: the other half's table
        b = np.array([bucket(int(d)) for d in X["dp"]])
        ce = np.array([cal(tabs, int(bi), X["ws"], ei) for bi, ei in zip(b, X["e"])]).astype(float).ravel()
        ce = np.clip(ce, 1e-9, 1.0)
        score = np.empty(len(ce))
        for i in range(len(ce)):                               # nodes are in pop order: a parent precedes its child
            score[i] = ce[i] * (score[X["pidx"][i]] if X["pidx"][i] >= 0 else 1.0)
        sel_raw = X["nd"] < NODES
        sel_cal = np.zeros(len(ce), bool); sel_cal[np.argsort(-score, kind="stable")[:NODES]] = True
        for sel, acc in ((sel_raw, acc_raw), (sel_cal, acc_cal)):
            h = X["ot"] & sel
            acc[X["fi"]][X["r"]] = int(X["dp"][h].max()) + 1 if h.any() else 0
    res = {}
    for nm, acc in (("raw EAGLE-2-16", acc_raw), ("calibrated EAGLE-2-16", acc_cal)):
        ks = nc = 0
        for w in wins:
            k_, n_ = walk(acc[w["fi"]], w["n"]); ks += k_; nc += n_
        res[nm] = ks / nc
    r = res["calibrated EAGLE-2-16"] / res["raw EAGLE-2-16"]
    say(f"== {s} ({tag}): raw EAGLE-2-16 kept/check {res['raw EAGLE-2-16']:.3f} | calibrated (cross-fit) "
        f"{res['calibrated EAGLE-2-16']:.3f} | x{r:.3f}")
open(OUT, "w").write("\n".join(lines) + "\n")
