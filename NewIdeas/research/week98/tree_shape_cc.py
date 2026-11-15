"""TREE SHAPE + TREES x CLOSE-CALL (NewIdeas, Sun 15 Nov 14:4x; the orchestrator's go 14:3x). CPU only, one process, small.
Data: BigMoE's real t30b EAGLE-2 tree dumps (tree_t30b_{heplus24,mbppbank}.npz, 32 best-first pops per anchor, on_true
flags), cells (dlp, m) and TOPK, in MTP Head's S9c layout (TrainLab sb98/tree_real125.py: load_cells, walk_acc, chain_acc
copied below, read-only). A 16-row check = 15 nodes; accepted = max{d + 1 : on_true node selected}; kept = accepted + 1;
the walk = certify98.kept_walk's (R0 = 1).
SCREEN 1, TREE SHAPE: re-select 15 of the 32 dumped nodes by a different score, then walk:
  - EAGLE-2 (the pop order; must reproduce S9c's real EAGLE-2-16 kept: 7.487 MBPP / 11.263 HE+);
  - DEPTH DISCOUNT score = path_p * lam^depth, lam 0.9 / 0.8 / 0.6 (wider, shallower trees);
  - ROOT TEMPER: the depth-0 edge probability p0 -> p0^(1/T), T 1.5 / 2 / 3 (more root siblings where the first guess is
    unsure: the check-start wall);
  - ORACLE-15: the true path's nodes among the 32, capped at 15 (headroom of any reshaping of these nodes).
  Every score is monotone along a path, so its top-15 is a closed subtree. Gains are split by the 125B's margin at the
  walk start's depth 0 (cells m: < 4 nats = unsure).
  LINE: PASS (a better shape exists) if any policy beats EAGLE-2-16's kept by >= 2 % on both sets; else EAGLE-2's
  path-probability order is the right shape for the build.
SCREEN 2, TREES x CLOSE-CALL (lossy; delta 1 nat, the passing delta): where does each lever rescue the chain's first miss
on the walk? Classes at the first miss depth d*: TREE (the 125B's token is a selected node under the true path),
CC (cells dlp >= -delta: the head's pick is within delta of the 125B's top-1), BOTH, NEITHER. Then kept for chain-16,
chain-16 + CC, tree-16, tree-16 + CC. For the CC arms we use the same approximation as the existing close-call walks: after
a close-call acceptance the walk goes on along the true text's cells. Tree arms are budget-correct: EAGLE-2-16's own 15
selected nodes. Tree + CC rescues the true path's first unselected depth only through the head's selected top-1 child
there, then follows that child's selected best-child chain (judged by the cells).
  LINE: ADDITIVE if gain(tree + CC) >= 0.8 x (gain(tree) + gain(CC)); else OVERLAPPING. Overlapping means the lossy
  line's gain shrinks once trees ship.
usage: BigMoE venv python -I -B tree_shape_cc.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
SETS = {"mbpp": ("mbppbank", 6.92), "he": ("heplus24", 13.77)}
DEP, K, NODES = 31, 8, 15
DELTA = 1.0
OUT = os.path.expanduser("~/NewIdeas/research/week98/tree_shape_cc.txt")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def load_cells(tag, published):                      # copied from TrainLab sb98/tree_real125.py (S9c), + m and dlp
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all(), f"{tag}: {k} differs between cells and topk"
    dep, right, yr = tz["depth"].astype(int), tz["right"].astype(int), tz["y_rank"].astype(int)
    yr = np.where((yr == 0) & (right == 0), 1, yr)
    row, fi = tz["row"].astype(np.int64), tz["fi"].astype(np.int64)
    mm, dl = cz["m"].astype(np.float32), cz["dlp"].astype(np.float32)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32 = [], []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); R = np.full((n0 + 1, DEP), -1, np.int16)
        M = np.full((n0 + 1, DEP), np.nan, np.float32); DL = np.full((n0 + 1, DEP), np.nan, np.float32)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; R[:c, j] = yr[s:t][m]; M[:c, j] = mm[s:t][m]; DL[:c, j] = dl[s:t][m]
        assert (rt == ref[wi][0]).all(), f"{tag}: window {wi} layout != certify98"
        R[rt < 0] = -1
        k32 += C98.kept_walk(rt, C98.R0, 31)
        an = row[s:t][dw == 0] - 1
        W.append({"rt": rt, "R": R, "M": M, "DL": DL, "fi": int(fi[s]), "an": an, "n": n0 + 1})
    say(f"SELF-CHECK {tag}: chain-32 {np.mean(k32):.2f} vs {published} -> {'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}")
    return W

def walk_acc(acc, n, r0=C98.R0):                       # copied from S9c
    p = r0; ks = nc = 0; starts = []
    while p < n:
        a = int(acc[p]); starts.append(p); ks += a + 1; nc += 1; p += a + 1
    return ks, nc, starts

def chain_acc(X, N):                                   # copied from S9c
    R = X["R"]
    k = np.where((R != 0).any(1), (R != 0).argmax(1), DEP)
    return np.minimum(k, N - 1)

RES = {}
for s, (tag, pub) in SETS.items():
    W = load_cells(tag, pub)
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz")
    nfi, nan_, nnode, ndep, ontrue, pp, par = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p", "parent"))
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    byfi = {X["fi"]: X for X in W}
    # per anchor arrays of its 32 nodes (pop order)
    order = np.lexsort((nnode, nan_, nfi))
    nfi, nan_, nnode, ndep, ontrue, pp, par = (a[order] for a in (nfi, nan_, nnode, ndep, ontrue, pp, par))
    bounds = np.flatnonzero(np.append(True, (np.diff(nfi) != 0) | (np.diff(nan_) != 0)))
    bounds = np.append(bounds, len(nfi))
    # depth-0 edge prob of each node = path_p of its depth-0 ancestor
    def policies():
        P = {"EAGLE-2": None}
        for lam in (0.9, 0.8, 0.6): P[f"discount {lam}"] = ("lam", lam)
        for T in (1.5, 2.0, 3.0): P[f"root temper {T}"] = ("T", T)
        P["ORACLE-15"] = ("oracle", None)
        return P
    POL = policies()
    acc = {name: {X["fi"]: np.zeros(X["n"], np.int64) for X in W} for name in POL}
    treepass = {X["fi"]: np.zeros((X["n"], DEP), bool) for X in W}     # EAGLE-2-16: on-true node selected at depth d
    tree_acc = {X["fi"]: np.zeros(X["n"], np.int64) for X in W}
    treecc_acc = {X["fi"]: np.zeros(X["n"], np.int64) for X in W}
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        f_, a_ = int(nfi[b0]), int(nan_[b0]); r = rowof[f_][a_]
        nd, dp, ot, ppp, pa = nnode[b0:b1], ndep[b0:b1], ontrue[b0:b1].astype(bool), pp[b0:b1].astype(np.float64), par[b0:b1]
        idx = {int(n): i for i, n in enumerate(nd)}
        p0 = np.empty(len(nd))
        for i in range(len(nd)):                       # walk up to the depth-0 ancestor
            j = i
            while dp[j] > 0 and int(pa[j]) in idx: j = idx[int(pa[j])]
            p0[i] = ppp[j]
        for name, pol in POL.items():
            if pol is None: sel = nd < NODES
            elif pol[0] == "lam": sel = np.zeros(len(nd), bool); sel[np.argsort(-(ppp * pol[1] ** dp), kind="stable")[:NODES]] = True
            elif pol[0] == "T": sel = np.zeros(len(nd), bool); sel[np.argsort(-(ppp * (p0 ** (1 / pol[1])) / np.maximum(p0, 1e-30)), kind="stable")[:NODES]] = True
            else: sel = ot.copy()
            hit = ot & sel
            a = int(dp[hit].max()) + 1 if hit.any() else 0
            acc[name][f_][r] = min(a, NODES) if pol is not None and pol[0] == "oracle" else a
        e = ot & (nd < NODES)
        treepass[f_][r, dp[e]] = True
        # SCREEN 2 tree arms, budget-correct: EAGLE-2-16's own selection (pops < 15)
        X = byfi[f_]; R_, DL_ = X["R"][r], X["DL"][r]
        selm = nd < NODES; hit = ot & selm
        A_t = int(dp[hit].max()) + 1 if hit.any() else 0
        tree_acc[f_][r] = A_t
        cc = A_t
        if A_t < min(DEP, NODES) and R_[A_t] > 0 and not np.isnan(DL_[A_t]) and DL_[A_t] >= -DELTA:
            pid = -1 if A_t == 0 else int(nd[np.flatnonzero(hit & (dp == A_t - 1))[0]])
            kids = np.flatnonzero(selm & (pa == pid) & (dp == A_t))
            if len(kids):
                cur = int(kids[np.argmax(ppp[kids])]); cont = 0; d2 = A_t
                while True:                           # follow the selected best-child chain below the close-called node
                    ch = np.flatnonzero(selm & (pa == nd[cur]))
                    if not len(ch): break
                    cur = int(ch[np.argmax(ppp[ch])]); d2 += 1
                    if d2 >= DEP or R_[d2] < 0: break
                    if R_[d2] == 0 or (not np.isnan(DL_[d2]) and DL_[d2] >= -DELTA): cont += 1
                    else: break
                cc = A_t + 1 + cont
        treecc_acc[f_][r] = cc
    say(f"== {s} ({tag}): 16-row check (15 nodes)")
    base = None; res = {}
    for name in POL:
        ks = nc = 0
        for X in W:
            k_, n_, _ = walk_acc(acc[name][X["fi"]], X["n"]); ks += k_; nc += n_
        kp = ks / nc; res[name] = kp
        if base is None: base = kp
        say(f"   {name:18s} kept/check {kp:7.3f}  x{kp / base:.3f} vs EAGLE-2-16")
    # gains by margin at the walk starts (EAGLE-2's own walk)
    for name in [n for n in POL if n not in ("EAGLE-2",)]:
        g = {"sure": [0.0, 0], "unsure": [0.0, 0]}
        for X in W:
            _, _, st = walk_acc(acc["EAGLE-2"][X["fi"]], X["n"])
            for p in st:
                m0 = X["M"][p, 0]
                if np.isnan(m0): continue
                b = "sure" if m0 >= 4 else "unsure"
                g[b][0] += acc[name][X["fi"]][p] - acc["EAGLE-2"][X["fi"]][p]; g[b][1] += 1
        say(f"   {name:18s} accepted change per EAGLE-2 walk start: sure {g['sure'][0] / max(g['sure'][1], 1):+.3f} "
            f"(n {g['sure'][1]}), unsure {g['unsure'][0] / max(g['unsure'][1], 1):+.3f} (n {g['unsure'][1]})")
    # SCREEN 2: classes at the chain's first miss on the chain-16 walk, and kept for the four arms
    cls = {"TREE only": 0, "CC only": 0, "BOTH": 0, "NEITHER": 0}
    def arm_acc(X, use_tree, use_cc):
        n = X["n"]; out = np.zeros(n, np.int64); R, DL, TP = X["R"], X["DL"], treepass[X["fi"]]
        for p in range(n):
            a = 0; on_path = True
            for d in range(min(DEP, NODES)):
                if R[p, d] < 0: break
                ok = R[p, d] == 0
                if not ok and use_tree and on_path and TP[p, d]: ok = True
                elif not ok and use_cc and not np.isnan(DL[p, d]) and DL[p, d] >= -DELTA:
                    ok = True; on_path = False if not (use_tree and TP[p, d]) else on_path
                if not ok: break
                a = d + 1
            out[p] = a
        return out
    for X in W:
        _, _, st = walk_acc(chain_acc(X, 16), X["n"])
        for p in st:
            R = X["R"][p]
            miss = np.flatnonzero(R[:NODES] > 0)
            if not len(miss): continue
            d = int(miss[0]); t_ = bool(treepass[X["fi"]][p, d]); c_ = (not np.isnan(X["DL"][p, d])) and X["DL"][p, d] >= -DELTA
            cls["BOTH" if t_ and c_ else "TREE only" if t_ else "CC only" if c_ else "NEITHER"] += 1
    tot = sum(cls.values())
    say(f"   first chain miss on the chain-16 walk, rescued by: " + ", ".join(f"{k} {100 * v / tot:.1f} %" for k, v in cls.items()) + f" (n {tot})")
    kk = {}
    for nm, ut, uc in (("chain-16", False, False), ("chain-16 + CC", False, True), ("tree-16", True, False), ("tree-16 + CC", True, True)):
        ks = nc = 0
        for X in W:
            a_ = arm_acc(X, False, uc) if not ut else (tree_acc if not uc else treecc_acc)[X["fi"]]
            k_, n_, _ = walk_acc(a_, X["n"]); ks += k_; nc += n_
        kk[nm] = ks / nc
        say(f"   {nm:14s} kept/check {kk[nm]:7.3f}  x{kk[nm] / kk['chain-16']:.3f}")
    gt, gc, gb = kk["tree-16"] - kk["chain-16"], kk["chain-16 + CC"] - kk["chain-16"], kk["tree-16 + CC"] - kk["chain-16"]
    say(f"   additivity: gain(tree + CC) {gb:.3f} vs gain(tree) + gain(CC) {gt + gc:.3f} -> {gb / max(gt + gc, 1e-9):.2f} "
        f"-> {'ADDITIVE' if gb >= 0.8 * (gt + gc) else 'OVERLAPPING'}")
    RES[s] = {"shape": res, "first_miss": cls, "kept": kk}
open(OUT, "w").write("\n".join(lines) + "\n")
json.dump(RES, open(OUT.replace(".txt", ".json"), "w"), indent=1)
