"""[tree_frontier.py: NewIdeas, Mon 16 Nov 2026 18:0x EET] THE LOSSY ACCEPTANCE FRONTIER: kept per check under relaxed
acceptance rules on chain / tree checks of 32 (and 16) rows, from CPU replays of the real t30b dumps. CPU only, one
process, small (< 1 GB). usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B tree_frontier.py
ORIGIN: a copy of tree_cc32.py (= tree_shape_cc.py with NODES = 31). Its SCREEN 1 (tree shape) is dropped; its SCREEN 2
arms (chain, chain + CC, tree, tree + CC; delta 1 nat) are kept verbatim as reference() and must reproduce the published
numbers to +-0.01 before anything else runs: 32 rows (tree_cc32.txt) HE+ 13.77 / 16.78 / 16.26 / 16.91, MBPP 6.92 / 8.52
/ 9.03 / 9.50; 16 rows (tree_shape_cc.txt) HE+ 10.625 / 12.097 / 11.263 / 11.762, MBPP 6.588 / 7.919 / 7.487 / 7.968.
A FAIL stops the script.
DATA (read-only): BigMoE data/deepdraft/ceiling: cells_t30b_{heplus24,mbppbank}.npz (per true-text cell: depth, right, m,
rank, dlp, row), topk_t30b_*.npz (the head's top-8 per cell: top_ids[:, 0] = the head's pick, y = the 125B's token,
y_rank), tree_t30b_*.npz (32 best-first pops per anchor, on_true, path_p, parent, token); the bank records the dumps were
made from (~/datasets/deepdraft/bank/<files>: ids, gen, lp = the 125B's captured top-32 log-softmax per row (float16),
tk = their ids); the 125B's tokenizer (~/BigMoE/models/bare/tokenizer.json, the one r57 decodes with); the engine's
taggers by path: ~/BigMoE/scripts/relax_core.py (CodeScan) and ~/BigMoE/data/r57/snap/cc1/inert_span.py (InertScan,
classify, arm CD32 with its default doctest guard ON).
GENERALIZATION: the close-call predicate (cells dlp >= -1) becomes a per-cell rule REL[p, d] (anchor p, depth d) and is
applied exactly where CC(1) was: chain arms keep the leading cells that are exact (y_rank 0) or REL (relaxed); tree arms
keep the EAGLE-2-N selection's exact accepted depth A (N = 31 or 15 nodes = pops < N), then, if the true path's cell at
depth A is a miss with REL true and the true-path node at A - 1 has a selected child, the head's selected top-1 child at A
and its selected best-child chain while each cell below is exact or REL. With REL = CC(1) the generalized arms must equal
reference() anchor for anchor (SELF-CHECK 'identical'). kept = accepted + 1; the walk = certify98.kept_walk's (R0 = 1).
APPROXIMATION (as in tree_cc32 / tree_shape_cc): after a relaxed acceptance the walk goes on along the TRUE text's cells
(the head's picks and the 125B's argmax given the true prefix, not the relaxed one). This flatters aggressive rules; the
'stop' column bounds it inside a check: every relaxed accept ends its check (the relaxed draft kept, then the bonus).
RULES (each REL is only consulted on misses, cells with y_rank > 0):
  R1(delta)   cells dlp >= -delta (= log p125(pick) - log p125(top1)); R1(1) = CC(1).
  TYP(tau)    p125(pick) >= tau, log p125(pick) = lp[row, 0] + dlp (= lp[row, rank]; record lp, checked equal); a pick
              outside the 125B's top-32 (rank 99) has p < 1/32 and never passes tau >= 0.05.
  TOP2        cells rank <= 1 (the pick's position in the 125B's top-32).
  SPAN (#8)   R1(3) on '#' comment / docstring rows, R1(1) on every other row; no line-end guard.
  INERT (#13) inert_span.classify(arm CD32) == 'whole': comment rows (CodeScan.in_comment) keep any pick unless the pick
              or the 125B's token holds a newline or the comment is / could become a tool comment; docstring rows
              (InertScan.in_doc) keep any pick unless the pick or the 125B's token holds the docstring's quote char or the
              row is a doctest line (the guard default); code and prose rows exact.
  INERT + X   INERT on comment / docstring rows, X on every other row (code and prose); 'INERT | X' (the JSON) = the union.
ROW CLASSES: InertScan fed the answer's text from its first token (ids[gen]), token by token (tokenizer.decode([id]), as
r57 feeds it); a cell's class = the scan's state before the cell's target token: comment, docstring, code (inside a ```
fence otherwise), prose (outside any fence)."""
import os, sys, json, math, importlib.util
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98
from tokenizers import Tokenizer

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
BANK = f"{H}/datasets/deepdraft/bank"
SETS = {"mbpp": ("mbppbank", 6.92), "he": ("heplus24", 13.77)}
LABEL = {"he": "HE+", "mbpp": "MBPP"}
DEP, K = 31, 8
OUT = os.path.expanduser("~/NewIdeas/research/week98/tree_frontier_t16.txt")
PUB = {31: {"he": (13.77, 16.78, 16.26, 16.91), "mbpp": (6.92, 8.52, 9.03, 9.50)},           # tree_cc32.txt
       15: {"he": (10.625, 12.097, 11.263, 11.762), "mbpp": (6.588, 7.919, 7.487, 7.968)}}   # tree_shape_cc.txt
ARMS4 = ("chain", "chain + CC", "tree", "tree + CC")
SHAPES = (("chain-32", "chain", 31), ("tree-32", "tree", 31), ("chain-16", "chain", 15), ("tree-16", "tree", 15))
CLASSES = ("code", "comment", "docstring", "prose")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def _load(name, path):                                 # the engine's taggers, imported read-only by path
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m
RXC = _load("relax_core", f"{H}/BigMoE/scripts/relax_core.py")
INM = _load("inert_span", f"{H}/BigMoE/data/r57/snap/cc1/inert_span.py")
ARM = INM.parse_arm("CD32")                            # whole comments + docstrings, guards on, doctest guard default ON
assert ARM["doctest"] is True and ARM["doc"] == "whole" and ARM["com"] == "whole"
TKZ = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
_TXT = {}
def txt(i):                                            # r57's _TKZ.decode([x]) (special tokens decode to '')
    i = int(i); t = _TXT.get(i)
    if t is None: t = _TXT[i] = TKZ.decode([i])
    return t

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

def tree_arrays(tag):
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz")
    return tuple(tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p", "parent", "token"))

def reference(W, T, NODES, DELTA=1.0):
    """tree_cc32.py's SCREEN 2 arms, verbatim but for NODES and DELTA as arguments (SCREEN 1's policies and the first-miss
    classes dropped). -> ({arm: kept/check}, {arm: {fi: accepted per anchor}})"""
    nfi, nan_, nnode, ndep, ontrue, pp, par = T[:7]
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    byfi = {X["fi"]: X for X in W}
    order = np.lexsort((nnode, nan_, nfi))
    nfi, nan_, nnode, ndep, ontrue, pp, par = (a[order] for a in (nfi, nan_, nnode, ndep, ontrue, pp, par))
    bounds = np.flatnonzero(np.append(True, (np.diff(nfi) != 0) | (np.diff(nan_) != 0)))
    bounds = np.append(bounds, len(nfi))
    treepass = {X["fi"]: np.zeros((X["n"], DEP), bool) for X in W}     # EAGLE-2-16: on-true node selected at depth d
    tree_acc = {X["fi"]: np.zeros(X["n"], np.int64) for X in W}
    treecc_acc = {X["fi"]: np.zeros(X["n"], np.int64) for X in W}
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        f_, a_ = int(nfi[b0]), int(nan_[b0]); r = rowof[f_][a_]
        nd, dp, ot, ppp, pa = nnode[b0:b1], ndep[b0:b1], ontrue[b0:b1].astype(bool), pp[b0:b1].astype(np.float64), par[b0:b1]
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
    kk, AC = {}, {}
    for nm, ut, uc in zip(ARMS4, (False, False, True, True), (False, True, False, True)):
        ks = nc = 0; AC[nm] = {}
        for X in W:
            a_ = arm_acc(X, False, uc) if not ut else (tree_acc if not uc else treecc_acc)[X["fi"]]
            AC[nm][X["fi"]] = np.asarray(a_, np.int64)
            k_, n_, _ = walk_acc(a_, X["n"]); ks += k_; nc += n_
        kk[nm] = ks / nc
    return kk, AC

def tree_struct(W, T, NODES):
    """per anchor of each window, for the EAGLE-2-NODES selection (pops < NODES): A = the exact tree's accepted depth;
    L = the length of the selected best-child chain below the head's selected top-1 child at depth A under the true path
    (-1: no selected child there, so no relaxed rescue); tok = that child's token (checked against the cell's pick)"""
    nfi, nan_, nnode, ndep, ontrue, pp, par, tok = T
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    order = np.lexsort((nnode, nan_, nfi))
    nfi, nan_, nnode, ndep, ontrue, pp, par, tok = (a[order] for a in (nfi, nan_, nnode, ndep, ontrue, pp, par, tok))
    bounds = np.flatnonzero(np.append(True, (np.diff(nfi) != 0) | (np.diff(nan_) != 0)))
    bounds = np.append(bounds, len(nfi))
    S = {X["fi"]: {"A": np.zeros(X["n"], np.int64), "L": np.full(X["n"], -1, np.int64), "tok": np.full(X["n"], -1, np.int64)}
         for X in W}
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        f_, a_ = int(nfi[b0]), int(nan_[b0]); r = rowof[f_][a_]
        nd, dp, ot, ppp, pa, tk_ = (nnode[b0:b1], ndep[b0:b1], ontrue[b0:b1].astype(bool), pp[b0:b1].astype(np.float64),
                                    par[b0:b1], tok[b0:b1])
        selm = nd < NODES; hit = ot & selm
        A_t = int(dp[hit].max()) + 1 if hit.any() else 0
        S[f_]["A"][r] = A_t
        if A_t < min(DEP, NODES):
            pid = -1 if A_t == 0 else int(nd[np.flatnonzero(hit & (dp == A_t - 1))[0]])
            kids = np.flatnonzero(selm & (pa == pid) & (dp == A_t))
            if len(kids):
                cur = int(kids[np.argmax(ppp[kids])]); S[f_]["tok"][r] = int(tk_[cur]); L = 0
                while True:
                    ch = np.flatnonzero(selm & (pa == nd[cur]))
                    if not len(ch): break
                    cur = int(ch[np.argmax(ppp[ch])]); L += 1
                S[f_]["L"][r] = L
    return S

def enrich(tag, W):
    """per window: the cells' rank / row / pick laid out as R, the record's lp (log p125 of the pick), the row classes and
    INERT's per-cell verdict; every field checked against the record it came from"""
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = cz["depth"].astype(int); rank, row = cz["rank"].astype(np.int64), cz["row"].astype(np.int64)
    pick, y = tz["top_ids"][:, 0].astype(np.int64), tz["y"].astype(np.int64)
    assert (cz["right"] == (pick == y)).all(), "topk's top-1 is not the cells' pick"
    files = [str(f) for f in cz["files"]]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    chk = dict(cells=0, top1_not_text_rows=0, inert_cells=0, prose_fence=0)
    for (s, t), X in zip(zip(st, en), W):
        assert int(cz["fi"][s]) == X["fi"]
        dw = dep[s:t]; n = X["n"]
        RK = np.full((n, DEP), -1, np.int64); RW = np.full((n, DEP), -1, np.int64)
        PK = np.full((n, DEP), -1, np.int64); YY = np.full((n, DEP), -1, np.int64)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            RK[:c, j] = rank[s:t][m]; RW[:c, j] = row[s:t][m]; PK[:c, j] = pick[s:t][m]; YY[:c, j] = y[s:t][m]
        v = RW >= 0
        assert (v == (X["R"] >= 0)).all() and ((X["R"] == 0) == (v & (X["rt"] == 1))).all()
        z = np.load(f"{BANK}/{files[X['fi']]}")
        ids = z["ids"].astype(np.int64); gen = int(z["gen"]); lp = z["lp"].astype(np.float32); tk = z["tk"].astype(np.int64)
        P_, D_ = np.nonzero(v)
        assert (RW[v] == gen - 1 + P_ + D_).all(), "cell (p, d) must target row gen - 1 + p + d"
        assert (YY[v] == ids[RW[v] + 1]).all(), "y must be the record's next token"
        assert np.allclose(X["M"][v], lp[RW[v], 0] - lp[RW[v], 1], atol=1e-4), "m != record lp[0] - lp[1]"
        i32, o32 = v & (RK < 32), v & (RK >= 32)
        assert (RK[o32] == 99).all() and np.isneginf(X["DL"][o32]).all()
        assert np.allclose(lp[RW[i32], RK[i32]] - lp[RW[i32], 0], X["DL"][i32], atol=1e-4), "dlp != record lp gap"
        assert (tk[RW[i32], RK[i32]] == PK[i32]).all(), "rank does not point at the pick"
        assert not (tk[RW[o32]] == PK[o32][:, None]).any(), "a rank-99 pick is in the top-32"
        LPP = np.full((n, DEP), np.nan, np.float32)
        LPP[i32] = lp[RW[i32], 0] + X["DL"][i32]; LPP[o32] = -np.inf          # log p125(pick)
        chk["cells"] += int(v.sum())
        rr = np.unique(RW[v]); chk["top1_not_text_rows"] += int((tk[rr, 0] != ids[rr + 1]).sum())
        # row classes: the engine's tagger fed the answer's text token by token from ids[gen]
        sc = INM.InertScan(RXC); snaps = {}; cpos = np.full(len(ids), 3, np.int8)
        for q in range(gen, len(ids)):
            snaps[q] = sc.copy()
            cpos[q] = 1 if sc.in_comment() else 2 if sc.in_doc() else 0 if sc.s.code else 3
            chk["prose_fence"] += int(bool(sc.s.code) and not sc.py)
            sc.feed(txt(ids[q]))
        Q = np.where(v, RW + 1, 0)
        CL = np.where(v, cpos[Q], 3).astype(np.int8)
        IN = np.zeros((n, DEP), bool); INL = np.zeros((n, DEP), bool)
        for p, d in zip(*np.nonzero(v & (X["R"] > 0) & ((CL == 1) | (CL == 2)))):
            q = int(Q[p, d]); dt, at = txt(PK[p, d]), txt(ids[q])
            kind, mode, _ = INM.classify(RXC, snaps[q], [dt], [at], ARM)[0]
            assert kind == ("comment" if CL[p, d] == 1 else "docstring")
            IN[p, d] = mode == "whole"
            INL[p, d] = IN[p, d] and "\n" not in dt and "\n" not in at
            chk["inert_cells"] += 1
        X.update(RK=RK, PK=PK, LPP=LPP, CL=CL, INERT=IN, INERTNL=INL, gen=gen, nids=len(ids), cpos=cpos,
                 E0=np.where((X["R"] == 0).all(1), DEP, (X["R"] != 0).argmax(1)))
    return chk

def rule_masks(X):
    R, DL, LPP, RK, CL, I = X["R"], X["DL"], X["LPP"], X["RK"], X["CL"], X["INERT"]
    span = (CL == 1) | (CL == 2)
    with np.errstate(invalid="ignore"):
        r1 = {d: DL >= -d for d in (1, 2, 3)}
        typ = {t: LPP >= math.log(t) for t in (0.05, 0.1, 0.2)}
    top2 = (RK >= 0) & (RK <= 1)
    M = {"exact": np.zeros_like(I), "R1(1) = CC(1)": r1[1], "R1(2)": r1[2], "R1(3)": r1[3],
         "TYP(0.05)": typ[0.05], "TYP(0.1)": typ[0.1], "TYP(0.2)": typ[0.2], "TOP2": top2,
         "SPAN": np.where(span, r1[3], r1[1]), "INERT": I,
         "INERT + R1(1)": I | (~span & r1[1]), "INERT + R1(2)": I | (~span & r1[2]),
         "INERT + TYP(0.1)": I | (~span & typ[0.1]), "INERT + TOP2": I | (~span & top2),
         # sensitivity rows (JSON + notes): INERT with the newline guard on docstrings too; the combos as unions
         "INERT-nl": X["INERTNL"], "INERT | R1(1)": I | r1[1], "INERT | R1(2)": I | r1[2],
         "INERT | TYP(0.1)": I | typ[0.1], "INERT | TOP2": I | top2}
    # GUARDED arms (Mon 18:3x): the engine forms after CD16's 11-06 quality KILL. R2(k, r) = rank < k and p >= r p1.
    with np.errstate(invalid="ignore"):
        r2_3 = (RK >= 0) & (RK < 3) & (DL >= math.log(0.1))
        r2_2 = (RK >= 0) & (RK < 2) & (DL >= math.log(0.25))
    com = (CL == 1)
    M.update({"BASE ruleC R2(2,.25)": com & r2_2,
              "T16 R2(3,.1) c+d": I & r2_3, "TC16 R2(3,.1) c": I & com & r2_3,
              "T16 + R1(1) code": (I & r2_3) | (~span & r1[1]), "BASE + R1(1) code": (com & r2_2) | (~span & r1[1])})
    return M
MAIN = ("exact", "R1(1) = CC(1)", "BASE ruleC R2(2,.25)", "BASE + R1(1) code", "TC16 R2(3,.1) c", "T16 R2(3,.1) c+d",
        "T16 + R1(1) code", "SPAN", "INERT")

def onehot(c):
    return (np.arange(4)[None, :] == np.asarray(c)[:, None]).astype(np.int64)

def gen_arms(X, St, REL, NODES):
    """the generalized arms of one window: {variant: (accepted per anchor, relaxed accepts per anchor by class [n, 4])}"""
    R, CL, n = X["R"], X["CL"], X["n"]; ar = np.arange(n); cap = min(DEP, NODES)
    OK = (R >= 0) & ((R == 0) | REL)
    RUN = np.zeros((n, DEP + 1), np.int64)                  # RUN[p, d]: consecutive kept cells from depth d on
    for d in range(DEP - 1, -1, -1):
        RUN[:, d] = np.where(OK[:, d], RUN[:, d + 1] + 1, 0)
    MISS = R > 0
    CUM = np.zeros((4, n, DEP + 1), np.int64)               # misses before depth d, by class (a kept miss = a relaxed accept)
    for c in range(4):
        CUM[c, :, 1:] = np.cumsum(MISS & (CL == c), axis=1)
    out = {}
    ch = np.minimum(RUN[:, 0], cap)
    out["chain"] = (ch, CUM[:, ar, ch].T)
    e = np.minimum(X["E0"], cap); ei = np.minimum(e, DEP - 1)
    rx = (e < cap) & MISS[ar, ei] & REL[ar, ei]
    out["chain stop"] = (e + rx, rx[:, None] * onehot(CL[ar, ei]))
    A, L = St["A"], St["L"]; Ai = np.minimum(A, DEP - 1); A1 = np.minimum(A + 1, DEP)
    resc = (A < cap) & (L >= 0) & MISS[ar, Ai] & REL[ar, Ai]
    cont = np.minimum(RUN[ar, A1], np.maximum(L, 0)); end = np.minimum(A1 + cont, DEP)
    out["tree"] = (np.where(resc, A + 1 + cont, A), resc[:, None] * (onehot(CL[ar, Ai]) + (CUM[:, ar, end] - CUM[:, ar, A1]).T))
    out["tree stop"] = (np.where(resc, A + 1, A), resc[:, None] * onehot(CL[ar, Ai]))
    return out

def walk_stats(X, acc, rel, cap):
    ks, nc, st = walk_acc(acc, X["n"])
    st = np.asarray(st, np.int64)
    CP = np.zeros((4, X["nids"] + 1), np.int64)             # committed answer positions by class: a check at anchor p
    for c in range(4):                                     # commits positions gen + p .. gen + p + accepted
        CP[c, 1:] = np.cumsum(X["cpos"] == c)
    lo = np.minimum(X["gen"] + st, X["nids"]); hi = np.minimum(X["gen"] + st + acc[st] + 1, X["nids"])
    return ks, nc, rel[st].sum(0), (CP[:, hi] - CP[:, lo]).sum(1), int((acc[st] >= cap).sum())   # + checks at the row cap

RES = {"meta": {}, "self_check": {}}
REF = {}
for s, (tag, pub) in SETS.items():
    W = load_cells(tag, pub)
    T = tree_arrays(tag)
    chk = enrich(tag, W)
    RES["meta"][s] = chk
    sc_ = RES["self_check"][s] = {}
    for NODES in (31, 15):
        kk, AC = reference(W, T, NODES)
        ok = all(abs(kk[a] - p_) < 0.01 + 1e-9 for a, p_ in zip(ARMS4, PUB[NODES][s]))
        say(f"SELF-CHECK {LABEL[s]} {NODES + 1} rows (reference = tree_cc32's code): " + ", ".join(
            f"{a} {kk[a]:.3f} vs {p_}" for a, p_ in zip(ARMS4, PUB[NODES][s])) + f" -> {'PASS' if ok else 'FAIL'}")
        St = tree_struct(W, T, NODES)
        same = True
        for X in W:
            M = rule_masks(X)
            for rn, (cn, tn) in (("exact", ("chain", "tree")), ("R1(1) = CC(1)", ("chain + CC", "tree + CC"))):
                g = gen_arms(X, St[X["fi"]], M[rn], NODES)
                same &= np.array_equal(g["chain"][0], AC[cn][X["fi"]]) and np.array_equal(g["tree"][0], AC[tn][X["fi"]])
        say(f"SELF-CHECK {LABEL[s]} {NODES + 1} rows: generalized arms (exact, R1(1)) vs the reference, anchor for anchor -> "
            f"{'identical' if same else 'DIFFERENT'}")
        sc_[f"{NODES + 1} rows"] = {"reference": kk, "published": dict(zip(ARMS4, PUB[NODES][s])), "pass": bool(ok),
                                     "generalized_identical": bool(same)}
        if not (ok and same):
            say("SELF-CHECK FAILED: stopping before the frontier")
            open(OUT, "w").write("\n".join(lines) + "\n"); json.dump(RES, open(OUT.replace(".txt", ".json"), "w"), indent=1)
            sys.exit(1)
        REF[(s, NODES)] = (kk, St)
    RES[s] = {"W": W}

def fmt_rel(v, kt):
    return f"{1000 * v.sum() / kt:6.1f} = " + " + ".join(f"{1000 * x / kt:5.1f}" for x in v)

TABLE = {}
for s in SETS:
    W = RES[s].pop("W"); out = RES[s]
    base_lossy, base_exact = REF[(s, 31)][0]["tree + CC"], REF[(s, 31)][0]["tree"]
    masks = {X["fi"]: rule_masks(X) for X in W}
    agree = [0, 0]
    for X in W:                                            # the tree rescue's child = the cell's pick (the approximation's
        St = REF[(s, 31)][1][X["fi"]]; ar = np.arange(X["n"])  # premise at the rescue row)
        mm = (St["A"] < DEP) & (St["L"] >= 0) & (X["R"][ar, np.minimum(St["A"], DEP - 1)] > 0)
        agree[0] += int((St["tok"][mm] == X["PK"][ar[mm], St["A"][mm]]).sum()); agree[1] += int(mm.sum())
    out["rescue_child_is_cell_pick"] = agree
    sp = {}
    for NODES in (31, 15):                                 # how deep the tree goes below the rescue child vs the chain
        Ls, rem = [], []
        for X in W:
            St = REF[(s, NODES)][1][X["fi"]]; ar = np.arange(X["n"]); cap = min(DEP, NODES)
            mm = (St["A"] < cap) & (St["L"] >= 0) & (X["R"][ar, np.minimum(St["A"], DEP - 1)] > 0)
            Ls += St["L"][mm].tolist(); rem += (cap - 1 - St["A"][mm]).tolist()
        sp[f"tree-{NODES + 1}"] = {"rescue_rows": len(Ls), "spine_mean": float(np.mean(Ls)), "spine_median": float(np.median(Ls)),
                                   "chain_room_mean": float(np.mean(rem))}
    out["spine_below_rescue"] = sp
    pos = np.zeros(4, np.int64)
    for X in W:
        pos += np.bincount(X["cpos"][X["gen"] + 1:], minlength=4)[:4]
    out["answer_positions_by_class"] = dict(zip(CLASSES, pos.tolist()))
    for shape, kind, NODES in SHAPES:
        St = REF[(s, NODES)][1]
        out[shape] = {}
        for rn in masks[W[0]["fi"]]:
            agg = {v: [0, 0, np.zeros(4, np.int64), np.zeros(4, np.int64), 0] for v in ("", " stop")}
            for X in W:
                g = gen_arms(X, St[X["fi"]], masks[X["fi"]][rn], NODES)
                for v in agg:
                    ks, nc, rel, cm, cp_ = walk_stats(X, *g[kind + v], min(DEP, NODES))
                    agg[v][0] += ks; agg[v][1] += nc; agg[v][2] += rel; agg[v][3] += cm; agg[v][4] += cp_
            ks, nc, rel, cm, capped = agg[""]
            kp = ks / nc; kst = agg[" stop"][0] / agg[" stop"][1]
            out[shape][rn] = {"kept": kp, "x_lossy_tree32": kp / base_lossy, "x_exact_tree32": kp / base_exact,
                              "kept_stop": kst, "checks": nc, "committed": ks, "checks_at_row_cap": capped,
                              "cap_share": capped / nc,
                              "relaxed_per_1k": {"all": 1000 * float(rel.sum()) / ks,
                                                 **{c: 1000 * float(x) / ks for c, x in zip(CLASSES, rel)}},
                              "relaxed": dict(zip(CLASSES, rel.tolist())), "committed_by_class": dict(zip(CLASSES, cm.tolist())),
                              "relaxed_per_1k_of_own_class": {c: 1000 * float(x) / max(int(y), 1) for c, x, y in zip(CLASSES, rel, cm)},
                              "stop_relaxed_per_1k": 1000 * float(agg[" stop"][2].sum()) / agg[" stop"][0]}
    TABLE[s] = (base_lossy, base_exact)

say()
say("LOSSY ACCEPTANCE FRONTIER (kept per check = accepted drafts + 1, on the walk from R0 = 1; CPU replay of the real t30b dumps)")
say("x lossy = / today's lossy (tree-32 + CC(1)); x exact = / exact tree-32; stop = kept if every relaxed accept ended its "
    "check (bounds the continue-along-true-text flattery); cap % = checks that keep every row (accepted = 31 or 15); relaxed / 1k = relaxed accepts per 1,000 committed tokens, "
    "all = code + comment + docstring + prose")
FLAGS = []
for s in ("he", "mbpp"):
    out = RES[s]; bl, be = TABLE[s]; pos = out["answer_positions_by_class"]; tp = sum(pos.values())
    say()
    say(f"=== {LABEL[s]} ({SETS[s][0]}): today's lossy tree-32 + CC(1) {bl:.3f}, exact tree-32 {be:.3f}; answer positions "
        f"{tp:,}: " + ", ".join(f"{c} {100 * pos[c] / tp:.1f} %" for c in CLASSES))
    say(f" {'shape':8s} {'rule':18s} {'kept':>7s} {'x lossy':>8s} {'x exact':>8s} {'stop':>7s} {'cap %':>5s}   relaxed / 1k: all = code + comment + docstring + prose")
    for shape, _, _ in SHAPES:
        for rn in MAIN:
            r = out[shape][rn]
            fl = s == "he" and r["x_lossy_tree32"] >= 1.40
            if fl: FLAGS.append(f"{LABEL[s]} {shape} {rn}: kept {r['kept']:.2f} = x{r['x_lossy_tree32']:.3f} (stop {r['kept_stop']:.2f} = x{r['kept_stop'] / bl:.3f})")
            say(f" {shape:8s} {rn:18s} {r['kept']:7.3f} {r['x_lossy_tree32']:8.3f} {r['x_exact_tree32']:8.3f} {r['kept_stop']:7.3f} "
                f"{100 * r['cap_share']:5.1f}   "
                + fmt_rel(np.array([r['relaxed'][c] for c in CLASSES]), r['committed']) + ("   <== x1.40" if fl else ""))
        say()
    SENS = ("INERT-nl", "INERT | R1(1)", "INERT | R1(2)", "INERT | TYP(0.1)", "INERT | TOP2")
    say(f" sensitivity ({LABEL[s]}; kept tree-32 / chain-32 / tree-16 (chain-32 stop); x lossy at chain-32):")
    for rn in SENS:
        r32, c32, t16 = out["tree-32"][rn], out["chain-32"][rn], out["tree-16"][rn]
        for shape in ("chain-32", "tree-32", "tree-16", "chain-16"):
            r = out[shape][rn]
            if s == "he" and r["x_lossy_tree32"] >= 1.40:
                FLAGS.append(f"{LABEL[s]} {shape} {rn} (sensitivity): kept {r['kept']:.2f} = x{r['x_lossy_tree32']:.3f} "
                             f"(stop {r['kept_stop']:.2f} = x{r['kept_stop'] / bl:.3f})")
        say(f"   {rn:18s} {r32['kept']:7.3f} / {c32['kept']:7.3f} / {t16['kept']:7.3f} ({c32['kept_stop']:.3f}); x{c32['x_lossy_tree32']:.3f}")
    a_ = out["rescue_child_is_cell_pick"]
    say(f" tree-32 rescue rows where the tree's top-1 child equals the cell's pick: {a_[0]} of {a_[1]}")
    for k_, v_ in out["spine_below_rescue"].items():
        say(f" {k_}: at the {v_['rescue_rows']:,} rescue-eligible anchors (first unselected true depth is a miss with a selected "
            f"top-1 child) the tree's selected spine below that child is {v_['spine_mean']:.2f} levels on average (median "
            f"{v_['spine_median']:.0f}); a chain of the same rows would have {v_['chain_room_mean']:.2f}")
say()
say("FLAGGED (>= x1.40 over today's lossy on HE+): " + ("; ".join(FLAGS) if FLAGS else "none"))
NOTES = """
NOTES (method, fields, assumptions)
- Walk and data as tree_cc32.py: certify98's decode walk from R0 = 1 over the 24 HE+ / 100 MBPP answers of the bank (the
  true text = the 125B's greedy answer); a check at anchor p drafts the true-text cells (p, 0..); kept = accepted + 1.
  Every rule walks the same true text, so committed tokens are fixed (HE+ 6,375 = 6,351 answer positions + one end row per
  answer, as certify98's walk counts it; MBPP 15,641 = 15,541 + 100) and kept/check = committed / checks.
- Shapes: chain-N = the head's top-1 chain to depth N - 1; tree-N = EAGLE-2's first N - 1 best-first pops of the 32 dumped
  per anchor. Tree + rule = the exact tree's accepted depth A, then ONE relaxed rescue at A through the head's selected
  top-1 child (the cell's pick: it matched on every rescue row), then that child's selected best-child spine while each
  cell is exact or relaxed. Siblings at A are not tried (as CC); the spine is the tree's own selection (budget-correct).
- APPROXIMATION (flatters aggressive rules): after a relaxed accept the walk continues along the TRUE text's cells, i.e.
  the head's picks and the 125B's argmax given the true prefix, not the relaxed one. 'stop' = the floor where every
  relaxed accept ends its check (the relaxed draft kept + the bonus). Chains depend on the continuation far more than
  trees (chain stop ~ exact chain + 1); the real engine sits somewhere between the two columns, unmeasured here.
- p125 fields: log p125(pick) = record lp[row, 0] + cells dlp (identical to lp[row, rank]; lp = the captured top-32
  log-softmax, float16, logsumexp <= +0.0003); TOP2 = cells rank <= 1 (the pick's index in the 125B's top-32, -> 99 off
  it). R1 = cells dlp. HE+: on 35 rows (0.5 %) the text token is not the captured top-1 (gap <= 0.75 nat, 8 exact ties);
  R1 / TYP / TOP2 measure the pick against the captured top-1 there, INERT's argmax guards use the text token.
- Classes: InertScan (relax_core.CodeScan + the docstring state) fed the answer's text from ids[gen], one decode([id]) per
  token as r57 does; a cell's class = the state before its target token. 'prose' = outside any ``` fence (here: the
  ```python line); prose rows follow the code rule in SPAN and the combos.
- INERT = inert_span.classify(arm CD32, doctest guard ON = its default): comments whole unless the pick / the 125B's token
  holds a newline or the comment is / could become a tool comment; docstrings whole unless the pick / the token holds the
  docstring's quote character or the row is a doctest line. No newline guard on docstrings (INERT-nl adds one).
- Combos 'INERT + X' (the table) = INERT on comment / docstring rows, X on code and prose rows; a guarded span row (line
  end, doctest line, quote) stays exact. 'INERT | X' (sensitivity) = the union: X may also relax those guarded rows.
- x lossy / x exact divide by this run's tree-32 + CC(1) and exact tree-32 of the same set (HE+ 16.910 / 16.263, MBPP
  9.502 / 9.031), also for the 16-row shapes."""
for l_ in NOTES.strip("\n").split("\n"):
    say(l_)
open(OUT, "w").write("\n".join(lines) + "\n")
json.dump(RES, open(OUT.replace(".txt", ".json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
