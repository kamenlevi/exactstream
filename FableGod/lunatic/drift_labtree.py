"""DRIFT LAB TREE BOUND (Lunatic, THINK_1116c, Mon 16 Nov 18:5x). CPU only, read-only. If drift were fully fixed (the
true-state head at every depth = the depth-0 guess of anchor a+d), and ties were covered by ONE sibling wherever the
125B's token is the true-state head's 2nd choice (or top-4), what would kept per 32-row check be? An upper bound for
'drift fix + trees' (siblings are not budgeted). Compared with the own-state chain and the same rescue on own states.
usage: ~/BigMoE/.venv/bin/python3 -I drift_labtree.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/drift_labtree.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
for tag in ("heplus24", "mbppbank"):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int); right = tz["right"].astype(int); yr = tz["y_rank"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    out = {k: [] for k in ("own chain", "own + rank<=1 sibling", "LAB chain", "LAB + rank<=1 sibling", "LAB + rank<=3 sibling")}
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        R = np.full((n0 + 1, DEP), -1, np.int8); Y = np.full((n0 + 1, DEP), 99)
        for j in range(DEP):
            m = dw == j; c = int(m.sum()); R[:c, j] = right[s:t][m]; Y[:c, j] = yr[s:t][m]
        L = np.full(R.shape, -1, np.int8); LY = np.full(R.shape, 99)
        for j in range(DEP):
            c = n0 - j
            if c > 0: L[:c, j] = R[j:j + c, 0]; LY[:c, j] = Y[j:j + c, 0]
        L[R < 0] = -1
        out["own chain"] += C98.kept_walk(R, C98.R0)
        r2 = R.copy(); r2[(r2 == 0) & (Y <= 1)] = 1; out["own + rank<=1 sibling"] += C98.kept_walk(r2, C98.R0)
        out["LAB chain"] += C98.kept_walk(L, C98.R0)
        l2 = L.copy(); l2[(l2 == 0) & (LY <= 1)] = 1; out["LAB + rank<=1 sibling"] += C98.kept_walk(l2, C98.R0)
        l3 = L.copy(); l3[(l3 == 0) & (LY <= 3)] = 1; out["LAB + rank<=3 sibling"] += C98.kept_walk(l3, C98.R0)
    say(f"== {tag} (kept per 32-row check; sibling rows unbudgeted = upper bounds)")
    for k, v in out.items(): say(f"   {k:24s} {np.mean(v):6.2f}")
open(OUT, "w").write("\n".join(lines) + "\n")
