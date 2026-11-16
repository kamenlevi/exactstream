"""DRIFT FORK CALIBRATION (Lunatic, THINK_1116c, Mon 16 Nov 18:5x). CPU only, read-only. Is the head's own top-1
probability still calibrated AFTER a fork (a passed step with p1 < 0.5)? Overconfidence there would mean a free
tree-scorer fix; calibration means the tree already prices the damage. Own guesses at depths 2-30, binned by p1.
usage: ~/BigMoE/.venv/bin/python3 -I drift_forkcal.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/drift_forkcal.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
for tag in ("heplus24", "mbppbank"):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int); right = tz["right"].astype(bool); p1 = tz["top_p"][:, 0].astype(np.float64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    P, Y, F = [], [], []
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        R = np.full((n0, DEP), -1, np.int8); P1 = np.full((n0, DEP), np.nan)
        for j in range(DEP):
            m = dw == j; c = int(m.sum()); R[:c, j] = right[s:t][m]; P1[:c, j] = p1[s:t][m]
        for d in range(2, DEP):
            ok = (R[:, d] >= 0) & np.isfinite(P1[:, :d]).all(1) & np.isfinite(P1[:, d])
            fork = (P1[:, :d] < 0.5).any(1)
            P.append(P1[ok, d]); Y.append(R[ok, d].astype(int)); F.append(fork[ok])
    P, Y, F = np.concatenate(P), np.concatenate(Y), np.concatenate(F)
    say(f"== {tag}: own guesses at depths 2-30, head p1 vs accuracy")
    for name, m in (("no fork passed", ~F), ("after a fork", F)):
        row = []
        for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)):
            k = m & (P >= lo) & (P < hi)
            if k.sum(): row.append(f"p1 {lo:.2f}-{min(hi,1):.2f}: n {k.sum():6d} mean p1 {P[k].mean():.3f} acc {Y[k].mean():.3f}")
        say(f"   {name:15s} all: n {m.sum():6d} mean p1 {P[m].mean():.3f} acc {Y[m].mean():.3f}")
        for r in row: say(f"      {r}")
open(OUT, "w").write("\n".join(lines) + "\n")
