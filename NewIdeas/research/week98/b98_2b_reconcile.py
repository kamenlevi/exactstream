#!/usr/bin/env python3
"""BREAK98b: reconcile FAR (2B right 67 % at first misses, 3,401 HE+) vs Lunatic (52 %, 337 HE+) on ONE script (NewIdeas, Tue 17 Nov;
time = the commit's). Same data for both: BigMoE's topk cells (topk_t30b_<tag>.npz: the head's top-8 per cell, y, right) and
MoonShots' rows{2b,4b}_<set>.npz (the auditor's greedy top-1 on the TRUE text per answer row), joined as Lunatic's limit_auditor.py
(cell (file, row) <-> rows (fname, row), checked by tk0 == the cell's y). Two FIRST-MISS populations:
  WALK   the plain chain-32 walk (the decoding process: each check starts where the last ended; one first miss per check);
  ALL    the first miss of the chain started at EVERY anchor (counts a hard position once per anchor whose chain first fails there).
For each: the 2B / 4B top-1 right, the head's #2 right, head #2 OR 2B, and the multiplicity of positions in ALL.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_2b_reconcile.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"; MS = f"{H}/MoonShots/week98"
DEP = 31
for tag, nm in (("heplus24", "heplus"), ("mbppbank", "mbpp")):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    files = [str(f) for f in tz["files"]]; dep = tz["depth"].astype(int); fi = tz["fi"].astype(int); row = tz["row"].astype(int)
    y = tz["y"].astype(np.int64); right = tz["right"].astype(np.int8); tid = tz["top_ids"].astype(np.int64); nc = len(dep)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    WIN = []
    for s, t_ in zip(st, en):
        dw = dep[s:t_]; n0 = int((dw == 0).sum()); C = np.full((n0 + 1, DEP), -1, np.int64)
        for j in range(DEP):
            idx = s + np.flatnonzero(dw == j); C[:len(idx), j] = idx
        WIN.append(C)
    walk, allm = [], []
    for C in WIN:
        n = C.shape[0]; p = 1
        while p < n:                                            # WALK (Lunatic's loop)
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
            if a < 31 and C[p, a] >= 0: walk.append(C[p, a])
            p += a + 1
        for p in range(1, n):                                   # ALL anchors
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
            if a < 31 and C[p, a] >= 0: allm.append(C[p, a])
    walk = np.array(walk); allm = np.array(allm)
    pos_all = np.array([fi[c] * 10_000_000 + row[c] for c in allm])
    _, cnt = np.unique(pos_all, return_counts=True)
    print(f"== {tag}: first misses WALK {len(walk)} | ALL anchors {len(allm)} at {len(cnt)} distinct positions "
          f"(each counted {cnt.mean():.1f}x on average, max {cnt.max()})")
    h2 = tid[:, 1] == y
    aud = {}
    for a_ in ("2b", "4b"):
        fpath = f"{MS}/rows{a_}_{nm}.npz"
        R = np.load(fpath)
        key = {(os.path.basename(str(n)), int(r)): (int(t), int(k0)) for n, r, t, k0 in zip(R["fname"], R["row"], R["top1"], R["tk0"])}
        top1 = np.full(nc, -1, np.int64); k0 = np.full(nc, -1, np.int64)
        for c in range(nc):
            v = key.get((os.path.basename(files[fi[c]]), int(row[c])))
            if v is not None: top1[c], k0[c] = v
        aud[a_] = top1
        have = top1 >= 0
        print(f"   {a_}: joined {100 * have.mean():.1f} % of cells; tk0 == y on {100 * (k0[have] == y[have]).mean():.2f} %")
    for name, M in (("WALK", walk), ("ALL", allm)):
        r2 = aud["2b"][M] == y[M]; r4 = aud["4b"][M] == y[M]; hh = h2[M]
        print(f"   {name:4s}: 2B right {100 * r2.mean():5.1f} % | 4B {100 * r4.mean():5.1f} % | head #2 {100 * hh.mean():5.1f} % | "
              f"head #2 OR 2B {100 * (hh | r2).mean():5.1f} % | 2B right where head #2 wrong {100 * (r2 & ~hh).mean():5.1f} %")
