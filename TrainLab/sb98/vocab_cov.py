"""WEEK98 S24 VOCABCOV (MTP Head; REPORT 'WEEK98 S24 VOCABCOV'), CPU: coverage of static vs per-check dynamic vocab screens on
t30b's TOPK dumps (true token y + the head's top-8 per cell), with the bank's ids / the 125B's top-32 and Leap's FR-Spec rank.
Usage: BigMoE venv python -I -B vocab_cov.py OUT.json"""
import os, sys, json
import numpy as np
H = os.path.expanduser
CE, BANK = H("~/BigMoE/data/deepdraft/ceiling"), H("~/datasets/deepdraft/bank")
RANK = np.load(H("~/Leap/data/frspec_rank.npy")).astype(np.int64)
V, DEP, R0 = 248320, 31, 1
STATIC = (8192, 16384, 32768)
DYN = [(B, R, k) for B in (4096, 8192) for R, k in ((64, 8), (256, 8), (10 ** 9, 32))]
OUT = {}
for st, tag in (("mbpp", "mbppbank"), ("he", "heplus24")):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep, right, row, fi, y, top = (tz[k] for k in ("depth", "right", "row", "fi", "y", "top_ids"))
    dep = dep.astype(int)
    s0 = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(s0[1:], len(dep))
    acc = {k: np.zeros(4) for k in [f"S{N}" for N in STATIC] + [f"D{B}_R{min(R, 9999)}_k{k}" for B, R, k in DYN]}
    sizes = {k: [] for k in acc}
    for w, (s, t) in enumerate(zip(s0, en)):
        f = str(tz["files"][int(fi[s])])
        z = np.load(os.path.join(BANK, f)); ids = np.asarray(z["ids"], np.int64); gen = int(z["gen"]); tk = np.asarray(z["tk"], np.int64)
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); Y = np.full((n0 + 1, DEP), -1, np.int64); TP = np.full((n0 + 1, DEP, 8), -1, np.int64)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; Y[:c, j] = y[s:t][m]; TP[:c, j] = top[s:t][m][:, :8]
        anc = row[s:t][dw == 0].astype(np.int64) - 1
        p = R0
        while p < n0:
            a = int(anc[p])
            sess = np.unique(ids[:a + 2])
            screens = {}
            for N in STATIC:
                m_ = np.zeros(V, bool); m_[RANK[:N]] = True; m_[sess] = True; screens[f"S{N}"] = m_
            for B, R, k in DYN:
                lo = max(gen - 1, a + 1 - R)
                rec = np.unique(tk[lo:a + 1, :k]) if a >= lo else np.zeros(0, np.int64)
                m_ = np.zeros(V, bool); m_[sess] = True; m_[rec] = True
                room = B - int(m_.sum())
                if room > 0:
                    core = RANK[~m_[RANK]][:room]; m_[core] = True
                screens[f"D{B}_R{min(R, 9999)}_k{k}"] = m_
            valid = rt[p] >= 0
            yy = Y[p][valid]; tt = TP[p][valid]
            for name, m_ in screens.items():
                acc[name] += [int((~m_[yy]).sum()), len(yy), int((~m_[tt[tt >= 0]]).sum()), int((tt >= 0).sum())]
                sizes[name].append(int(m_.sum()))
            k_ = int(np.argmin(np.append(rt[p] == 1, False))); p += k_ + 1
    OUT[st] = {n: {"m1_true_miss": v[0] / max(v[1], 1), "m2_top8_out": v[2] / max(v[3], 1), "cells": int(v[1]),
                   "mean_size": float(np.mean(sizes[n]))} for n, v in acc.items()}
    print(f"{st}:")
    for n, r in OUT[st].items():
        print(f"  {n:14s} size {r['mean_size']:8.0f}  true-token miss {100 * r['m1_true_miss']:.3f} %  head top-8 outside {100 * r['m2_top8_out']:.3f} %")
ref = {st: OUT[st]["S32768"] for st in OUT}
win = [n for n in OUT["he"] if n.startswith("D") and all(OUT[st][n]["m1_true_miss"] <= ref[st]["m1_true_miss"]
       and OUT[st][n]["m2_top8_out"] <= ref[st]["m2_top8_out"] for st in OUT)]
print("VERDICT S24: " + (f"PASS ({', '.join(win)} cover as well as static 32K on both sets)" if win else "FAIL (no dynamic <= 8K screen matches static 32K on both sets)"))
json.dump({"res": OUT, "pass": win}, open(sys.argv[1], "w"), indent=1)
