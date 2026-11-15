"""CHECK-START MARGINS (NewIdeas, Sun 15 Nov 02:4x; the orchestrator's question (a) after CERTIFY98 amendment 4):
at the decode walk's check starts (certify98.kept_walk from row 1), what is the 125B's own margin mix at depth 0, against
all anchors, and how does t30b's depth-0 accuracy split by margin bin there? Reads BigMoE's cells files via EvalSuite's
certify98.windows (read-only import). CPU, numpy. usage: python -I checkstart_margins.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C
BINS = [(0, 0.25), (0.25, 1), (1, 2), (2, 4), (4, 1e9)]
def lab(a, b): return f"{a:g}-{b:g}" if b < 1e8 else f">={a:g}"
for nm in ("heplus", "mbpp100"):
    path = os.path.expanduser(f"~/BigMoE/data/deepdraft/ceiling/cells_t30b_{nm}.npz")
    W = C.windows(path)
    allm, allr, stm, str_, kept = [], [], [], [], []
    for rt, mt in W:
        ok = rt[:, 0] >= 0
        allm += list(mt[ok, 0]); allr += list(rt[ok, 0])
        sub = rt[C.R0:]; p = 0
        while p < sub.shape[0]:
            if sub[p, 0] >= 0:
                stm.append(mt[C.R0 + p, 0]); str_.append(sub[p, 0])
            k = 0
            while k < 31 and sub[p, k] == 1: k += 1
            kept.append(k + 1); p += k + 1
    allm, allr, stm, str_ = map(np.array, (allm, allr, stm, str_))
    print(f"== {nm}: windows {len(W)}; depth-0 cells: all anchors {len(allm)}, check starts {len(stm)}; kept/32 {np.mean(kept):.2f}")
    print(f"   depth-0 accuracy: all anchors {100 * allr.mean():.2f} %, check starts {100 * str_.mean():.2f} %")
    print(f"   {'margin bin':>10} {'all: rows %':>11} {'acc':>6}   {'starts: rows %':>14} {'acc':>6}")
    for a, b in BINS:
        sa = (allm >= a) & (allm < b); ss = (stm >= a) & (stm < b)
        print(f"   {lab(a, b):>10} {100 * sa.mean():11.1f} {100 * allr[sa].mean():6.2f}   {100 * ss.mean():14.1f} {100 * str_[ss].mean():6.2f}")
    # how much of the check-start loss is mix vs within-bin accuracy
    mix_only = sum(((stm >= a) & (stm < b)).mean() * allr[(allm >= a) & (allm < b)].mean() for a, b in BINS)
    print(f"   check-start accuracy if only the margin MIX changed (all-anchor accuracy per bin): {100 * mix_only:.2f} %")
