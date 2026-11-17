#!/usr/bin/env python3
"""BREAK98 bound for FAR's FUTURE-STATE ROWS (NewIdeas, Tue 17 Nov; time = the commit's), via b98sim: per block (anchor),
with probability c (FAR's end coverage) the head gets placeholder-state information at depths 0..k-1; inside that band
each depth flips to the true-state guess (LAB) with probability G (the fraction of the true-state effect a frozen-model
placeholder carries; G = 1 is the ceiling). Point delivery (the carried state is not refreshed). Chain-32 kept, cells.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_far.py"""
import numpy as np
import os, sys; sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C98
import b98sim as S

def far(W, k, c, G, draws=12, seed=17):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(draws):
        ks = []
        for rt in W:
            lab = S.shifted_lab(rt)
            blk = rng.random(rt.shape[0]) < c                      # this anchor's block end is covered
            band = np.zeros(rt.shape, bool); band[:, :k] = True
            m = band & blk[:, None] & (rng.random(rt.shape) < G)
            g = np.where(m, lab, rt).astype(np.int8); g[rt < 0] = -1
            ks += C98.kept_walk(g, C98.R0)
        vals.append(np.mean(ks))
    return float(np.mean(vals))

for tag in ("heplus24", "mbppbank"):
    W = S.load(tag); base = S.kept(W, S.refresh(32))
    print(f"== {tag}: real chain-32 {base:.2f}; LAB {S.kept(W, S.refresh(1)):.2f}")
    for k in (4, 8, 12, 16):
        row = []
        for c in (0.71, 0.855, 1.0):
            for G in (1.0, 0.5, 0.3):
                v = far(W, k, c, G); row.append(f"c{c} G{G} {v:.2f} ({100 * (v / base - 1):+.0f} %)")
        print(f"  k={k:2d}: " + "; ".join(row))
