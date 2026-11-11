"""Fit12 T3R K1 side checks (2026-10-25, after fit_t3r_k1.py; information, not the decisive line): at the 12 GB budgets,
(1) 4-row checks accepting 3 (NewIdeas' b10_12gb.txt MTP-like rows; B12LL's engine runs LRU: LFU was killed at C_m 51)
and (2) the C_r grid extended below fit_t3r_k1's lower edge (C_r 1-4) where its best sat on the edge (C_m 59 budget).
Usage: ~/BigMoE/.venv/bin/python -B fit_t3r_k1side.py   (CPU, 1 core, ~1 min)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_t3r_k1 as K
import fit_v3replay as V

tr = {s: [np.load(os.path.join(V.TR, f"{s}_800_g.npz"))[f"L{L}"][:, :10].astype(np.int64) for L in range(V.NL)] for s in V.SETS}
for cm, cl in ((51, 20), (59, 20)):
    budget = cm * K.MSB + cl * K.LSB
    for proto in ((4, 3, 0.0), (4, 3, 0.99), K.DEC):
        crs = range(1, 3 * cl + 1)
        a, _ = K.mb(tr, cm, cl, K.MSB, K.LSB, proto)
        (vb, cmb, clb), _ = K.best(tr, budget, K.MSB, K.LSB, crs, proto)
        (b, cb, cr), _ = K.best(tr, budget, K.BASE, K.RES, crs, proto)
        print(f"budget {cm} / {cl}, rows {proto[0]} accept {proto[1]} {'LRU' if not proto[2] else 'LFU ' + str(proto[2])} "
              f"(C_r 1..{3 * cl}): v3 {a:.1f} (best split {cmb} / {clb}: {vb:.1f}) | T3R {b:.1f} (C_b {cb} / C_r {cr}) = "
              f"{100 * (1 - b / a):.1f} % fewer than v3 as configured, {100 * (1 - b / vb):.1f} % than v3's best split", flush=True)
