#!/usr/bin/env python3
"""BREAK98 bound for IN-CHECK REPAIR rounds (ZERO's CHECK-AS-TAP-PASS; NewIdeas' early repair), via the cells (NewIdeas, Tue
17 Nov; time = the commit's). Per anchor p the chain runs on the head's own guesses; at the first wrong depth e a repair
round re-predicts depth e WITH the true state (lab = the depth-0 guess of anchor p + e) and refreshes the carried state
there (later depths use anchor p + e's own chain). R rounds per block = R repairs. UPPER BOUND: perfect error location and
true-state quality at the repair point (a layer-8/16 tap delivers less). Each round also costs a partial recompute of the
re-drafted rows inside the check. Chain-32 kept (cap 31), certify98 walk.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_repair.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C98  # noqa: E402
import b98sim as S  # noqa: E402

def accepted(rt, p, R, q=1.0, rng=None):
    """accepted depth for anchor p with R repair rounds; q = P(a repair's re-prediction carries true-state quality)"""
    n = rt.shape[0]; r = 0; d = 0; left = R
    while d < S.DEP:
        a = p + r
        if a >= n: break
        g = rt[a, d - r]
        if g < 0: break
        if g == 1:
            d += 1; continue
        if left > 0 and p + d < n and rt[p + d, 0] >= 0 and (rng is None or rng.random() < q):
            left -= 1
            if rt[p + d, 0] == 1:            # the true-state re-prediction at depth d is right
                r = d; d += 1; continue       # refresh: later depths use anchor p + d's own chain
        break
    return d

def kept(W, R, q=1.0, seed=17):
    rng = np.random.default_rng(seed) if q < 1.0 else None
    ks = []
    for rt in W:
        sub = rt[C98.R0:]; p = 0
        while p < sub.shape[0]:
            a = accepted(sub, p, R, q, rng); ks.append(a + 1); p += a + 1
    return float(np.mean(ks))

for tag in ("heplus24", "mbppbank"):
    W = S.load(tag); base = kept(W, 0); lab = S.kept(W, S.refresh(1))
    print(f"== {tag}: own chain-32 {base:.2f} (published 13.77 / 6.92); LAB {lab:.2f}")
    for R in (1, 2, 3, 5, 31):
        row = [f"q1.0 {kept(W, R):.2f} ({100 * (kept(W, R) / base - 1):+.0f} %, {100 * (kept(W, R) - base) / (lab - base):.0f} % of the gap)"]
        for q in (0.5, 0.3):
            v = kept(W, R, q); row.append(f"q{q} {v:.2f} ({100 * (v / base - 1):+.0f} %)")
        print(f"  {R:2d} repair round(s) per block: " + "; ".join(row))
