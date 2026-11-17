#!/usr/bin/env python3
"""BREAK98 bound for Breakthrough's LANDMARK-ANS (NewIdeas, Tue 17 Nov; time = the commit's): a landmark gives the head a
(near-)true 125B state at ~38 % of answer positions (k-gram match in the answer; cos 0.795 to the true state, like the
true state one position back). Modelled as RANDOM REFRESHES: each absolute position carries a landmark with probability
pi (one mask per window, shared by every anchor); a landmark at position q replaces the head's carried state there, so the
guess for q + 1 is the true-state guess and later depths run on anchor q's own chain until the next landmark. Upper bound:
the head treats a landmark exactly like the true state (Breakthrough's ceiling statement). Chain-32 kept, cells.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_landmark.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C98  # noqa: E402
import b98sim as S  # noqa: E402

def kept_landmark(W, pi, draws=8, seed=17):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(draws):
        ks = []
        for rt in W:
            n = rt.shape[0]
            lm = rng.random(n + S.DEP + 2) < pi                   # landmark at the state after anchor index q
            g = np.full(rt.shape, -1, np.int8)
            for p in range(n):
                r = 0
                for d in range(S.DEP):
                    if d > 0 and lm[p + d]:                       # a landmark at position p + d refreshes the state
                        r = d
                    a = p + r
                    if a >= n or d - r >= S.DEP:
                        break
                    g[p, d] = rt[a, d - r]
            g[rt < 0] = -1
            ks += C98.kept_walk(g, C98.R0)
        vals.append(np.mean(ks))
    return float(np.mean(vals))

for tag in ("heplus24", "mbppbank"):
    W = S.load(tag); base = S.kept(W, S.refresh(32)); lab = S.kept(W, S.refresh(1))
    row = []
    for pi in (0.2, 0.38, 0.5):
        v = kept_landmark(W, pi)
        row.append(f"pi {pi}: {v:.2f} ({100 * (v / base - 1):+.0f} %, {100 * (v - base) / (lab - base):.0f} % of the gap)")
    print(f"== {tag}: own {base:.2f}, LAB {lab:.2f}; random landmark refreshes: " + "; ".join(row))
