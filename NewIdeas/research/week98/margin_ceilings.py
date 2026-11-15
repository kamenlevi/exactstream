"""MARGIN CEILINGS under the walk (NewIdeas, Sun 15 Nov 03:0x): kept/32 by certify98's decode walk if the head were made
right on every cell whose 125B margin is >= T nats (the drift / reader errors on rows the 125B is sure of), keeping its
real cells below T. T = inf is t30b as is; T = 4 fixes only sure rows; T = 0.25 fixes everything but near-ties.
Also: the same with depth 0 left as is (only depths >= 1 fixed), which is what a drift fix alone could buy.
CPU, read-only. usage: python -I margin_ceilings.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C
for nm in ("heplus", "mbpp100"):
    W = C.windows(os.path.expanduser(f"~/BigMoE/data/deepdraft/ceiling/cells_t30b_{nm}.npz"))
    print(f"== {nm} ({len(W)} windows)")
    for T in (np.inf, 4, 2, 1, 0.25, 0.0):
        for keep0 in (False, True):
            ks = []
            for rt, mt in W:
                r = rt.copy()
                fix = (r >= 0) & (mt >= T)
                if keep0: fix[:, 0] = False
                r[fix] = 1
                ks += C.kept_walk(r, C.R0)
            tag = "as is" if T == np.inf else f"fixed where m >= {T:g}" + (" (depth >= 1 only)" if keep0 else "")
            if T == np.inf and keep0: continue
            print(f"   {tag:42s} kept/32 {np.mean(ks):6.2f}")
