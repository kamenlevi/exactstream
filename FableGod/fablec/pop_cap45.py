#!/usr/bin/env python3 -I
"""pop_cap45.py (Fable C, C4 §5): the popularity table for the COLD-CUT map screen: share of cap45's live checks (all six
prompts, t 0.3 transplant seed 0) whose EXACT layer-union holds the expert; [48, 512] float32 -> masks/pop_cap45.npy.
A prior from other prompts than the map's windows, as the engine's eviction prior would be. CPU, ~10 s."""
import os, sys
sys.argv = ["replay_c4.py"]
import numpy as np
src = open("replay_c4.py").read().split("res = {}")[0]          # definitions and data loading only (no replay)
exec(compile(src, "replay_c4.py", "exec"))
T = transplant(0)
Q = needs(T, R_exact, [[None] * NL] * len(P))
cnt = np.zeros((NL, E), np.float32); n = 0
for p in range(len(P)):
    for t in range(len(P[p][1])):
        n += 1
        for l in range(NL):
            cnt[l, list(Q[p][t][l][0])] += 1
pop = cnt / n
np.save("masks/pop_cap45.npy", pop)
print(f"masks/pop_cap45.npy: {pop.shape}, {n} checks; experts with pop >= 0.05 per layer: mean {np.mean((pop >= 0.05).sum(1)):.0f} of 512; "
      f"share of EXACT pairs on cold experts (< 0.05): {np.mean([(pop[l, c['rid'][l].ravel()] < 0.05).mean() for _, cks, _ in P for c in cks for l in range(NL)]):.3f}")
