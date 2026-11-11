"""NewIdeas Wed 4 Nov 2026 03:3x: shows the keep-mask bug in ~/FableGod/fablec/bytes10_replay.py on one code block (base
union sizes per layer under its EXACT / Q1 0 t3 rules vs the true unions). Usage (BigMoE venv): python -B bytes10_bug_check.py"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
import numpy as np
import bytes10_replay as F
W = F.windows(); w = W["code"][0]
bl = F.live_blocks(w)
rows, cfs, kp = bl[0]
for rule in ("EXACT", "Q1 0 t3"):
    Ub, Ul, Ua, kp_, nr = F.block_sets(w, [bl[0]], rule, None)[0]
    nz = [l for l in range(48) if Ub[l]]
    print(rule, "rows", len(rows), "| layers with base experts:", len(nz), "first/last", nz[:3], nz[-3:],
          "| union sizes L0 / L9 / L10 / L47:", len(Ub[0]), len(Ub[9]), len(Ub[10]), len(Ub[47]))
E = w["ids"][rows]; print("true union sizes L10 / L47 (all 10 ranks):", len(np.unique(E[:, 10])), len(np.unique(E[:, 47])))
