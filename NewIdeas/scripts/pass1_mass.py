"""Pass-1 candidates for the 12 GB two-pass check (NewIdeas, Tue 3 Nov 2026): gate mass a row keeps under each mask set,
from the masks' kept ranks and Breakthrough's measured mean profile (data/gmass/gate_mass.txt: rank 0 0.212, 0-1 0.358,
0-3 0.575, 50/70/90 % at 3.74/5.91/8.79 ranks; ranks 2-9 split by a smooth fit). An estimate: it assumes a union member
carries its rank's mean mass. Realized rows only (each check's A .. first miss), as the masks.
Usage (BigMoE venv): python pass1_mass.py <mask_root> <arm> [...]"""
import os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
PROF = np.array([0.212, 0.146, 0.115, 0.102, 0.088, 0.079, 0.071, 0.066, 0.062, 0.059]); PROF /= PROF.sum()
root = sys.argv[1]
for arm in sys.argv[2:]:
    st = {}
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        for i, nm in enumerate(str(x) for x in Z["names"]):
            p = os.path.join(root, arm, nm)
            if not os.path.exists(p): continue
            keep = np.load(p)["keep"]; N = len(keep)
            rows = sorted({r for (k, m, A) in Z[f"{i}_checks"].tolist() for r in range(A, min(A + k, N - 1) + 1)})
            K = keep[rows]                                   # [rows, 48, 10]
            d = st.setdefault(dom, [np.zeros(10), 0, []])
            d[0] += K.sum(axis=(0, 1)); d[1] += K.shape[0] * K.shape[1]; d[2].append((K * PROF).sum(-1).ravel())
    for dom, (rk, n, ms) in st.items():
        m = np.concatenate(ms)
        print(f"{arm} {dom}: mean kept mass {m.mean():.3f} (P10 {np.percentile(m, 10):.3f}), ranks kept % "
              + " ".join(f"{100 * x / n:.0f}" for x in rk), flush=True)
print("reference: top-1 0.212, top-2 0.358, top-3 0.473, top-4 0.575 (profile)")
