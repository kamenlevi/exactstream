#!/usr/bin/env python3
"""a2_tail_tf.py (drafterwall, DESCRIPTIVE, after T1; no line). The best a non-derailing tail could do: after the own
chain's first miss at depth k (every anchor of tree3_map), how often is the reader's TEACHER-FORCED top-1 / top-3 (it is
told the true previous token) right at the later depths of the same block? This bounds ledger ideas 4 and 15."""
import numpy as np, os, glob
TD = os.path.expanduser("~/TrainLab/data/tree3_map"); out = []
for dom in ("code", "ifeval", "multi"):
    acc = {k: [0, 0] for k in ("dfl tf top-1", "dfl tf top-3", "mtp tf top-1", "mtp tf top-3", "either tf top-1")}
    run = []
    for f in sorted(glob.glob(f"{TD}/{dom}/*.npz")):
        z = np.load(f); true, own, mm = z["true"], z["dfl_own"], z["m"]
        d1 = z["dfl_t3"][..., 0] == true; d3 = (z["dfl_t3"] == true[..., None]).any(-1)
        m1 = z["mtp_t3"][..., 0] == true; m3 = (z["mtp_t3"] == true[..., None]).any(-1)
        for a in range(true.shape[0]):
            m = int(mm[a])
            if m < 15: continue
            eq = own[a, :15] == true[a, :15]
            if eq.all(): continue
            k = int(np.argmin(eq)); sl = slice(k + 1, 15)
            if k + 1 >= 15: continue
            for nm, tab in (("dfl tf top-1", d1), ("dfl tf top-3", d3), ("mtp tf top-1", m1), ("mtp tf top-3", m3), ("either tf top-1", d1 | m1)):
                acc[nm][0] += int(tab[a, sl].sum()); acc[nm][1] += 15 - (k + 1)
            e = (d1 | m1)[a, sl]; r = 0
            for v in e:
                if v: r += 1
                else: break
            run.append(r)
    s = f"{dom}: after the own chain's first miss, later depths of the same block, told the true previous token: " + ", ".join(f"{k} {100 * a / max(b, 1):.1f} %" for k, (a, b) in acc.items()) + f"; mean right run of the either-oracle {np.mean(run):.2f} positions"
    print(s); out.append(s)
open(os.path.join(os.path.dirname(__file__), "..", "out", "a2_tail_tf.txt"), "w").write("\n".join(out) + "\n")
