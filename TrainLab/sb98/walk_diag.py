"""WEEK98 S5 WALKW read-out add-on (MTP Head): for each run, on the MBPP-50 and HE-30 read-out windows (evalx's first
50 / 30, sorted), its own walk's check starts: depth-0 accuracy there vs over all anchors, strict depths 0-14 along the
walk (right at depth j given right at 0..j-1, pooled over the walk's checks), and kept/32. Sandbox98's core.chain_table
and kept_walk logic, read-only. Usage: cd ~/Sandbox98 && ~/BigMoE/.venv/bin/python ~/TrainLab/sb98/walk_diag.py OUT.json RUN..."""
import os, sys, glob, json
sys.path.insert(0, os.path.expanduser("~/Sandbox98"))
import numpy as np
import mlx.core as mx
from sb import core as C

C.limit_memory(4.5)                      # Sandbox98's caps: 1 GB buffer cache, 4.5 GB memory (06:06 incident: without them the
                                         # cache grew to ~50 GB over the window loop)

OUT, RUNS = sys.argv[1], sys.argv[2:]


def files(d, n):
    return sorted(f for f in glob.glob(os.path.join(C.ROOT, "data", d, "*.npz")) if not f.endswith(".taps.npz"))[:n]


R = {}
for run in RUNS:
    D = C.load_drafter(run)
    for st, fl in (("mbpp", files("mbpp", 50)), ("he", files("he", 30))):
        d0s = [0, 0]; d0a = [0, 0]; strict = np.zeros((2, 15)); kept = []
        for f in fl:
            rt, gen, a0 = C.chain_table(D, f, 31)
            mx.clear_cache()
            r0 = max(0, gen - 1 - a0); sub = rt[r0:]; p = 0
            v = rt[:, 0] >= 0
            d0a[0] += int((rt[v, 0] == 1).sum()); d0a[1] += int(v.sum())
            while p < sub.shape[0]:
                k = 0
                while k < 31 and sub[p, k] == 1:
                    k += 1
                if sub[p, 0] >= 0:
                    d0s[0] += int(sub[p, 0] == 1); d0s[1] += 1
                for j in range(15):                        # strict: reached depth j (right at 0..j-1) and has a target
                    if j <= k and sub[p, j] >= 0:
                        strict[1, j] += 1; strict[0, j] += int(sub[p, j] == 1)
                kept.append(k + 1); p += k + 1
        R[f"{run}|{st}"] = {"depth0_at_check_starts": d0s[0] / max(d0s[1], 1), "n_starts": d0s[1],
                            "depth0_all": d0a[0] / max(d0a[1], 1), "kept_per_check": float(np.mean(kept)),
                            "walk_strict_0_14": (strict[0] / np.maximum(strict[1], 1)).round(4).tolist()}
        r = R[f"{run}|{st}"]
        print(f"{run} {st}: depth 0 at its check starts {100 * r['depth0_at_check_starts']:.2f} % (n {r['n_starts']}) vs all "
              f"{100 * r['depth0_all']:.2f} %; kept/check {r['kept_per_check']:.3f}; walk-strict 0-14 mean "
              f"{100 * float(np.mean(r['walk_strict_0_14'])):.2f} %", flush=True)
json.dump(R, open(OUT, "w"), indent=1)
