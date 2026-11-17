#!/usr/bin/env python3
"""ADAPTIVE K: per-window spread + window bootstrap of the policy's gain (NewIdeas, Tue 17 Nov; label from date in the
output). Frozen tau (0.9 = MBPP's, applied to HE+; 0.5 = HE+'s, applied to MBPP), KERN2 (16 / 32 rows) and ONEPASS,
L 0.40 / 0.66 / 0.92 ms per level. Is the HE+ gain broad across windows, or a few copy-heavy windows?"""
import json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tree_audit as TA
import adaptk_draft as AD
from adaptk_oracle import MODELS
from adaptk_sens import VAR
rng = np.random.default_rng(17)
lines = [f"run {time.strftime('%a %H:%M')}"]
def say(s=""):
    print(s, flush=True); lines.append(s)
def per_window(S, choose, A, cost):
    out = []
    for ids in S:
        kept = tm = 0.0; n = len(ids); p = 1
        while p < n:
            i = ids[p]; r = choose(i) if i >= 0 else 16; a = int(A[r][i]) if i >= 0 else 0
            kept += a + 1; tm += cost(r, i); p += a + 1
        out.append((kept, tm))
    return np.array(out)
res = {}
for s, (tag, _rt, pub32, _p16) in TA.SETS.items():
    W = TA.load_windows(tag, pub32)
    z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
    idx, PP, A, Hh, mono = AD.dense({k: z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p")})
    S = AD.steps(W, idx)
    for vn in ("KERN2", "ONEPASS"):
        rows, T = VAR[vn]
        for L in (0.40, 0.66, 0.92):
            cost = (lambda r, i, T=T, L=L, Hh=Hh: T[r] + (L * float(Hh[r][i]) if i >= 0 else 0.0))
            base = per_window(S, (lambda i: 16), A, cost)
            for tau in (0.9, 0.5):
                npop = np.maximum(15, (PP >= tau).sum(1))
                ro = np.array([next((r for r in rows if r - 1 >= k), rows[-1]) for k in npop])
                pol = per_window(S, (lambda i, ro=ro: int(ro[i])), A, cost)
                g_w = (pol[:, 0] / pol[:, 1]) / (base[:, 0] / base[:, 1]) - 1
                pooled = (pol[:, 0].sum() / pol[:, 1].sum()) / (base[:, 0].sum() / base[:, 1].sum()) - 1
                bs = []
                for _ in range(2000):
                    j = rng.integers(0, len(S), len(S))
                    bs.append((pol[j, 0].sum() / pol[j, 1].sum()) / (base[j, 0].sum() / base[j, 1].sum()) - 1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                say(f"{s} {vn} L {L:.2f} tau {tau}: pooled {100 * pooled:+.1f} % [95 % window-bootstrap {100 * lo:+.1f}, "
                    f"{100 * hi:+.1f}]; windows up {int((g_w > 0.005).sum())} / flat {int((abs(g_w) <= 0.005).sum())} / "
                    f"down {int((g_w < -0.005).sum())} of {len(S)}; per-window median {100 * np.median(g_w):+.1f} %, max "
                    f"{100 * g_w.max():+.1f} %, min {100 * g_w.min():+.1f} %")
                res.setdefault(s, {}).setdefault(vn, {})[f"L{L}_tau{tau}"] = {"pooled": pooled, "ci": [lo, hi],
                    "per_window": g_w.tolist()}
json.dump(res, open(os.path.join(HERE, "adaptk_windows.json"), "w"), indent=1)
open(os.path.join(HERE, "adaptk_windows.txt"), "w").write("\n".join(lines) + "\n")
