#!/usr/bin/env python3
"""ADAPTIVE K sensitivity (NewIdeas, Tue 17 Nov 01:0x): what survives WITHOUT new check kernels.
 (a) KERN2: rows {16, 32} only (today's A16 + C32one);
 (b) MULTIPASS: > 32 rows as repeated passes of today's kernels (48 = 32 + 16, 64 = 2 x 32, 96 = 3 x 32, 128 = 4 x 32;
     24 pads to 32), i.e. no one-pass K-row forms;
 (c) ONEPASS: Leap's v2 one-pass growth rescaled to E14 (adaptk_draft's e14), for reference.
Draft cost L 0.40 / 0.92 ms per level; POLICY = pop while path_p >= tau (in-sample + cross-set tau); ORACLE = hindsight DP.
"""
import json, os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tree_audit as TA
import adaptk_draft as AD
from adaptk_oracle import MODELS
A16, OV, R32 = 62.49, 4.0, 1.4372
c = {16: 1.025, 24: 1.035, 32: 1.04, 48: 1.055, 64: 1.07, 96: 1.09, 128: 1.11}
t16, t32 = 1.0 * c[16] * A16, R32 * c[32] * A16
MP = {16: t16 + OV, 24: t32 + OV, 32: t32 + OV, 48: t32 + t16 + OV, 64: 2 * t32 + OV, 96: 3 * t32 + OV, 128: 4 * t32 + OV}
VAR = {"KERN2": ([16, 32], MODELS["e14"]), "MULTIPASS": (AD.ROWS, MP), "ONEPASS": (AD.ROWS, MODELS["e14"])}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
data = {}
for s, (tag, _rt, pub32, _p16) in TA.SETS.items():
    W = TA.load_windows(tag, pub32)
    z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
    idx, PP, A, Hh, mono = AD.dense({k: z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p")})
    data[s] = (AD.steps(W, idx), A, Hh, PP)
res = {}
for vn, (rows, T) in VAR.items():
    AD.ROWS[:] = rows if vn == "KERN2" else [16, 24, 32, 48, 64, 96, 128]
    for L in (0.40, 0.92):
        best = {}
        for s, (S, A, Hh, PP) in data.items():
            cost = (lambda r, i, T=T, L=L, Hh=Hh: T[r] + (L * float(Hh[r][i]) if i >= 0 else 0.0))
            fx = {r: AD.walk(S, (lambda i, r=r: r), A, cost)[0] for r in AD.ROWS}
            rb = max(fx, key=fx.get)
            pol = {}
            for t in AD.TAUS:
                npop = np.maximum(15, (PP >= t).sum(1))
                ro = np.array([next((r for r in AD.ROWS if r - 1 >= k), AD.ROWS[-1]) for k in npop])
                pol[t] = AD.walk(S, (lambda i, ro=ro: int(ro[i])), A, cost)
            tb = max(pol, key=lambda t: pol[t][0]); best[s] = (tb, pol, fx, rb)
            o = AD.oracle(S, A, cost, fx[rb])
            say(f"[{vn} L {L:.2f}] {s}: best fixed {rb} {fx[rb]:.1f} | ORACLE {o:.1f} ({100 * (o / fx[rb] - 1):+.1f} %) | "
                f"POLICY tau {tb} {pol[tb][0]:.1f} ({100 * (pol[tb][0] / fx[rb] - 1):+.1f} %) use {pol[tb][1]}")
            res.setdefault(vn, {}).setdefault(str(L), {})[s] = {"best_fixed": rb, "fixed": fx, "oracle": o, "tau": tb,
                                                                "policy": pol[tb][0], "use": pol[tb][1]}
        for s in data:
            other = [x for x in data if x != s][0]; t = best[other][0]; tb, pol, fx, rb = best[s]
            v = pol[t][0]; res[vn][str(L)][s]["cross"] = {"tau": t, "tps": v}
            say(f"[{vn} L {L:.2f}] {s}: CROSS tau {t} {v:.1f} ({100 * (v / fx[rb] - 1):+.1f} % vs best fixed)")
json.dump(res, open(os.path.join(HERE, "adaptk_sens.json"), "w"), indent=1)
open(os.path.join(HERE, "adaptk_sens.txt"), "w").write("\n".join(lines) + "\n")
