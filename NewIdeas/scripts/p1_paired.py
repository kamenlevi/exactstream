"""P1-OEA paired check (NewIdeas, Tue 3 Nov 2026): the same blocks under the control and an arm. For each domain and
theta: blocks needing pass 2 in the control only / the arm only (McNemar z), and certified-row flips in each. Same
definitions as p1_cert_read.py (dm_cert_read's CPU columns). Reads ~/FableGod/decision_map/out (read-only); writes nothing.
Usage (BigMoE venv): python p1_paired.py [control] [arm ...] (defaults: p1_top_k2 vs p1_oea_k2, p1_oea_k1)."""
import json, math, os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
runs = sys.argv[1:] or ["p1_top_k2", "p1_oea_k2", "p1_oea_k1"]
CHECKS = {}
for dom, wf in dm_gate.MAPW.items():
    M = np.load(os.path.join(dm_gate.TD, wf))
    for s, nm in enumerate([str(x) for x in M["names"]][:12]):
        CHECKS[nm] = [(int(A), int(k)) for (k, m, A) in M[f"{s}_checks"]]


def vecs(S, dom, thr):
    need, flip_c = [], 0
    for nm, b in base.items():
        if b["dom"] != dom: continue
        m = min(len(b["tk"]), len(S[nm]["tk"]))
        full = np.exp(np.array(S[nm]["lp"])[:, 0]) >= thr
        f = np.array(S[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        flip_c += int(f[full[:m]].sum())
        a, n = b["a"], b["a"] + len(full) + 1
        cert = np.r_[np.full(a, True), full, False]
        need += [bool((~cert[A:min(A + k, n - 2) + 1]).any()) for A, k in CHECKS[nm]]
    return np.array(need), flip_c


R = {r: json.load(open(os.path.join(OUT, r + ".json"))) for r in runs if os.path.exists(os.path.join(OUT, r + ".json"))}
ctl = runs[0]
for r in runs[1:]:
    if r not in R: print(f"{r}: not run"); continue
    for dom in ("code", "ifeval", "multi"):
        for thr in (0.8, 0.9):
            c, fc = vecs(R[ctl], dom, thr); x, fx = vecs(R[r], dom, thr)
            oc, ox = int((c & ~x).sum()), int((~c & x).sum()); z = (ox - oc) / math.sqrt(max(oc + ox, 1))
            print(f"{r} vs {ctl} {dom} theta {thr}: blocks {len(c)}, pass 2 {int(c.sum())} -> {int(x.sum())} "
                  f"({100 * (x.mean() - c.mean()):+.1f} pt); control-only {oc}, arm-only {ox}, McNemar z {z:+.2f}; "
                  f"certified flips {fc} -> {fx}")
