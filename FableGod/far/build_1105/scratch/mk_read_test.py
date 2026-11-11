"""Synthetic inputs for nw_read.py / nw_selfcheck.py's read-outs (logic test only: stand-ins made from Thursday's JSONs)."""
import json, os, sys, copy
import numpy as np
DMO = os.path.expanduser("~/FableGod/decision_map/out"); T = sys.argv[1]; mode = sys.argv[2]
os.makedirs(T, exist_ok=True)
tw = json.load(open(os.path.join(DMO, "qom_0_live_ball+code.json"))); s99 = json.load(open(os.path.join(DMO, "qom_0_split0.99_live_ball+code.json")))
s9 = json.load(open(os.path.join(DMO, "qom_0_split0.9_live_ball+code.json")))
def nw(R, d, writes, flips=0, noise=0.0, seed=0, warn=False):
    R = copy.deepcopy(R); r = np.random.RandomState(seed); k = 0
    for w in R:
        n = len(R[w]["tk"]); R[w]["nw"] = dict(d=d, late=10 * n, rows=n, writes=writes, terms=10 * n)
        if noise:
            lp = np.array(R[w]["lp"]); lp[:, 0] += r.randn(n) * noise; R[w]["lp"] = lp.tolist()
        while k < flips:
            j = r.randint(n); R[w]["tk"][j][0], R[w]["tk"][j][1] = R[w]["tk"][j][1], R[w]["tk"][j][0]; k += 1
            if k % 2 == 0: break
        if warn: R[w]["warn"] = "; WARNING never-wait used 1 late terms, the mask holds 2 (rows misaligned)"; warn = False
    return R
def put(name, R): json.dump(R, open(os.path.join(T, name + ".json"), "w"))
put("qom_0_live_ball+code", tw); put("qom_0_split0.99_live_ball+code", s99); put("qom_0_live_ball_c208+code", tw)
put("qom_0_live_ball_c208_nwz+code", nw(tw, 1, 0, flips=3 if mode == "badz" else 0))
put("qom_0_live_ball_c208_nw0+code", nw(tw, 0, 300, flips=2, noise=0.003, seed=1))
put("qom_0_live_ball_c208_nw1+code", nw(s99, 1, 300, warn=mode == "warn"))
put("qom_0_live_ball_c208_nw1r1+code", nw(tw, 1, 200, flips=4, seed=2))
put("qom_0_live_ball_nw1+code", nw(tw, 1, 100, flips=2, seed=3))
put("qom_0_split0.99_live_ball_nw1+code", nw(s9, 1, 100))
two = sorted(tw, key=lambda w: len(tw[w]["tk"]))[:2]; sub = lambda R: {w: R[w] for w in two}
for tag, R in (("twin", tw), ("twin2", tw), ("nwz", nw(tw, 1, 0)), ("nw0", nw(tw, 0, 300, flips=1, noise=0.003, seed=1)), ("nw1", nw(s99, 1, 300))):
    json.dump(sub(R), open(os.path.join(T, "sc_" + tag + ".json"), "w"))
print("synthetic read-out inputs in", T, "mode", mode)
