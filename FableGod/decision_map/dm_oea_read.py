"""Read-out of the OEA / union-stack arms (NewIdeas Tue 07:0x; masks ~/NewIdeas/data/dm_masks; pre-registered in SCOREBOARD
before the runs): oea_k8 / oea_k6 (every row keeps top-k0 plus the rank > k0 experts already in its check's union; OEA,
arXiv 2511.02237) and hyb_0.9_u8 / hyb_0.9_u6 (sure rows, the drafter's cf >= 0.9, k0 2; the others k0 8 / 6). Lines per
domain on all-row flips vs the baseline argmax: PASS <= Q1 at 0.99's level (gate_top2_cf099: code 1.34, IFEval 5.42,
multi 3.69 %), BETWEEN up to 1.5x that, KILL above. Gated code flips (cf >= 0.99) beside. Union bytes per check vs exact
are NewIdeas' CPU numbers (same blocks), quoted from its proposal.
Usage (BigMoE venv): python dm_oea_read.py -> out/OEA_read.txt"""
import json, os
import numpy as np
import dm_gate
OUT = os.path.expanduser(os.environ.get("DM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")))
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
BYTES = {"oea_k8": "x0.828 / 0.834 / 0.814", "oea_k6": "x0.647 / 0.661 / 0.631", "hyb_0.9_u8": "x0.527 / 0.756 / 0.762",
         "hyb_0.9_u6": "x0.435 / 0.603 / 0.594", "q1u_cf09": "x0.618 / 0.903 / 0.934", "gate_top2_cf099": "-",
         "hyb_0.9_u8_t3": "hyb_0.9_u8's union; sure rows' ranks 0-1 at 3 bits", "q1u_cf09_t3": "q1u_cf09's; sure rows' ranks 0-1 at 3 bits",
         "res_c256": "no demand reads (only resident experts; misses load for later checks)"}
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
def flips(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); out = {"g": [0, 0]}
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"])); f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        d = out.setdefault(b["dom"], [0, 0]); d[0] += int(f.sum()); d[1] += m
        if b["dom"] == "code":
            g = np.nan_to_num(G[nm], nan=-1.0)[b["a"]:b["a"] + m] >= 0.99; out["g"][0] += int(f[g].sum()); out["g"][1] += int(g.sum())
    return out
ref = flips("gate_top2_cf099"); lvl = {d: 100 * ref[d][0] / ref[d][1] for d in ("code", "ifeval", "multi")}
say(f"OEA / union-stack arms: all-row flips per domain vs Q1 at 0.99's level (code {lvl['code']:.2f} / IFEval {lvl['ifeval']:.2f} / "
    f"multi {lvl['multi']:.2f} %); PASS <= 1x, BETWEEN <= 1.5x, KILL above")
say("| arm | union bytes vs exact (code / IFEval / multi; NewIdeas CPU) | code | IFEval | multi | gated code flips (cf >= 0.99) | verdict per domain |")
say("|---|---|---|---|---|---|---|")
for arm in ("oea_k8", "oea_k6", "hyb_0.9_u8", "hyb_0.9_u6", "hyb_0.9_u8_t3", "q1u_cf09_t3", "q1u_cf09", "res_c256", "gate_top2_cf099"):
    X = flips(arm)
    if X is None: say(f"| {arm} | {BYTES[arm]} | not run | | | | |"); continue
    v = []
    for d in ("code", "ifeval", "multi"):
        r = (100 * X[d][0] / X[d][1]) / lvl[d]; v.append("PASS" if r <= 1 else "BETWEEN" if r <= 1.5 else "KILL")
    say(f"| {arm} | {BYTES[arm]} | " + " | ".join(f"{100 * X[d][0] / X[d][1]:.2f} %" for d in ("code", "ifeval", "multi"))
        + f" | {100 * X['g'][0] / max(X['g'][1], 1):.3f} % ({X['g'][0]}/{X['g'][1]}) | {' / '.join(v)} |")
open(os.path.join(OUT, "OEA_read.txt"), "w").write("\n".join(lines) + "\n")
