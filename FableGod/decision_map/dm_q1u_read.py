"""Read-out of Q1-UNION (pre-registered SCOREBOARD a6dd04e): gated rows (the drafter's cf >= theta on the measured rows)
and their flips vs the baseline argmax, against Q1 at the same theta (gate_top2_cf099 / gate_top2_cf09); all rows by
domain beside. Lines: PASS = gated code flips <= 0.03 % at 0.99 and <= 0.5 % at 0.9; KILL = >= Q1's at the same theta.
Usage (BigMoE venv): python dm_q1u_read.py -> out/Q1U_read.txt"""
import json, os, math
import numpy as np
import dm_gate
OUT = os.path.expanduser(os.environ.get("DM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")))
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
def load(n):
    p = os.path.join(OUT, n + ".json"); return json.load(open(p)) if os.path.exists(p) else None
def wilson(k, n, z=1.96):
    if not n: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), 100 * (c + h)
def count(R, thr, dom):
    gk = gn = ak = an = 0
    for nm, b in base.items():
        if b["dom"] != dom: continue
        a = b["a"]; m = min(len(b["tk"]), len(R[nm]["tk"]))
        f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        g = np.nan_to_num(G[nm], nan=-1.0)[a:a + m] >= thr
        gk += int(f[g].sum()); gn += int(g.sum()); ak += int(f.sum()); an += m
    return gk, gn, ak, an
say("Q1-UNION vs Q1 on the decision map (gated rows = the drafter's cf >= theta; flips vs the baseline argmax)")
say("| run | theta | code gated flips [95 % CI] | code all rows | IFEval all rows | multi all rows | verdict |")
say("|---|---|---|---|---|---|---|")
for thr, run, ref in ((0.99, "q1u_cf099", "gate_top2_cf099"), (0.9, "q1u_cf09", "gate_top2_cf09")):
    q = {}
    for name in (ref, run):
        R = load(name)
        if not R: say(f"| {name} | {thr} | not run | | | | |"); continue
        gk, gn, ak, an = count(R, thr, "code"); q[name] = 100 * gk / max(gn, 1); lo, hi = wilson(gk, gn)
        oth = [count(R, thr, d) for d in ("ifeval", "multi")]
        v = ""
        if name == run and ref in q:
            line = 0.03 if thr == 0.99 else 0.5
            v = "PASS" if q[run] <= line else "KILL" if q[run] >= q[ref] else "BETWEEN"
        say(f"| {name} | {thr} | {q[name]:.3f} % ({gk}/{gn}) [{lo:.3f}-{hi:.3f}] | {100 * ak / an:.2f} % | "
            + " | ".join(f"{100 * o[2] / o[3]:.2f} %" for o in oth) + f" | {v} |")
open(os.path.join(OUT, "Q1U_read.txt"), "w").write("\n".join(lines) + "\n")
