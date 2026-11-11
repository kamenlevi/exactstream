"""Read-out of the live-cut union arms (NewIdeas, Tue 3 Nov 2026; pre-registered in ~/NewIdeas/QUEUE.md, commit 7a0339d):
q1u_0.9_live (Q1-UNION 0.9 with unions over live-cut checks) and reach_0.9_0.5_live (+ REACH-CUT). Lines (the union arms'):
per domain all-row flips <= Q1 at 0.99's level (gate_top2_cf099) = PASS, <= 1.5x = BETWEEN, above = KILL. Paired (McNemar)
vs q1u_cf09 (the 16-row map arm) and REACH vs q1u_0.9_live. Reads ~/FableGod/decision_map/out (read-only); writes
~/NewIdeas/data/LIVE_read.txt when run without args. Usage (BigMoE venv): python live_read.py [live_run reach_run]
(defaults: the first existing of q1u_0.9_live / q1u_cf09_live, and reach_0.9_0.5_live)."""
import json, math, os, sys
import numpy as np
OUT = os.path.expanduser("~/FableGod/decision_map/out"); DST = os.path.expanduser("~/NewIdeas/data/LIVE_read.txt")
base = json.load(open(os.path.join(OUT, "baseline.json"))); DOMS = ("code", "ifeval", "multi")
lines = []
def say(s=""): print(s, flush=True); lines.append(s)


def first(*names):
    for n in names:
        if os.path.exists(os.path.join(OUT, n + ".json")): return n
    return names[0]


def flips(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); v = {}
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"]))
        f = np.r_[np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0], np.zeros(len(b["tk"]) - m, bool)]
        v.setdefault(b["dom"], []).append(f)
    return {d: np.concatenate(x) for d, x in v.items()}


live = sys.argv[1] if len(sys.argv) > 1 else first("q1u_0.9_live", "q1u_cf09_live", "p1_q1u_0.9_live")
reach = sys.argv[2] if len(sys.argv) > 2 else first("reach_0.9_0.5_live", "p1_reach_0.9_0.5_live")
F = {a: flips(a) for a in ("gate_top2_cf099", "gate_top2_cf09", "q1u_cf09", live, reach)}
lvl = {d: 100 * F["gate_top2_cf099"][d].mean() for d in DOMS}
say(f"Live-cut union arms: all-row flips per domain vs Q1 at 0.99's level (code {lvl['code']:.2f} / IFEval {lvl['ifeval']:.2f} / "
    f"multi {lvl['multi']:.2f} %); PASS <= 1x, BETWEEN <= 1.5x, KILL above")
say("| arm | code | IFEval | multi | verdict per domain |")
say("|---|---|---|---|---|")
for a in ("q1u_cf09", "gate_top2_cf09", live, reach):
    X = F[a]
    if X is None: say(f"| {a} | not run | | | |"); continue
    v = []
    for d in DOMS:
        r = 100 * X[d].mean() / lvl[d]; v.append("PASS" if r <= 1 else "BETWEEN" if r <= 1.5 else "KILL")
    say(f"| {a} | " + " | ".join(f"{100 * X[d].mean():.2f} % ({int(X[d].sum())})" for d in DOMS) + f" | {' / '.join(v)} |")
say("\nPaired (McNemar: only-A, only-B, z):")
for a, b in ((live, "q1u_cf09"), (reach, live), (live, "gate_top2_cf099"), (reach, "gate_top2_cf099")):
    if F[a] is None or F[b] is None: continue
    for d in DOMS:
        x, y = F[a][d], F[b][d]; oa, ob = int((x & ~y).sum()), int((~x & y).sum())
        say(f"  {a} vs {b} {d}: {oa} / {ob}, z {(oa - ob) / math.sqrt(max(oa + ob, 1)):+.2f}")
if len(sys.argv) == 1: open(DST, "w").write("\n".join(lines) + "\n")
