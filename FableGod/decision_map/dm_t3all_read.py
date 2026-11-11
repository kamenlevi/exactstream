"""Read-out of t3_all (pre-registered SCOREBOARD b17bf5f; 45 GB's plan 3333333333, the 1-bit sidecar dropped): every row's
ten routed experts from records3 at 3 bits vs the t4_all control (the same path at 4 bits). Flips vs the baseline argmax by
the 125B's own class (sure p1 >= 0.99 / tie p1 - p2 <= 0.1 / mid) on code, IFEval and multi. Lines: PASS if all-row code
flips stay within +0.3 points of t4_all AND code-sure rows flip <= 1 %; KILL if code flips go > +1 point over t4_all.
Usage (BigMoE venv): python dm_t3all_read.py -> out/T3ALL_read.txt"""
import json, os, math
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
def wilson(k, n, z=1.96):
    if not n: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), 100 * (c + h)
def cells(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); out = {}
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"])); lp = np.array(b["lp"][:m]); p1, p2 = np.exp(lp[:, 0]), np.exp(lp[:, 1])
        c = np.where(p1 >= 0.99, 0, np.where(p1 - p2 <= 0.1, 1, 2))
        f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        d = out.setdefault(b["dom"], [[0, 0] for _ in range(4)])
        for k in range(3): d[k][0] += int(f[c == k].sum()); d[k][1] += int((c == k).sum())
        d[3][0] += int(f.sum()); d[3][1] += m
    return out
def fmt(k, n, ci=False):
    if not n: return "-"
    s = f"{100 * k / n:.2f} % ({k}/{n})"
    if ci: lo, hi = wilson(k, n); s += f" [{lo:.2f}-{hi:.2f}]"
    return s
say("t3_all (all ranks, all rows at 3 bits) vs the t4_all control: flips vs the baseline argmax by the 125B's own class")
say("| run | domain | sure | tie | mid | all rows |")
say("|---|---|---|---|---|---|")
C = {n: cells(n) for n in ("t4_all", "t3_all")}
for n, X in C.items():
    if X is None: say(f"| {n} | not run | | | | |"); continue
    for dom in ("code", "ifeval", "multi"):
        d = X[dom]; say(f"| {n} | {dom} | {fmt(*d[0], ci=True)} | {fmt(*d[1])} | {fmt(*d[2])} | {fmt(*d[3], ci=True)} |")
if C["t3_all"] and C["t4_all"]:
    a3, a4 = C["t3_all"]["code"], C["t4_all"]["code"]
    d_all = 100 * a3[3][0] / a3[3][1] - 100 * a4[3][0] / a4[3][1]; sure = 100 * a3[0][0] / max(a3[0][1], 1)
    v = "KILL" if d_all > 1 else "PASS" if d_all <= 0.3 and sure <= 1 else "BETWEEN"
    say(f"\nVerdict (b17bf5f): code all-row flips t3_all - t4_all = {d_all:+.2f} points, code-sure rows {sure:.2f} % -> {v} "
        f"(PASS <= +0.3 and sure <= 1 %; KILL > +1)")
    for dom in ("ifeval", "multi"):
        b3, b4 = C["t3_all"][dom], C["t4_all"][dom]
        say(f"- {dom}: all rows {100 * b3[3][0] / b3[3][1] - 100 * b4[3][0] / b4[3][1]:+.2f} points over t4_all; sure rows "
            f"{fmt(*b3[0])} vs {fmt(*b4[0])}")
open(os.path.join(OUT, "T3ALL_read.txt"), "w").write("\n".join(lines) + "\n")
