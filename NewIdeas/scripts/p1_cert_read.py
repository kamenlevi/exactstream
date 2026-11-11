"""P1-OEA read-out (NewIdeas, Tue 3 Nov 2026; pre-registered in ~/NewIdeas/QUEUE.md, commit 37f8e37): the 12 GB two-pass
check's certificate from three all-rows map runs (Breakthrough's harness, my masks): p1_top_k2 (control: plain top-2 on the
realized rows, prompt exact), p1_oea_k2 (top-2 + union members), p1_oea_k1 (top-1 + union members). The CPU columns of
Breakthrough's dm_cert_read.py, unchanged in method: certified share (p1 >= theta under the run's cut), blocks needing
pass 2 (an uncertified row among A .. A + k of the exact walk's checks; prompt rows are prefill's), pass-1-alone flips on
the certified rows (Wilson 95 %), all-row flips. Reads ~/FableGod/decision_map/out (read-only); writes
~/NewIdeas/data/P1OEA_read.txt. Usage (BigMoE venv): python p1_cert_read.py [control] [arm ...] (defaults: the three runs;
any all-rows run, e.g. topk2, can stand in for a self-test)."""
import json, math, os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out"); DST = os.path.expanduser("~/NewIdeas/data/P1OEA_read.txt")
base = json.load(open(os.path.join(OUT, "baseline.json"))); DOMS = ("code", "ifeval", "multi"); THR = (0.8, 0.9, 0.95, 0.99)
runs = sys.argv[1:] or ["p1_top_k2", "p1_oea_k2", "p1_oea_k1"]
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def wilson(k, n, z=1.96):
    if not n: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), 100 * (c + h)

CHECKS = {}
for dom, wf in dm_gate.MAPW.items():
    M = np.load(os.path.join(dm_gate.TD, wf))
    for s, nm in enumerate([str(x) for x in M["names"]][:12]):
        CHECKS[nm] = [(int(A), int(k)) for (k, m, A) in M[f"{s}_checks"]]

def blocks(nm, ok):
    a, n = base[nm]["a"], base[nm]["a"] + len(ok) + 1
    cert = np.r_[np.full(a, True), ok, False]
    return sum(bool((~cert[A:min(A + k, n - 2) + 1]).any()) for A, k in CHECKS[nm]), len(CHECKS[nm])

def cols(S, dom, thr):
    A = dict(n=0, nc=0, f1=0, fa=0, bn=0, b=0)
    for nm, b in base.items():
        if b["dom"] != dom: continue
        m = min(len(b["tk"]), len(S[nm]["tk"]))
        p1 = np.exp(np.array(S[nm]["lp"])[:, 0]); full = p1 >= thr; cert = full[:m]
        f = np.array(S[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        A["n"] += m; A["nc"] += int(cert.sum()); A["f1"] += int(f[cert].sum()); A["fa"] += int(f.sum())
        bn, bt = blocks(nm, full); A["bn"] += bn; A["b"] += bt
    return A

R = {}
for r in runs:
    p = os.path.join(OUT, r + ".json")
    if os.path.exists(p): R[r] = json.load(open(p))
say(f"P1-OEA read-out ({', '.join(f'{r}: ' + ('done' if r in R else 'NOT RUN') for r in runs)}); flips vs the baseline argmax")
say("| run | domain | theta | certified share | blocks needing pass 2 | pass 1 alone: certified-row flips [95 %] | all rows flip |")
say("|---|---|---|---|---|---|---|")
T = {}
for r in runs:
    if r not in R: continue
    for dom in DOMS:
        for thr in THR:
            A = cols(R[r], dom, thr); T[(r, dom, thr)] = A; lo, hi = wilson(A["f1"], A["nc"])
            say(f"| {r} | {dom} | {thr} | {100 * A['nc'] / A['n']:.1f} % | {100 * A['bn'] / A['b']:.0f} % ({A['bn']}/{A['b']}) | "
                f"{100 * A['f1'] / max(A['nc'], 1):.2f} % ({A['f1']}/{A['nc']}) [{lo:.2f}-{hi:.2f}] | {100 * A['fa'] / A['n']:.2f} % |")
ctl = runs[0]
if ctl in R:
    say(f"\n## Verdicts (pre-registered 37f8e37; code, theta 0.8, vs {ctl})")
    C = T[(ctl, "code", 0.8)]; cu = wilson(C["f1"], C["nc"])[1]; cb = 100 * C["bn"] / C["b"]
    for r in runs[1:]:
        if r not in R: say(f"- {r}: not run"); continue
        A = T[(r, "code", 0.8)]; f = 100 * A["f1"] / max(A["nc"], 1); b = 100 * A["bn"] / A["b"]
        if "k1" in r:
            v = "KILL" if f > 2 * cu or b > cb + 20 else "PASS" if f <= cu and b <= cb + 10 else "BETWEEN"
            rule = "PASS: flips <= control's upper bound and blocks <= control + 10 points; KILL: > 2x the bound or > +20 points"
        else:
            v = "PASS" if b <= cb - 5 and f <= cu else "KILL"
            rule = "PASS: blocks <= control - 5 points with flips <= the control's upper bound; else KILL as a lever"
        say(f"- {r}: certified flips {f:.2f} % vs control upper bound {cu:.2f} %; blocks needing pass 2 {b:.0f} % vs control "
            f"{cb:.0f} % -> {v} ({rule})")
os.makedirs(os.path.dirname(DST), exist_ok=True)
if len(sys.argv) == 1: open(DST, "w").write("\n".join(lines) + "\n")
