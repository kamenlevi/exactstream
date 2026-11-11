"""DAY9 read-out (Breakthrough, Mon 2026-11-02): conditions 1-3 of ROUTE_350 on the decision map.
Flip = the run's top-1 differs from the baseline's on a measured row. Rows are grouped by the expert count the drafter
gate gave them (dm_gate.py: DFlash2's cf for the draft the row verifies), with the 125B's own class as the oracle
column (sure p1 >= 0.99 / tie p1 - p2 <= 0.1 / mid). 95 % Wilson intervals on the line numbers.
Usage (BigMoE venv): python dm_day9_read.py -> out/DAY9_read.txt"""
import json, os, math
import numpy as np
import dm_gate
OUT = dm_gate.OUT
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
DOMS = ("code", "ifeval", "multi")
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def wilson(k, n, z=1.96):
    if not n: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def rows(name):
    """per measured row: domain, flip, oracle class, gate cf (-1 = none)"""
    run = json.load(open(os.path.join(OUT, name + ".json"))); out = []
    for nm, b in base.items():
        r = run[nm]; m = min(len(b["tk"]), len(r["tk"])); lp = np.array(b["lp"][:m])
        p1, p2 = np.exp(lp[:, 0]), np.exp(lp[:, 1]); c = np.where(p1 >= 0.99, 0, np.where(p1 - p2 <= 0.1, 1, 2))
        f = np.array(r["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        g = np.nan_to_num(G[nm][b["a"]:b["a"] + m], nan=-1.0)
        out.append((b["dom"], f, c, g))
    return out

def k_of(spec, g):
    k = np.full(len(g), 10)
    if spec[0] == "top": k[g >= spec[1]] = spec[2]
    else: k[g >= 0.9] = 8; k[g >= 0.99] = 4; k[g >= 0.999] = 2
    return k

def cell(f, sel):
    n = int(sel.sum()); k = int(f[sel].sum())
    return (k, n, 100 * k / n if n else float("nan"))

def fmt(k, n, p, ci=False):
    if not n: return "-"
    s = f"{p:.2f} ({k}/{n})"
    if ci: lo, hi = wilson(k, n); s += f" [{lo:.2f}-{hi:.2f}]"
    return s

def oracle_split(f, c, sel):
    return " / ".join(fmt(*cell(f, sel & (c == x))) for x in range(3))

verdict = {}
say("DAY9 conditions 1-3 on the decision map (36 windows, D-02a configuration, 4-bit experts; flips vs the baseline argmax)")
say("Columns: flip % (flips / rows); oracle split = the 125B's own class sure / tie / mid. Gate = DFlash2 cf (HPD-0 walk).")
for name, spec in (("gate_top2_cf099", ("top", 0.99, 2)), ("gate_top2_cf0999", ("top", 0.999, 2)),
                   ("gate_top2t4_cf099", ("top", 0.99, 2)), ("gate_top2t3_cf099", ("top", 0.99, 2)),
                   ("gate_top2t3_cf0999", ("top", 0.999, 2))):
    if not os.path.exists(os.path.join(OUT, name + ".json")): say(f"\n{name}: not run"); continue
    R = rows(name); say(f"\n## Q1 {name}: top-2 at the model's own weights on rows with cf >= {spec[1]}, the rest exact")
    say("| domain | gated rows (top-2): flip % [95 % CI] | gated, oracle sure / tie / mid | ungated rows (exact; context only) | all rows | gated share |")
    say("|---|---|---|---|---|---|")
    for dom in DOMS:
        f = np.concatenate([x[1] for x in R if x[0] == dom]); c = np.concatenate([x[2] for x in R if x[0] == dom])
        g = np.concatenate([x[3] for x in R if x[0] == dom]); k = k_of(spec, g); gt = k == 2
        kg = cell(f, gt)
        say(f"| {dom} | {fmt(*kg, ci=True)} | {oracle_split(f, c, gt)} | {fmt(*cell(f, ~gt))} | {fmt(*cell(f, np.ones_like(gt)))} | "
            f"{100 * gt.mean():.1f} % |")
        if dom == "code": verdict[name] = kg
if os.path.exists(os.path.join(OUT, "gate_bands.json")):
    R = rows("gate_bands"); spec = ("bands",)
    say("\n## Q2 gate_bands (NEW-1): cf >= 0.999 top-2 | 0.99-0.999 top-4 | 0.9-0.99 top-8 | < 0.9 or none exact")
    say("| domain | band | rows (share) | flip % [95 % CI] | oracle sure / tie / mid |")
    say("|---|---|---|---|---|")
    for dom in DOMS:
        f = np.concatenate([x[1] for x in R if x[0] == dom]); c = np.concatenate([x[2] for x in R if x[0] == dom])
        g = np.concatenate([x[3] for x in R if x[0] == dom]); k = k_of(spec, g); res = {}
        for kk, lab in ((2, ">= 0.999 top-2"), (4, "0.99-0.999 top-4"), (8, "0.9-0.99 top-8"), (10, "< 0.9 exact")):
            sel = k == kk; res[kk] = cell(f, sel)
            say(f"| {dom} | {lab} | {int(sel.sum())} ({100 * sel.mean():.1f} %) | {fmt(*res[kk], ci=True)} | {oracle_split(f, c, sel)} |")
        s12 = cell(f, (k == 2) | (k == 4))
        say(f"| {dom} | all rows | {len(f)} | {fmt(*cell(f, np.ones(len(f), bool)), ci=True)} | mid band / top-2 band "
            f"{res[8][2] / res[2][2] if res[2][2] else float('inf'):.2f}x; mid / (>= 0.99) {res[8][2] / s12[2] if s12[2] else float('inf'):.2f}x |")
        if dom == "code": verdict["gate_bands"] = (res, s12)
for name in ("hc8w", "hc8w_mix", "capture"):
    if not os.path.exists(os.path.join(OUT, name + ".json")): continue
    R = rows(name); say(f"\n## {name}: flip % by the 125B's own class (sure / tie / mid), all rows")
    for dom in DOMS:
        f = np.concatenate([x[1] for x in R if x[0] == dom]); c = np.concatenate([x[2] for x in R if x[0] == dom])
        say(f"- {dom}: sure {fmt(*cell(f, c == 0), ci=True)}; tie {fmt(*cell(f, c == 1))}; mid {fmt(*cell(f, c == 2))}; "
            f"all {fmt(*cell(f, np.ones(len(f), bool)))}")
        if dom == "code": verdict[name] = cell(f, c == 0)
say("\n## Verdicts against DAY9's lines")
for name in ("gate_top2_cf099", "gate_top2_cf0999", "gate_top2t4_cf099", "gate_top2t3_cf099", "gate_top2t3_cf0999"):
    if name in verdict:
        k, n, p = verdict[name]; lo, hi = wilson(k, n)
        v = "PASS" if p <= 1 else "KILL" if p > 5 else "BETWEEN"
        say(f"- Q1 {name}: code drafter-sure rows flip {p:.2f} % [{lo:.2f}-{hi:.2f}] ({k}/{n}) -> {v} (line <= 1 %, kill > 5 %)")
if "gate_bands" in verdict:
    res, s12 = verdict["gate_bands"]; r1 = res[8][2] / res[2][2] if res[2][2] else float("inf")
    r2 = res[8][2] / s12[2] if s12[2] else float("inf")
    say(f"- Q2 gate_bands: code mid band (0.9-0.99, top-8) {res[8][2]:.2f} % vs top-2 band {res[2][2]:.2f} % = {r1:.2f}x, vs all "
        f">= 0.99 rows {s12[2]:.2f} % = {r2:.2f}x -> {'KILL' if min(r1, r2) > 2 else 'PASS' if max(r1, r2) <= 2 else 'BETWEEN (the two readings of the sure band disagree)'} (kill > 2x)")
for name in ("hc8w", "hc8w_mix"):
    if name in verdict:
        k, n, p = verdict[name]
        say(f"- Q3 {name}: code-sure flip {p:.2f} % ({k}/{n}); the condition's gate is EvalSuite's (static > 5 % is the map kill)")
if "capture" in verdict: say(f"- capture self-check: code-sure flips {verdict['capture'][0]} (must be 0)")
if all(os.path.exists(os.path.join(OUT, n + ".json")) for n in ("gate_top2_cf099", "gate_top2t4_cf099")):
    a, b = json.load(open(os.path.join(OUT, "gate_top2_cf099.json"))), json.load(open(os.path.join(OUT, "gate_top2t4_cf099.json")))
    d = n = 0
    for nm in a:
        x, y = np.array(a[nm]["tk"])[:, 0], np.array(b[nm]["tk"])[:, 0]; m = min(len(x), len(y)); d += int((x[:m] != y[:m]).sum()); n += m
    say(f"- control: gate_top2t4_cf099 (top-2 4-bit through dm_t3's path) vs gate_top2_cf099 (the stock path): top-1 differs on {d} of {n} rows")
if os.path.exists(os.path.join(OUT, "gate_top2t4_cf099.json")):      # the 45 GB read-across (orchestrator Mon 23:2x)
    say("\n## 3-bit ranks 0-1, read across to 45 GB: flips by the 125B's own class, t3 (MSB-3 records) vs the t4 control")
    say("The arm puts the gated rows' ranks 0-1 at 3 bits (ranks 2-9 dropped, as Q1); t4 runs the same path at 4 bits, so"
        " t3 - t4 is the 3-bit effect alone. 45 GB's plan 3333333333 puts every rank of every row at 3 bits: these are its"
        " ranks-0-1-on-sure-rows part, a lower bound.")
    say("| run | rows | domain | sure | tie | mid | all |")
    say("|---|---|---|---|---|---|---|")
    T4 = rows("gate_top2t4_cf099")
    for name, thr in (("gate_top2t3_cf099", 0.99), ("gate_top2t3_cf0999", 0.999)):
        if not os.path.exists(os.path.join(OUT, name + ".json")): continue
        R3 = rows(name)
        for tag, R in ((name, R3), ("gate_top2t4_cf099 (control)", T4)):
            for which in ("gated", "all"):
                for dom in DOMS:
                    X = [x for x in R if x[0] == dom]
                    f = np.concatenate([x[1] for x in X]); c = np.concatenate([x[2] for x in X]); g = np.concatenate([x[3] for x in X])
                    sel = (g >= thr) if which == "gated" else np.ones(len(f), bool)
                    say(f"| {tag} | {which} (cf >= {thr}) | {dom} | {fmt(*cell(f, sel & (c == 0)))} | {fmt(*cell(f, sel & (c == 1)))} | "
                        f"{fmt(*cell(f, sel & (c == 2)))} | {fmt(*cell(f, sel))} |" if which == "gated" else
                        f"| {tag} | all rows | {dom} | {fmt(*cell(f, c == 0))} | {fmt(*cell(f, c == 1))} | {fmt(*cell(f, c == 2))} | "
                        f"{fmt(*cell(f, sel))} |")
open(os.path.join(OUT, "DAY9_read.txt"), "w").write("\n".join(lines) + "\n")
