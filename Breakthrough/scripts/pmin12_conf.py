"""PMIN12 confirmation read-out (lines in SCOREBOARD 'PMIN12 CONFIRM LINES', written before the run): per prompt and arm,
the faster of its two ABBA runs (a stall only slows a run) and, as a check, their mean; per type, the geometric mean of the
ratios to p_min 0; rows and tokens per pass; token equality with the first p_min 0 run.
Usage: [BigMoE venv] python pmin12_conf.py [data/pmin12/ab_conf.jsonl] -> stdout + data/pmin12/conf.txt"""
import json, os, sys
import numpy as np

F = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Breakthrough/data/pmin12/ab_conf.jsonl")
R = [json.loads(l) for l in open(F)]
PS = sorted({r["p_min"] for r in R})
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


say(f"PMIN12 confirmation: {len(R)} runs, {len({r['key'] for r in R})} prompts, arms {PS} (shared GPU)")
best = {}
for typ in ("chat", "code", "math", "agent"):
    ks = sorted({r["key"] for r in R if r["type"] == typ})
    if not ks:
        continue
    rb, rm = {p: [] for p in PS}, {p: [] for p in PS}
    for k in ks:
        x = {p: [r for r in R if r["key"] == k and r["p_min"] == p] for p in PS}
        for p in PS:
            rb[p].append(max(r["tok_s"] for r in x[p]) / max(r["tok_s"] for r in x[0.0]))
            rm[p].append(np.mean([r["tok_s"] for r in x[p]]) / np.mean([r["tok_s"] for r in x[0.0]]))
    g = lambda v: 100 * (float(np.exp(np.mean(np.log(v)))) - 1)
    best[typ] = {p: g(rb[p]) for p in PS}
    rows = {p: np.mean([r["rows"] / max(1, r["passes"]) for r in R if r["type"] == typ and r["p_min"] == p]) for p in PS}
    tpp = {p: np.mean([r["ntok"] / max(1, r["passes"]) for r in R if r["type"] == typ and r["p_min"] == p]) for p in PS}
    eq = {p: sum(r["first_diff"] is None for r in R if r["type"] == typ and r["p_min"] == p) for p in PS}
    n = {p: sum(1 for r in R if r["type"] == typ and r["p_min"] == p) for p in PS}
    say(f"  {typ:5s} ({len(ks)} prompts): " + "; ".join(
        f"p_min {p}: best-of-2 {g(rb[p]):+.1f} % (mean-of-2 {g(rm[p]):+.1f} %, range {100 * (min(rb[p]) - 1):+.0f}..{100 * (max(rb[p]) - 1):+.0f}), "
        f"rows / pass {rows[p]:.2f}, tok / pass {tpp[p]:.2f}, equal to run 0 {eq[p]}/{n[p]}" for p in PS if p > 0)
        + f"; p_min 0: rows / pass {rows[0.0]:.2f}, tok / pass {tpp[0.0]:.2f}")
ok = [p for p in PS if p > 0 and best.get("chat", {}).get(p, -99) >= 8 and best.get("code", {}).get(p, 0) >= -2
      and best.get("math", {}).get(p, 0) >= -2]
kill = all(best.get("chat", {}).get(p, -99) < 3 for p in PS if p > 0)
say(f"VERDICT (PASS: chat >= +8 % at one p_min with code and math >= -2 %; KILL: chat < +3 %): "
    + (f"PASS at p_min {ok}" if ok else "KILL" if kill else "BETWEEN"))
open(os.path.expanduser("~/Breakthrough/data/pmin12/conf.txt"), "w").write("\n".join(lines) + "\n")
