#!/usr/bin/env python3
"""Read the AWQ-style 3-bit GDN + attention screen (Breakthrough, Sun 15 Nov 2026 04:0x; pre-registered
docs/LOSSY_CHECK_PRICE.md 544a1f1): bt_ga3awq (activation-aware per-linear scaling, diagonal-Hessian alpha search, E[x^2]
from 6 IFEval windows) and its alpha-0 control bt_ga3w0, against bt_ga3 (plain RTN 3-bit) on the 12 code windows.
Code flips vs baseline.json (all rows / non-tie, floor_rows.json), 95 % window bootstrap; validity = bt_ga3w0's per-row
top-1 against bt_ga3's. Output: ~/Breakthrough/data/m1/awq_screen_read.out.
Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B awq_screen_read.py
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
rng = np.random.default_rng(20261115)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(M, "baseline.json")))
TIE = {(r["nm"], int(r["i"])): r["cls"] == "tie" for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))}
A = {a: json.load(open(os.path.join(M, a + ".json"))) for a in ("bt_ga3", "bt_ga3w0", "bt_ga3awq") if os.path.exists(os.path.join(M, a + ".json"))}


def flips(arm):
    per = []
    for nm, v in A[arm].items():
        if v["dom"] != "code" or nm not in base:
            continue
        x = np.array([t[0] for t in v["tk"]]); y = np.array([t[0] for t in base[nm]["tk"]]); k = min(len(x), len(y))
        tie = np.array([TIE.get((nm, base[nm]["off"] + base[nm]["a"] + j), False) for j in range(k)])
        f = x[:k] != y[:k]; per.append((f.sum(), k, (f & ~tie).sum(), (~tie).sum()))
    P = np.array(per, float); idx = rng.integers(0, len(P), (10000, len(P))); out = []
    for c, d in ((0, 1), (2, 3)):
        r = 100 * P[idx, c].sum(1) / P[idx, d].sum(1)
        out.append((100 * P[:, c].sum() / P[:, d].sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)))
    return out


say("AWQ-style 3-bit GDN + attention, code windows: flips % vs baseline (all rows / non-tie) [95 % window bootstrap]")
say("| arm | all-row (level 1.34) | non-tie (level 0.87) |")
say("|---|---|---|")
R = {}
for arm in ("bt_ga3", "bt_ga3w0", "bt_ga3awq"):
    if arm in A:
        al, nt = flips(arm); R[arm] = (al, nt)
        say(f"| {arm} | {al[0]:.2f} [{al[1]:.2f}, {al[2]:.2f}] | {nt[0]:.2f} [{nt[1]:.2f}, {nt[2]:.2f}] |")
if "bt_ga3" in A and "bt_ga3w0" in A:
    same = tot = 0
    for nm, v in A["bt_ga3w0"].items():
        if nm in A["bt_ga3"]:
            x = [t[0] for t in v["tk"]]; y = [t[0] for t in A["bt_ga3"][nm]["tk"]]; k = min(len(x), len(y))
            same += sum(1 for i in range(k) if x[i] == y[i]); tot += k
    say(f"VALIDITY: bt_ga3w0 (alpha 0, wrapper path) vs bt_ga3 (RTN, bare module): top-1 identical on {100 * same / tot:.2f} % of "
        f"{tot} rows (pre-registered: bit for bit; a few kernel-lane tie differences are possible, a bypass would differ widely)")
if "bt_ga3awq" in R:
    nt = R["bt_ga3awq"][1][0]
    act = ("ENTERS the plan -> screen bt_comb_awq next" if nt <= 0.50 else "BETWEEN" if nt <= 1.0 else
           "CLOSED for scaling-only quantizers (a full-Hessian GPTQ or trellis is the next try)")
    say(f"(a) bt_ga3awq non-tie {nt:.2f} %: {act}")
open(os.path.expanduser("~/Breakthrough/data/m1/awq_screen_read.out"), "w").write("\n".join(lines) + "\n")
