#!/usr/bin/env python3
"""Read-out of the optional comment-QOM arms (Breakthrough, Mon 16 Nov 2026 13:4x; amendment (f), information only):
flips vs ~/FableGod/decision_map/out/baseline.json on its 12 code windows (all rows / non-tie; ties = the baseline's own
p1 - p2 <= 0.1), 95 % window bootstrap (10,000, seed 20261116), the INVALID rule (an arm equal to the baseline on every
row), the int8 census, and each arm's change paired against bt_cq_ctrl (the same windows and rows).
Output ~/Breakthrough/data/m1/comqom_map.out. Usage: ~/BigMoE/.venv/bin/python -I -B comqom_read.py"""
import json
import os

import numpy as np

O = os.path.expanduser(os.environ.get("CQ_O", "~/FableGod/decision_map/out"))      # CQ_O: tests only
OUTF = os.path.expanduser(os.environ.get("CQ_OUTF", "~/Breakthrough/data/m1/comqom_map.out"))
ARMS = [("bt_cq_ctrl", "1c tiers + hc8w_mix + int8 (the matched control)"),
        ("bt_cq", "1c with comment rows on QOM + hc8w_mix + int8"),
        ("bt_cq_res", "1c with comment rows on QOM + RES + hc8w_mix + int8")]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(O, "baseline.json")))
code = sorted(nm for nm, v in base.items() if v["dom"] == "code")
rng = np.random.default_rng(20261116)
say(f"comment-QOM map arms (information only): code flips % vs the exact baseline, {len(code)} code windows of the original "
    f"map, [95 % window bootstrap]; ties = the baseline's p1 - p2 <= 0.1")
cp = os.path.join(O, "bt_int8_census.jsonl")
if os.path.exists(cp):
    for i, ln in enumerate(open(cp)):
        c = json.loads(ln)
        ok = c["windows"] == len(code) and not c["bad"] and c["rows_q"] == c["rows_q_want"] > 0
        say(f"VALIDITY int8 census line {i + 1}: {c['windows']} windows, quantized {c['rows_q']} (want {c['rows_q_want']}), "
            f"mismatches {len(c['bad'])}: {'PASS' if ok else 'FAIL'}")
F = {}
say("| arm | all-row % | non-tie % | identical rows | paired vs bt_cq_ctrl (all-row / non-tie pt) |")
say("|---|---|---|---|---|")
for arm, lab in ARMS:
    p = os.path.join(O, arm + ".json")
    if not os.path.exists(p):
        say(f"| {arm}: {lab} | NOT RUN | | | |")
        continue
    A = json.load(open(p)); per = []; same = tot = 0; flips = {}
    for nm in code:
        if nm not in A:
            continue
        b = base[nm]; y = np.array([t[0] for t in b["tk"]])
        pp = np.exp(np.array(b["lp"])[:, :2]); tie = (pp[:, 0] - pp[:, 1]) <= 0.1
        x = np.array([row[0] for row in A[nm]["tk"]]); k = min(len(x), len(y)); f = x[:k] != y[:k]
        flips[nm] = (f, tie[:k])
        per.append((f.sum(), k, (f & ~tie[:k]).sum(), (~tie[:k]).sum()))
        same += sum(1 for i in range(k) if A[nm]["tk"][i] == b["tk"][i] and A[nm]["lp"][i] == b["lp"][i]); tot += k
    P = np.array(per, float); idx = rng.integers(0, len(P), (10000, len(P)))
    st = []
    for c_, d_ in ((0, 1), (2, 3)):
        r = 100 * P[idx, c_].sum(1) / P[idx, d_].sum(1)
        st.append(f"{100 * P[:, c_].sum() / P[:, d_].sum():.2f} [{np.percentile(r, 2.5):.2f}, {np.percentile(r, 97.5):.2f}]")
    F[arm] = flips
    pair = ""
    if arm != "bt_cq_ctrl" and "bt_cq_ctrl" in F:
        a0 = sum(F["bt_cq_ctrl"][nm][0].sum() for nm in flips); a1 = sum(flips[nm][0].sum() for nm in flips)
        n0 = sum((F["bt_cq_ctrl"][nm][0] & ~F["bt_cq_ctrl"][nm][1]).sum() for nm in flips)
        n1 = sum((flips[nm][0] & ~flips[nm][1]).sum() for nm in flips)
        rows = sum(len(flips[nm][0]) for nm in flips); nt = sum((~flips[nm][1]).sum() for nm in flips)
        pair = f"{100 * (a1 - a0) / rows:+.2f} / {100 * (n1 - n0) / nt:+.2f}"
    say(f"| {arm}: {lab} | {st[0]} | {st[1]} | {same}/{tot}{' INVALID' if same == tot else ''} | {pair} |")


def rate(arm, nontie=False):
    fl = F[arm]
    if nontie:
        return 100 * sum((f & ~t).sum() for f, t in fl.values()) / sum((~t).sum() for f, t in fl.values())
    return 100 * sum(f.sum() for f, t in fl.values()) / sum(len(f) for f, t in fl.values())


# amendment (f) + NewIdeas' refinement 14:0x (before data): the reading (a decision aid for G1's re-plan, not a gate line)
# is on NON-TIE flips (hc8w + int8 in both arms dilute the all-row ratio); all-row printed beside
if "bt_cq_ctrl" in F and "bt_cq" in F:
    c0, c1 = rate("bt_cq_ctrl", True), rate("bt_cq", True)
    r = c1 / c0 if c0 > 0 else float("inf")
    v = "CONFIRMED (G1's fix = comment-QOM)" if r <= 0.5 else ("NOT CARRIED (the re-plan falls back to a lower lambda)"
                                                                 if r >= 0.8 else "BETWEEN (Fit12's 4-key kill test decides)")
    a0, a1 = rate("bt_cq_ctrl"), rate("bt_cq")
    say(f"READING (non-tie): bt_cq / bt_cq_ctrl = {c1:.2f} / {c0:.2f} = {r:.2f} -> {v}; all-row beside {a1:.2f} / {a0:.2f} "
        f"= {a1 / a0 if a0 > 0 else float('inf'):.2f} (Fit12's [P] for comment-QOM ~0.42 / ~0.06)")
if "bt_cq" in F and "bt_cq_res" in F:
    d = rate("bt_cq_res") - rate("bt_cq")
    say(f"RES's paired increment (bt_cq_res - bt_cq): all-row {d:+.2f} pt, non-tie "
        f"{rate('bt_cq_res', True) - rate('bt_cq', True):+.2f} pt -> {'within' if d <= 0.3 else 'OVER'} G2's <= +0.3 pt")
open(OUTF, "w").write("\n".join(lines) + "\n")
