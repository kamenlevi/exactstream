#!/usr/bin/env python3
"""Read the WEEK98 goal-B map screens (Breakthrough, Sun 15 Nov 2026 01:2x; pre-registered docs/LOSSY_CHECK_PRICE.md
0d90b27): bt_comb, bt_mix4, bt_ga3, bt_1c on the 12 code windows. Code flips vs baseline.json's top-1, all rows and
non-tie rows (~/FableGod/floor_rows.json), with the 95 % window bootstrap. The 16K lm_head shortlist (the deepdraft bank's
16,384 most frequent top-1 tokens) is applied offline to an arm's recorded top-8: a top-1 outside the list is replaced by
the first listed token in the top-8 (if none, the row counts as a flip). Prints each arm against its prediction and its
pre-registered action. Output: ~/Breakthrough/data/m1/bt_screen_read.out.
Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B bt_screen_read.py
"""
import collections
import glob
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
LEVEL, NT_LEVEL = 1.34, 0.87
rng = np.random.default_rng(20261115)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(M, "baseline.json")))
TIE = {(r["nm"], int(r["i"])): r["cls"] == "tie" for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))}
cnt = collections.Counter()
for f in sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/*.npz"))):
    cnt.update(np.load(f)["tk"][:, 0].tolist())
SL = set(t for t, _ in cnt.most_common(16384))


def read(arm, shortlist=False):
    a = json.load(open(os.path.join(M, arm + ".json")))
    per = []
    for nm, v in a.items():
        if v["dom"] != "code" or nm not in base:
            continue
        tk = v["tk"]; y = np.array([t[0] for t in base[nm]["tk"]]); k = min(len(tk), len(y))
        if shortlist:
            x = np.array([next((t for t in row if t in SL), -1) for row in tk[:k]])
        else:
            x = np.array([row[0] for row in tk[:k]])
        tie = np.array([TIE.get((nm, base[nm]["off"] + base[nm]["a"] + j), False) for j in range(k)])
        f = x != y[:k]
        per.append((f.sum(), k, (f & ~tie).sum(), (~tie).sum()))
    P = np.array(per, float)
    idx = rng.integers(0, len(P), (10000, len(P)))
    out = []
    for c, d in ((0, 1), (2, 3)):
        r = 100 * P[idx, c].sum(1) / P[idx, d].sum(1)
        out.append((100 * P[:, c].sum() / P[:, d].sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)))
    return out, len(P)


PRED = {"bt_comb": "1.53 additive / ~1.23 tie-aware; non-tie 0.65 [0.61, 0.68]", "bt_mix4": "+0.42 / +0.12 (confounded)",
        "bt_ga3": "none (D3 bound up to +7.4 non-tie)", "bt_1c": "1.08 [0.99, 1.15] / 0.50 [0.46, 0.53]"}
say("WEEK98 goal-B map screens, code windows: flips % vs baseline (all rows / non-tie), [95 % window bootstrap]")
say("| arm | windows | all-row (level 1.34) | non-tie (level 0.87) | prediction |")
say("|---|---|---|---|---|")
R = {}
for arm, sl in (("bt_comb", True), ("bt_comb", False), ("bt_mix4", False), ("bt_ga3", False), ("bt_1c", False), ("hc8w", False), ("t4_all", False)):
    if not os.path.exists(os.path.join(M, arm + ".json")):
        say(f"| {arm} | (no result) | | | |"); continue
    (al, nt), n = read(arm, sl)
    lab = arm + (" + shortlist" if sl else "")
    R[lab] = (al, nt)
    say(f"| {lab} | {n} | {al[0]:.2f} [{al[1]:.2f}, {al[2]:.2f}] | {nt[0]:.2f} [{nt[1]:.2f}, {nt[2]:.2f}] | {PRED.get(arm, 'reference')} |")
say("")
if "bt_comb + shortlist" in R:
    al, nt = R["bt_comb + shortlist"]
    act = ("BOTH LINES MET: 1c + 4-bit mixers + shortlist is the code plan (24.9 ms at 84 %) -> BigMoE integration + EvalSuite code gate"
           if al[0] <= LEVEL and nt[0] <= NT_LEVEL else
           "NON-TIE MET, ALL-ROW OVER: the tie-aware reading fails -> fall back to lambda 1.2e-2 (23.6 / 28.1 ms), screen that combination"
           if nt[0] <= NT_LEVEL else "NON-TIE OVER: additivity fails -> 8-bit mixers + lambda 1.2e-2, re-screen")
    say(f"(a) bt_comb: {act}")
if "bt_mix4" in R:
    al, nt = R["bt_mix4"]
    say(f"(b) bt_mix4: {'4-bit mixers ADOPTED in the plan' if nt[0] <= 0.20 and al[0] <= 0.60 else 'keep 8-bit mixers (HC8, shipped)'}")
if "bt_ga3" in R:
    al, nt = R["bt_ga3"]
    say(f"(c) bt_ga3: {'GDN + attention 3-bit ENTERS the plan (read chat next)' if nt[0] <= 0.50 else ('CLOSED until a better quantizer' if nt[0] > 1.0 else 'BETWEEN')}")
if "bt_1c" in R and "bt_comb" in R:
    al, nt = R["bt_1c"]; ca, cn = R["bt_comb"]
    ok = 0.54 <= al[0] <= 1.62 and 0.25 <= nt[0] <= 0.75
    say(f"(d) bt_1c: expert calibration {'HOLDS' if ok else 'OFF'} (prediction 1.08 / 0.50, +-50 %); dense increment bt_comb - bt_1c = "
        f"{ca[0] - al[0]:+.2f} all-row / {cn[0] - nt[0]:+.2f} non-tie (bt_mix4 alone: {R['bt_mix4'][0][0] if 'bt_mix4' in R else float('nan'):.2f} / "
        f"{R['bt_mix4'][1][0] if 'bt_mix4' in R else float('nan'):.2f})")
open(os.path.expanduser("~/Breakthrough/data/m1/bt_screen_read.out"), "w").write("\n".join(lines) + "\n")
