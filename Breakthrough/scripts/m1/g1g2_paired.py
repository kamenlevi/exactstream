#!/usr/bin/env python3
"""Paired attribution of the G1 / G2 map arms (Breakthrough, Mon 16 Nov 2026 19:5x; pre-registered in
docs/CHECK_BYTES_900.md section 7, before computing). CPU only; it reads the arm files in out_code24 and writes nothing
there. Section 2's flip rule (top-1 vs the baseline's top-1), ties (the baseline's p1 - p2 <= 0.1), pooled ratio and window
bootstrap (10,000 draws, seed 20261116; one draw matrix shared by every read, so the differences are paired).
Usage (BigMoE venv, cwd ~/Breakthrough):  python -I -B scripts/m1/g1g2_paired.py   -> data/m1/g1g2_paired.out
"""
import json
import os

import numpy as np

O = os.path.expanduser("~/FableGod/decision_map/out_code24")
OUT = os.path.expanduser("~/Breakthrough/data/m1/g1g2_paired.out")
LV, NT = 1.34, 0.87
out = []


def say(s=""):
    print(s, flush=True)
    out.append(s)


base = json.load(open(os.path.join(O, "baseline.json")))
order = sorted(base)
V32 = set(np.load(os.path.expanduser("~/Breakthrough/data/m1/lmsl_code32k.npy")).tolist())
TOK = {nm: np.load(os.path.join(O, "routing", nm))["tokens"].astype(np.int64) for nm in order}
ARMS = ["bt_g1h8", "bt_g1", "bt_c6comb", "bt_l20comb", "bt_l20g2"]
A = {a: json.load(open(os.path.join(O, a + ".json"))) for a in ARMS}
reg = json.load(open(os.path.expanduser("~/Breakthrough/data/m1/g1g2_map.json")))["arms"]


def pick(arm, nm, mode):   # == g1g2_map.py read()'s pick for "full" and "v1-32k + session rows"
    rows = A[arm][nm]["tk"]
    if mode == "full":
        return np.array([r[0] for r in rows])
    a = base[nm]["a"]
    sess = set(TOK[nm][:a + 1].tolist())
    res = []
    for j, row in enumerate(rows):
        if j:
            sess.add(int(TOK[nm][a + j]))
        res.append(next((t for t in row if t in V32 or t in sess), -1))
    return np.array(res)


def counts(arm, mode):   # per window: flips, rows, non-tie flips, non-tie rows
    P = []
    for nm in order:
        b = base[nm]
        y = np.array([t[0] for t in b["tk"]])
        pp = np.exp(np.array(b["lp"])[:, :2])
        tie = (pp[:, 0] - pp[:, 1]) <= 0.1
        x = pick(arm, nm, mode)
        k = min(len(x), len(y))
        f = x[:k] != y[:k]
        P.append((f.sum(), k, (f & ~tie[:k]).sum(), (~tie[:k]).sum()))
    return np.array(P, float)


rng = np.random.default_rng(20261116)
IDX = rng.integers(0, len(order), (10000, len(order)))


def rate(P):
    res = []
    for c, d in ((0, 1), (2, 3)):
        r = 100 * P[IDX, c].sum(1) / P[IDX, d].sum(1)
        res.append((100 * P[:, c].sum() / P[:, d].sum(), np.percentile(r, 2.5), np.percentile(r, 97.5), int(P[:, c].sum())))
    return res


def diff(P, Q):
    res = []
    for c, d in ((0, 1), (2, 3)):
        r = 100 * (P[IDX, c].sum(1) - Q[IDX, c].sum(1)) / P[IDX, d].sum(1)
        res.append((100 * (P[:, c].sum() - Q[:, c].sum()) / P[:, d].sum(), np.percentile(r, 2.5), np.percentile(r, 97.5),
                    int(P[:, c].sum() - Q[:, c].sum())))
    return res


C = {(a, m): counts(a, m) for a in ARMS for m in ("full", "v32")
     if not (a == "bt_g1h8" and m == "v32")}
say("Paired attribution of the G1 / G2 map arms (docs/CHECK_BYTES_900.md section 7, pre-registered): code flips % vs the "
    "exact baseline on out_code24's 24 windows; [95 % window bootstrap, one shared draw matrix]; lines 1.34 / 0.87")
say(f"rows {int(C[('bt_g1', 'full')][:, 1].sum())}, non-tie {int(C[('bt_g1', 'full')][:, 3].sum())}")
say()
say("FULL-HEAD reads (the recorded top-1; V1 arms only had in-sample and out-of-sample list reads before):")
say("| arm | all-row % [CI] (rows) | non-tie % [CI] (rows) | vs lines | in-sample read (rows) | full - in-sample (rows) |")
say("|---|---|---|---|---|---|")
for a in ARMS[1:]:
    (ar, alo, ahi, an), (nr, nlo, nhi, nn) = rate(C[(a, "full")])
    v = "PASS" if ar <= LV and nr <= NT else "OVER (" + ", ".join(x for x, c in (("all-row", ar > LV), ("non-tie", nr > NT)) if c) + ")"
    ins = round(reg[f"{a} | sl-registered (in sample)"]["all_row"][0] * 6327 / 100)
    say(f"| {a} | {ar:.2f} [{alo:.2f}, {ahi:.2f}] ({an}) | {nr:.2f} [{nlo:.2f}, {nhi:.2f}] ({nn}) | {v} | {ins} | {an - ins:+d} |")
say()
say("PAIRED differences (pt; rows in brackets after the CI):")
PAIRS = [
    ("1. V3 vs HC8 mixers: bt_g1 (full) - bt_g1h8", ("bt_g1", "full"), ("bt_g1h8", "full")),
    ("2a. int8: bt_g1 - bt_c6comb (full)", ("bt_g1", "full"), ("bt_c6comb", "full")),
    ("2b. int8: bt_l20g2 - bt_l20comb (full)", ("bt_l20g2", "full"), ("bt_l20comb", "full")),
    ("3a. lambda 2.0e-2 vs C6: bt_l20comb - bt_c6comb (full)", ("bt_l20comb", "full"), ("bt_c6comb", "full")),
    ("3b. lambda 2.0e-2 vs C6: bt_l20g2 - bt_g1 (full)", ("bt_l20g2", "full"), ("bt_g1", "full")),
] + [(f"4. V1-32k: {a} 32K - full", (a, "v32"), (a, "full")) for a in ARMS[1:]]
say("| difference | all-row pt [CI] (rows) | non-tie pt [CI] (rows) |")
say("|---|---|---|")
for lab, p, q in PAIRS:
    (d1, l1, h1, r1), (d2, l2, h2, r2) = diff(C[p], C[q])
    say(f"| {lab} | {d1:+.3f} [{l1:+.3f}, {h1:+.3f}] ({r1:+d}) | {d2:+.3f} [{l2:+.3f}, {h2:+.3f}] ({r2:+d}) |")
say()
(ar, _, _, an), (nr, _, _, nn) = rate(C[("bt_l20g2", "full")])
say(f"5. G2 no-V4 without V1 (bt_l20g2, full head): {ar:.2f} / {nr:.2f} ({an} / {nn} rows; lines <= 84 / <= 54 rows): "
    f"{'PASS' if ar <= LV and nr <= NT else 'OVER'}")
open(OUT, "w").write("\n".join(out) + "\n")
