#!/usr/bin/env python3
"""Read-out of the per-layer lambda band screen (Breakthrough, Mon 16 Nov 2026 14:0x; docs/CHECK_BYTES_900.md section 6).
Each band arm (lambda 3.2e-2 on layers 0-15 / 16-31 / 32-47) vs bt_c6tiers, paired on the same rows of the 24 code
windows (out_code24/baseline.json; ties = the baseline's p1 - p2 <= 0.1): the all-row / non-tie increments, the expert
GB saved (from the masks, real pieces), the measured cost per 0.1 GB vs the uniform law's (L0-15 +0.27 pt / 0.70 GB,
L16-31 +0.23 / 0.61, L32-47 +0.22 / 0.56), and the pre-registered verdict: TOLERANT if <= 0.5x the uniform cost,
SENSITIVE if >= 2x; PASS = at least one TOLERANT band, else KILL. Paired increments get a 95 % window bootstrap.
Output ~/Breakthrough/data/m1/lband_map.out (+ .json). Usage: ~/BigMoE/.venv/bin/python -I -B lband_read.py"""
import json
import os

import numpy as np

O = os.path.expanduser(os.environ.get("LB_O", "~/FableGod/decision_map/out_code24"))      # LB_O: tests only
OUTF = os.path.expanduser(os.environ.get("LB_OUTF", "~/Breakthrough/data/m1/lband_map.out"))
UNI = {"bt_lb0": (0.27, 0.70, "L0-15"), "bt_lb1": (0.23, 0.61, "L16-31"), "bt_lb2": (0.22, 0.56, "L32-47")}
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(O, "baseline.json")))
order = sorted(base)
rng = np.random.default_rng(20261117)


def flips(arm):
    A = json.load(open(os.path.join(O, arm + ".json"))); r = {}; same = tot = 0
    for nm in order:
        if nm not in A:
            continue
        b = base[nm]; y = np.array([t[0] for t in b["tk"]])
        pp = np.exp(np.array(b["lp"])[:, :2]); tie = (pp[:, 0] - pp[:, 1]) <= 0.1
        x = np.array([row[0] for row in A[nm]["tk"]]); k = min(len(x), len(y))
        r[nm] = (x[:k] != y[:k], tie[:k])
        same += sum(1 for i in range(k) if A[nm]["tk"][i] == b["tk"][i] and A[nm]["lp"][i] == b["lp"][i]); tot += k
    return r, same, tot


say("Per-layer lambda band screen (experts only, 24 code windows): each band arm vs bt_c6tiers, paired; ties = the "
    "baseline's p1 - p2 <= 0.1")
if not os.path.exists(os.path.join(O, "bt_c6tiers.json")):
    say("bt_c6tiers NOT RUN: no paired read")
else:
    R, s0, t0 = flips("bt_c6tiers")
    r_all = 100 * sum(f.sum() for f, t in R.values()) / sum(len(f) for f, t in R.values())
    r_nt = 100 * sum((f & ~t).sum() for f, t in R.values()) / sum((~t).sum() for f, t in R.values())
    say(f"reference bt_c6tiers: {r_all:.2f} / {r_nt:.2f} % (all-row / non-tie; the uniform law predicted 1.21 / 0.59); "
        f"identical rows {s0}/{t0}{' INVALID' if s0 == t0 else ''}")
    say("| band arm | all-row increment pt [95 %] | non-tie increment pt | GB saved | cost pt per 0.1 GB (uniform law) | ratio | class |")
    say("|---|---|---|---|---|---|---|")
    res = {}
    tol = 0
    for arm, (uni_pt, gb, lab) in UNI.items():
        if not os.path.exists(os.path.join(O, arm + ".json")):
            say(f"| {arm} ({lab}) | NOT RUN | | | | | |")
            continue
        X, s1, t1 = flips(arm)
        ws = [nm for nm in R if nm in X]
        da = np.array([(X[nm][0].sum() - R[nm][0].sum()) for nm in ws], float)
        dn = np.array([((X[nm][0] & ~X[nm][1]).sum() - (R[nm][0] & ~R[nm][1]).sum()) for nm in ws], float)
        nr = np.array([len(R[nm][0]) for nm in ws], float); nn = np.array([(~R[nm][1]).sum() for nm in ws], float)
        inc_a, inc_n = 100 * da.sum() / nr.sum(), 100 * dn.sum() / nn.sum()
        idx = rng.integers(0, len(ws), (10000, len(ws)))
        bs = 100 * da[idx].sum(1) / nr[idx].sum(1)
        cost = inc_a / (gb * 10); ucost = uni_pt / (gb * 10); ratio = cost / ucost
        cls = "TOLERANT" if ratio <= 0.5 else ("SENSITIVE" if ratio >= 2 else "as the uniform law")
        tol += cls == "TOLERANT"
        res[arm] = dict(band=lab, inc_all=round(inc_a, 3), ci=[round(float(np.percentile(bs, 2.5)), 3),
                        round(float(np.percentile(bs, 97.5)), 3)], inc_nontie=round(inc_n, 3), gb_saved=gb,
                        cost=round(cost, 4), uniform_cost=round(ucost, 4), ratio=round(ratio, 2), cls=cls,
                        identical=f"{s1}/{t1}")
        say(f"| {arm} ({lab}) | {inc_a:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] | {inc_n:+.2f} | "
            f"{gb:.2f} | {cost:.3f} ({ucost:.3f}) | {ratio:.2f} | **{cls}** |")
    v = "PASS (at least one TOLERANT band: the lever exists; a confirm arm on G2's stack follows)" if tol else \
        "KILL (no TOLERANT band: per-layer lambda gives no lever)"
    say(f"VERDICT (pre-registered, a decision aid): {v}")
    json.dump(dict(ref=[round(r_all, 3), round(r_nt, 3)], arms=res, verdict=v), open(OUTF.replace(".out", ".json"), "w"),
              indent=1)
open(OUTF, "w").write("\n".join(lines) + "\n")
