#!/usr/bin/env python3
"""The combined code plan on all 24 code windows (Breakthrough, Sun 15 Nov 2026 01:4x; the orchestrator 01:3x: tighten
the evidence before BigMoE's integration). Two steps around the map runs in DM_OUT=~/FableGod/decision_map/out_code24
(so the map's own out/ and baseline.json are never touched):
  masks   after 'baseline scapture' (DM_SETS=code, --max 24): the 1c tier masks from out_code24/scores/<window>.npz (every
          row's top-10 scores + ids, the same normalised weights as xcap), the same rule as build_tiers_masks.py
          (s 0.98, lambda 1.6e-2, 32-row blocks from the first answer row) -> out_code24/bt/tiers_1c/<window>.npz
  read    after 'bt_comb': flips vs out_code24/baseline.json, all rows and non-tie (ties: the baseline's own p1 - p2 <=
          0.1, floor_rows.json's rule), the 16K shortlist applied offline, 95 % window bootstrap; windows 1-12 and 13-24
          apart (1-12 must reproduce the 12-window screen: 1.25 / 0.60). Output ~/Breakthrough/data/m1/code24_comb.out.
Usage: ~/BigMoE/.venv/bin/python -I -B code24_comb.py masks|read
"""
import collections
import glob
import json
import os
import sys

import numpy as np

O = os.path.expanduser("~/FableGod/decision_map/out_code24")
S_, LAM, ROWS, NL = 0.98, 1.6e-2, 32, 48
T4, T3 = 2.775 * LAM, 0.8457 * LAM


def masks():
    od = os.path.join(O, "bt", "tiers_1c"); os.makedirs(od, exist_ok=True)
    base = json.load(open(os.path.join(O, "baseline.json")))
    for f in sorted(os.listdir(os.path.join(O, "scores"))):
        z = np.load(os.path.join(O, "scores", f)); a = int(z["a"])
        ids, w = z["ids"].astype(np.int64), z["scores"].astype(np.float64)
        n = ids.shape[0]; m = n - a
        assert base[f]["a"] == a and len(base[f]["tk"]) == m, (f, a, m, len(base[f]["tk"]))
        assert np.allclose(w[a:].sum(-1), 1, atol=2e-2), f
        keep = np.ones((n, NL, 10), bool); keep3 = np.zeros((n, NL, 10), bool)
        for b0 in range(a, n, ROWS):
            rows = np.arange(b0, min(b0 + ROWS, n)); rw = S_ ** np.arange(len(rows))
            for l in range(NL):
                u, inv = np.unique(ids[rows, l].ravel(), return_inverse=True)
                imp = np.bincount(inv, weights=(w[rows, l] ** 2 * rw[:, None]).ravel(), minlength=len(u))
                tr = np.where(imp > T4, 0, np.where(imp > T3, 1, 2))[inv].reshape(len(rows), 10)
                keep[rows, l] = tr == 0; keep3[rows, l] = tr == 1
        np.savez_compressed(os.path.join(od, f), keep=keep, keep3=keep3)
    old = os.path.expanduser("~/FableGod/decision_map/out/bt/tiers_1c")
    same = [f for f in os.listdir(od) if os.path.exists(os.path.join(old, f))]
    eq = all(np.array_equal(np.load(os.path.join(od, f))["keep"], np.load(os.path.join(old, f))["keep"]) and
             np.array_equal(np.load(os.path.join(od, f))["keep3"], np.load(os.path.join(old, f))["keep3"]) for f in same)
    print(f"masks: {len(os.listdir(od))} windows -> {od}; the {len(same)} windows shared with the 12-window screen identical "
          f"to its masks: {'PASS' if eq else 'FAIL'}")


def read():
    rng = np.random.default_rng(20261115); lines = []
    base = json.load(open(os.path.join(O, "baseline.json"))); arm = json.load(open(os.path.join(O, "bt_comb.json")))
    cnt = collections.Counter()
    for f in sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/*.npz"))):
        cnt.update(np.load(f)["tk"][:, 0].tolist())
    SL = set(t for t, _ in cnt.most_common(16384))
    order = sorted(base)
    per = []
    for nm in order:
        if nm not in arm:
            continue
        b = base[nm]; y = np.array([t[0] for t in b["tk"]])
        p = np.exp(np.array(b["lp"])[:, :2]); tie = (p[:, 0] - p[:, 1]) <= 0.1
        x = np.array([next((t for t in row if t in SL), -1) for row in arm[nm]["tk"]]); k = min(len(x), len(y))
        f = x[:k] != y[:k]
        per.append((nm, f.sum(), k, (f & ~tie[:k]).sum(), (~tie[:k]).sum()))
    def stat(P):
        P = np.array([r[1:] for r in P], float); idx = rng.integers(0, len(P), (10000, len(P))); out = []
        for c, d in ((0, 1), (2, 3)):
            r = 100 * P[idx, c].sum(1) / P[idx, d].sum(1)
            out.append(f"{100 * P[:, c].sum() / P[:, d].sum():.2f} [{np.percentile(r, 2.5):.2f}, {np.percentile(r, 97.5):.2f}]")
        return out
    first = [r for r in per if r[0] in order[:12]] if len(order) >= 12 else per
    say = lambda s: (print(s, flush=True), lines.append(s))
    say("The combined code plan (1c + 4-bit mixers + 16K shortlist) on the code windows: flips % vs the exact baseline, "
        "[95 % window bootstrap]; ties = the baseline's p1 - p2 <= 0.1")
    say("| windows | all-row (level 1.34) | non-tie (level 0.87) |")
    say("|---|---|---|")
    for lab, P in (("all 24", per), ("1-12 (the screen's windows)", per[:12]), ("13-24 (new)", per[12:])):
        if P:
            a_, n_ = stat(P); say(f"| {lab} ({len(P)} windows, {int(sum(r[2] for r in P))} rows) | {a_} | {n_} |")
    open(os.path.expanduser("~/Breakthrough/data/m1/code24_comb.out"), "w").write("\n".join(lines) + "\n")


{"masks": masks, "read": read}[sys.argv[1]]()
