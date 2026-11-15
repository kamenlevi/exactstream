#!/usr/bin/env python3
"""lm_head shortlist with context, offline from the decision map's recorded top-8 (Breakthrough, Sun 15 Nov 2026 03:4x;
pre-registered docs/LOSSY_CHECK_PRICE.md 03c71ba; Sandbox98's prompt-aware list, 69c93b2) [K: exact from the top-8,
conservative when no top-8 token is listed].
S_K = the deepdraft bank's K most frequent top-1 tokens (code); C_K = S_K + every window token before the row (prompt +
answer so far). Misses = the baseline top-1 outside the list. For code, the combined arm (bt_comb, out/ 12 windows and
out_code24/ 24 windows) with the list applied (the first listed token of its top-8, else a flip).
Output: ~/Breakthrough/data/m1/lmhead_ctx_shortlist.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B lmhead_ctx_shortlist.py
"""
import collections
import glob
import json
import os

import numpy as np

H = os.path.expanduser("~")
M, M24 = f"{H}/FableGod/decision_map/out", f"{H}/FableGod/decision_map/out_code24"
SETS = [("code", "d02b_walk_code.npz"), ("ifeval", "d02b_walk_d02a_ifeval.npz"), ("multi", "d02b_walk_d02a_multi.npz")]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


cnt = collections.Counter()
for f in sorted(glob.glob(f"{H}/datasets/deepdraft/bank/*.npz")):
    cnt.update(np.load(f)["tk"][:, 0].tolist())
order = [t for t, _ in cnt.most_common()]
IDS = {}
for dom, wf in SETS:
    z = np.load(f"{H}/TrainLab/data/{wf}")
    for s, nm in enumerate(z["names"]):
        IDS[str(nm)] = (np.asarray(z[f"{s}_ids"]), int(z[f"{s}_off"]))
TIE = {(r["nm"], int(r["i"])): r["cls"] == "tie" for r in json.load(open(f"{H}/FableGod/floor_rows.json"))}


def listed(K, ctx, ids, row):
    """membership test factory for row `row` (predicting ids[row + 1]): S_K, plus the tokens ids[0..row] with ctx"""
    return None


def run(base_path, arm_path, ties_from_lp):
    base = json.load(open(base_path)); arm = json.load(open(arm_path)) if arm_path else None
    return base, arm


say("lm_head shortlist with context: misses = the baseline top-1 outside the list (%); rows read per check = K + mean unique context")
base = json.load(open(f"{M}/baseline.json"))
say("| list | code misses | ifeval misses | multi misses | mean unique context tokens (code / ifeval / multi) |")
say("|---|---|---|---|---|")
for K in (4096, 8192, 16384):
    S = set(order[:K])
    for ctx in (False, True):
        mis = {d: [0, 0] for d, _ in SETS}; uc = {d: [] for d, _ in SETS}
        for nm, b in base.items():
            ids, off = IDS[nm]; a = b["a"]; d = b["dom"]
            top1 = [t[0] for t in b["tk"]]
            seen = set(ids[:a].tolist()) if ctx else set()
            for j, t in enumerate(top1):
                row = a + j                                          # row a + j predicts ids[a + j + 1]
                if ctx:
                    seen.add(int(ids[row]))
                ok = (t in S) or (ctx and t in seen)
                mis[d][0] += not ok; mis[d][1] += 1
            if ctx:
                uc[d].append(len(seen))
        say(f"| {'C' if ctx else 'S'}_{K} | " + " | ".join(f"{100 * mis[d][0] / mis[d][1]:.3f}" for d, _ in SETS) + " | "
            + (" / ".join(f"{np.mean(uc[d]):.0f}" for d, _ in SETS) if ctx else "-") + " |")


def comb_flips(bpath, apath, K, ctx, lp_ties):
    b_ = json.load(open(bpath)); a_ = json.load(open(apath)); S = set(order[:K]); per = []
    for nm, b in b_.items():
        if b["dom"] != "code" or nm not in a_:
            continue
        ids, off = IDS[nm]; a = b["a"]; y = np.array([t[0] for t in b["tk"]])
        if lp_ties:
            p = np.exp(np.array(b["lp"])[:, :2]); tie = (p[:, 0] - p[:, 1]) <= 0.1
        else:
            tie = np.array([TIE.get((nm, off + a + j), False) for j in range(len(y))])
        seen = set(ids[:a].tolist()) if ctx else set(); x = []
        for j, row8 in enumerate(a_[nm]["tk"][:len(y)]):
            if ctx:
                seen.add(int(ids[a + j]))
            x.append(next((t for t in row8 if t in S or (ctx and t in seen)), -1))
        x = np.array(x); k = len(x); f = x != y[:k]
        per.append((f.sum(), k, (f & ~tie[:k]).sum(), (~tie[:k]).sum()))
    P = np.array(per, float)
    return 100 * P[:, 0].sum() / P[:, 1].sum(), 100 * P[:, 2].sum() / P[:, 3].sum(), len(P)


say("")
say("The combined code arm (1c + 4-bit mixers) with each list, flips % (all-row / non-tie; levels 1.34 / 0.87):")
say("| list | 12 code windows (out/) | 24 code windows (out_code24/) |")
say("|---|---|---|")
for K in (4096, 8192, 16384):
    for ctx in (False, True):
        r12 = comb_flips(f"{M}/baseline.json", f"{M}/bt_comb.json", K, ctx, False)
        r24 = comb_flips(f"{M24}/baseline.json", f"{M24}/bt_comb.json", K, ctx, True)
        say(f"| {'C' if ctx else 'S'}_{K} | {r12[0]:.2f} / {r12[1]:.2f} | {r24[0]:.2f} / {r24[1]:.2f} |")
say("(no list: 12 windows 1.22 / 0.57, 24 windows 1.06 / 0.59 approx; 16K static was the plan's choice)")
say(f"lm_head bytes per check: K x 2560 x 4.5 bits = {4096 * 1440 / 1e6:.1f} / {8192 * 1440 / 1e6:.1f} / {16384 * 1440 / 1e6:.1f} MB "
    f"for 4K / 8K / 16K (+ ~1.4 MB per 1K context rows), vs the full head 358 MB")
open(os.path.expanduser("~/Breakthrough/data/m1/lmhead_ctx_shortlist.out"), "w").write("\n".join(lines) + "\n")
