#!/usr/bin/env python3
"""comqom_engage.py (Breakthrough, Tue 17 Nov 2026 01:2x): engagement add-on for the comment-QOM map read-out
(comqom_read.py, which the armed runner calls and which stays untouched). BigMoE's G1 key test (00:48) failed only on
its COMQOM probe because the probe answer had no comment line to tag. So this says how much the lever ENGAGES on the
map, and where its flips land.
  1. Per code window: positions, and positions whose tier masks differ between cq_comqom and tiers_1c ("touched"),
     split into prompt positions and scored rows (scored row j = absolute position off + a + j; dm_common.py:618).
  2. Per arm present (bt_cq, bt_cq_res) against bt_cq_ctrl, all-row and non-tie flips vs the exact baseline:
     (a) on touched rows; (b) on untouched rows of touched windows; (c) on windows with no touched row.
  3. NULL CHECK: on windows with no touched row the masks are identical, so bt_cq must equal bt_cq_ctrl row for row
     (any difference = run nondeterminism or a setup difference).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/m1/comqom_engage.py   (CQ_O / CQ_ENGF: tests only)."""
import json
import os

import numpy as np

O = os.path.expanduser(os.environ.get("CQ_O", "~/FableGod/decision_map/out"))
OUTF = os.path.expanduser(os.environ.get("CQ_ENGF", "~/Breakthrough/data/m1/comqom_engage.out"))
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


base = json.load(open(os.path.join(O, "baseline.json")))
code = sorted(nm for nm, v in base.items() if v["dom"] == "code")
touched = {}
say(f"comment-QOM engagement on the map's {len(code)} code windows (masks bt/cq_comqom vs bt/tiers_1c)")
say("| window | positions | touched positions | of them in the prompt | scored rows | touched scored rows |")
say("|---|---|---|---|---|---|")
for nm in code:
    a = np.load(os.path.join(O, "bt", "cq_comqom", nm))
    t = np.load(os.path.join(O, "bt", "tiers_1c", nm))
    d = ((a["keep"] != t["keep"]) | (a["keep3"] != t["keep3"])).any(axis=(1, 2))
    touched[nm] = d
    s0 = int(base[nm]["off"]) + int(base[nm]["a"])
    ns = len(base[nm]["tk"])
    say(f"| {nm} | {len(d)} | {int(d.sum())} | {int(d[:s0].sum())} | {ns} | {int(d[s0:s0 + ns].sum())} |")
R = sum(len(d) for d in touched.values())
T = sum(int(d.sum()) for d in touched.values())
quiet_w = [nm for nm in code if not touched[nm].any()]
TS = sum(int(touched[nm][int(base[nm]["off"]) + int(base[nm]["a"]):][:len(base[nm]["tk"])].sum()) for nm in code)
NS = sum(len(base[nm]["tk"]) for nm in code)
say(f"TOTAL: {T} of {R} positions touched ({100 * T / R:.1f} %); scored rows touched {TS} of {NS} "
    f"({100 * TS / NS:.1f} %); windows with no touched position: {len(quiet_w)} "
    f"({', '.join(w.replace('tf_code_', '').replace('.npz', '') for w in quiet_w)})")


def arm_flips(arm):
    p = os.path.join(O, arm + ".json")
    if not os.path.exists(p):
        return None
    A = json.load(open(p))
    out = {}
    for nm in code:
        if nm not in A:
            continue
        b = base[nm]
        y = np.array([t[0] for t in b["tk"]])
        pp = np.exp(np.array(b["lp"])[:, :2])
        tie = (pp[:, 0] - pp[:, 1]) <= 0.1
        x = np.array([row[0] for row in A[nm]["tk"]])
        s0 = int(b["off"]) + int(b["a"])
        tch = touched[nm][s0:s0 + len(y)]
        k = min(len(x), len(y), len(tch))
        out[nm] = dict(f=x[:k] != y[:k], tie=tie[:k], tch=tch[:k], tk=[A[nm]["tk"][i] for i in range(k)],
                       lp=[A[nm]["lp"][i] for i in range(k)])
    return out


def part(F, sel):
    """(all-row flips, rows, non-tie flips, non-tie rows) over the rows sel(nm, rec) picks."""
    a = n = an = nn = 0
    for nm, r in F.items():
        m = sel(nm, r)
        a += int((r["f"] & m).sum())
        n += int(m.sum())
        an += int((r["f"] & ~r["tie"] & m).sum())
        nn += int((~r["tie"] & m).sum())
    return a, n, an, nn


def pct(x, y):
    return f"{100 * x / y:.2f}" if y else "-"


F = {arm: arm_flips(arm) for arm in ("bt_cq_ctrl", "bt_cq", "bt_cq_res")}
if F["bt_cq_ctrl"] is None:
    say("arms: bt_cq_ctrl NOT RUN -> no flip breakdown")
else:
    say("")
    say("flips % vs the exact baseline (all-row / non-tie), by where the rows sit:")
    say("| arm | (a) touched scored rows | (b) untouched scored rows, touched windows | (c) windows with no touched "
        "position |")
    say("|---|---|---|---|")
    sels = [lambda nm, r: r["tch"], lambda nm, r: (~r["tch"]) & bool(touched[nm].any()),
            lambda nm, r: np.full(len(r["f"]), not touched[nm].any())]
    for arm in ("bt_cq_ctrl", "bt_cq", "bt_cq_res"):
        if F[arm] is None:
            say(f"| {arm} | NOT RUN | | |")
            continue
        cells = []
        for s in sels:
            a, n, an, nn = part(F[arm], s)
            cells.append(f"{pct(a, n)} / {pct(an, nn)} ({a} / {an} of {n} / {nn})")
        say(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} |")
    # Tue 07:2x fix: a null window is one where THIS arm's masks equal the control's everywhere. RES changes the tier
    # masks on every window (cq_comqom_res vs tiers_1c differs at 119-301 positions on the 4 COMQOM-quiet windows), so
    # bt_cq_res has no null window; the first version compared it on COMQOM's quiet windows (901 / 901 rows differ,
    # expected, not a run problem). bt_cq's null (the pre-registered one, docstring item 3) is unchanged.
    MDIR = {"bt_cq": "cq_comqom", "bt_cq_res": "cq_comqom_res"}
    for arm in ("bt_cq", "bt_cq_res"):
        if F[arm] is None:
            continue
        qw = []
        for nm in code:
            a = np.load(os.path.join(O, "bt", MDIR[arm], nm))
            t = np.load(os.path.join(O, "bt", "tiers_1c", nm))
            if not ((a["keep"] != t["keep"]) | (a["keep3"] != t["keep3"])).any():
                qw.append(nm)
        if not qw:
            say(f"NULL CHECK {arm} vs bt_cq_ctrl: no window has masks identical to the control's (its masks differ on "
                f"every window), so no null check applies")
            continue
        diff_rows = tot = 0
        for nm in qw:
            if nm in F[arm] and nm in F["bt_cq_ctrl"]:
                c, x = F["bt_cq_ctrl"][nm], F[arm][nm]
                k = min(len(c["tk"]), len(x["tk"]))
                diff_rows += sum(1 for i in range(k) if c["tk"][i] != x["tk"][i] or c["lp"][i] != x["lp"][i])
                tot += k
        say(f"NULL CHECK {arm} vs bt_cq_ctrl on the {len(qw)} windows where its masks equal the control's: {diff_rows} of "
            f"{tot} rows differ -> {'CLEAN (identical, as the masks are)' if diff_rows == 0 else 'NOT CLEAN: run noise or a setup difference'}")
open(OUTF, "w").write("\n".join(lines) + "\n")
