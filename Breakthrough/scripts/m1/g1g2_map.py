#!/usr/bin/env python3
"""G1 / G2 map pre-screens (Breakthrough, Mon 16 Nov 2026 13:xx; pre-registered docs/CHECK_BYTES_900.md section 2,
9ac2850). Two steps around the map run in DM_OUT=~/FableGod/decision_map/out_code24 (the 24 code windows of f755d70):
  masks   C6 (lambda 1.8e-2) and lambda 2.0e-2 tier masks with code24_comb.py's rule (s 0.98, 3 tiers, 32-row blocks
          from the first answer row) from out_code24/scores -> out_code24/bt/tiers_c6, tiers_l20. Self-checks: the rule at
          lambda 1.6e-2 reproduces out_code24/bt/tiers_1c bit for bit; the full blocks' 4-bit-eq on the 12 screen windows
          equals the recount's code values (35.72 / 32.7) within 0.1.
  read    flips vs out_code24/baseline.json (all rows and non-tie; ties = the baseline's p1 - p2 <= 0.1); the 16K code
          shortlist applied offline for the arms with V1; 95 % window bootstrap (10,000, seed 20261116); validity
          (INVALID = top-8 ids and log-probs equal to the baseline on every row; bt_restore == baseline bit for bit);
          verdicts as pre-registered. Output ~/Breakthrough/data/m1/g1g2_map.out (+ .json).
Usage: ~/BigMoE/.venv/bin/python -I -B g1g2_map.py masks|read
"""
import collections
import glob
import json
import os
import sys

import numpy as np

O = os.path.expanduser("~/FableGod/decision_map/out_code24")
S_, ROWS, NL = 0.98, 32, 48
OUTF = os.path.expanduser("~/Breakthrough/data/m1/g1g2_map.out")
REF_EQ = {"tiers_c6": 35.72, "tiers_l20": 32.7}           # recount (5fc4c34 / ddd7a0b): full code blocks, 12 windows
LAMS = {"tiers_1c": 1.6e-2, "tiers_c6": 1.8e-2, "tiers_l20": 2.0e-2}
LV, NT = 1.34, 0.87
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def build(lam):
    """{window: (keep, keep3)} and the full blocks' 4-bit-eq per layer on each window"""
    T4, T3 = 2.775 * lam, 0.8457 * lam
    base = json.load(open(os.path.join(O, "baseline.json")))
    out, eq = {}, {}
    for f in sorted(os.listdir(os.path.join(O, "scores"))):
        z = np.load(os.path.join(O, "scores", f)); a = int(z["a"])
        ids, w = z["ids"].astype(np.int64), z["scores"].astype(np.float64)
        n = ids.shape[0]; m = n - a
        assert base[f]["a"] == a and len(base[f]["tk"]) == m, (f, a, m, len(base[f]["tk"]))
        keep = np.ones((n, NL, 10), bool); keep3 = np.zeros((n, NL, 10), bool)
        e_sum, e_n = 0.0, 0
        for b0 in range(a, n, ROWS):
            rows = np.arange(b0, min(b0 + ROWS, n)); rw = S_ ** np.arange(len(rows))
            for l in range(NL):
                u, inv = np.unique(ids[rows, l].ravel(), return_inverse=True)
                imp = np.bincount(inv, weights=(w[rows, l] ** 2 * rw[:, None]).ravel(), minlength=len(u))
                tu = np.where(imp > T4, 0, np.where(imp > T3, 1, 2))
                tr = tu[inv].reshape(len(rows), 10)
                keep[rows, l] = tr == 0; keep3[rows, l] = tr == 1
                if len(rows) == ROWS:
                    e_sum += (tu == 0).sum() + 0.778 * (tu == 1).sum(); e_n += 1
        out[f] = (keep, keep3)
        eq[f] = (e_sum, e_n)
    return out, eq


def masks():
    names = sorted(os.listdir(os.path.join(O, "scores")))
    xcap = os.path.expanduser("~/FableGod/decision_map/out/xcap")
    first12 = [f for f in names if os.path.exists(os.path.join(xcap, f))]     # the 12-window screen's code windows
    assert len(first12) == 12, first12
    chk, _ = build(LAMS["tiers_1c"])
    old = os.path.join(O, "bt", "tiers_1c")
    same = all(np.array_equal(chk[f][0], np.load(os.path.join(old, f))["keep"]) and
               np.array_equal(chk[f][1], np.load(os.path.join(old, f))["keep3"]) for f in names)
    say(f"SELF-CHECK masks (a): the rule at lambda 1.6e-2 == out_code24/bt/tiers_1c on all {len(names)} windows: "
        f"{'PASS' if same else 'FAIL'}")
    ok = same
    for nm in ("tiers_c6", "tiers_l20"):
        mk, eq = build(LAMS[nm])
        e12 = sum(eq[f][0] for f in first12) / max(1, sum(eq[f][1] for f in first12))
        e24 = sum(v[0] for v in eq.values()) / max(1, sum(v[1] for v in eq.values()))
        good = abs(e12 - REF_EQ[nm]) <= 0.1
        ok &= good
        say(f"SELF-CHECK masks (b) {nm} (lambda {LAMS[nm]:.1e}): full blocks' 4-bit-eq on the 12 screen windows {e12:.2f} "
            f"vs the recount's {REF_EQ[nm]} (+-0.1): {'PASS' if good else 'FAIL'}; all 24 windows {e24:.2f}")
        if good and same:
            od = os.path.join(O, "bt", nm); os.makedirs(od, exist_ok=True)
            for f, (k4, k3) in mk.items():
                np.savez_compressed(os.path.join(od, f), keep=k4, keep3=k3)
            say(f"  wrote {len(mk)} masks -> {od}")
    open(OUTF.replace(".out", "_masks.out"), "w").write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


ARMS = [("bt_g1h8", False, "G1 as built (g1x45g1a) = C6 + HC8 + int8 (answer rows); full head"),
        ("bt_l20g2", True, "G2 no-V4 route = lambda 2.0e-2 + V3 + int8 (answer rows) + V1"),
        ("bt_g1", True, "G1 with the plan's dense = C6 + V3 + int8 (answer rows) + V1"),
        ("bt_c6comb", True, "C6 combined = C6 + V3 + V1 (G1's bytes)"),
        ("bt_dense3", False, "DENSE-3 alone (V4's quality; exact experts, no shortlist)"),
        ("bt_l20comb", True, "lambda 2.0e-2 combined (+ V3 + V1)")]


def read():
    rng = np.random.default_rng(20261116)
    base = json.load(open(os.path.join(O, "baseline.json")))
    cnt = collections.Counter()
    for f in sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/*.npz"))):
        cnt.update(np.load(f)["tk"][:, 0].tolist())
    SL = set(t for t, _ in cnt.most_common(16384))
    order = sorted(base)

    def stat(P):
        P = np.array([r[1:] for r in P], float); idx = rng.integers(0, len(P), (10000, len(P))); out = []
        for c, d in ((0, 1), (2, 3)):
            r = 100 * P[idx, c].sum(1) / P[idx, d].sum(1)
            out.append((100 * P[:, c].sum() / P[:, d].sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)))
        return out
    say("G1 / G2 map pre-screens: code flips % vs the exact baseline (out_code24, 24 windows), [95 % window bootstrap]; "
        "ties = the baseline's p1 - p2 <= 0.1; lines all-row 1.34 / non-tie 0.87")
    res = {}
    rs = os.path.join(O, "bt_restore.json")
    restore_ok = None
    if os.path.exists(rs):
        r = json.load(open(rs))
        restore_ok = all(nm in r and r[nm]["tk"] == base[nm]["tk"] and r[nm]["lp"] == base[nm]["lp"] for nm in order)
        say(f"VALIDITY bt_restore == baseline bit for bit (top-8 ids and log-probs, every row of {len(order)} windows): "
            f"{'PASS' if restore_ok else 'FAIL (arms after the first unrestored one are VOID)'}")
    else:
        say("VALIDITY bt_restore: not run (restore unverified)")
    cp = os.path.join(O, "bt_int8_census.jsonl")
    if os.path.exists(cp):
        for i, ln in enumerate(open(cp)):
            c = json.loads(ln)
            ok = c["windows"] == len(order) and not c["bad"] and c["rows_q"] == c["rows_q_want"] > 0
            say(f"VALIDITY int8 arm {i + 1} (run order: bt_g1h8, bt_l20g2, bt_g1): {c['windows']} windows, answer rows x modules "
                f"quantized {c['rows_q']} (want {c['rows_q_want']}), windows with a mismatch {len(c['bad'])}, census "
                f"{c['census']}: {'PASS' if ok else 'FAIL (that arm is VOID)'}")
    say("| arm | windows | all-row % | non-tie % | verdict | identical-to-baseline rows (INVALID if all) |")
    say("|---|---|---|---|---|---|")
    for arm, sl, lab in ARMS:
        p = os.path.join(O, arm + ".json")
        if not os.path.exists(p):
            say(f"| {arm}: {lab} | - | - | - | NOT RUN | - |")
            continue
        A = json.load(open(p)); per = []; same_rows = tot_rows = 0
        for nm in order:
            if nm not in A:
                continue
            b = base[nm]; y = np.array([t[0] for t in b["tk"]])
            pp = np.exp(np.array(b["lp"])[:, :2]); tie = (pp[:, 0] - pp[:, 1]) <= 0.1
            if sl:
                x = np.array([next((t for t in row if t in SL), -1) for row in A[nm]["tk"]])
            else:
                x = np.array([row[0] for row in A[nm]["tk"]])
            k = min(len(x), len(y)); f = x[:k] != y[:k]
            per.append((nm, f.sum(), k, (f & ~tie[:k]).sum(), (~tie[:k]).sum()))
            same_rows += sum(1 for i in range(min(len(A[nm]["tk"]), len(b["tk"])))
                             if A[nm]["tk"][i] == b["tk"][i] and A[nm]["lp"][i] == b["lp"][i])
            tot_rows += len(b["tk"])
        (ar, alo, ahi), (nt, nlo, nhi) = stat(per)
        invalid = same_rows == tot_rows
        if invalid:
            v = "INVALID"
        elif arm == "bt_dense3":
            v = "PASS" if ar <= LV else ("BETWEEN" if ar <= 2.01 else "FAIL")
        else:
            v = "PASS" if (ar <= LV and nt <= NT) else "OVER (" + ", ".join(
                x for x, c in (("all-row", ar > LV), ("non-tie", nt > NT)) if c) + ")"
        res[arm] = dict(all_row=[round(ar, 3), round(alo, 3), round(ahi, 3)], non_tie=[round(nt, 3), round(nlo, 3),
                        round(nhi, 3)], verdict=v, windows=len(per), rows=int(sum(r[2] for r in per)), same_rows=same_rows)
        say(f"| {arm}: {lab} | {len(per)} ({int(sum(r[2] for r in per))} rows) | {ar:.2f} [{alo:.2f}, {ahi:.2f}] | "
            f"{nt:.2f} [{nlo:.2f}, {nhi:.2f}] | **{v}** | {same_rows}/{tot_rows} |")
    json.dump(dict(restore_ok=restore_ok, arms=res), open(OUTF.replace(".out", ".json"), "w"), indent=1)
    open(OUTF, "w").write("\n".join(lines) + "\n")


{"masks": masks, "read": read}[sys.argv[1]]()
