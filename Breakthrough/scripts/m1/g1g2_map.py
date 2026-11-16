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

O = os.path.expanduser(os.environ.get("G1G2_O", "~/FableGod/decision_map/out_code24"))    # G1G2_O: tests only
S_, ROWS, NL = 0.98, 32, 48
OUTF = os.path.expanduser(os.environ.get("G1G2_OUTF", "~/Breakthrough/data/m1/g1g2_map.out"))
REF_EQ = {"tiers_c6": 35.72, "tiers_l20": 32.7}           # recount (5fc4c34 / ddd7a0b): full code blocks, 12 windows
LAMS = {"tiers_1c": 1.6e-2, "tiers_c6": 1.8e-2, "tiers_l20": 2.0e-2}
LV, NT = 1.34, 0.87
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def build(lam):
    """{window: (keep, keep3)} and the full blocks' 4-bit-eq per layer on each window; lam: a scalar, or one lambda
    per layer (section 6's band screen)"""
    lam = np.broadcast_to(np.asarray(lam, np.float64), (NL,))
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
                tu = np.where(imp > T4[l], 0, np.where(imp > T3[l], 1, 2))
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


BANDS = [(0, 16), (16, 32), (32, 48)]
LB_HI = 3.2e-2


def band_bytes(mk, a_of):
    """expert MB per check by band (full 32-row answer blocks; real pieces 2.784 / 2.064 MB), from masks"""
    ids_dir = os.path.join(O, "scores")
    tot = np.zeros(NL); nb = 0
    for f, (k4, k3) in mk.items():
        ids = np.load(os.path.join(ids_dir, f))["ids"].astype(np.int64); a = a_of[f]; n = ids.shape[0]
        for b0 in range(a, n - ROWS + 1, ROWS):
            rows = np.arange(b0, b0 + ROWS); nb += 1
            for l in range(NL):
                I = ids[rows, l]
                n4 = len(np.unique(I[k4[rows, l]])); n3 = len(np.setdiff1d(np.unique(I[k3[rows, l]]), np.unique(I[k4[rows, l]])))
                tot[l] += n4 * 2.784 + n3 * 2.064
    return tot / max(1, nb), nb


def lbmasks():
    """section 6: lambda 3.2e-2 on one band (0-15 / 16-31 / 32-47), C6 (1.8e-2) elsewhere -> out_code24/bt/tiers_lb{0,1,2}.
    Self-checks: an all-1.8e-2 vector reproduces tiers_c6 bit for bit; C6's per-band GB == 1.67 / 1.48 / 1.46 (+-0.02)."""
    a_of = {f: int(np.load(os.path.join(O, "scores", f))["a"]) for f in os.listdir(os.path.join(O, "scores"))}
    c6, _ = build(np.full(NL, 1.8e-2))
    old = os.path.join(O, "bt", "tiers_c6")
    same = all(np.array_equal(c6[f][0], np.load(os.path.join(old, f))["keep"]) and
               np.array_equal(c6[f][1], np.load(os.path.join(old, f))["keep3"]) for f in c6)
    pb0, nb = band_bytes(c6, a_of)
    gb0 = [pb0[a:b].sum() / 1e3 for a, b in BANDS]
    ok = same and all(abs(x - y) <= 0.02 for x, y in zip(gb0, (1.67, 1.48, 1.46)))
    say(f"SELF-CHECK lbmasks: per-layer build at 1.8e-2 == tiers_c6 bit for bit: {same}; C6 per-band GB "
        f"{[round(x, 3) for x in gb0]} (want 1.67 / 1.48 / 1.46, {nb} full blocks): {'PASS' if ok else 'FAIL'}")
    for i, (a, b) in enumerate(BANDS):
        lam = np.full(NL, 1.8e-2); lam[a:b] = LB_HI
        mk, _ = build(lam)
        pb, _ = band_bytes(mk, a_of)
        od = os.path.join(O, "bt", f"tiers_lb{i}"); os.makedirs(od, exist_ok=True)
        for f, (k4, k3) in mk.items():
            np.savez_compressed(os.path.join(od, f), keep=k4, keep3=k3)
        say(f"  tiers_lb{i}: lambda {LB_HI:.1e} on L{a}-{b - 1}: experts {pb.sum() / 1e3:.2f} GB per check (C6 "
            f"{pb0.sum() / 1e3:.2f}): saves {(pb0.sum() - pb.sum()) / 1e3:.2f} GB; -> {od}")
    open(OUTF.replace(".out", "_lbmasks.out"), "w").write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


def resmasks():
    """amendment (e): C6 + RES (Fit12's resident-only 3-bit tier at C 256, ~/Fit12/scripts/res1c_price.py's rule): per
    32-row answer block and layer, an MSB-3-tier expert with any non-resident slot (out/res_c256, Breakthrough's LRU
    replay of the 45 GB base pool, dm_res_masks.py) is dropped; the 4-bit tier stays. Residency exists for the 12 screen
    windows only (out/res_c256); windows 13-24 copy tiers_c6 unchanged, so the arm's RES read is paired on windows 1-12."""
    src, od = os.path.join(O, "bt", "tiers_c6"), os.path.join(O, "bt", "tiers_c6res")
    resd = os.path.expanduser("~/FableGod/decision_map/out/res_c256")
    os.makedirs(od, exist_ok=True)
    n3 = n3nr = 0
    done = []
    for f in sorted(os.listdir(src)):
        t = np.load(os.path.join(src, f)); k4, k3 = t["keep"].copy(), t["keep3"].copy()
        if os.path.exists(os.path.join(resd, f)):
            res = np.load(os.path.join(resd, f))["keep"].astype(bool)
            ids = np.load(os.path.join(O, "routing", f))["ids"].astype(np.int64)
            a = json.load(open(os.path.join(O, "baseline.json")))[f]["a"]
            n = min(ids.shape[0], res.shape[0], k4.shape[0])
            assert not (k4 & k3).any(), f
            for b0 in range(a, n, ROWS):
                rows = np.arange(b0, min(b0 + ROWS, n))
                for l in range(NL):
                    I, R, K3 = ids[rows, l], res[rows, l], k3[rows, l]
                    for e in np.unique(I[K3]):
                        sel = I == e
                        n3 += 1
                        if not R[sel].all():
                            n3nr += 1
                            k3[rows, l] &= ~sel
            done.append(f)
        np.savez_compressed(os.path.join(od, f), keep=k4, keep3=k3)
    pct = 100 * n3nr / max(1, n3)
    ok = len(done) == 12 and 1.0 <= pct <= 8.0
    say(f"SELF-CHECK resmasks: residency on {len(done)} windows (want the 12 screen windows); MSB-3-tier experts per "
        f"block-layer dropped as non-resident {pct:.2f} % ({n3nr} of {n3}; Fit12 at 1c: 3.58 %): {'PASS' if ok else 'FAIL'}")
    open(OUTF.replace(".out", "_resmasks.out"), "w").write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


ARMS = [("bt_g1h8", False, "G1 as built (g1x45g1a) = C6 + HC8 + int8 (answer rows); full head"),
        ("bt_g1h8_res", False, "G1 as built + RES (resident-only MSB-3 tier at C 256; RES on windows 1-12 only)"),
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
            say(f"VALIDITY int8 arm {i + 1} (run order: bt_g1h8, bt_l20g2, bt_g1h8_res, bt_g1): {c['windows']} windows, answer rows x modules "
                f"quantized {c['rows_q']} (want {c['rows_q_want']}), windows with a mismatch {len(c['bad'])}, census "
                f"{c['census']}: {'PASS' if ok else 'FAIL (that arm is VOID)'}")
    # amendment (d), Mon 13:2x, before any run: the registered shortlist SL is in sample (all-bank top-1 tokens: the bank
    # holds the 24 code windows themselves; v1_lmsl_lists.py), so V1 arms also read the out-of-sample lists V1 would
    # ship (data/m1/lmsl_code16k / 32k.npy: Magicoder top-1 + FR-Spec fill) WITH session rows (prompt + committed answer
    # prefix, as SVH's screened_head_add feeds them; drafts not added: conservative). G2's V1 decision reads v1-32k.
    V1L = {k: set(np.load(os.path.expanduser(f"~/Breakthrough/data/m1/lmsl_code{k}.npy")).tolist()) for k in ("16k", "32k")}
    TOK = {nm: np.load(os.path.join(O, "routing", nm))["tokens"].astype(np.int64) for nm in order}

    def pick(A, nm, mode):
        if mode == "full":
            return np.array([row[0] for row in A[nm]["tk"]])
        if mode == "sl-registered (in sample)":
            return np.array([next((t for t in row if t in SL), -1) for row in A[nm]["tk"]])
        L = V1L[mode.split()[0].split("-")[1]]
        a = base[nm]["a"]; sess = set(TOK[nm][:a + 1].tolist()); out = []
        for j, row in enumerate(A[nm]["tk"]):
            if j:
                sess.add(int(TOK[nm][a + j]))
            out.append(next((t for t in row if t in L or t in sess), -1))
        return np.array(out)

    say("| arm | head read | windows | all-row % | non-tie % | verdict | identical-to-baseline rows (INVALID if all) |")
    say("|---|---|---|---|---|---|---|")
    for arm, sl, lab in ARMS:
        p = os.path.join(O, arm + ".json")
        if not os.path.exists(p):
            say(f"| {arm}: {lab} | - | - | - | - | NOT RUN | - |")
            continue
        A = json.load(open(p))
        same_rows = tot_rows = 0
        for nm in order:
            if nm in A:
                same_rows += sum(1 for i in range(min(len(A[nm]["tk"]), len(base[nm]["tk"])))
                                 if A[nm]["tk"][i] == base[nm]["tk"][i] and A[nm]["lp"][i] == base[nm]["lp"][i])
                tot_rows += len(base[nm]["tk"])
        modes = (["sl-registered (in sample)", "v1-16k + session rows", "v1-32k + session rows"] if sl else ["full"])
        for mode in modes:
            per = []
            for nm in order:
                if nm not in A:
                    continue
                b = base[nm]; y = np.array([t[0] for t in b["tk"]])
                pp = np.exp(np.array(b["lp"])[:, :2]); tie = (pp[:, 0] - pp[:, 1]) <= 0.1
                x = pick(A, nm, mode)
                k = min(len(x), len(y)); f = x[:k] != y[:k]
                per.append((nm, f.sum(), k, (f & ~tie[:k]).sum(), (~tie[:k]).sum()))
            (ar, alo, ahi), (nt, nlo, nhi) = stat(per)
            if same_rows == tot_rows:
                v = "INVALID"
            elif arm == "bt_dense3":
                v = "PASS" if ar <= LV else ("BETWEEN" if ar <= 2.01 else "FAIL")
            else:
                v = "PASS" if (ar <= LV and nt <= NT) else "OVER (" + ", ".join(
                    x for x, c in (("all-row", ar > LV), ("non-tie", nt > NT)) if c) + ")"
            res[f"{arm} | {mode}"] = dict(all_row=[round(ar, 3), round(alo, 3), round(ahi, 3)],
                                          non_tie=[round(nt, 3), round(nlo, 3), round(nhi, 3)], verdict=v,
                                          windows=len(per), rows=int(sum(r[2] for r in per)), same_rows=same_rows)
            say(f"| {arm}: {lab} | {mode} | {len(per)} ({int(sum(r[2] for r in per))} rows) | {ar:.2f} [{alo:.2f}, {ahi:.2f}] | "
                f"{nt:.2f} [{nlo:.2f}, {nhi:.2f}] | **{v}** | {same_rows}/{tot_rows} |")
    # amendment (e): RES's increment, paired on the windows that have residency (the 12 screen windows)
    pa, pb = os.path.join(O, "bt_g1h8.json"), os.path.join(O, "bt_g1h8_res.json")
    if os.path.exists(pa) and os.path.exists(pb):
        resd = os.path.expanduser("~/FableGod/decision_map/out/res_c256")
        w12 = [nm for nm in order if os.path.exists(os.path.join(resd, nm))]
        Aa, Ab = json.load(open(pa)), json.load(open(pb))
        tot = fa = fb = nta = ntb = ntt = 0
        for nm in w12:
            b = base[nm]; y = np.array([t[0] for t in b["tk"]])
            pp = np.exp(np.array(b["lp"])[:, :2]); tie = (pp[:, 0] - pp[:, 1]) <= 0.1
            xa, xb = pick(Aa, nm, "full"), pick(Ab, nm, "full")
            k = min(len(xa), len(xb), len(y))
            ga, gb = xa[:k] != y[:k], xb[:k] != y[:k]
            tot += k; fa += ga.sum(); fb += gb.sum(); ntt += (~tie[:k]).sum()
            nta += (ga & ~tie[:k]).sum(); ntb += (gb & ~tie[:k]).sum()
        say(f"RES increment, paired on the {len(w12)} windows with residency ({tot} rows): all-row {100 * fa / tot:.2f} -> "
            f"{100 * fb / tot:.2f} ({100 * (fb - fa) / tot:+.2f} pt; Fit12's price +0.046), non-tie {100 * nta / ntt:.2f} -> "
            f"{100 * ntb / ntt:.2f} ({100 * (ntb - nta) / ntt:+.2f} pt; price +0.033)")
    json.dump(dict(restore_ok=restore_ok, arms=res), open(OUTF.replace(".out", ".json"), "w"), indent=1)
    open(OUTF, "w").write("\n".join(lines) + "\n")


{"masks": masks, "read": read, "resmasks": resmasks, "lbmasks": lbmasks}[sys.argv[1]]()
