#!/usr/bin/env python3
"""Lossy 32-row check at 45 GB: bytes, ms and predicted map flips for union caps, bit-plane tiers (HOBBIT-style) and
acceptance weighting (AcceptMoE-style) (Breakthrough, Sun 15 Nov 2026 00:5x; WEEK98 goal B; pre-registered
docs/LOSSY_CHECK_PRICE.md f4d546b) [P: byte floors + an energy-law calibration on the decision map's measured arms].
Data (read-only): ~/FableGod/decision_map/out/xcap/<window>.npz (answer rows: top-10 ids + top-10-normalised weights w,
rank-sorted), out/<arm>.json + baseline.json (top-1 per answer row), the arms' keep masks, ~/FableGod/floor_rows.json
(tie rows: p1 - p2 <= 0.1).
Bytes (Leap cheapcheck.md): dense 3.44 GB + 2.135 MB per distinct expert per layer; 280 GB/s floor, 84 % realistic.
Quality: per row E = sum over layers and ranks of err x w^2 / 48; flips = a E^b per domain, fitted on drop arms (all
rows, and again on non-tie rows); bit-plane errors calibrated on t3_all vs its 4-bit control t4_all (v2, 00:5x).
Output: ~/Breakthrough/data/m1/lossy_check_price.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B lossy_check_price.py
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
DENSE, EXP_MB, BW, EFF = 3.44, 2.135, 280.0, 0.84
ROWS, NL = 32, 48
LEVEL = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}          # all-row levels (gate_top2_cf099)
NT_LEVEL = {"code": 0.87, "ifeval": 2.22, "multi": 1.63}       # non-tie levels (the VERDICT2 second line)
# bit-plane errors calibrated on the map: t3_all (MSB-3 everywhere) net of t4_all (the same path at 4 bits) gives an
# effective MSB-3 error of 0.065-0.09 per w^2 (rd_sim's 0.02 fails t3_all 3-6x); MSB-2 = 4x per plane [P]
TIERS = [("4-bit", 0.0, 1.0), ("MSB-3", 0.08, 0.778), ("MSB-2", 0.32, 0.556), ("drop", 1.0, 0.0)]
DOMS = ["code", "ifeval", "multi"]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(M, "baseline.json")))
TIE = {}
for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))):
    TIE[(r["nm"], int(r["i"]))] = r["cls"] == "tie"
W = {}
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f))
    a = int(z["a"]); m = z["ids"].shape[0]
    tie = np.array([TIE.get((f, a + j), False) for j in range(m)])
    W[f] = (base[f]["dom"], z["ids"].astype(np.int32), z["w"].astype(np.float32), a, tie)
say(f"Lossy 32-row check pricing v2: {len(W)} map windows, {sum(v[1].shape[0] for v in W.values())} answer rows, "
    f"{sum(int(v[4].sum()) for v in W.values())} tie rows")


# ---- calibration ----
def arm_rows(arm):
    a = json.load(open(os.path.join(M, arm + ".json")))
    out = {}
    for nm, (dom, ids, w, aa, tie) in W.items():
        if nm in a:
            p = np.array([t[0] for t in a[nm]["tk"]]); b = np.array([t[0] for t in base[nm]["tk"]]); k = min(len(p), len(b), len(tie))
            out[nm] = np.r_[(p[:k] != b[:k]), np.zeros(len(tie) - k, bool)]
    return out


def e_rows(kind, nm):
    dom, ids, w, a, tie = W[nm]
    w2 = w ** 2
    if kind[0] == "topk":
        return w2[:, :, kind[1]:].sum((1, 2)) / NL
    if kind[0] == "mask":
        keep = np.load(os.path.join(M, kind[1], nm))["keep"].astype(bool)[a:a + w.shape[0]]
        return (w2 * ~keep).sum((1, 2)) / NL
    return kind[1] * w2.sum((1, 2)) / NL                         # lowbit


def pooled(nt):
    """per domain: (mean E, flips %) over all rows or non-tie rows"""
    def f(kind, F):
        out = {}
        for d in DOMS:
            es, fs = [], []
            for nm, (dom, ids, w, a, tie) in W.items():
                if dom != d or nm not in F:
                    continue
                sel = ~tie if nt else np.ones(len(tie), bool)
                es.append(e_rows(kind, nm)[sel]); fs.append(F[nm][sel])
            E, FL = np.concatenate(es), np.concatenate(fs)
            out[d] = (float(E.mean()), 100 * float(FL.mean()), len(E))
        return out
    return f


CAL = [("topk2", ("topk", 2)), ("topk5", ("topk", 5)), ("top_k8", ("topk", 8)), ("q1u_cf09", ("mask", "q1u_cf09")),
       ("q1u_cf099", ("mask", "q1u_cf099")), ("qom_0_live_c51", ("mask", "bt/qom_0_live_c51")),
       ("qom_0_live_c147", ("mask", "ni/qom_0_live_c147"))]
VAL = [("t4_all", ("lowbit", 0.0)), ("t3_all", ("lowbit", 0.02))]
FIT = {}
PTS = {}
for nt in (False, True):
    tag = "NON-TIE rows" if nt else "ALL rows"
    say("")
    say(f"CALIBRATION ({tag}): measured flips % vs missed energy E per row-layer")
    say("| arm | " + " | ".join(f"{d} E | {d} flips" for d in DOMS) + " |")
    say("|---|" + "---|---|" * len(DOMS))
    pts = {d: [] for d in DOMS}; vals = {}
    for arm, kind in CAL + VAL:
        F = arm_rows(arm); P = pooled(nt)(kind, F)
        say(f"| {arm} | " + " | ".join(f"{P[d][0]:.5f} | {P[d][1]:.2f}" for d in DOMS) + " |")
        for d in DOMS:
            if (arm, kind) in CAL and P[d][1] > 0:
                pts[d].append((P[d][0], P[d][1]))
            elif (arm, kind) in VAL:
                vals.setdefault(arm, {})[d] = P[d]
    for d in DOMS:
        Pn = np.array(pts[d]); lx, ly = np.log(Pn[:, 0]), np.log(Pn[:, 1])
        b, la = np.polyfit(lx, ly, 1); r = ly - (la + b * lx)
        FIT[(nt, d)] = (np.exp(la), b); PTS[(nt, d)] = Pn
        e3 = vals["t3_all"][d][0] / 0.02 * TIERS[1][1]           # t3_all's E at the calibrated MSB-3 error
        say(f"fit {d} ({tag}): flips% = {np.exp(la):.3g} E^{b:.3f}; log residual rms {np.sqrt((r ** 2).mean()):.2f}; "
            f"t4_all path noise {vals['t4_all'][d][1]:.2f} %; t3_all measured {vals['t3_all'][d][1]:.2f} % vs predicted "
            f"{np.exp(la) * e3 ** b:.2f} at err 0.08 (+ path noise)")


def pred(nt, d, E):
    a, b = FIT[(nt, d)]
    return a * E ** b if E > 0 else 0.0


# ---- 32-row blocks with per-row bookkeeping ----
GIDX, W2, RDOM, RTIE, RD = [], [], [], [], []
S_parts, G_of = [], []
g0 = 0; nblk = 0; distinct = []
for nm, (dom, ids, w, a, tie) in W.items():
    for bi in range(ids.shape[0] // ROWS):
        sl = slice(bi * ROWS, (bi + 1) * ROWS)
        I, w2 = ids[sl], w[sl] ** 2
        gi = np.empty((ROWS, NL, 10), np.int64)
        for l in range(NL):
            u, inv = np.unique(I[:, l].ravel(), return_inverse=True)
            gi[:, l, :] = (inv + g0).reshape(ROWS, 10)
            G_of.append(np.full(len(u), len(distinct))); distinct.append(len(u)); g0 += len(u)
        GIDX.append(gi); W2.append(w2); RDOM.append(np.full(ROWS, DOMS.index(dom))); RTIE.append(tie[sl]); RD.append(np.arange(ROWS))
        nblk += 1
GIDX = np.concatenate(GIDX); W2 = np.concatenate(W2); RDOM = np.concatenate(RDOM); RTIE = np.concatenate(RTIE)
RD = np.concatenate(RD); GROUP = np.concatenate(G_of); NG = g0; NBL = len(distinct)
distinct = np.array(distinct)
say("")
say(f"BLOCKS: {nblk} full 32-row blocks ({len(RD)} rows); exact distinct experts per layer mean {distinct.mean():.1f} "
    f"(p99 {np.percentile(distinct, 99):.0f}, max {distinct.max()}); SELF-CHECK vs Leap's 100.5 (+-10 %): "
    f"{'PASS' if abs(distinct.mean() / 100.5 - 1) <= 0.10 else 'FAIL'}")


def S_of(s):
    rw = np.ones(len(RD)) if s is None else s ** RD
    return np.bincount(GIDX.ravel(), weights=(W2 * rw[:, None, None]).ravel(), minlength=NG)


def evaluate(errg, sizeg, s=None):
    """errg / sizeg: per global expert. Returns 4-bit-eq experts per layer and per domain (E all, E non-tie, E reach)"""
    keq = sizeg.sum() / NBL
    Erow = (errg[GIDX] * W2).sum((1, 2)) / NL
    rw = np.ones(len(RD)) if s is None else s ** RD
    out = {}
    for i, d in enumerate(DOMS):
        sel = RDOM == i
        out[d] = (float(Erow[sel].mean()), float(Erow[sel & ~RTIE].mean()),
                  float((Erow[sel] * rw[sel]).sum() / rw[sel].sum()))
    return keq, out


def tok_s(check_ms, kept):
    return kept / ((check_ms + 7.0) / 1000)


HEAD = ("| lever | 4-bit-eq / layer | GB | ms floor / 84 % | tok/s HE+ today / 98 % | flips % all (code / ifeval / multi) | "
        "non-tie (code / ifeval / multi) | all-row levels | non-tie levels | code-only |\n|---|---|---|---|---|---|---|---|---|---|")


def row(label, keq, E, reach=False, extra=""):
    gb = DENSE + EXP_MB * NL * keq / 1000
    a, b = 1000 * gb / BW, 1000 * gb / (BW * EFF)
    k = 2 if reach else 0
    fa = {d: pred(False, d, E[d][k]) for d in DOMS}
    fn = {d: pred(True, d, E[d][1] if not reach else E[d][2]) for d in DOMS}
    va = "within" if all(fa[d] <= LEVEL[d] for d in DOMS) else ("<=1.5x" if all(fa[d] <= 1.5 * LEVEL[d] for d in DOMS) else "over")
    vn = "within" if all(fn[d] <= NT_LEVEL[d] for d in DOMS) else ("<=1.5x" if all(fn[d] <= 1.5 * NT_LEVEL[d] for d in DOMS) else "over")
    vc = "within" if fa["code"] <= LEVEL["code"] and fn["code"] <= NT_LEVEL["code"] else "over"
    say(f"| {label} | {keq:.1f} | {gb:.2f} | {a:.1f} / {b:.1f} | {tok_s(b, 13.8):.0f} / {tok_s(b, 23.3):.0f} | "
        + " / ".join(f"{fa[d]:.2f}" for d in DOMS) + " | " + " / ".join(f"{fn[d]:.2f}" for d in DOMS)
        + f" | {va} | {vn} | {vc} |{extra}")


def cap(N, S):
    order = np.lexsort((-S, GROUP)); gs = GROUP[order]
    start = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    rank = np.arange(NG) - np.repeat(start, np.diff(np.r_[start, NG]))
    kept = np.zeros(NG, bool); kept[order] = rank < N
    return np.where(kept, 0.0, 1.0), kept.astype(float)


def tiers(lam, S):
    err = np.array([t[1] for t in TIERS]); size = np.array([t[2] for t in TIERS])
    t = (err[None, :] * S[:, None] + lam * size[None, :]).argmin(1)
    cnt = np.bincount(t, minlength=len(TIERS)) / NBL
    return err[t], size[t], " " + " ".join(f"{n} {c:.1f}" for (n, _, _), c in zip(TIERS, cnt))


S0 = S_of(None)
say(f"exact: {DENSE + EXP_MB * NL * distinct.mean() / 1000:.2f} GB, {1000 * (DENSE + EXP_MB * NL * distinct.mean() / 1000) / BW:.1f} ms "
    f"at the floor. Targets: 25 ms = 34.7 4-bit-eq experts per layer at the floor, 23.8 at 84 %.")
say("")
say("(c) UNION CAPS (keep the N largest sums of w^2 per block-layer; others dropped):")
say(HEAD)
for N in (80, 60, 50, 40, 35, 30, 24):
    e, sz = cap(N, S0); keq, E = evaluate(e, sz); row(f"cap {N}", keq, E)
say("")
say("(b) BIT-PLANE TIERS, calibrated errors (MSB-3 0.08, MSB-2 0.32), RD per block-layer:")
say(HEAD)
for lam in (2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2):
    e, sz, ex = tiers(lam, S0); keq, E = evaluate(e, sz); row(f"tiers lam {lam:.1e}", keq, E, extra=ex)
for s, nm in ((0.935, "today's code drafter s 0.935"), (0.98, "98 % drafter s 0.98")):
    Ss = S_of(s)
    say("")
    say(f"(a)+(b)+(c) ACCEPTANCE-WEIGHTED ({nm}): decisions on sum s^d w^2; quality = the reach-weighted E:")
    say(HEAD)
    for N in (40, 30, 24):
        e, sz = cap(N, Ss); keq, E = evaluate(e, sz, s); row(f"acc-cap {N}", keq, E, reach=True)
    for lam in (4e-3, 8e-3, 1.6e-2, 3.2e-2):
        e, sz, ex = tiers(lam, Ss); keq, E = evaluate(e, sz, s); row(f"acc-tiers lam {lam:.1e}", keq, E, reach=True, extra=ex)
open(os.path.expanduser("~/Breakthrough/data/m1/lossy_check_price.out"), "w").write("\n".join(lines) + "\n")

# ---- refinement (01:0x): the code-only 25 ms point and a 3-tier variant without MSB-2 (simpler to build) ----
say("")
say("REFINE: acceptance-weighted tiers near 25 ms at the floor (34.7 4-bit-eq / layer), and without MSB-2:")
say(HEAD)
T4 = TIERS
for s, nm in ((0.935, "s 0.935"), (0.98, "s 0.98")):
    Ss = S_of(s)
    for lam in (1.2e-2, 1.4e-2, 1.8e-2, 2.2e-2):
        e, sz, ex = tiers(lam, Ss); keq, E = evaluate(e, sz, s); row(f"acc-tiers {nm} lam {lam:.1e}", keq, E, reach=True, extra=ex)
    TIERS = [t for t in T4 if t[0] != "MSB-2"]
    for lam in (8e-3, 1.6e-2):
        e, sz, ex = tiers(lam, Ss); keq, E = evaluate(e, sz, s); row(f"acc-3tiers {nm} lam {lam:.1e}", keq, E, reach=True, extra=ex)
    TIERS = T4
open(os.path.expanduser("~/Breakthrough/data/m1/lossy_check_price.out"), "w").write("\n".join(lines) + "\n")

# ---- uncertainty (01:0x): leave-one-arm-out refits of the flip calibration, for the plan's key levers ----
def loo(nt, d, E):
    P = PTS[(nt, d)]; out = []
    for i in range(len(P)):
        Q = np.delete(P, i, 0); b, la = np.polyfit(np.log(Q[:, 0]), np.log(Q[:, 1]), 1)
        out.append(np.exp(la) * E ** b)
    return min(out), max(out)
say("")
say("UNCERTAINTY: predicted flips % with the leave-one-arm-out range [min, max] (7 refits per domain)")
say("| lever | code all | code non-tie | ifeval all | multi all |")
say("|---|---|---|---|---|")
KEY = [("1a acc-tiers s 0.935 lam 1.2e-2", 0.935, 1.2e-2, TIERS), ("1b acc-tiers s 0.98 lam 1.8e-2", 0.98, 1.8e-2, TIERS),
       ("1c acc-3tiers s 0.98 lam 1.6e-2", 0.98, 1.6e-2, [x for x in TIERS if x[0] != "MSB-2"]),
       ("1d acc-tiers s 0.98 lam 8e-3", 0.98, 8e-3, TIERS)]
for label, s, lam, tl in KEY:
    T0 = TIERS; TIERS = tl
    e, sz, ex = tiers(lam, S_of(s)); keq, E = evaluate(e, sz, s)
    TIERS = T0
    ca, cn, ia, ma = E["code"][2], E["code"][2], E["ifeval"][2], E["multi"][2]
    f = lambda nt, d, x: f"{pred(nt, d, x):.2f} [{loo(nt, d, x)[0]:.2f}, {loo(nt, d, x)[1]:.2f}]"
    say(f"| {label} ({keq:.1f} eq) | {f(False, 'code', ca)} | {f(True, 'code', cn)} | {f(False, 'ifeval', ia)} | {f(False, 'multi', ma)} |")
open(os.path.expanduser("~/Breakthrough/data/m1/lossy_check_price.out"), "w").write("\n".join(lines) + "\n")
