#!/usr/bin/env python3
"""ADAPTK-LOSSY (NewIdeas, Tue 17 Nov; the time is the commit's): does ADAPTIVE K still pay on the LOSSY line?
ADAPTIVE K = KERN2: 16 or 32 rows per check, picked by the drafter's own path_p (pop while path_p >= tau, >= 15 pops,
rows round up to 16 / 32). Measured so far only with EXACT acceptance (ADAPTK.md: HE+ +8.0-10.3 % on E14 costs).
PLAN900's conditional rung assumed it applies to the lossy line "the same way". But the lossy line runs trees WITH
close-call acceptance CC(1), and TREES-x-CLOSECALL found trees and CC overlapping (additivity 0.54-0.62). So CC could
shrink the 16-vs-32 kept gap, or grow it (longer runs reward deep trees). This screen measures it.

Acceptance per anchor, at 15 and 31 pops of BigMoE's 127-node k8 dumps (the first K - 1 pops = EAGLE-2-K's nodes):
  EXACT: accepted = max{depth + 1 : on_true node among the pops} (adaptk_draft.dense);
  CC(1): tree_cc32.py's tree + CC rule, unchanged (delta 1 nat): at the true path's first unselected depth, the head's
         selected top-1 child is kept if the 125B's log-prob gap for it (cells dlp) >= -1, then its selected best-child
         chain while each step is right or within delta (judged by the true text's cells, as every close-call walk here).
Walk: certify98's (R0 = 1), kept = accepted + 1 per check, along the true text.
Costs per check (ms) = check(K) + L x tree height (one draft forward per level):
  floor25 : PUSH800's lossy byte floor, 25 x R_floor(K) / R_floor(32) x c(K) + 4 (Leap v2; adaptk_oracle "lossy");
  v4pass  : LADDER v4's modules-PASS check, 40 ms at 32 rows, 16 rows at E14's measured ratio (1.025 / (1.4372 x 1.04)),
            overhead 4; L 0.42 (8-bit + 8K screen, ~42 MB per level, f 0.269);
  v4stretch: 32 ms at 32 rows, same ratio, overhead 1.5, L 0.25 (f 0.10).
POLICY tau: the frozen 0.9 (adaptk_fresh), the in-set best and the cross-set tau (the other set's best).
ORACLE: hindsight DP over 16 / 32 per check (Dinkelbach), the ceiling of any KERN2 policy.
Line (registered here, before the run): the rung's lossy claim HOLDS if the HE+ policy gain over the better fixed K is
>= +5 % under CC(1) at the frozen tau on v4pass; FAILS below +2 %; else PARTIAL. Information only otherwise; this does
not touch the exact gate (adaptk_fresh.py) nor any running job.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_lossy.py
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98  # noqa: E402

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
SETS = {"he": ("heplus24", 13.77), "mbpp": ("mbppbank", 6.92)}
DEP, DELTA = 31, 1.0
TAUS = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
TAU0 = 0.9
V2 = json.load(open(os.path.expanduser("~/Leap/data/week98/treek_costs_v2.json")))
FL, CK = V2["R"]["floor"], V2["c"]
E14R = 1.025 / (1.4372 * 1.04)
COSTS = {   # name: (T16, T32, L per level)
    "floor25": (25.0 * FL["16"] / FL["32"] * CK["16"] + 4.0, 25.0 * CK["32"] + 4.0, 0.42),
    "v4pass": (40.0 * E14R + 4.0, 40.0 + 4.0, 0.42),
    "v4stretch": (32.0 * E14R + 1.5, 32.0 + 1.5, 0.25),
}
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def load_cells(tag, published):
    """tree_cc32.load_cells (S9c layout + m and dlp), unchanged: per window rt / R (y rank) / DL (dlp), fi, an, n"""
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all(), f"{tag}: {k} differs between cells and topk"
    dep, right, yr = tz["depth"].astype(int), tz["right"].astype(int), tz["y_rank"].astype(int)
    yr = np.where((yr == 0) & (right == 0), 1, yr)
    row, fi = tz["row"].astype(np.int64), tz["fi"].astype(np.int64)
    dl = cz["dlp"].astype(np.float32)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32 = [], []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); R = np.full((n0 + 1, DEP), -1, np.int16)
        DL = np.full((n0 + 1, DEP), np.nan, np.float32)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; R[:c, j] = yr[s:t][m]; DL[:c, j] = dl[s:t][m]
        assert (rt == ref[wi][0]).all(), f"{tag}: window {wi} layout != certify98"
        R[rt < 0] = -1
        k32 += C98.kept_walk(rt, C98.R0, 31)
        an = row[s:t][dw == 0] - 1
        W.append({"R": R, "DL": DL, "fi": int(fi[s]), "an": an, "n": n0 + 1})
    say(f"SELF-CHECK {tag}: chain-32 {np.mean(k32):.2f} vs {published} -> "
        f"{'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}")
    return W


def anchor_arrays(tag, W):
    """per window position: exact and CC(1) accepted at 15 / 31 pops, tree height, and the pops with path_p >= tau"""
    z = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
    fi, an, nd, pa, dp, pp, ot = (z[k] for k in ("fi", "an", "node", "parent", "depth", "path_p", "on_true"))
    order = np.lexsort((nd, an, fi))
    fi, an, nd, pa, dp, pp, ot = (a[order] for a in (fi, an, nd, pa, dp, pp, ot))
    b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    byfi = {X["fi"]: X for X in W}
    out = {X["fi"]: {"ex16": np.full(X["n"], -1), "ex32": np.full(X["n"], -1), "cc16": np.full(X["n"], -1),
                     "cc32": np.full(X["n"], -1), "h16": np.zeros(X["n"]), "h32": np.zeros(X["n"]),
                     "npop": {t: np.zeros(X["n"], np.int64) for t in TAUS}} for X in W}
    mono = nanc = 0
    for b0, b1 in zip(b[:-1], b[1:]):
        f_, a_ = int(fi[b0]), int(an[b0])
        if f_ not in rowof or a_ not in rowof[f_]:
            continue
        r = rowof[f_][a_]; X = byfi[f_]; o = out[f_]
        n_, p_, d_, ppp, t_ = nd[b0:b1], pa[b0:b1], dp[b0:b1].astype(np.int64), pp[b0:b1].astype(np.float64), ot[b0:b1].astype(bool)
        nanc += 1; mono += int(np.all(np.diff(ppp) <= 1e-6))
        for t in TAUS:
            o["npop"][t][r] = int((ppp >= t).sum())
        R_, DL_ = X["R"][r], X["DL"][r]
        for NODES, key in ((15, "16"), (31, "32")):
            selm = n_ < NODES; hit = t_ & selm
            A_t = int(d_[hit].max()) + 1 if hit.any() else 0
            o["ex" + key][r] = A_t
            o["h" + key][r] = int(d_[selm].max()) + 1 if selm.any() else 0
            cc = A_t
            if A_t < min(DEP, NODES) and R_[A_t] > 0 and not np.isnan(DL_[A_t]) and DL_[A_t] >= -DELTA:
                pid = -1 if A_t == 0 else int(n_[np.flatnonzero(hit & (d_ == A_t - 1))[0]])
                kids = np.flatnonzero(selm & (p_ == pid) & (d_ == A_t))
                if len(kids):
                    cur = int(kids[np.argmax(ppp[kids])]); cont = 0; d2 = A_t
                    while True:
                        ch = np.flatnonzero(selm & (p_ == n_[cur]))
                        if not len(ch):
                            break
                        cur = int(ch[np.argmax(ppp[ch])]); d2 += 1
                        if d2 >= DEP or R_[d2] < 0:
                            break
                        if R_[d2] == 0 or (not np.isnan(DL_[d2]) and DL_[d2] >= -DELTA):
                            cont += 1
                        else:
                            break
                    cc = A_t + 1 + cont
            o["cc" + key][r] = cc
    say(f"  {tag}: {nanc} anchors matched; path_p non-increasing in pop order on {100 * mono / max(nanc, 1):.1f} %")
    return out


def walk(W, O, acc, choose, cost):
    """per window (kept, ms); acc 'ex' / 'cc'; choose(o, p) -> 16 / 32; an anchor missing from the dump: 16 rows, 0"""
    res = []
    for X in W:
        o = O[X["fi"]]; n = X["n"]; p = 1; kept = tm = 0.0
        while p < n:
            r = choose(o, p)
            a = int(o[acc + str(r)][p])
            if a < 0:
                r, a, h = 16, 0, 0.0
            else:
                h = o["h" + str(r)][p]
            kept += a + 1; tm += cost(r, h); p += a + 1
        res.append((kept, tm))
    return np.array(res)


def oracle(W, O, acc, cost, iters=12):
    """hindsight DP over 16 / 32 per check (Dinkelbach on tokens per ms)"""
    base = walk(W, O, acc, lambda o, p: 16, cost); lam = base[:, 0].sum() / base[:, 1].sum()
    for _ in range(iters):
        kept = tm = 0.0
        for X in W:
            o = O[X["fi"]]; n = X["n"]; V = np.zeros(n + 40); ch = np.zeros(n, np.int64)
            for p in range(n - 1, 0, -1):
                best = -1e18
                for r in (16, 32):
                    a = int(o[acc + str(r)][p]); h = o["h" + str(r)][p]
                    if a < 0:
                        a, h = 0, 0.0
                    v = (a + 1) - lam * cost(r, h) + (V[p + a + 1] if p + a + 1 < n else 0.0)
                    if v > best:
                        best, ch[p] = v, r
                V[p] = best
            p = 1
            while p < n:
                r = int(ch[p]); a = int(o[acc + str(r)][p]); h = o["h" + str(r)][p]
                if a < 0:
                    a, h = 0, 0.0
                kept += a + 1; tm += cost(r, h); p += a + 1
        new = kept / tm
        if abs(new - lam) < 1e-12:
            break
        lam = new
    return 1000 * lam


def tps(X):
    return 1000 * X[:, 0].sum() / X[:, 1].sum()


def main():
    rng = np.random.default_rng(17)
    say("ADAPTK-LOSSY: KERN2 (16 / 32 rows by path_p >= tau) with EXACT vs CC(1) acceptance, lossy cost models")
    say("costs (T16, T32, L ms/level): " + "; ".join(f"{k} {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f}" for k, v in COSTS.items()))
    data = {}
    for s, (tag, pub) in SETS.items():
        W = load_cells(tag, pub)
        O = anchor_arrays(tag, W)
        data[s] = (W, O)
        for acc in ("ex", "cc"):
            k16 = walk(W, O, acc, lambda o, p: 16, lambda r, h: 1.0)
            k32 = walk(W, O, acc, lambda o, p: 32, lambda r, h: 1.0)
            say(f"  {s} {acc}: kept per check 16 rows {k16[:, 0].sum() / k16[:, 1].sum():.3f}, 32 rows "
                f"{k32[:, 0].sum() / k32[:, 1].sum():.3f}")
    res = {}
    for cname, (T16, T32, L) in COSTS.items():
        cost = (lambda r, h, T16=T16, T32=T32, L=L: (T16 if r == 16 else T32) + L * h)
        for acc in ("ex", "cc"):
            best_tau = {}
            for s, (W, O) in data.items():
                f16 = walk(W, O, acc, lambda o, p: 16, cost); f32 = walk(W, O, acc, lambda o, p: 32, cost)
                fb = max(tps(f16), tps(f32)); fbk = 16 if tps(f16) >= tps(f32) else 32
                pol = {t: walk(W, O, acc, (lambda o, p, t=t: 16 if max(15, o["npop"][t][p]) <= 15 else 32), cost)
                       for t in TAUS}
                tb = max(pol, key=lambda t: tps(pol[t])); best_tau[s] = tb
                o_ = oracle(W, O, acc, cost)
                P0 = pol[TAU0]; F = f16 if fbk == 16 else f32
                bs = []
                for _ in range(2000):
                    j = rng.integers(0, len(W), len(W))
                    bs.append((P0[j, 0].sum() / P0[j, 1].sum()) / (F[j, 0].sum() / F[j, 1].sum()) - 1)
                r_ = {"fixed16": tps(f16), "fixed32": tps(f32), "best_fixed": fbk, "oracle": o_,
                      "policy": {str(t): tps(v) for t, v in pol.items()}, "tau_in": tb,
                      "gain_tau0": tps(P0) / fb - 1, "ci95_tau0": [float(np.percentile(bs, 2.5)),
                                                                     float(np.percentile(bs, 97.5))]}
                res.setdefault(cname, {}).setdefault(acc, {})[s] = r_
                say(f"[{cname} {acc}] {s}: fixed 16 {tps(f16):.1f} / 32 {tps(f32):.1f} (best {fbk}) | ORACLE {o_:.1f} "
                    f"({100 * (o_ / fb - 1):+.1f} %) | POLICY tau {TAU0} {tps(P0):.1f} ({100 * r_['gain_tau0']:+.1f} % "
                    f"[{100 * r_['ci95_tau0'][0]:+.1f}, {100 * r_['ci95_tau0'][1]:+.1f}]) | in-set tau {tb} "
                    f"{tps(pol[tb]):.1f} ({100 * (tps(pol[tb]) / fb - 1):+.1f} %)")
            for s in data:
                other = [x for x in data if x != s][0]; t = best_tau[other]; r_ = res[cname][acc][s]
                fb = max(r_["fixed16"], r_["fixed32"]); v = r_["policy"][str(t)]; r_["cross"] = {"tau": t, "tps": v}
                say(f"[{cname} {acc}] {s}: CROSS tau {t} (from {other}) {v:.1f} ({100 * (v / fb - 1):+.1f} %)")
    g = res["v4pass"]["cc"]["he"]["gain_tau0"]
    verdict = "HOLDS" if g >= 0.05 else ("FAILS" if g < 0.02 else "PARTIAL")
    say(f"LINE (HE+, CC(1), v4pass, frozen tau {TAU0}): {100 * g:+.1f} % vs the better fixed K -> the rung's lossy claim "
        f"{verdict}")
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(HERE, "adaptk_lossy.json"), "w"), indent=1)
    open(os.path.join(HERE, "adaptk_lossy.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
