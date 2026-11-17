#!/usr/bin/env python3
"""ADAPTIVE K DOWN: verify 8 rows when the drafter is unsure (NewIdeas, Tue 17 Nov; the time is the commit's).

ADAPTIVE K (ADAPTK.md) picks 16 or 32 rows per check. Its policy never goes below 16 rows (at least 15 pops), and it gains
~0 on MBPP. The idea here is the other direction: when the 8 marginal pops (rows 9-16) carry little expected kept, verify
8 rows. That needs a smaller expert union and a shallower draft tree.

Costs: E14 (A16 62.49 ms, R32 1.4372, c 1.025 / 1.04, overhead 4 ms, costs_e14.json). R8 is NOT measured (E14 ran 16 and
32 rows only), so it is bracketed [est]:
  0.71  affine in check bytes through the 16- and 32-row points (dense ~66 MB + union x 2.135 MB per layer);
  0.78  linear in rows through the same two points;
  0.86  only the expert union shrinks (73.2 -> ~48.8 per layer by the 16-128 power law, ~280 GB/s), compute as at 16.
Policies (one parameter each, cross-fit: the parameter comes from the OTHER set); otherwise ADAPTIVE K as frozen (32 rows
iff > 15 pops have path_p >= 0.9, else 16):
  SUM:  8 rows iff the calibrated expected kept of pops 8-15, sum(path_p[7:15]), < theta. Deciding needs all 15 pops
        drafted, so an 8-row check still pays the 16-row draft height.
  PEEK: 8 rows iff the 8th pop's path_p < tau8 (path_p is non-increasing, so this bounds pops 8-15). Deciding needs only
        7 pops + the peek, so an 8-row check pays the 8-row draft height.
Ceiling: a hindsight DP over {8, 16, 32} per window (Dinkelbach on the rate).
KILL: the cross-fit MBPP gain over frozen KERN2 < +2 % at R8 0.78 and L 0.66. Otherwise the engine kill test is one
  8-row tree-check timing (~5 min under a quiet lock), against the break-even R8 printed here.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B adaptk_k8.py
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tree_audit as TA  # noqa: E402
import adaptk_draft as AD  # noqa: E402

NP = 127
ROWS3 = (8, 16, 32)
SETS = {"he": ("heplus24", 13.77), "mbpp": ("mbppbank", 6.92)}
TAU, L0, LS = 0.9, 0.66, (0.40, 0.66, 0.92)
A16, OV, R32, C16, C32 = 62.49, 4.0, 1.4372, 1.025, 1.04
R8S = (0.71, 0.78, 0.86)
THETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
TAU8S = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def T(r8):
    return {8: r8 * C16 * A16 + OV, 16: C16 * A16 + OV, 32: R32 * C32 * A16 + OV}


def dense3(z):
    fi, an, nd, dp, ot, pp = (z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p"))
    key = fi.astype(np.int64) * 10_000_000 + an.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    PP = np.zeros((n, NP), np.float32); DP = np.zeros((n, NP), np.int16); OT = np.zeros((n, NP), bool)
    PP[inv, nd] = pp; DP[inv, nd] = dp; OT[inv, nd] = ot
    acc = np.maximum.accumulate(np.where(OT, DP.astype(np.int64) + 1, 0), axis=1)
    hh = np.maximum.accumulate(DP.astype(np.int64) + 1, axis=1)
    A = {r: np.minimum(acc[:, r - 2], r - 1) for r in ROWS3}
    H = {r: hh[:, r - 2] for r in ROWS3}
    return {int(k): i for i, k in enumerate(uk)}, PP, A, H


def load(name):
    tag, pub = SETS[name]
    W = TA.load_windows(tag, pub)
    z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
    idx, PP, A, H = dense3(z)
    S = AD.steps(W, idx)
    ro2 = np.where((PP >= TAU).sum(1) > 15, 32, 16)
    E8 = PP[:, 7:15].sum(1)
    pos = np.array([i for ids in S for i in ids if i >= 0])
    say(f"{name} ({tag}): windows {len(S)}, anchors {len(idx)}, walk positions with a dump {len(pos)}; calibration "
        f"E[kept8] {np.mean(1 + PP[pos, :7].sum(1)):.2f} vs real {np.mean(A[8][pos] + 1):.2f}, E[kept16] "
        f"{np.mean(1 + PP[pos, :15].sum(1)):.2f} vs real {np.mean(A[16][pos] + 1):.2f}; mean height h8 "
        f"{np.mean(H[8][pos]):.2f} / h16 {np.mean(H[16][pos]):.2f} levels")
    return {"S": S, "A": A, "H": H, "ro2": ro2, "E8": E8, "P7": PP[:, 7].copy()}


def per_window(D, TT, L, rows, hd):
    """rows / hd: per anchor, the rows verified and the draft height paid; -> per window (kept, ms)"""
    A, out = D["A"], []
    for ids in D["S"]:
        kept = tm = 0.0; n = len(ids); p = 1
        while p < n:
            i = ids[p]
            if i >= 0:
                r = int(rows[i]); a = int(A[r][i]); c = TT[r] + L * float(hd[i])
            else:
                a = 0; c = TT[16]
            kept += a + 1; tm += c; p += a + 1
        out.append((kept, tm))
    return np.array(out)


def policies(d):
    """name -> function(param) -> (rows, draft height)"""
    H, ro2 = d["H"], d["ro2"]
    hk2 = np.where(ro2 == 32, H[32], H[16])

    def s(th):
        m = (ro2 == 16) & (d["E8"] < th)
        return np.where(m, 8, ro2), hk2                       # the 16-row draft is paid to decide

    def k(t8):
        m = d["P7"] < t8                                       # t8 <= 0.9 implies ro2 == 16
        return np.where(m, 8, ro2), np.where(m, H[8], hk2)
    return {"SUM": (s, THETAS), "PEEK": (k, TAU8S)}


def tps(X):
    return 1000.0 * X[:, 0].sum() / X[:, 1].sum()


def oracle(D, cost, iters=8):
    """hindsight DP over {8, 16, 32} per position, Dinkelbach on the rate"""
    A, lam = D["A"], 0.1
    for _ in range(iters):
        K = M = 0.0
        for ids in D["S"]:
            n = len(ids)
            V = np.zeros(n + 40); best = np.zeros(n + 40, np.int64)
            for p in range(n - 1, 0, -1):
                i = ids[p]
                if i < 0:
                    V[p] = 1 - lam * cost(16, -1) + V[p + 1]; best[p] = 16
                    continue
                vs = [(int(A[r][i]) + 1 - lam * cost(r, i) + V[min(p + int(A[r][i]) + 1, n)], r) for r in ROWS3]
                V[p], best[p] = max(vs)
            p = 1
            while p < n:
                i = ids[p]; r = int(best[p]); a = int(A[r][i]) if i >= 0 else 0
                K += a + 1; M += cost(r, i); p += a + 1
        lam = K / M
    return 1000.0 * lam


def main():
    D = {k: load(k) for k in SETS}
    res = {}
    for L in LS:
        for r8 in R8S:
            TT = T(r8)
            ev = {}
            for k in D:
                d = D[k]
                base = per_window(d, TT, L, np.full(len(d["ro2"]), 16), d["H"][16])
                k2 = per_window(d, TT, L, d["ro2"], np.where(d["ro2"] == 32, d["H"][32], d["H"][16]))
                pol = {name: {prm: per_window(d, TT, L, *f(prm)) for prm in grid} for name, (f, grid) in policies(d).items()}
                ev[k] = (base, k2, pol)
            for k in D:
                other = "mbpp" if k == "he" else "he"
                base, k2, pol = ev[k]
                d = D[k]
                pos = np.array([i for ids in d["S"] for i in ids if i >= 0])
                orc = float("nan")
                if L == L0:
                    H = d["H"]
                    orc = oracle(d, lambda r, i, TT=TT, H=H: TT[r] + (L0 * float(H[r][i]) if i >= 0 else 0.0))
                msg = [f"{k:4s} L {L:.2f} R8 {r8:.2f}: fixed16 {tps(base):.2f} KERN2 {tps(k2):.2f} ({100 * (tps(k2) / tps(base) - 1):+.1f} %)"]
                for name, (f, grid) in policies(d).items():
                    px = max(grid, key=lambda t: tps(ev[other][2][name][t]))
                    pin = max(grid, key=lambda t: tps(pol[name][t]))
                    X = pol[name][px]
                    gx = tps(X) / tps(k2) - 1; gin = tps(pol[name][pin]) / tps(k2) - 1
                    rng = np.random.default_rng(17); bs = []
                    for _ in range(2000):
                        j = rng.integers(0, len(X), len(X))
                        bs.append((X[j, 0].sum() / X[j, 1].sum()) / (k2[j, 0].sum() / k2[j, 1].sum()) - 1)
                    rows = f(px)[0]
                    sh = float(np.mean(rows[pos] == 8))
                    res[f"{k} L{L} R8 {r8} {name}"] = {"param_x": px, "gain_x": gx, "ci95": [float(np.percentile(bs, 2.5)),
                        float(np.percentile(bs, 97.5))], "param_in": pin, "gain_in": gin, "share8": sh,
                        "kern2": tps(k2), "fixed16": tps(base), "oracle": orc}
                    msg.append(f"{name} ({px}) {100 * gx:+.1f} % [{100 * np.percentile(bs, 2.5):+.1f}, "
                               f"{100 * np.percentile(bs, 97.5):+.1f}] at {100 * sh:.0f} % 8-row (in-sample {100 * gin:+.1f} %)")
                if orc == orc:
                    msg.append(f"hindsight {{8,16,32}} {100 * (orc / tps(k2) - 1):+.1f} %")
                say("; ".join(msg))
    say("")
    for name in ("SUM", "PEEK"):
        for k in D:
            other = "mbpp" if k == "he" else "he"
            be = None
            for r8 in np.arange(0.60, 1.001, 0.01):
                TT = T(float(r8))
                f_o, grid = policies(D[other])[name]
                px = max(grid, key=lambda t: tps(per_window(D[other], TT, L0, *f_o(t))))
                f_k, _ = policies(D[k])[name]
                d = D[k]
                g = tps(per_window(d, TT, L0, *f_k(px))) / tps(per_window(d, TT, L0, d["ro2"],
                        np.where(d["ro2"] == 32, d["H"][32], d["H"][16]))) - 1
                if g >= 0.01:
                    be = float(r8)
            res[f"breakeven {name} {k}"] = be
            say(f"break-even R8, {name} ({k}, L {L0}, cross-fit): the largest R8 with >= +1 % over KERN2 = "
                f"{'none in 0.60-1.00' if be is None else f'{be:.2f}'}")
    g = max(res[f"mbpp L{L0} R8 0.78 SUM"]["gain_x"], res[f"mbpp L{L0} R8 0.78 PEEK"]["gain_x"])
    verdict = "KILL" if g < 0.02 else "ALIVE (engine timing next)"
    say(f"VERDICT (MBPP cross-fit at R8 0.78, L {L0}, the better rule): {100 * g:+.1f} % vs the +2 % line -> {verdict}")
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(HERE, "adaptk_k8.json"), "w"), indent=1)
    open(os.path.join(HERE, "adaptk_k8.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
