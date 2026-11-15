"""CERTIFY98's scorer (docs/PREREG_certify98_2026-11-15.md; ~/FableGod/deepdraft/CERTIFY98.md). CPU only, numpy only.
Reads a cells file (.npz) as BigMoE's dd_ceiling.py writes them (cells_t30b_mbpp100.npz, cells_t30b_heplus.npz):
  depth [n] (0-30), right [n] (1 = the drafter's top-1 equals the target's runtime top-1, true prefix tokens fed), and
  optionally m [n] (the target's own top-1 minus top-2 log-prob, nats) and fi [n] (the window index).
  Cells are depth-major per window: depth 0's block lists the window's anchors 0, 1, ... in order, then depth 1's, ...
  (dd_ttt.py's eval layout: rt[:len(right_j), j] = right_j). A window starts where depth returns to 0.
  .venv/bin/python tools/certify98.py score <cells.npz> [--label HE+] [--line 97.0 | --report]
  .venv/bin/python tools/certify98.py reproduce     # validity 1: t30b's stored cells against the published read-outs
  .venv/bin/python tools/certify98.py selftest      # the walk, the CR1 bound and the layout on hand-made cells
Per band (dd_ttt.py's BANDS) and per depth: the ratio estimate (right cells / cells over windows), the window-clustered
CR1 SE and the one-sided 95 % lower bound (estimate - t(K-1, 0.95) x SE; K = windows with cells in that band); the
margin bins (sure m >= 4 nats / unsure); kept/32 and kept/16 by dd_ttt.kept_walk (the decode-path replay from the first
answer row, row 1 of the cells) with window-clustered 95 % intervals. Last line: `CERTIFY98 <label>: CERTIFIED [98 met] | NOT CERTIFIED (...)`
with --line, or `CERTIFY98 <label>: REPORT ...` with --report; exit 0 CERTIFIED / REPORT, 1 NOT CERTIFIED, 2 bad input."""
import json, os, sys
import numpy as np

BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]    # dd_ttt.py BANDS
DEPTHS, SURE, GUARD, MET = 31, 4.0, 96.5, 98.0
GATE = [(0, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]   # amendment 4: the walk-strict gate bands (0-2 merged)
GUARD_ROWS = 200                                                  # the per-depth guard applies where a depth has >= 200 rows
R0 = 1   # the cells' first anchor is the prompt's last row (a0 = gen - 2); the prefill makes the first answer token, so the
         # decode-path walk starts at anchor gen - 1, i.e. row 1 (dd_taps.py's eval: kept_walk(rt, gen - 1 - a0))
H = os.path.expanduser("~")
REPRO = [  # (cells, published read-out json, run key): validity 1 of the PREREG
    (f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz", f"{H}/BigMoE/data/deepdraft/tapsq/t1_chain.json",
     "t30b+wd+k1000"),
    (f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus.npz",
     f"{H}/BigMoE/data/deepdraft/tapsq/mh_t30b_heplus_bankctx.json", "t30b+wd+k1000+bankctx")]


def t_one(df, z=1.6448536):
    """t(df) quantile by Cornish-Fisher (error < 1e-3 for df >= 10; no scipy in this venv); z 1.6449 = one-sided 95 %"""
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def kept_walk(rt, r0, cap=31):
    """dd_ttt.kept_walk (cap 31: a 32-row check) and dd_taps.kept_walk_n (cap 15: a 16-row check), verbatim logic"""
    sub = rt[r0:]; out = []; p = 0
    while p < sub.shape[0]:
        k = 0
        while k < cap and sub[p, k] == 1:
            k += 1
        out.append(k + 1); p += k + 1
    return out


def windows(path):
    """[(rt [n0, 31] int8 1/0/-1, mt [n0, 31] float margins or nan)] per window, checked against the layout"""
    d = np.load(path)
    dep, right = d["depth"].astype(int), d["right"].astype(int)
    m = d["m"].astype(float) if "m" in d.files else np.full(len(dep), np.nan)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    if "fi" in d.files:                                       # the window index must agree with the depth resets
        fi = d["fi"]
        assert all(len(set(fi[s:t].tolist())) == 1 for s, t in zip(st, en)), "fi changes inside a window"
        assert len(set(int(fi[s]) for s in st)) == len(st), "a window index repeats"
    out = []
    for s, t in zip(st, en):
        dw, rw, mw = dep[s:t], right[s:t], m[s:t]
        assert (np.diff(dw) >= 0).all(), "depths not depth-major inside a window"
        n0 = int((dw == 0).sum())
        # one more row than depth 0's cells: the answer's last row has no stored target, and dd_taps.py's eval sizes rt
        # as len(ids) - 1 - a0 = n0 + 1 (all -1 there; the walk counts a check of 1 if it lands on it)
        rt = np.full((n0 + 1, DEPTHS), -1, np.int8); mt = np.full((n0 + 1, DEPTHS), np.nan)
        an0 = d["an"][s:t][dw == 0] if "an" in d.files else None
        for j in range(DEPTHS):
            k = dw == j
            c = int(k.sum())
            assert c <= n0, "a depth block is longer than depth 0's"
            if an0 is not None:                                # the walk needs depth j's cells on depth 0's first c anchors
                assert (d["an"][s:t][k] == an0[:c]).all(), f"depth {j}'s anchors are not depth 0's first {c}"
            rt[:c, j] = rw[k]; mt[:c, j] = mw[k]
        out.append((rt, mt))
    return out


def cr1(y, x):
    """ratio estimate sum(y) / sum(x) over windows, and its CR1 SE (clustered by window)"""
    y, x = np.asarray(y, float), np.asarray(x, float)
    K = len(x); p = y.sum() / x.sum()
    se = np.sqrt(K / (K - 1) * ((y - p * x) ** 2).sum()) / x.sum() if K > 1 else float("nan")
    return p, se


def bandstat(W, lo, hi, sel=None):
    """per window: right and cell counts in depths lo..hi (sel: an optional mask on margins)"""
    y, x = [], []
    for rt, mt in W:
        c = rt[:, lo:hi + 1]; ok = c >= 0
        if sel is not None:
            ok = ok & sel(mt[:, lo:hi + 1])
        if ok.sum():
            y.append(int((c[ok] == 1).sum())); x.append(int(ok.sum()))
    return np.array(y), np.array(x)


def strict_counts(W, mode="all", cap=31):
    """per window (num[k], den[k]) of the STRICT conditional per-depth accuracy P(right at k | right at 0..k-1), what the
    engine pays on (the orchestrator 02:2x; PREREG_certify98 amendment 4). mode "all": every anchor from row R0 starts a
    check; a missing cell (the answer's end) leaves the denominator, since truncation is not a miss. mode "walk": only
    the decode walk's check starts, a missing cell counting as the run's end, so 1 + sum_k prod_{j<=k} c_j = kept_walk."""
    out = []
    for rt, _ in W:
        num, den = np.zeros(DEPTHS), np.zeros(DEPTHS)
        if mode == "walk":
            starts, p = [], R0
            while p < rt.shape[0]:
                k = 0
                while k < cap and rt[p, k] == 1:
                    k += 1
                starts.append(p); p += k + 1
        else:
            starts = range(R0, rt.shape[0])
        for a in starts:
            for k in range(cap):
                c = rt[a, k]
                if c == -1:
                    den[k] += mode == "walk"
                    break
                den[k] += 1
                if c != 1:
                    break
                num[k] += 1
        out.append((num, den))
    return out


def strict_stats(C, lo, hi):
    """pooled ratio, CR1 SE and one-sided 95 % lower bound of the strict accuracy over depths lo..hi"""
    y = np.array([n[lo:hi + 1].sum() for n, d in C if d[lo:hi + 1].sum() > 0])
    x = np.array([d[lo:hi + 1].sum() for n, d in C if d[lo:hi + 1].sum() > 0])
    p, se = cr1(y, x); K = len(x)
    return {"acc": 100 * p, "se": 100 * se, "lcb": 100 * (p - t_one(K - 1) * se), "K": K, "rows": int(x.sum())}


def kept_from(C, cap):
    """1 + sum_{k<cap} prod_{j<=k} c_j from pooled strict counts (equals kept_walk for the walk's counts)"""
    num, den = sum(n for n, _ in C), sum(d for _, d in C)
    c = [num[k] / den[k] if den[k] else 0.0 for k in range(cap)]
    return 1 + float(np.sum(np.cumprod(c)))


def summary(W):
    """bands, depths, margin bins and kept: the numbers every report quotes"""
    R = {"windows": len(W), "bands": {}, "depths": [], "bins": {}}
    for lo, hi in BANDS:
        y, x = bandstat(W, lo, hi)
        p, se = cr1(y, x); K = len(x)
        R["bands"][f"{lo}-{hi}"] = {"acc": 100 * p, "se": 100 * se, "lcb": 100 * (p - t_one(K - 1) * se), "K": K,
                                    "cells": int(x.sum())}
        bins = {}
        for lab, f in (("sure", lambda mm: mm >= SURE), ("unsure", lambda mm: mm < SURE)):
            yb, xb = bandstat(W, lo, hi, f)
            bins[lab] = {"share": 100 * xb.sum() / max(x.sum(), 1), "acc": 100 * yb.sum() / max(xb.sum(), 1)}
        R["bins"][f"{lo}-{hi}"] = bins
    for j in range(DEPTHS):
        y, x = bandstat(W, j, j)
        p, se = cr1(y, x)
        R["depths"].append({"depth": j, "acc": 100 * p, "lcb": 100 * (p - t_one(len(x) - 1) * se), "K": len(x)})
    for cap, lab in ((31, "kept_per_32"), (15, "kept_per_16")):
        y, x = [], []
        for rt, _ in W:
            k = kept_walk(rt, R0, cap)
            y.append(sum(k)); x.append(len(k))
        p, se = cr1(y, x); K = len(x)
        half = t_one(K - 1, 1.959964) * se
        R[lab] = {"mean": p, "lo": p - half, "hi": p + half, "lcb": p - t_one(K - 1) * se}
    C = strict_counts(W, "all")
    R["strict"] = {"bands": {f"{lo}-{hi}": strict_stats(C, lo, hi) for lo, hi in BANDS},
                   "depths": [dict(depth=j, **strict_stats(C, j, j)) for j in range(DEPTHS)]}
    R["strict"]["kept_approx"] = kept_from(C, 31)
    CW, CW16 = strict_counts(W, "walk", 31), strict_counts(W, "walk", 15)
    R["strict_walk"] = {"depths": [dict(depth=j, **strict_stats(CW, j, j)) for j in range(DEPTHS)],
                        "bands": {f"{lo}-{hi}": strict_stats(CW, lo, hi) for lo, hi in GATE},
                        "kept_per_32": kept_from(CW, 31), "kept_per_16": kept_from(CW16, 15)}
    return R


def report(R, label):
    print(f"{label}: {R['windows']} windows")
    print("  PRIMARY: strict P(right at k | right at 0..k-1) at the decode walk's check starts (what the engine pays on)")
    print(f"  {'band':>6} {'rows':>8} {'acc %':>7} {'SE pt':>6} {'LCB95 %':>8} {'K':>4}")
    for b, v in R["strict_walk"]["bands"].items():
        print(f"  {b:>6} {v['rows']:>8} {v['acc']:7.2f} {v['se']:6.3f} {v['lcb']:8.2f} {v['K']:>4}")
    print("  walk-strict per depth (acc %): " + " ".join(f"{r['acc']:.2f}" for r in R["strict_walk"]["depths"]))
    print("  walk-strict rows per depth: " + " ".join(str(r["rows"]) for r in R["strict_walk"]["depths"]))
    print("  INFORMATION: strict at every anchor (acc %, overstates the engine: misses cluster at check starts): "
          + " ".join(f"{b} {v['acc']:.2f}" for b, v in R["strict"]["bands"].items()))
    print(f"  kept/32 from the walk's strict accuracies {R['strict_walk']['kept_per_32']:.6f} (kept_walk {R['kept_per_32']['mean']:.6f}); "
          f"from every anchor's {R['strict']['kept_approx']:.3f}")
    print("  INFORMATION: dd_ttt's bands (every row at every depth, rows past an earlier miss included)")
    print(f"  {'band':>6} {'cells':>8} {'acc %':>7} {'SE pt':>6} {'LCB95 %':>8} {'K':>4} | sure share / acc | unsure share / acc")
    for b, v in R["bands"].items():
        s, u = R["bins"][b]["sure"], R["bins"][b]["unsure"]
        print(f"  {b:>6} {v['cells']:>8} {v['acc']:7.2f} {v['se']:6.3f} {v['lcb']:8.2f} {v['K']:>4} | {s['share']:5.1f} % / "
              f"{s['acc']:6.2f} | {u['share']:5.1f} % / {u['acc']:6.2f}")
    lo = min(R["depths"], key=lambda r: r["acc"])
    print("  per depth (acc %): " + " ".join(f"{r['acc']:.2f}" for r in R["depths"]))
    print(f"  lowest depth: {lo['depth']} at {lo['acc']:.2f} % (guard >= {GUARD})")
    for lab in ("kept_per_32", "kept_per_16"):
        k = R[lab]
        print(f"  {lab.replace('_per_', '/')}: {k['mean']:.3f} (95 % CI {k['lo']:.3f}-{k['hi']:.3f})")
    print(f"  kept/32 over kept/16: {R['kept_per_32']['mean'] / R['kept_per_16']['mean']:.3f} (a 32-row check pays at "
          f">= ~1.35-1.4)")


def kept_line(line):
    """kept/32 of a drafter flat at `line` % per guess: 1 + sum_{k=1..31} (line/100)^k (98 -> 23.80, 97 -> 20.76)"""
    q = line / 100
    return 1 + sum(q ** k for k in range(1, 32))


def verdict(R, label, line):
    S = R["strict_walk"]
    low = [b for b, v in S["bands"].items() if not v["lcb"] >= line]
    dip = [r["depth"] for r in S["depths"] if r["rows"] >= GUARD_ROWS and not r["acc"] >= GUARD]
    kl = kept_line(line); kept = R["kept_per_32"]
    short = not kept["lcb"] >= kl
    met = all(v["acc"] >= MET for v in S["bands"].values())
    if low or dip or short:
        why = ([f"strict bands {', '.join(low)} below {line}"] if low else []) + \
              ([f"strict depths {dip} below {GUARD}"] if dip else []) + \
              ([f"kept/32 lower bound {kept['lcb']:.2f} below {kl:.2f}"] if short else [])
        print(f"CERTIFY98 {label}: NOT CERTIFIED ({'; '.join(why)})")
        return 1
    print(f"CERTIFY98 {label}: CERTIFIED{' [98 met]' if met else ''} (every walk-strict band's 95 % lower bound >= {line}; "
          f"every walk-strict depth with >= {GUARD_ROWS} rows >= {GUARD}; kept/32 lower bound {kept['lcb']:.2f} >= {kl:.2f})")
    return 0


def reproduce():
    worst, ok = 0.0, True
    for cells, pub, key in REPRO:
        R = summary(windows(cells))
        P = json.load(open(pub))[key]
        for b, v in R["bands"].items():
            dlt = abs(v["acc"] - 100 * P[b]); worst = max(worst, dlt); ok &= dlt <= 0.01
        dk = [abs(R[k]["mean"] - P[k]) for k in ("kept_per_32", "kept_per_16")]
        ok &= max(dk) <= 1e-6
        di = [abs(R["strict_walk"][k] - R[k]["mean"]) for k in ("kept_per_32", "kept_per_16")]   # the strict identity
        ok &= max(di) <= 1e-9
        print(f"  strict (walk) -> kept/32 {R['strict_walk']['kept_per_32']:.9f}, kept/16 {R['strict_walk']['kept_per_16']:.9f}: "
              f"identity off by {max(di):.1e}; strict bands (every anchor) "
              + " ".join(f"{b} {v['acc']:.2f}" for b, v in R["strict"]["bands"].items()))
        print(f"{os.path.basename(cells)} vs {os.path.basename(pub)} [{key}]: bands "
              + " ".join(f"{b} {v['acc']:.2f}/{100 * P[b]:.2f}" for b, v in R["bands"].items())
              + f"; kept/32 {R['kept_per_32']['mean']:.6f}/{P['kept_per_32']:.6f}; kept/16 "
              f"{R['kept_per_16']['mean']:.6f}/{P['kept_per_16']:.6f}")
    print(f"CERTIFY98 reproduce: {'PASS' if ok else 'FAIL'} (largest band difference {worst:.4f} pt; line 0.01 pt; "
          f"kept to 1e-6; the strict walk reproduces kept_walk to 1e-9)")
    return 0 if ok else 1


def selftest():
    rt = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1], [1, -1, -1]], np.int8)
    assert kept_walk(rt, 0, 2) == [3, 2], kept_walk(rt, 0, 2)         # row 0: 2 rights (cap 2) -> 3; row 3: 1 right -> 2
    assert kept_walk(rt, 1, 31) == [2, 2], kept_walk(rt, 1, 31)       # row 1: 1 right -> 2; row 3: 1 right -> 2
    p, se = cr1([9, 18], [10, 20])                                      # equal rates: SE 0
    assert abs(p - 0.9) < 1e-12 and se < 1e-12
    p, se = cr1([10, 0], [10, 10])                                      # 1.0 and 0.0: p 0.5, SE = sqrt(2 * (25 + 25)) / 20
    assert abs(p - 0.5) < 1e-12 and abs(se - np.sqrt(2 * 50) / 20) < 1e-12
    import tempfile
    dep = np.array([0, 0, 0, 1, 1, 2, 0, 0, 1]); right = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1])
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "c.npz"); np.savez(f, depth=dep, right=right, m=np.full(len(dep), 5.0))
        W = windows(f)
    assert len(W) == 2 and W[0][0][:, :3].tolist() == [[1, 1, 1], [1, 0, -1], [0, -1, -1], [-1, -1, -1]], W[0][0][:, :3].tolist()
    assert W[1][0][:, :2].tolist() == [[1, 1], [1, -1], [-1, -1]]
    C = strict_counts([(rt, None)], "all", 3)
    assert C[0][1][:3].tolist() == [3, 1, 0] and C[0][0][:3].tolist() == [2, 0, 0], C      # from row 1: [1,0,-1], [0,..], [1,-1,..]
    CW = strict_counts([(rt, None)], "walk", 2)                                          # walk from row 1: starts 1, 3
    assert CW[0][1][:2].tolist() == [2, 2] and CW[0][0][:2].tolist() == [2, 0], CW       # row 3's end counts in the walk
    assert abs(kept_from(CW, 2) - np.mean(kept_walk(rt, 1, 2))) < 1e-12
    print("CERTIFY98 selftest: PASS (walk, CR1, layout, strict counts, the strict -> kept identity)")
    return 0


def main(argv):
    if not argv or argv[0] not in ("score", "reproduce", "selftest"):
        print(__doc__); return 2
    if argv[0] == "reproduce":
        return reproduce()
    if argv[0] == "selftest":
        return selftest()
    if len(argv) < 2 or not os.path.isfile(argv[1]):
        print("score <cells.npz> [--label L] [--line X | --report]"); return 2
    label = argv[argv.index("--label") + 1] if "--label" in argv else os.path.basename(argv[1])
    R = summary(windows(argv[1]))
    report(R, label)
    if "--line" in argv:
        return verdict(R, label, float(argv[argv.index("--line") + 1]))
    print(f"CERTIFY98 {label}: REPORT (no line; band 0 {R['bands']['0-0']['acc']:.2f}, 23-30 "
          f"{R['bands']['23-30']['acc']:.2f}, kept/32 {R['kept_per_32']['mean']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
