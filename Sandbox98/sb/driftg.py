"""Drift-gap report (THINK_1116c, orchestrator 18:2x: close >= half of the own -> true gap at depths 3-30).

For the base chain (own), the true-state oracle (true) and each arm, the walk-strict rate per band: P(right at d |
right at 0..d-1), pooled over the band's depths and windows (the SIB read-out's definition). Each arm gets
  delta = arm - own (pt)   and   G = (arm - own) / (true - own), the share of the own -> true gap it closes,
with a paired bootstrap over windows (windows resampled jointly for own, true and the arm). Inputs are sb/evalx.py
outputs: their "pcond" per-window counts. Older files without them give point values only (no CI).

usage: driftg.py --own 'results/x.json:s30|{set}|chain' --true 'results/x.json:s30|{set}|true'
         --arm 'NAME=results/y.json:RUN|{set}|chain' [--arm ...] [--sets he,mbpp]
         [--bands 1-2,3-6,7-14,15-30,3-14,3-30] [--line 3-30 --pass 0.5 --kill 0.25] [--boot 2000 --seed 1116]
         [--out results/drift_NAME.json]
The verdict line printed here is a default only: each screen's pre-registered line (results/LEVERS_A.md) decides."""
import argparse, json, sys
import numpy as np

_cache = {}


def load(spec, st):
    """'path:tag' (tag may hold {set}) -> (keys, r [W, D], v [W, D], res-summary or None)"""
    path, _, tag = spec.partition(":")
    tag = tag.replace("{set}", st)
    if path not in _cache:
        _cache[path] = json.load(open(path))
    J = _cache[path]
    res = J.get("res", {}).get(tag)
    pc = J.get("pcond", {}).get(tag)
    if pc:
        keys = sorted(pc)
        D = max(len(pc[k][1]) for k in keys)
        r = np.zeros((len(keys), D)); v = np.zeros((len(keys), D))
        for i, k in enumerate(keys):
            r[i, :len(pc[k][0])] = pc[k][0]; v[i, :len(pc[k][1])] = pc[k][1]
        return keys, r, v, res
    if res is None:
        sys.exit(f"{path}: no '{tag}' (have: {', '.join(sorted(J.get('res', {})))})")
    n = np.array(res["cond_n"], float)                      # aggregate only: one pseudo-window, point values
    c = np.array([x if x is not None else 0.0 for x in res["cond"]], float)
    return None, np.round(c * n)[None], n[None], res


def band(lo_hi):
    lo, hi = (int(x) for x in lo_hi.split("-"))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", required=True); ap.add_argument("--true", required=True)
    ap.add_argument("--arm", action="append", default=[], help="NAME=path:tag ({set} is replaced)")
    ap.add_argument("--sets", default="he,mbpp")
    ap.add_argument("--bands", default="1-2,3-6,7-14,15-30,3-14,3-30")
    ap.add_argument("--line", default="3-30"); ap.add_argument("--pass", dest="pas", type=float, default=0.5)
    ap.add_argument("--kill", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=1116)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    bands = a.bands.split(",")
    if a.line not in bands:
        bands.append(a.line)
    arms = [s.partition("=")[::2] for s in a.arm]
    out = {"args": vars(a), "sets": {}}
    gline = {name: [] for name, _ in arms}
    for st in a.sets.split(","):
        src = {"own": load(a.own, st), "true": load(a.true, st)}
        for name, spec in arms:
            src[name] = load(spec, st)
        paired = all(s[0] is not None for s in src.values())
        if paired:
            common = sorted(set.intersection(*(set(s[0]) for s in src.values())))
            diff = {k: len(s[0]) - len(common) for k, s in src.items() if len(s[0]) != len(common)}
            if diff:
                print(f"[{st}] warning: windows differ, using the {len(common)} common ones (dropped {diff})")
            D = min(s[1].shape[1] for s in src.values())
            M = {k: (s[1][[s[0].index(w) for w in common], :D], s[2][[s[0].index(w) for w in common], :D])
                 for k, s in src.items()}
        else:
            print(f"[{st}] note: a file has no per-window counts (old evalx output): point values, no CI")
            D = min(s[1].shape[1] for s in src.values())
            M = {k: (s[1][:, :D], s[2][:, :D]) for k, s in src.items()}
            common = []
        nW = next(iter(M.values()))[0].shape[0]
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, nW, (a.boot, nW)) if paired and a.boot else None
        so = {"windows": len(common) if paired else None, "bands": {}}
        print(f"\n[{st}] walk-strict per band (pooled over the band's depths), {so['windows'] or 'aggregate'} windows")
        print(f"  {'band':>6} {'own':>7} {'true':>7} {'gap':>6} | " + " | ".join(f"{n}: value, delta [CI], G [CI]" for n, _ in arms))
        for b in bands:
            lo, hi = band(b)
            if hi >= D:
                continue
            sums = {k: (r[:, lo:hi + 1].sum(1), v[:, lo:hi + 1].sum(1)) for k, (r, v) in M.items()}
            val = {k: float(rs.sum() / vs.sum()) if vs.sum() else float("nan") for k, (rs, vs) in sums.items()}
            bo = {}
            if idx is not None:
                for k, (rs, vs) in sums.items():
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bo[k] = rs[idx].sum(1) / vs[idx].sum(1)
            row = {"own": val["own"], "true": val["true"], "cells": {k: int(s[1].sum()) for k, s in sums.items()}, "arms": {}}
            txt = []
            for name, _ in arms:
                d = val[name] - val["own"]; den = val["true"] - val["own"]
                g = d / den if abs(den) > 1e-9 else float("nan")
                ra = {"value": val[name], "delta_pt": 100 * d, "G": g}
                if bo:
                    dd = bo[name] - bo["own"]; den_b = bo["true"] - bo["own"]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        gb = np.where(np.abs(den_b) > 1e-9, dd / den_b, np.nan)
                    ra["delta_ci"] = [float(100 * np.nanpercentile(dd, 2.5)), float(100 * np.nanpercentile(dd, 97.5))]
                    ra["G_ci"] = [float(np.nanpercentile(gb, 2.5)), float(np.nanpercentile(gb, 97.5))]
                    ra["G_nan_share"] = float(np.isnan(gb).mean())
                    txt.append(f"{100 * val[name]:6.2f} {100 * d:+6.2f} [{ra['delta_ci'][0]:+.2f}, {ra['delta_ci'][1]:+.2f}]"
                               f" G {g:+.2f} [{ra['G_ci'][0]:+.2f}, {ra['G_ci'][1]:+.2f}]")
                else:
                    txt.append(f"{100 * val[name]:6.2f} {100 * d:+6.2f} G {g:+.2f}")
                row["arms"][name] = ra
                if b == a.line:
                    gline[name].append((st, g))
            so["bands"][b] = row
            print(f"  {b:>6} {100 * val['own']:7.2f} {100 * val['true']:7.2f} {100 * (val['true'] - val['own']):6.2f} | " + " | ".join(txt))
        marg = {}
        for k, s in src.items():
            res = s[3] or {}
            marg[k] = {bb: res.get(bb) for bb in ("0-0", "1-2", "3-6", "7-10", "11-14", "15-22", "23-30") if res.get(bb) is not None}
            marg[k]["kept"] = res.get("kept_per_32")
        so["marginal_and_kept"] = marg
        print("  info (marginal bands, kept per check): " + "; ".join(
            f"{k} " + " ".join(f"{bb}:{100 * x:.1f}" for bb, x in m.items() if bb != "kept" and x is not None)
            + (f" kept {m['kept']:.2f}" if m.get("kept") is not None else "") for k, m in marg.items()))
        out["sets"][st] = so
    print()
    out["verdict"] = {}
    for name, gs in gline.items():
        if not gs:
            continue
        if all(g >= a.pas for _, g in gs):
            v = "PASS"
        elif all(g < a.kill for _, g in gs):
            v = "KILL"
        else:
            v = "BETWEEN"
        out["verdict"][name] = v
        print(f"VERDICT {name} (default line: G({a.line}) >= {a.pas} on every set = PASS, < {a.kill} on every set = KILL):"
              f" {v} -- " + ", ".join(f"{st} G {g:+.2f}" for st, g in gs))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
