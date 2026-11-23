"""LANDMARK-ANS screen 1 verdict (Breakthrough's pre-registration + ADDENDUM + 08:05 alignment note; NewIdeas 07:4x-07:5x;
my reading in results/LEVERS_A.md section LANDMARK, written before the run). Input: sb/landmark.py's JSON.
Per set and cell group (all covered cells; by longest k: k>=4, k<=3, k8 .. k2), walk-strict over covered cells pooled over
the band, G = (mode - own) / (true - own), paired window bootstrap (2000, seed 1116). G_copy = the STRONGER of copy (i)
and copy (ii). LINES, per arm (lm1 = w 1.0, lm05 = w 0.5), HE, band 1-8, all covered cells:
  KILL        G < 0.10
  KILL-COPY   G <= G_copy + 0.10   (lookup in disguise)
  PASS        G >= 0.50 on HE AND >= 0.25 on MBPP
  BETWEEN     otherwise;  WARN (report only): G < 0.5 x G_stale1
Screen: PASS needs an arm PASS whose k<=3 cells also clear KILL-COPY (NewIdeas: the idea lives only if G at k <= 3 clears
it); an arm PASS that clears it only at k>=4 reads BETWEEN (k>=4 only). KILL if both arms are KILL or KILL-COPY.
usage: landmark_verdict.py [--in results/landmark_s1.json] [--out results/landmark_verdict.json]"""
import argparse, json
import numpy as np

BANDS = ("1-2", "3-6", "7-8", "1-8", "9-14", "15-30")
ARMS = ("lm1", "lm05")
SHOW = ("lm1", "lm05", "stale1", "copy", "copy2")


def arr(per, sec, g, mode):
    keys = sorted(per)
    r = np.array([per[k][sec][g][mode][0] for k in keys], float); v = np.array([per[k][sec][g][mode][1] for k in keys], float)
    return r, v


def gband(per, sec, g, mode, b, idx):
    lo, hi = (int(x) for x in b.split("-"))
    S = {}
    for m in ("own", "true", mode):
        r, v = arr(per, sec, g, m)
        S[m] = (r[:, lo:hi + 1].sum(1), v[:, lo:hi + 1].sum(1))
    if S["own"][1].sum() == 0:
        return None
    val = {m: s[0].sum() / s[1].sum() for m, s in S.items()}
    gap = val["true"] - val["own"]
    out = {"own": float(val["own"]), "true": float(val["true"]), "mode": float(val[mode]), "cells": int(S["own"][1].sum()),
           "delta_pt": float(100 * (val[mode] - val["own"]))}
    if abs(gap) > 1e-9:
        out["G"] = float((val[mode] - val["own"]) / gap)
        with np.errstate(invalid="ignore", divide="ignore"):
            bo = {m: s[0][idx].sum(1) / s[1][idx].sum(1) for m, s in S.items()}
            gb = (bo[mode] - bo["own"]) / (bo["true"] - bo["own"])
        out["G_ci"] = np.nanpercentile(gb, [2.5, 97.5]).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/landmark_s1.json"); ap.add_argument("--out", default="results/landmark_verdict.json")
    a = ap.parse_args()
    J = json.load(open(a.inp))
    res = {"landmark_source_sha256": J.get("landmark_source_sha256"), "sets": {}}
    for st, X in J["sets"].items():
        per = X["per"]; nW = len(per)
        idx = np.random.default_rng(1116).integers(0, nW, (2000, nW))
        groups = list(next(iter(per.values()))["cov"].keys())
        R = {"groups": {}, "coverage_by_band": X["coverage_by_band"]}
        for g in groups:
            if g == "all_shift1":
                continue
            R["groups"][g] = {m: {b: gband(per, "cov", g, m, b, idx) for b in BANDS} for m in SHOW}
        R["copy1_info"] = {b: gband(per, "cov", "all_shift1", "copy1", b, idx) for b in ("1-2", "3-6", "1-8")}
        R["all_cells"] = {}                                   # info: walk-strict over ALL cells (each mode's own chain)
        for m in ("own", "true", "lm1", "lm05", "stale1", "copy", "copy2"):
            rr = np.array([per[k]["all"][m][0] for k in sorted(per)], float); vv = np.array([per[k]["all"][m][1] for k in sorted(per)], float)
            R["all_cells"][m] = {b: float(rr[:, int(b.split("-")[0]):int(b.split("-")[1]) + 1].sum()
                                          / max(vv[:, int(b.split("-")[0]):int(b.split("-")[1]) + 1].sum(), 1)) for b in BANDS}
        kept = {}
        for m in ("own", "true", "lm1", "lm05", "stale1", "copy", "copy2"):
            ks = np.array([per[k]["kept"][m] for k in sorted(per)], float)
            kept[m] = float(ks[:, 0].sum() / ks[:, 1].sum())
        R["kept_per_32"] = kept
        an = np.array(X["angles"]["lm"]); ak = np.array(X["angles"]["lm_k"]); a1 = np.array(X["angles"]["stale1"])
        R["angles"] = {"lm_within_0.40": float((an <= 0.40).mean()) if len(an) else None,
                       "lm_cos_median": float(np.median(np.cos(an))) if len(an) else None,
                       "stale1_within_0.40": float((a1 <= 0.40).mean()) if len(a1) else None,
                       "stale1_cos_median": float(np.median(np.cos(a1))) if len(a1) else None,
                       "lm_within_0.40_by_k": {int(k): float((an[ak == k] <= 0.40).mean()) for k in (8, 6, 4, 3, 2) if (ak == k).any()},
                       "n": int(len(an))}
        res["sets"][st] = R
        print(f"\n[{st}] {nW} windows; coverage of cells by band (1..8): "
              + " ".join(f"{100 * c:.0f}" for c in X["coverage_by_band"][1:9]) + " %; kept/32 " + ", ".join(f"{m} {v:.3f}" for m, v in kept.items()))
        print(f"  angles: landmark within 0.40 rad {100 * R['angles']['lm_within_0.40']:.1f} % (cos median {R['angles']['lm_cos_median']:.3f});"
              f" stale1 {100 * R['angles']['stale1_within_0.40']:.1f} % (cos {R['angles']['stale1_cos_median']:.3f}); by k "
              + " ".join(f"k{k} {100 * v:.1f}%" for k, v in R["angles"]["lm_within_0.40_by_k"].items()))
        for g in ("all", "k>=4", "k<=3", "k8", "k6", "k4", "k3", "k2"):
            if g not in R["groups"]:
                continue
            G = R["groups"][g]; o = G["lm1"]["1-8"]
            if o is None:
                continue
            print(f"  {g:5s} covered 1-8: own {100 * o['own']:.2f} true {100 * o['true']:.2f} ({o['cells']} cells); " + "; ".join(
                f"{m} {100 * G[m]['1-8']['mode']:.2f} G {G[m]['1-8'].get('G', float('nan')):+.2f}"
                + (f" [{G[m]['1-8']['G_ci'][0]:+.2f}, {G[m]['1-8']['G_ci'][1]:+.2f}]" if "G_ci" in G[m]["1-8"] else "") for m in SHOW))
    V = {}
    for arm in ARMS:
        def g18(st, g, m):
            x = res["sets"][st]["groups"][g][m]["1-8"]
            return x.get("G", float("nan")) if x else float("nan")
        Gh, Gm = g18("he", "all", arm), g18("mbpp", "all", arm) if "mbpp" in res["sets"] else float("nan")
        Gc = max(g18("he", "all", "copy"), g18("he", "all", "copy2"))
        Gs = g18("he", "all", "stale1")
        if Gh < 0.10:
            v = "KILL"
        elif Gh <= Gc + 0.10:
            v = "KILL-COPY"
        elif Gh >= 0.50 and Gm >= 0.25:
            v = "PASS"
        else:
            v = "BETWEEN"
        lo3 = g18("he", "k<=3", arm); c3 = max(g18("he", "k<=3", "copy"), g18("he", "k<=3", "copy2"))
        hi4 = g18("he", "k>=4", arm); c4 = max(g18("he", "k>=4", "copy"), g18("he", "k>=4", "copy2"))
        clears3 = lo3 > c3 + 0.10
        V[arm] = {"verdict": v, "G_he": Gh, "G_mbpp": Gm, "G_copy_he": Gc, "G_stale1_he": Gs, "warn": bool(Gh < 0.5 * Gs),
                  "k<=3": {"G": lo3, "G_copy": c3, "clears_kill_copy": bool(clears3)},
                  "k>=4": {"G": hi4, "G_copy": c4, "clears_kill_copy": bool(hi4 > c4 + 0.10)}}
        print(f"VERDICT {arm}: {v} -- HE covered G(1-8) {Gh:+.3f} (copy, the stronger of (i)/(ii): {Gc:+.3f}; stale1 {Gs:+.3f}),"
              f" MBPP {Gm:+.3f}; k<=3 G {lo3:+.3f} vs copy {c3:+.3f} ({'clears' if clears3 else 'FAILS'} KILL-COPY);"
              f" k>=4 G {hi4:+.3f} vs copy {c4:+.3f}" + ("; WARN: G < 0.5 x G_stale1" if V[arm]["warn"] else ""), flush=True)
    vs = [x["verdict"] for x in V.values()]
    if any(x["verdict"] == "PASS" and x["k<=3"]["clears_kill_copy"] for x in V.values()):
        sv = "PASS"
    elif all(x in ("KILL", "KILL-COPY") for x in vs):
        sv = "KILL" if all(x == "KILL" for x in vs) else "KILL-COPY"
    elif any(x["verdict"] == "PASS" for x in V.values()):
        sv = "BETWEEN (k>=4 only)"
    else:
        sv = "BETWEEN"
    res["arms"] = V; res["screen_verdict"] = sv
    print(f"VERDICT SCREEN: {sv} -- " + ", ".join(f"{k} {x['verdict']}" for k, x in V.items()), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
