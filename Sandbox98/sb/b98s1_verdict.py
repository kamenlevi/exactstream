"""BREAK98 Screen 1 verdict (Fable FAR's FUTURE-STATE ROWS on the 2B; results/LEVERS_A.md section B98S1, pre-registered
before data). Inputs (LAB, teacher-forced 2B):
  results/b98s1_ctl_eval.json   evalx: wmix_ctl chain + true (own / true for G), MBPP-50 / HE-30, per-window counts
  results/b98s1_eval_V.json     evalx: b98s1V_fV chain (the arm with future states at depths 1..8)
  results/b98s1_tree_V.json     tree8.py (MTP Head's, read-only): the arm's tree-32 / tree-16 / chain-32 kept per window
  ctl's tree: results/b98s1_tree_ctl.json if present, else WALK-MIX's results/walkmix_tree.json (wmix_ctl: the same
              weights; used only when this run's ctl chain read-out equals WALK-MIX's exactly, --detcheck)
  results/b98s1_zeroshot_V.json the 2B's own top-1 at placeholder offsets 1..8 (sb/futzs.py)
LINES, per variant (FAR's):
  PASS  HE G(1-8) >= 0.30 with its 95 % CI's lower end > 0 AND HE tree-32 kept >= x1.05 of ctl
  KILL  HE G(1-8) < 0.10
  BETWEEN otherwise.   Screen: PASS if any variant passes; KILL only if all three are KILL.
G = (arm - own) / (true - own), walk-strict pooled over the band's depths (driftg.py / SIB's definition); bands are
evalx's labels (band d = chain depth d + 1; the future states feed depths 1..8 = bands 0..7). Paired window bootstrap
(2000, seed 1116) for G; tree ratios as tree8.py's (ratio of sums, 5000 window resamples, seed 1115 per comparison).
usage: b98s1_verdict.py [--variants A,B,C] [--out results/b98s1_verdict.json] | --detcheck"""
import argparse, json, os, sys
import numpy as np

SETS = ("mbpp", "he")
BANDS = ("0-0", "1-2", "3-6", "7-8", "1-8", "9-14", "15-30")
CTL = "wmix_ctl"


def pc(J, tag):
    p = J["pcond"][tag]
    keys = sorted(p)
    D = max(len(p[k][1]) for k in keys)
    r = np.zeros((len(keys), D)); v = np.zeros((len(keys), D))
    for i, k in enumerate(keys):
        r[i, :len(p[k][0])] = p[k][0]; v[i, :len(p[k][1])] = p[k][1]
    return keys, r, v


def detcheck():
    """this run's ctl chain read-out == WALK-MIX's (same weights, same code): then WALK-MIX's ctl tree is reused"""
    A = json.load(open("results/b98s1_ctl_eval.json")); B = json.load(open("results/walkmix_eval.json"))
    same = all(A["pcond"][f"{CTL}|{st}|chain"] == B["pcond"][f"{CTL}|{st}|chain"]
               and A["res"][f"{CTL}|{st}|chain"]["kept_per_32"] == B["res"][f"{CTL}|{st}|chain"]["kept_per_32"] for st in SETS)
    print(f"DETCHECK {'SAME' if same else 'DIFFERS'}: this run's {CTL} chain read-out vs WALK-MIX's (per-window walk-strict"
          f" counts + kept/32, MBPP-50 / HE-30)")
    return same


def tree_ratio(A, B, seed=1115, boot=5000):
    A, B = np.array(A, float), np.array(B, float)
    assert A.shape == B.shape, "tree windows differ"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), (boot, len(A)))
    bs = (B[idx, 0].sum(1) / B[idx, 1].sum(1)) / (A[idx, 0].sum(1) / A[idx, 1].sum(1))
    return float((B[:, 0].sum() / B[:, 1].sum()) / (A[:, 0].sum() / A[:, 1].sum())), np.percentile(bs, [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C"); ap.add_argument("--out", default="results/b98s1_verdict.json")
    ap.add_argument("--detcheck", action="store_true")
    a = ap.parse_args()
    if a.detcheck:
        sys.exit(0 if detcheck() else 1)
    E0 = json.load(open("results/b98s1_ctl_eval.json"))
    if os.path.exists("results/b98s1_tree_ctl.json"):
        TC, tsrc = json.load(open("results/b98s1_tree_ctl.json")), "results/b98s1_tree_ctl.json"
    else:
        assert detcheck(), "ctl's read-out differs from WALK-MIX's and there is no results/b98s1_tree_ctl.json"
        TC, tsrc = json.load(open("results/walkmix_tree.json")), "results/walkmix_tree.json (WALK-MIX's ctl, same weights)"
    out = {"ctl": CTL, "ctl_tree_source": tsrc, "variants": {}, "ctl_bands": {}}
    rng0 = 1116
    for st in SETS:
        ko, ro, vo = pc(E0, f"{CTL}|{st}|chain"); kt, rt, vt = pc(E0, f"{CTL}|{st}|true")
        assert ko == kt
        out["ctl_bands"][st] = {}
        for b in BANDS:
            lo, hi = (int(x) for x in b.split("-"))
            o = ro[:, lo:hi + 1].sum() / vo[:, lo:hi + 1].sum(); t = rt[:, lo:hi + 1].sum() / vt[:, lo:hi + 1].sum()
            out["ctl_bands"][st][b] = {"own": float(o), "true": float(t)}
        res = E0["res"][f"{CTL}|{st}|chain"]
        out["ctl_bands"][st]["marginal"] = {k: res.get(k) for k in ("0-0", "1-2", "3-6", "7-10")}
    print(f"ctl = {CTL} (WALK-MIX's control: s30o + 400 steps, the same setup); tree source: {tsrc}")
    for st in SETS:
        cb = out["ctl_bands"][st]
        print(f"  {st} own -> true walk-strict (the gap the future states can close): " + "; ".join(
            f"{b} {100 * cb[b]['own']:.2f} -> {100 * cb[b]['true']:.2f}" for b in BANDS))
    verdicts = {}
    for V in a.variants.split(","):
        run = f"b98s1{V}_f{V}"
        p_e = f"results/b98s1_eval_{V}.json"; p_t = f"results/b98s1_tree_{V}.json"
        if not (os.path.exists(p_e) and os.path.exists(p_t)):
            print(f"  variant {V}: read-out missing ({p_e}, {p_t}): skipped"); continue
        E, T = json.load(open(p_e)), json.load(open(p_t))
        r = {"sets": {}}
        for st in SETS:
            ko, ro, vo = pc(E0, f"{CTL}|{st}|chain"); _, rt, vt = pc(E0, f"{CTL}|{st}|true")
            ka, ra, va = pc(E, f"{run}|{st}|chain")
            assert ko == ka, "the arm and ctl were read on different windows"
            D = min(ro.shape[1], ra.shape[1], rt.shape[1])
            rng = np.random.default_rng(rng0)
            idx = rng.integers(0, len(ko), (2000, len(ko)))
            bands = {}
            for b in BANDS:
                lo, hi = (int(x) for x in b.split("-"))
                if hi >= D:
                    continue
                S = {k: (x[:, lo:hi + 1].sum(1), y[:, lo:hi + 1].sum(1)) for k, (x, y) in
                     {"own": (ro, vo), "true": (rt, vt), "arm": (ra, va)}.items()}
                val = {k: s[0].sum() / s[1].sum() for k, s in S.items()}
                bo = {k: s[0][idx].sum(1) / s[1][idx].sum(1) for k, s in S.items()}
                dlt = val["arm"] - val["own"]; gap = val["true"] - val["own"]
                x = {"own": float(val["own"]), "true": float(val["true"]), "arm": float(val["arm"]), "delta_pt": float(100 * dlt),
                     "delta_ci": (100 * np.percentile(bo["arm"] - bo["own"], [2.5, 97.5])).tolist()}
                if abs(gap) > 1e-9:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        gb = (bo["arm"] - bo["own"]) / (bo["true"] - bo["own"])
                    x["G"] = float(dlt / gap); x["G_ci"] = np.nanpercentile(gb, [2.5, 97.5]).tolist()
                bands[b] = x
            tr = {}
            for key in ("K32|g1.0", "K16|g1.0", "chain"):
                tr[key] = tree_ratio(TC["res"][f"{CTL}|{st}"][key]["per_window"], T["res"][f"{run}|{st}"][key]["per_window"])
            band = [T["res"][f"{run}|{st}"][f"K32|g{g}"]["kept"] / TC["res"][f"{CTL}|{st}"][f"K32|g{g}"]["kept"] for g in ("0.5", "2.0")]
            r["sets"][st] = {"bands": bands, "tree32": {"ctl": TC["res"][f"{CTL}|{st}"]["K32|g1.0"]["kept"],
                                                         "arm": T["res"][f"{run}|{st}"]["K32|g1.0"]["kept"],
                                                         "ratio": tr["K32|g1.0"][0], "ci95": tr["K32|g1.0"][1]},
                             "tree16_ratio": tr["K16|g1.0"][0], "chain32_ratio": tr["chain"][0], "tree32_ratio_g0.5_g2": band}
        zf = f"results/b98s1_zeroshot_{V}.json"
        r["zeroshot"] = json.load(open(zf))["sets"] if os.path.exists(zf) else None
        ck = f"ckpt/{run}.safetensors"
        if os.path.exists(ck):
            import mlx.core as mx
            w = mx.load(ck); d = w["fc.weight"].shape[0]
            r["P_norm_ratio"] = float(mx.linalg.norm(w["fut_fc.weight"]) / mx.linalg.norm(w["fc.weight"][:, d:]))
        H = r["sets"]["he"]
        g18 = H["bands"]["1-8"]["G"]; glo = H["bands"]["1-8"]["G_ci"][0]; t32 = H["tree32"]["ratio"]
        if g18 >= 0.30 and glo > 0 and t32 >= 1.05:
            v = "PASS"
        elif g18 < 0.10:
            v = "KILL"
        else:
            v = "BETWEEN"
        r["verdict"] = v; verdicts[V] = v
        out["variants"][V] = r
        for st in SETS:
            x = r["sets"][st]; bb = x["bands"]
            print(f"  {V} {st}: " + "; ".join(
                f"{b} {100 * bb[b]['own']:.2f}->{100 * bb[b]['arm']:.2f} ({bb[b]['delta_pt']:+.2f} pt"
                + (f", G {bb[b]['G']:+.2f} [{bb[b]['G_ci'][0]:+.2f}, {bb[b]['G_ci'][1]:+.2f}]" if "G" in bb[b] else "") + ")"
                for b in BANDS if b in bb)
                  + f"; tree-32 {x['tree32']['ctl']:.3f} -> {x['tree32']['arm']:.3f} = x{x['tree32']['ratio']:.4f}"
                    f" [{x['tree32']['ci95'][0]:.4f}, {x['tree32']['ci95'][1]:.4f}] (gamma 0.5 / 2 x{x['tree32_ratio_g0.5_g2'][0]:.4f}"
                    f" / x{x['tree32_ratio_g0.5_g2'][1]:.4f}), tree-16 x{x['tree16_ratio']:.4f}, chain-32 x{x['chain32_ratio']:.4f}")
            if r["zeroshot"]:
                zs = r["zeroshot"][st]["acc"]; cm = out["ctl_bands"][st]["marginal"]
                print(f"  {V} {st}: zero-shot lookahead (the 2B's own top-1 at offset 1..8) " + " ".join(f"{100 * z:.1f}" for z in zs)
                      + f" %; ctl head marginal: depth 1 {100 * cm['0-0']:.1f}, bands 1-2 {100 * cm['1-2']:.1f}, 3-6 {100 * cm['3-6']:.1f},"
                        f" 7-10 {100 * cm['7-10']:.1f} %")
        print(f"VERDICT {V}: {v} -- HE G(1-8) {g18:+.3f} [{glo:+.3f}, {H['bands']['1-8']['G_ci'][1]:+.3f}],"
              f" HE tree-32 x{t32:.4f} [{H['tree32']['ci95'][0]:.4f}, {H['tree32']['ci95'][1]:.4f}];"
              f" MBPP G(1-8) {r['sets']['mbpp']['bands']['1-8']['G']:+.3f}, tree-32 x{r['sets']['mbpp']['tree32']['ratio']:.4f}"
              + (f"; |P| / |fc W-cols| {r['P_norm_ratio']:.4f}" + (" (may be under-trained)" if r["P_norm_ratio"] < 0.01 else "")
                 if "P_norm_ratio" in r else ""), flush=True)
    if verdicts:
        sv = "PASS" if "PASS" in verdicts.values() else ("KILL" if len(verdicts) == 3 and all(v == "KILL" for v in verdicts.values())
                                                          else "BETWEEN")
        out["screen_verdict"] = sv
        print(f"VERDICT SCREEN: {sv} -- " + ", ".join(f"{k} {v}" for k, v in verdicts.items()), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
