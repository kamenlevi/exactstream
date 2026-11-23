"""BREAK98c verdict: Fable FAR's PREFIX EYES on the 2B (results/LEVERS_A.md section B98C, pre-registered before data).
Inputs (LAB, teacher-forced 2B):
  results/b98s1_ctl_eval.json   evalx: wmix_ctl chain + true (own / true for G), MBPP-50 / HE-30, per-window counts
  results/walkmix_tree.json     tree8.py: wmix_ctl's tree-32 / tree-16 / chain-32 per window (used when the ctl read-out
                                above equals WALK-MIX's exactly: the same weights, the same code)
  results/b98c_eval.json        evalx: b98c_kvx3 / b98c_kvx10 chain (the arms with the cross-attention block)
  results/b98c_tree.json        tree8.py (MTP Head's, read-only): the arms' trees
  ckpt/b98c.json, ckpt/b98c_ARM.safetensors   training log (loss, gradient norms) and the block's weights (information)
LINES (FAR's), per arm; HE is the primary set:
  PASS     HE G(9-30) >= 0.30 AND HE tree-32 kept >= x1.08 of ctl
  KILL     HE G < 0.10 at every band (1-2, 3-6, 7-10, 11-14, 15-22, 23-30)
  BETWEEN  otherwise.   Screen: PASS if either arm passes; KILL only if both are KILL.
G = (arm - own) / (true - own), walk-strict pooled over the band's depths (driftg.py / SIB / B98S1's definition); bands are
evalx's labels (band d = chain depth d + 1). Paired window bootstrap (2000, seed 1116) for G; tree ratios as tree8.py's
(ratio of sums, 5000 window resamples, seed 1115 per comparison).
usage: b98c_verdict.py [--arms kvx3,kvx10] [--out results/b98c_verdict.json]"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb.b98s1_verdict import pc, detcheck, tree_ratio

SETS = ("mbpp", "he")
KILL_BANDS = ("1-2", "3-6", "7-10", "11-14", "15-22", "23-30")
BANDS = KILL_BANDS + ("1-8", "9-30")
CTL = "wmix_ctl"
LAYERS = (15, 19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="kvx3,kvx10"); ap.add_argument("--out", default="results/b98c_verdict.json")
    a = ap.parse_args()
    E0 = json.load(open("results/b98s1_ctl_eval.json"))
    assert detcheck(), "ctl's read-out differs from WALK-MIX's: its tree cannot be reused"
    TC = json.load(open("results/walkmix_tree.json"))
    E, T = json.load(open("results/b98c_eval.json")), json.load(open("results/b98c_tree.json"))
    TR = json.load(open("ckpt/b98c.json")) if os.path.exists("ckpt/b98c.json") else None
    WM = json.load(open("ckpt/wmix.json"))["arms"]["ctl"]["loss"] if os.path.exists("ckpt/wmix.json") else None
    out = {"ctl": CTL, "ctl_tree_source": "results/walkmix_tree.json (WALK-MIX's ctl, same weights)", "arms": {}, "ctl_bands": {}}
    for st in SETS:
        ko, ro, vo = pc(E0, f"{CTL}|{st}|chain"); kt, rt, vt = pc(E0, f"{CTL}|{st}|true")
        assert ko == kt
        out["ctl_bands"][st] = {}
        for b in BANDS:
            lo, hi = (int(x) for x in b.split("-"))
            out["ctl_bands"][st][b] = {"own": float(ro[:, lo:hi + 1].sum() / vo[:, lo:hi + 1].sum()),
                                       "true": float(rt[:, lo:hi + 1].sum() / vt[:, lo:hi + 1].sum())}
    print(f"ctl = {CTL} (WALK-MIX's control: s30o + 400 steps, the same setup); tree: walkmix_tree.json")
    for st in SETS:
        cb = out["ctl_bands"][st]
        print(f"  {st} own -> true walk-strict (the gap): " + "; ".join(f"{b} {100 * cb[b]['own']:.2f} -> {100 * cb[b]['true']:.2f}"
                                                                          for b in BANDS))
    verdicts = {}
    for arm in a.arms.split(","):
        run = f"b98c_{arm}"
        r = {"sets": {}}
        for st in SETS:
            ko, ro, vo = pc(E0, f"{CTL}|{st}|chain"); _, rt, vt = pc(E0, f"{CTL}|{st}|true")
            ka, ra, va = pc(E, f"{run}|{st}|chain")
            assert ko == ka, "the arm and ctl were read on different windows"
            D = min(ro.shape[1], ra.shape[1], rt.shape[1])
            idx = np.random.default_rng(1116).integers(0, len(ko), (2000, len(ko)))
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
            tr = {k: tree_ratio(TC["res"][f"{CTL}|{st}"][k]["per_window"], T["res"][f"{run}|{st}"][k]["per_window"])
                  for k in ("K32|g1.0", "K16|g1.0", "chain")}
            gband = [T["res"][f"{run}|{st}"][f"K32|g{g}"]["kept"] / TC["res"][f"{CTL}|{st}"][f"K32|g{g}"]["kept"] for g in ("0.5", "2.0")]
            r["sets"][st] = {"bands": bands, "tree32": {"ctl": TC["res"][f"{CTL}|{st}"]["K32|g1.0"]["kept"],
                                                         "arm": T["res"][f"{run}|{st}"]["K32|g1.0"]["kept"],
                                                         "ratio": tr["K32|g1.0"][0], "ci95": tr["K32|g1.0"][1]},
                             "tree16_ratio": tr["K16|g1.0"][0], "chain32_ratio": tr["chain"][0], "tree32_ratio_g0.5_g2": gband}
        ck = f"ckpt/{run}.safetensors"
        if os.path.exists(ck):
            import mlx.core as mx
            w = mx.load(ck)
            on = float(mx.linalg.norm(w["o_proj.weight"]))
            r["o_x_over_o_proj"] = {L: float(mx.linalg.norm(w[f"kvx_o{L}.weight"])) / on for L in LAYERS}
        if TR is not None and arm in TR["arms"]:
            L_ = np.array(TR["arms"][arm]["loss"])
            r["train"] = {"loss_last100": float(L_[-100:].mean())}
            if WM is not None and len(WM) == len(L_):
                r["train"]["ctl_loss_last100"] = float(np.mean(WM[-100:]))
            gn = TR["arms"][arm].get("gnorm")
            if gn:
                gn = np.array(gn); r["train"]["gnorm_head_median"] = float(np.median(gn[:, 0]))
                r["train"]["gnorm_block_median"] = float(np.median(gn[:, 1]))
        H = r["sets"]["he"]["bands"]
        g930 = H["9-30"]["G"]; t32 = r["sets"]["he"]["tree32"]["ratio"]
        if g930 >= 0.30 and t32 >= 1.08:
            v = "PASS"
        elif all(H[b]["G"] < 0.10 for b in KILL_BANDS):
            v = "KILL"
        else:
            v = "BETWEEN"
        r["verdict"] = v; verdicts[arm] = v
        out["arms"][arm] = r
        for st in SETS:
            x = r["sets"][st]; bb = x["bands"]
            print(f"  {arm} {st}: " + "; ".join(
                f"{b} {100 * bb[b]['own']:.2f}->{100 * bb[b]['arm']:.2f} ({bb[b]['delta_pt']:+.2f} pt"
                + (f", G {bb[b]['G']:+.2f} [{bb[b]['G_ci'][0]:+.2f}, {bb[b]['G_ci'][1]:+.2f}]" if "G" in bb[b] else "") + ")"
                for b in BANDS if b in bb)
                  + f"; tree-32 {x['tree32']['ctl']:.3f} -> {x['tree32']['arm']:.3f} = x{x['tree32']['ratio']:.4f}"
                    f" [{x['tree32']['ci95'][0]:.4f}, {x['tree32']['ci95'][1]:.4f}] (gamma 0.5 / 2 x{x['tree32_ratio_g0.5_g2'][0]:.4f}"
                    f" / x{x['tree32_ratio_g0.5_g2'][1]:.4f}), tree-16 x{x['tree16_ratio']:.4f}, chain-32 x{x['chain32_ratio']:.4f}")
        info = ""
        if "o_x_over_o_proj" in r:
            info += "; |o_x| / |o_proj| " + " / ".join(f"L{L} {v_:.4f}" for L, v_ in r["o_x_over_o_proj"].items())
        if "train" in r:
            t_ = r["train"]
            info += f"; train loss (last 100) {t_['loss_last100']:.4f}" + (f" vs ctl {t_['ctl_loss_last100']:.4f}" if "ctl_loss_last100" in t_ else "")
        hb = r["sets"]["he"]
        print(f"VERDICT {arm}: {v} -- HE G(9-30) {g930:+.3f} [{H['9-30']['G_ci'][0]:+.3f}, {H['9-30']['G_ci'][1]:+.3f}],"
              f" HE tree-32 x{t32:.4f} [{hb['tree32']['ci95'][0]:.4f}, {hb['tree32']['ci95'][1]:.4f}]; HE G by band "
              + " ".join(f"{b} {H[b]['G']:+.2f}" for b in KILL_BANDS)
              + f"; MBPP G(9-30) {r['sets']['mbpp']['bands']['9-30']['G']:+.3f}, tree-32 x{r['sets']['mbpp']['tree32']['ratio']:.4f}" + info,
              flush=True)
    if verdicts:
        sv = "PASS" if "PASS" in verdicts.values() else ("KILL" if all(v == "KILL" for v in verdicts.values()) else "BETWEEN")
        out["screen_verdict"] = sv
        print(f"VERDICT SCREEN: {sv} -- " + ", ".join(f"{k} {v}" for k, v in verdicts.items()), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
