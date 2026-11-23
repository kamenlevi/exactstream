"""TREEMASK verdict (the tree-aware mask, NewIdeas 20:3x; results/LEVERS_A.md section TREEMASK, pre-registered before
data). Reads results/treemask_eval.json (sb/evalx.py: s30, tmask_ctl, tmask_tm2, tmaskb_tm4 on MBPP-50 / HE-30, with
per-window walk-strict counts) and results/treemask_tree.json (~/TrainLab/sb98/tree8.py, read-only: tree-32 kept per run
and each arm / ctl ratio with its window-bootstrap CI). Lines, per arm:
  PASS  tree-32 kept (gamma 1) >= x1.03 of ctl on BOTH sets, with the shallow line held on both sets: pooled walk-strict
        at band depths 0-3 (the first four guesses; d0 = the first guess) AND at band depths 1-4 each within -0.2 pt of ctl
  KILL  tree-32 kept < x1.01 on HE
  BETWEEN otherwise
Information: the off-path part (tree-32 kept minus chain-32 kept, the share the mask aims at) per arm minus ctl with a
paired window-bootstrap CI; FAR's shallow-loss flag (> 0.5 pt down); the control against s30 (MBPP 7-14 marginal must not
fall > 1.5 pt); whether ctl's weights equal WALK-MIX's ctl (sha256), and if so each arm's off-path part against WALK-MIX's
mix5 (k = all, w 0.5); chain-32, tree-16, the gamma 0.5 / 2 band.
usage: treemask_verdict.py [--eval results/treemask_eval.json] [--tree results/treemask_tree.json]
                           [--out results/treemask_verdict.json] [--ckpt ckpt]"""
import argparse, hashlib, json, os
import numpy as np

ARMS = (("tm2", "tmask_tm2"), ("tm4", "tmaskb_tm4"))
CTL, BASE, SETS = "tmask_ctl", "s30", ("mbpp", "he")
WM = ("results/walkmix_tree.json", "wmix_ctl", "wmixb_mix5")


def strict(E, run, st, lo, hi):
    pc = E["pcond"][f"{run}|{st}|chain"]
    r = sum(sum(v[0][lo:hi + 1]) for v in pc.values()); n = sum(sum(v[1][lo:hi + 1]) for v in pc.values())
    return r / n if n else float("nan"), sorted(pc)


def marg(E, run, st, bands):
    per = E["per"][f"{run}|{st}|chain"]
    r = sum(per[w][b][0] for w in per for b in bands); n = sum(per[w][b][1] for w in per for b in bands)
    return r / n if n else float("nan")


def offpath(T, run, st):
    """per-window (tree-32 kept sum, blocks, chain-32 kept sum, blocks) -> arrays [n_windows, 4]"""
    a = np.array(T["res"][f"{run}|{st}"]["K32|g1.0"]["per_window"], float)
    b = np.array(T["res"][f"{run}|{st}"]["chain"]["per_window"], float)
    return np.concatenate([a, b], 1)


def off_value(X):
    return X[..., 0].sum(-1) / X[..., 1].sum(-1) - X[..., 2].sum(-1) / X[..., 3].sum(-1)


def off_delta(XA, XC, boot=2000, seed=1116):
    """off-path part of A minus C, paired over windows (same window order), with a 95 % bootstrap CI"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(XC), (boot, len(XC)))
    bs = off_value(XA[idx]) - off_value(XC[idx])
    return float(off_value(XA) - off_value(XC)), np.percentile(bs, [2.5, 97.5]).tolist()


def sha(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/treemask_eval.json"); ap.add_argument("--tree", default="results/treemask_tree.json")
    ap.add_argument("--out", default="results/treemask_verdict.json"); ap.add_argument("--ckpt", default="ckpt")
    a = ap.parse_args()
    E, T = json.load(open(a.eval)), json.load(open(a.tree))
    out = {"control": {}, "arms": {}}
    hold = {}
    for st in SETS:
        b0, c0 = marg(E, BASE, st, ("7-10", "11-14")), marg(E, CTL, st, ("7-10", "11-14"))
        out["control"][st] = {"s30_7-14": b0, "ctl_7-14": c0, "delta_pt": 100 * (c0 - b0)}
        hold[st] = 100 * (c0 - b0) >= -1.5
    print("control vs s30 (marginal 7-14; MBPP must not fall > 1.5 pt): " + "; ".join(
        f"{st} {100 * v['s30_7-14']:.2f} -> {100 * v['ctl_7-14']:.2f} ({v['delta_pt']:+.2f})" for st, v in out["control"].items())
        + f" -> {'HOLDS' if hold['mbpp'] else 'DRIFTED (read the arms against ctl only)'}")
    h_ctl, h_wm = sha(os.path.join(a.ckpt, f"{CTL}.safetensors")), sha(os.path.join(a.ckpt, f"{WM[1]}.safetensors"))
    same_ctl = h_ctl is not None and h_ctl == h_wm
    out["ctl_equals_walkmix_ctl"] = same_ctl
    print(f"ctl's weights {'EQUAL' if same_ctl else 'differ from'} WALK-MIX's ctl (sha256 {str(h_ctl)[:12]} vs {str(h_wm)[:12]})")
    WT = json.load(open(WM[0])) if same_ctl and os.path.exists(WM[0]) else None
    for nm, run in ARMS:
        r = {"sets": {}}
        for st in SETS:
            sh = {}
            for lo, hi in ((0, 3), (1, 4)):
                s_c, kc = strict(E, CTL, st, lo, hi); s_a, ka = strict(E, run, st, lo, hi)
                assert kc == ka, "the arm and ctl were read on different windows"
                sh[f"{lo}-{hi}"] = {"ctl": s_c, nm: s_a, "delta_pt": 100 * (s_a - s_c)}
            cmp32 = T["cmp"][f"{run}@{CTL}|{st}|K32|g1.0"]; cmp16 = T["cmp"][f"{run}@{CTL}|{st}|K16|g1.0"]
            cmpch = T["cmp"][f"{run}@{CTL}|{st}|chain"]
            band = [T["res"][f"{run}|{st}"][f"K32|g{g}"]["kept"] / T["res"][f"{CTL}|{st}"][f"K32|g{g}"]["kept"] for g in ("0.5", "2.0")]
            XA, XC = offpath(T, run, st), offpath(T, CTL, st)
            od, oci = off_delta(XA, XC)
            x = {"tree32_ratio": cmp32["ratio"], "tree32_ci95": cmp32["ci95"], "tree32_ratio_g0.5_g2": band,
                 "tree16_ratio": cmp16["ratio"], "chain32_ratio": cmpch["ratio"],
                 "tree32_kept": {"ctl": T["res"][f"{CTL}|{st}"]["K32|g1.0"]["kept"], nm: T["res"][f"{run}|{st}"]["K32|g1.0"]["kept"]},
                 "offpath": {"ctl": float(off_value(XC)), nm: float(off_value(XA)), "delta": od, "ci95": oci},
                 "strict": sh}
            if WT is not None:
                XW = offpath(WT, WM[2], st)
                x["offpath"]["walkmix_mix5"] = float(off_value(XW))
                x["offpath"]["delta_vs_mix5"], x["offpath"]["ci95_vs_mix5"] = off_delta(XA, XW)
            r["sets"][st] = x
        S = r["sets"]
        shallow = all(S[st]["strict"][b]["delta_pt"] >= -0.2 for st in SETS for b in ("0-3", "1-4"))
        if all(S[st]["tree32_ratio"] >= 1.03 for st in SETS) and shallow:
            v = "PASS"
        elif S["he"]["tree32_ratio"] < 1.01:
            v = "KILL"
        else:
            v = "BETWEEN"
        r["verdict"], r["shallow_line_held"] = v, shallow
        r["shallow_loss_flag"] = any(S[st]["strict"][b]["delta_pt"] < -0.5 for st in SETS for b in ("0-3", "1-4"))
        out["arms"][nm] = r
        for st in SETS:
            x = S[st]; o = x["offpath"]
            print(f"  {nm} {st}: tree-32 {x['tree32_kept']['ctl']:.3f} -> {x['tree32_kept'][nm]:.3f} = x{x['tree32_ratio']:.4f}"
                  f" [{x['tree32_ci95'][0]:.4f}, {x['tree32_ci95'][1]:.4f}] (gamma 0.5 / 2: x{x['tree32_ratio_g0.5_g2'][0]:.4f} /"
                  f" x{x['tree32_ratio_g0.5_g2'][1]:.4f}), tree-16 x{x['tree16_ratio']:.4f}, chain-32 x{x['chain32_ratio']:.4f};"
                  f" off-path part {o['ctl']:.3f} -> {o[nm]:.3f} ({o['delta']:+.3f} [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}])"
                  + (f", vs WALK-MIX mix5 {o['walkmix_mix5']:.3f} ({o['delta_vs_mix5']:+.3f} [{o['ci95_vs_mix5'][0]:+.3f},"
                     f" {o['ci95_vs_mix5'][1]:+.3f}])" if "walkmix_mix5" in o else "")
                  + "; walk-strict " + ", ".join(f"{b} {100 * x['strict'][b]['ctl']:.2f} -> {100 * x['strict'][b][nm]:.2f}"
                                                f" ({x['strict'][b]['delta_pt']:+.2f} pt)" for b in ("0-3", "1-4")))
        print(f"VERDICT {nm}: {v} -- tree-32 x{S['mbpp']['tree32_ratio']:.4f} MBPP-50, x{S['he']['tree32_ratio']:.4f} HE-30;"
              f" shallow line {'held' if shallow else 'NOT held'}; off-path part {S['mbpp']['offpath']['delta']:+.3f} /"
              f" {S['he']['offpath']['delta']:+.3f} tokens per block"
              + ("; FAR's shallow-loss flag (> 0.5 pt down)" if r["shallow_loss_flag"] else ""))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
