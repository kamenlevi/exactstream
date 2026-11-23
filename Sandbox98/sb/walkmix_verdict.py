"""WALKMIX verdict (drift screen 2; results/LEVERS_A.md section WALKMIX, pre-registered before data). Reads
results/walkmix_eval.json (sb/evalx.py: s30, wmix_ctl, wmix_mix3, wmixb_mix5 on MBPP-50 / HE-30, with per-window
walk-strict counts) and results/walkmix_tree.json (~/TrainLab/sb98/tree8.py, read-only: tree-32 kept per run and each
arm / ctl ratio with its window-bootstrap CI). Lines, per arm, on each set:
  PASS  tree-32 kept (gamma 1) >= x1.03 of ctl on BOTH sets, with walk-strict at depths 1-4 within -0.2 pt of ctl on both
  KILL  tree-32 kept < x1.01 on both sets
  BETWEEN otherwise
Information: FAR's shallow-loss flag (walk-strict 1-4 down > 0.5 pt), the control against s30 (MBPP 7-14 marginal must not
fall > 1.5 pt), chain-32 kept, tree-16, the gamma 0.5 / 2 band.
usage: walkmix_verdict.py [--eval results/walkmix_eval.json] [--tree results/walkmix_tree.json] [--out results/walkmix_verdict.json]"""
import argparse, json
import numpy as np

ARMS = (("mix3", "wmix_mix3"), ("mix5", "wmixb_mix5"))
CTL, BASE, SETS = "wmix_ctl", "s30", ("mbpp", "he")


def strict(E, run, st, lo, hi):
    pc = E["pcond"][f"{run}|{st}|chain"]
    r = sum(sum(v[0][lo:hi + 1]) for v in pc.values()); n = sum(sum(v[1][lo:hi + 1]) for v in pc.values())
    return r / n if n else float("nan"), sorted(pc)


def marg(E, run, st, bands):
    per = E["per"][f"{run}|{st}|chain"]
    r = sum(per[w][b][0] for w in per for b in bands); n = sum(per[w][b][1] for w in per for b in bands)
    return r / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/walkmix_eval.json"); ap.add_argument("--tree", default="results/walkmix_tree.json")
    ap.add_argument("--out", default="results/walkmix_verdict.json")
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
    for nm, run in ARMS:
        r = {"sets": {}}
        for st in SETS:
            s_c, kc = strict(E, CTL, st, 1, 4); s_a, ka = strict(E, run, st, 1, 4)
            assert kc == ka, "the arm and ctl were read on different windows"
            cmp32 = T["cmp"][f"{run}@{CTL}|{st}|K32|g1.0"]; cmp16 = T["cmp"][f"{run}@{CTL}|{st}|K16|g1.0"]
            cmpch = T["cmp"][f"{run}@{CTL}|{st}|chain"]
            band = [T["res"][f"{run}|{st}"][f"K32|g{g}"]["kept"] / T["res"][f"{CTL}|{st}"][f"K32|g{g}"]["kept"] for g in ("0.5", "2.0")]
            r["sets"][st] = {"tree32_ratio": cmp32["ratio"], "tree32_ci95": cmp32["ci95"], "tree32_ratio_g0.5_g2": band,
                             "tree16_ratio": cmp16["ratio"], "chain32_ratio": cmpch["ratio"],
                             "tree32_kept": {"ctl": T["res"][f"{CTL}|{st}"]["K32|g1.0"]["kept"],
                                             nm: T["res"][f"{run}|{st}"]["K32|g1.0"]["kept"]},
                             "strict_1-4": {"ctl": s_c, nm: s_a, "delta_pt": 100 * (s_a - s_c)}}
        S = r["sets"]
        if all(S[st]["tree32_ratio"] >= 1.03 and S[st]["strict_1-4"]["delta_pt"] >= -0.2 for st in SETS):
            v = "PASS"
        elif all(S[st]["tree32_ratio"] < 1.01 for st in SETS):
            v = "KILL"
        else:
            v = "BETWEEN"
        r["verdict"] = v
        r["shallow_loss_flag"] = any(S[st]["strict_1-4"]["delta_pt"] < -0.5 for st in SETS)
        out["arms"][nm] = r
        for st in SETS:
            x = S[st]
            print(f"  {nm} {st}: tree-32 x{x['tree32_ratio']:.4f} [{x['tree32_ci95'][0]:.4f}, {x['tree32_ci95'][1]:.4f}]"
                  f" (gamma 0.5 / 2: x{x['tree32_ratio_g0.5_g2'][0]:.4f} / x{x['tree32_ratio_g0.5_g2'][1]:.4f}),"
                  f" tree-16 x{x['tree16_ratio']:.4f}, chain-32 x{x['chain32_ratio']:.4f};"
                  f" walk-strict 1-4 {100 * x['strict_1-4']['ctl']:.2f} -> {100 * x['strict_1-4'][nm]:.2f}"
                  f" ({x['strict_1-4']['delta_pt']:+.2f} pt)")
        print(f"VERDICT {nm}: {v} -- tree-32 x{S['mbpp']['tree32_ratio']:.4f} MBPP-50, x{S['he']['tree32_ratio']:.4f} HE-30;"
              f" walk-strict 1-4 {S['mbpp']['strict_1-4']['delta_pt']:+.2f} / {S['he']['strict_1-4']['delta_pt']:+.2f} pt"
              + ("; FAR's shallow-loss flag (> 0.5 pt down)" if r["shallow_loss_flag"] else ""))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
