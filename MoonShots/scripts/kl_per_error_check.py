"""KLPERERR (MoonShots, Thu 2026-11-05; the orchestrator 07:1x, NewIdeas' ask before Kamen sees SWEEP v1.3): independent
re-derivation of 'data-aware codes need a KL screen' (NewIdeas QUEUE Wed 22:5x, a2c2a52): on identical reads and inputs,
the screen-7 trellis store's errors cost ~2-3.5x more KL per unit of expert-output error than truncation (3.5x / 1.8x /
2.4x on code / IFEval / multi).
NewIdeas' computation: predicted trl - ref = D_cal x ((r - 1) T01 + r E29) / (4 T01 + 5 E29), D_cal = cal - ref (MSB-2 at
screen 7's 4 layers), r = Fit12's whole-expert output-error ratio of the store vs MSB-3 (layer means 1.23 / 1.68 / 1.67),
T01 / E29 = the ranks 0-1 / 2-9 MSB-3 terms; ratio = measured trl - ref / predicted.
Mine:
  KL: my own per-row KL(REF2-T || arm) (sweep_quality_check.row_kl, M-36's 'mine'; NewIdeas-style 'theirs' beside it) on
      the decision map's arm outputs, paired differences over the rows both arms have, per-row and per-window SEs.
  (1) NewIdeas' formula with my KL numbers and Fit12's r (the replication);
  (2) the formula with the MEASURED output-error ratios of data/msb2_judge.json (scripts/msb2_judge.py: MSB-2's ratio vs
      MSB-3 instead of the assumed 5, the store's per rank group, router-weighted), the KL terms as in (1);
  (3) the direct per-unit comparison, no T01 / E29 model: (dKL_trl / dKL_cal) / (dE_trl / dE_cal), dE = the router-
      weighted expert-output error added at the 4 layers over s7_ref (ranks 0-1: store - MSB-3; ranks 2-9: store vs
      4-bit), pooled over all answer rows (the arms touch only gated rows' kept experts: assumed proportional).
Arms (FableGod decision_map/dm_ablations.py): s7_ref = ranks 0-1 of the gated rows at MSB-3, kept 2-9 at 4 bits;
s7_trl / s7_cal = every kept expert of the gated rows at screen 7's layers 7 / 18 / 30 / 41 from the TRL mix / MSB-2;
s7_cal_all = MSB-2 at all 48; s7_ref3 = MSB-3 at all 48 (E29 = ref3 - ref); T01 = qom_0_live_bq_t3 - qom_0_live_bq.
usage: [BigMoE venv] python kl_per_error_check.py -> data/kl_per_error_check.txt"""
import json, math, os, sys
import numpy as np

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/MoonShots/scripts")
import sweep_quality_check as SQ                                   # noqa: E402  (M-36: load, row_kl)

DOMS = ("code", "ifeval", "multi")
ARMS = ("s7_ref", "s7_trl", "s7_cal", "s7_cal_all", "s7_ref3", "qom_0_live_bq", "qom_0_live_bq_t3")
THEIRS = {"trl": (0.00060, 0.00296, 0.00419), "cal": (0.00126, 0.00673, 0.00721), "E29": (0.00114, 0.00649, 0.00886),
          "pred": (0.00017, 0.00161, 0.00177), "ratio": (3.5, 1.8, 2.4)}
R_FIT12 = (1.23, 1.68, 1.67)
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def per_row(mode):
    """arm -> {window: (dom, [KL per row])}"""
    base = SQ.load("baseline")
    out = {}
    for a in ARMS:
        arm = SQ.load(a)
        d = {}
        for w, b in base.items():
            if w in arm:
                ar = arm[w]
                d[w] = (b["dom"], np.array([SQ.row_kl(tp, np.array(lp), tq, np.array(lq), mode)
                                            for tp, lp, tq, lq in zip(b["tk"], b["lp"], ar["tk"], ar["lp"])]))
        out[a] = d
    return out


def diff(K, a, b, dom):
    """per-row a - b over the windows / rows both have, in one domain -> (per-row array, per-window means, weights)"""
    xs, wm, wn = [], [], []
    for w, (dm, ka) in K[a].items():
        if dm != dom or w not in K[b]:
            continue
        kb = K[b][w][1]
        n = min(len(ka), len(kb))
        x = ka[:n] - kb[:n]
        xs.append(x); wm.append(x.mean()); wn.append(n)
    return np.concatenate(xs), np.array(wm), np.array(wn)


def se(x, wm, wn):
    row = x.std(ddof=1) / math.sqrt(len(x))
    m = np.average(wm, weights=wn)
    win = math.sqrt(np.average((wm - m) ** 2, weights=wn) / (len(wm) - 1))
    return row, win


def main():
    J = json.load(open(f"{H}/MoonShots/data/msb2_judge.json")) if os.path.exists(f"{H}/MoonShots/data/msb2_judge.json") else None
    for mode in ("mine", "theirs"):
        K = per_row(mode)
        say(f"== KL terms ({mode} KL approximation): mean per-row paired difference, SE per row / per window; NewIdeas' in brackets")
        T = {}
        for di, d in enumerate(DOMS):
            row = {}
            for nm, a, b in (("trl", "s7_trl", "s7_ref"), ("cal", "s7_cal", "s7_ref"), ("cal_all", "s7_cal_all", "s7_ref"),
                             ("E29", "s7_ref3", "s7_ref"), ("T01", "qom_0_live_bq_t3", "qom_0_live_bq")):
                x, wm, wn = diff(K, a, b, d)
                row[nm] = (x.mean(), *se(x, wm, wn), x)
            T[d] = row
            say(f"  {d}: " + "; ".join(f"{nm} {v[0]:+.5f} ({v[1]:.5f} / {v[2]:.5f})" + (f" [{THEIRS[nm][di]:+.5f}]" if nm in THEIRS else "")
                                       for nm, v in row.items()))
        say(f"  (1) NewIdeas' formula, r = Fit12's layer means {R_FIT12}, MSB-2 = 5 units: predicted trl - ref, measured / predicted")
        for di, d in enumerate(DOMS):
            t = T[d]; r = R_FIT12[di]
            comp = lambda r01, r29: (r01 - 1) * t["T01"][0] + r29 * t["E29"][0]
            pred = t["cal"][0] * comp(r, r) / comp(5, 5)
            ratio = t["trl"][0] / pred
            # delta-method SE of trl / cal (paired rows: both differences share s7_ref)
            a_, b_ = t["trl"][3], t["cal"][3]
            n = min(len(a_), len(b_))
            cov = np.cov(a_[:n], b_[:n])
            q = a_[:n].mean() / b_[:n].mean()
            vq = q * q * (cov[0, 0] / a_[:n].mean() ** 2 + cov[1, 1] / b_[:n].mean() ** 2
                          - 2 * cov[0, 1] / (a_[:n].mean() * b_[:n].mean())) / n
            say(f"    {d}: predicted {pred:+.5f} [{THEIRS['pred'][di]:+.5f}], ratio {ratio:.2f}x +- {ratio * math.sqrt(vq) / q:.2f} "
                f"(per-row SE) [{THEIRS['ratio'][di]}x]; weights T01 {t['T01'][0]:+.5f} vs E29 {t['E29'][0]:+.5f}")
        if J is None:
            say("  (2) / (3): data/msb2_judge.json missing (run scripts/msb2_judge.py first)")
            continue
        sums = {L: np.array(v["sums"]) for L, v in J["layers"].items()}          # [view m3 / m2 / trl, (e, ew), dom, rank]
        say(f"  (2) / (3) with the measured output errors (data/msb2_judge.json, layers {sorted(sums, key=int)}; "
            "router-weighted unless noted)")
        for di, d in enumerate(DOMS):
            t = T[d]
            G = lambda v, wi, rk: sum(sums[L][v, wi, di][rk].sum() for L in sums)
            r01, r29 = list(range(2)), list(range(2, 10))
            rr = {v: (G(i, 1, r01) / G(0, 1, r01), G(i, 1, r29) / G(0, 1, r29)) for i, v in ((1, "m2"), (2, "trl"))}
            rr_u = {v: (G(i, 0, r01) / G(0, 0, r01), G(i, 0, r29) / G(0, 0, r29)) for i, v in ((1, "m2"), (2, "trl"))}
            comp = lambda a, b: (a - 1) * t["T01"][0] + b * t["E29"][0]
            pred2 = t["cal"][0] * comp(*rr["trl"]) / comp(*rr["m2"])
            dE_trl = (G(2, 1, r01) - G(0, 1, r01)) + G(2, 1, r29)
            dE_cal = (G(1, 1, r01) - G(0, 1, r01)) + G(1, 1, r29)
            direct = (t["trl"][0] / t["cal"][0]) / (dE_trl / dE_cal)
            say(f"    {d}: output-error ratios vs MSB-3, ranks 0-1 / 2-9: MSB-2 {rr['m2'][0]:.2f} / {rr['m2'][1]:.2f} "
                f"(unweighted {rr_u['m2'][0]:.2f} / {rr_u['m2'][1]:.2f}; assumed 5), TRL {rr['trl'][0]:.2f} / {rr['trl'][1]:.2f} "
                f"(unweighted {rr_u['trl'][0]:.2f} / {rr_u['trl'][1]:.2f}; Fit12's layer mean {R_FIT12[di]})")
            say(f"      (2) formula with measured ratios: predicted {pred2:+.5f}, ratio {t['trl'][0] / pred2:.2f}x; "
                f"(3) direct: dKL trl / cal = {t['trl'][0] / t['cal'][0]:.3f}, dE trl / cal = {dE_trl / dE_cal:.3f} -> "
                f"KL per unit of output error, trellis / truncation = {direct:.2f}x")
    open(f"{H}/MoonShots/data/kl_per_error_check.txt", "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
