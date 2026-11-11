"""Independent re-derivation of THE SWEEP v1.1's quality column in the engine frame (NewIdeas 63f25fe; the orchestrator's
rule before Kamen sees numbers). My own code: KL approximation, flip measurement, the flips ~ sqrt(KL) fit, the model.
Inputs: the decision map's arms (~/FableGod/decision_map/out/<arm>.json) vs baseline.json (REF2-T): per answer row, the top-8
ids and log-probs. Only the slot-count keep shares and gap closures (g) are NewIdeas' replay output
(~/NewIdeas/data/ledger/trellis_draft17.txt, 24 GB rows), used as given.
KL(REF2-T || arm) per row, mine: the reference's top-8 + its tail; an id of the reference's top-8 missing from the arm's
top-8 gets an equal share of the arm's tail mass (capped at the arm's 8th probability), and the arm's mass outside the
reference's top-8 meets the reference's tail. NewIdeas' variant (a missing id gets the arm's 8th probability) runs as a
sensitivity. Flips: the arm's top-1 differs from the reference's top-1, per row, measured. flips = a + b sqrt(KL) is MY
fit over the 7 arms per domain.
Model (NewIdeas' frame, recomputed): E29 = KL(s7_ref3) - KL(s7_ref); T01 = KL(bq_t3) - KL(bq); D17 = KL(screen 6) - KL(C) on
code, 0 on chat; map(g) piecewise linear through (0, screen 6), (g_ref, s7_ref), (1, bq_t3 + D17); row KL = map(g(c)) +
(r - 1) T01 + r E29 keep(c) / keep(239). Check: MSB-2 at 239 = s7_ref + 4 T01 + 5 E29 vs s7_cal_all.
usage: [BigMoE venv] python sweep_quality_check.py -> data/sweep_quality_check.txt"""
import json, math, os
import numpy as np

OUT = os.path.expanduser("~/FableGod/decision_map/out")
ARMS = {"screen6": "qom_0_live_ball_c147_t3", "C": "qom_0_live_bq_c147_t3", "s7_ref": "s7_ref", "s7_ref3": "s7_ref3",
        "s7_cal_all": "s7_cal_all", "bq_t3": "qom_0_live_bq_t3", "bq": "qom_0_live_bq"}
DOMS = ("code", "ifeval", "multi")
KEEP = {171: (80.2, 82.6, 83.6), 185: (82.0, 83.9, 84.7), 239: (87.9, 88.2, 88.2), 276: (91.1, 90.3, 89.8)}   # % (NewIdeas)
GAP = {171: (0.21, 0.23, 0.25), 185: (0.32, 0.36, 0.38), 239: (0.67, 0.75, 0.81), 276: (0.87, 0.94, 1.00)}
ROWS = [("today MSB-3", 1.0, 171), ("TRL-2", 1.854, 276), ("mix (1.62)", 1.62, 239), ("mix (1.87)", 1.87, 239),
        ("TRL-3", 0.52, 185), ("GPTQ-3", 0.61, 171)]
THEIRS = {"today MSB-3": (1.18, 8.22, 6.82), "TRL-2": (1.30, 8.52, 7.96), "mix (1.62)": (1.24, 8.29, 7.57),
          "mix (1.87)": (1.31, 8.71, 8.02), "TRL-3": (0.99, 7.07, 5.66), "GPTQ-3": (1.06, 7.53, 5.99)}
THEIR_AB = {"code": (-0.078, 15.68), "ifeval": (0.420, 39.04), "multi": (-0.173, 36.17)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def row_kl(tp, lp, tq, lq, mode):
    p = np.exp(lp); q = np.exp(lq)
    pt, qt = max(0.0, 1 - p.sum()), max(0.0, 1 - q.sum())
    qmap = dict(zip(tq, q))
    miss = [i for i in tp if i not in qmap]
    share = min(q[-1], qt / (len(miss) + 1)) if mode == "mine" else q[-1]
    qh = np.array([qmap.get(i, share) for i in tp])
    kl = float((p * (lp - np.log(np.maximum(qh, 1e-30)))).sum())
    if pt > 1e-12:
        qr = max(1e-12, 1 - qh.sum())
        kl += pt * (math.log(pt) - math.log(qr))
    return kl


def load(name):
    d = json.load(open(os.path.join(OUT, name + ".json")))
    return {k[:-4] if k.endswith(".npz") else k: v for k, v in d.items()}


def main():
    base = load("baseline")
    stats = {}
    for tag, name in ARMS.items():
        arm = load(name)
        for mode in ("mine", "theirs"):
            acc = {d: [0.0, 0, 0] for d in DOMS}
            for w, b in base.items():
                if w not in arm:
                    continue
                a = arm[w]
                assert a["dom"] == b["dom"] and len(a["tk"]) == len(b["tk"]) and a.get("a") == b.get("a"), (tag, w)
                for tp, lp, tq, lq in zip(b["tk"], b["lp"], a["tk"], a["lp"]):
                    acc[b["dom"]][0] += row_kl(tp, np.array(lp), tq, np.array(lq), mode)
                    acc[b["dom"]][1] += tq[0] != tp[0]
                    acc[b["dom"]][2] += 1
            stats[(tag, mode)] = {d: (acc[d][0] / acc[d][2], 100 * acc[d][1] / acc[d][2], acc[d][2]) for d in DOMS}
    say("== THE SWEEP v1.1 quality column, independent re-derivation (MoonShots): mean per-row KL(REF2-T || arm), "
        "mine [NewIdeas-style approx], and measured flips % (code / IFEval / multi)")
    for tag in ARMS:
        m, t = stats[(tag, "mine")], stats[(tag, "theirs")]
        say(f"  {tag:10s}: KL " + " / ".join(f"{m[d][0]:.4f} [{t[d][0]:.4f}]" for d in DOMS)
            + " | flips " + " / ".join(f"{m[d][1]:.2f}" for d in DOMS) + f"  (rows {m['code'][2]} / {m['ifeval'][2]} / {m['multi'][2]})")
    for mode in ("mine", "theirs"):
        say(f"  -- model with the {mode} KL approximation --")
        K = {tag: stats[(tag, mode)] for tag in ARMS}
        ab = {}
        for di, d in enumerate(DOMS):
            x = np.array([math.sqrt(K[t][d][0]) for t in ARMS]); y = np.array([K[t][d][1] for t in ARMS])
            b_, a_ = np.polyfit(x, y, 1)
            r2 = 1 - ((y - (a_ + b_ * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            ab[d] = (a_, b_)
            say(f"    flips fit {d}: {a_:+.3f} + {b_:.2f} sqrt(KL)  (R2 {r2:.3f}; NewIdeas {THEIR_AB[d][0]:+.3f} + {THEIR_AB[d][1]:.2f})")
        terms = {}
        for di, d in enumerate(DOMS):
            kl = lambda t: K[t][d][0]
            E29, T01 = kl("s7_ref3") - kl("s7_ref"), kl("bq_t3") - kl("bq")
            D17 = kl("screen6") - kl("C") if d == "code" else 0.0
            terms[d] = (E29, T01, D17, kl("screen6"), kl("s7_ref"), kl("bq_t3") + D17)
            msb2 = kl("s7_ref") + 4 * T01 + 5 * E29
            say(f"    {d}: E29 {E29:+.4f}, T01 {T01:+.4f}, D17 {D17:+.4f}; MSB-2 check: model {msb2:.4f} vs measured "
                f"s7_cal_all {kl('s7_cal_all'):.4f} (x{kl('s7_cal_all') / msb2:.2f})")
        for name, r, c in ROWS:
            out = []
            for di, d in enumerate(DOMS):
                E29, T01, D17, k0, kr, k1 = terms[d]
                g, gr = GAP[c][di], GAP[239][di]
                mp = k0 + (kr - k0) * g / gr if g <= gr else kr + (k1 - kr) * (g - gr) / (1 - gr)
                kl = mp + (r - 1) * T01 + r * E29 * KEEP[c][di] / KEEP[239][di]
                a_, b_ = ab[d]
                out.append((kl, a_ + b_ * math.sqrt(max(kl, 0.0))))
            say(f"    {name:12s} (r {r}, {c} slots): KL " + " / ".join(f"{k:.4f}" for k, _ in out) + " | flips "
                + " / ".join(f"{f:.2f}" for _, f in out) + "  [NewIdeas " + " / ".join(f"{x:.2f}" for x in THEIRS[name]) + "]")
    open(os.path.expanduser("~/MoonShots/data/sweep_quality_check.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
