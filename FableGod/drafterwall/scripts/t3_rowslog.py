#!/usr/bin/env python3
"""t3_rowslog.py (drafterwall T3, pre-registered in PREREG.md, commit 602d66d). CPU.
Can any tree or picker over the two existing readers reach the target? Data: tree3_map (teacher-forced top-3 with
probabilities along the true path, depths 0-14 = a 16-row check).
(i) ceilings with unlimited rows: the official loop where a depth is passed when the true token is inside a candidate set.
(ii) threshold trees: verify every node whose path probability under the reader is >= tau. On the true path the
     reader gives the true token q_j (0 when it is outside its top-3); P_j = prod q_i; kept = 1 + #{j: P_j >= tau}.
     Rows: R = 1 + sum_j E[1{P_j >= tau} / P_j] (exact for a calibrated reader: each node of probability P is the true
     one with probability P, so counting true-path nodes with weight 1 / P counts all nodes) [P].
(iii) the slope of kept against ln R."""
import numpy as np, os, glob
TD = os.path.expanduser("~/TrainLab/data/tree3_map"); out = []
def say(s=""):
    print(s); out.append(s)
TAUS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
summary = {}
for dom in ("code", "ifeval", "multi"):
    files = sorted(glob.glob(f"{TD}/{dom}/*.npz"))
    ceil = {k: [] for k in ("dfl top-1", "dfl top-3", "mtp top-1", "mtp top-3", "either top-1", "union top-3")}
    tree = {r: {t: [[], []] for t in TAUS} for r in ("dfl", "mtp", "mix")}
    cal = {r: [0.0, 0.0, 0] for r in ("dfl", "mtp")}
    for f in files:
        z = np.load(f); true = z["true"]; n = true.shape[0]; mm = z["m"]
        dt, dp, mt, mp = z["dfl_t3"], z["dfl_p3"].astype(np.float64), z["mtp_t3"], z["mtp_p3"].astype(np.float64)
        ok = true >= 0
        sets = {"dfl top-1": (dt[..., 0] == true), "dfl top-3": (dt == true[..., None]).any(-1), "mtp top-1": (mt[..., 0] == true),
                "mtp top-3": (mt == true[..., None]).any(-1)}
        sets["either top-1"] = sets["dfl top-1"] | sets["mtp top-1"]; sets["union top-3"] = sets["dfl top-3"] | sets["mtp top-3"]
        for name, tab in sets.items():
            tab = tab & ok; p = 0
            while p < n:
                m = min(15, int(mm[p])); k = 0
                while k < m and tab[p, k]: k += 1
                ceil[name].append(k + 1); p += k + 1
        # the reader's probability of the true token (0 outside its top-3)
        qd = (dp * (dt == true[..., None])).sum(-1); qm = (mp * (mt == true[..., None])).sum(-1)
        cal["dfl"][0] += float(dp[..., 0][ok].sum()); cal["dfl"][1] += float(sets["dfl top-1"][ok].sum()); cal["dfl"][2] += int(ok.sum())
        cal["mtp"][0] += float(mp[..., 0][ok].sum()); cal["mtp"][1] += float(sets["mtp top-1"][ok].sum()); cal["mtp"][2] += int(ok.sum())
        Q = {"dfl": qd, "mtp": qm, "mix": 0.5 * qd + 0.5 * qm}
        for r, q in Q.items():
            q = np.where(ok, q, 0.0)
            for t in TAUS:
                p = 0
                while p < n:
                    m = min(15, int(mm[p])); P = np.cumprod(q[p, :m]); inside = P >= t
                    k = int(inside.sum())
                    tree[r][t][0].append(k + 1)
                    tree[r][t][1].append(1.0 + float((1.0 / P[inside]).sum()))
                    p += k + 1
    say(f"\n== {dom} ({len(files)} map windows; the official loop at 16 rows on the tables)")
    say("   ceilings with unlimited rows, kept per check: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in ceil.items()))
    say("   calibration (mean top-1 probability vs top-1 right): " + ", ".join(f"{r} {c[0] / c[2]:.3f} vs {c[1] / c[2]:.3f}" for r, c in cal.items()))
    for r in ("dfl", "mtp", "mix"):
        rows = [(np.mean(tree[r][t][1]), np.mean(tree[r][t][0]), t) for t in TAUS]
        say(f"   threshold tree on {r}: " + "; ".join(f"R {R:.0f} -> kept {K:.2f}" for R, K, t in rows))
        R = np.array([x[0] for x in rows]); K = np.array([x[1] for x in rows])
        sl = np.polyfit(np.log(R), K, 1)[0]
        say(f"      slope: +{sl:.2f} kept per e-fold of rows (+{sl * np.log(2):.2f} per doubling); kept at R <= 128: {max([k for r_, k, _ in rows if r_ <= 128], default=float('nan')):.2f}; at R <= 256: {max([k for r_, k, _ in rows if r_ <= 256], default=float('nan')):.2f}; at R <= 1024: {max([k for r_, k, _ in rows if r_ <= 1024], default=float('nan')):.2f}")
        summary[(dom, r)] = (max([k for r_, k, _ in rows if r_ <= 128], default=0), max([k for r_, k, _ in rows if r_ <= 256], default=0), sl)
    summary[(dom, "ceil")] = float(np.mean(ceil["union top-3"]))
c = summary[("code", "ceil")]; b128 = max(summary[("code", r)][0] for r in ("dfl", "mtp", "mix")); b256 = max(summary[("code", r)][1] for r in ("dfl", "mtp", "mix"))
v = "PASS" if (c >= 14.5 and b128 >= 13.5) else ("KILL" if (c < 14.5 or b256 < 13) else "BETWEEN")
say(f"\nT3 LINE (PASS: union-top-3 ceiling >= 14.5 and a tree >= 13.5 at R <= 128; KILL: ceiling < 14.5 or kept at R <= 256 < 13): "
    f"code ceiling {c:.2f}, best tree at R <= 128 {b128:.2f}, at R <= 256 {b256:.2f} -> {v}")
open(os.path.join(os.path.dirname(__file__), "..", "out", "t3_rowslog.txt"), "w").write("\n".join(out) + "\n")
