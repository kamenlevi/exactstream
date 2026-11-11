"""MAP KL (NewIdeas, Wed 4 Nov 2026 14:3x; CPU only, read-only on the map outputs): a continuous read-out for map arms.
Per row: the reference (baseline = REF2-T) and the arm each store the top-8 token ids and log-probs. KL(ref || arm) is
approximated on the reference's top-8 plus one tail bucket; an id missing from the arm's top-8 gets the arm's 8th
log-prob (an upper bound on its probability, so KL is slightly underestimated). Also: flips (top-1 changed) and the
mean drop of the reference top-1's log-prob. Usage: python map_kl.py <arm> [<arm> ...] -> prints a table."""
import json, os, sys
import numpy as np
OUT = os.path.expanduser("~/FableGod/decision_map/out")
BASE = json.load(open(os.path.join(OUT, "baseline.json")))


def rows_kl(b, r):
    out = []
    m = min(len(b["tk"]), len(r["tk"]))
    for i in range(m):
        tb, lb, ta, la = b["tk"][i], np.array(b["lp"][i]), r["tk"][i], np.array(r["lp"][i])
        pa = dict(zip(ta, la)); floor = la[-1]
        lpa = np.array([pa.get(t, floor) for t in tb])
        pb = np.exp(lb); qa = np.exp(lpa)
        tb_ = max(1e-9, 1 - pb.sum()); ta_ = max(1e-9, 1 - np.exp(la).sum())
        kl = float((pb * (lb - lpa)).sum() + tb_ * np.log(tb_ / ta_))
        out.append((kl, float(lb[0] - lpa[0]), int(ta[0] != tb[0])))
    for i in range(m, len(b["tk"])): out.append((np.nan, np.nan, 1))   # rows past the arm's end count as flips (qom_read)
    return out


def arm(name):
    R = json.load(open(os.path.join(OUT, name + ".json"))); v = {}
    for nm, b in BASE.items():
        if nm in R: v.setdefault(b["dom"], []).extend(rows_kl(b, R[nm]))
    res = {}
    for d, x in v.items():
        a = np.array(x, float)
        res[d] = (100 * np.nanmean(a[:, 2]), np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), len(a))
    return res


if __name__ == "__main__":
    print("| arm | code flips % / KL / dlp | ifeval | multi |")
    for n in sys.argv[1:]:
        if not os.path.exists(os.path.join(OUT, n + ".json")): print(f"| {n} | missing |"); continue
        r = arm(n)
        print(f"| {n} | " + " | ".join(f"{r[d][0]:.2f} / {r[d][1]:.4f} / {r[d][2]:.4f}" for d in ("code", "ifeval", "multi")) + " |")
