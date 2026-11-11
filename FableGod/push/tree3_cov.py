"""tree3_cov.py (Fable B, PUSH-B refinement): on MTP Head's aligned tree3_map data (36 map windows, every anchor, depths 0-14):
top-1 rate per depth and the top-3 coverage CONDITIONAL on a top-1 miss, for DFlash2 L1-20, the MTP head b5 and their union;
and the value of a cheap sibling fan at depth 0 on the MTP chain: P(miss at 0) x P(true in top-3 | miss) x (a - q_0).
Run: nice -n 15 ~/BigMoE/.venv/bin/python tree3_cov.py > tree3_cov.out"""
import numpy as np, glob, os
R = os.path.expanduser("~/TrainLab/data/tree3_map")
for dom in ("code", "ifeval", "multi"):
    T, D3, M3, DR, MR_ = [], [], [], [], []
    for f in sorted(glob.glob(f"{R}/{dom}/*.npz")):
        z = np.load(f)
        m = z["m"]; n = len(m)
        for i in range(n):
            k = int(m[i])
            T.append(z["true"][i, :k]); D3.append(z["dfl_t3"][i, :k]); M3.append(z["mtp_t3"][i, :k])
            DR.append(z["dfl_rank"][i, :k]); MR_.append(z["mtp_rank"][i, :k])
    # per depth arrays (ragged): collect by depth
    dep = {}
    for t, d3, m3, dr, mr in zip(T, D3, M3, DR, MR_):
        for j in range(len(t)):
            if t[j] < 0 or mr[j] < 0:
                continue
            dep.setdefault(j, []).append((int(d3[j, 0] == t[j]), int(t[j] in d3[j]), int(m3[j, 0] == t[j]), int(t[j] in m3[j]), int(t[j] in d3[j] or t[j] in m3[j])))
    print(f"==== {dom}: {sum(len(v) for v in dep.values())} (anchor, depth) pairs ====")
    print("  depth |   n   | dfl top1 | dfl top3 | dfl top3|miss | mtp top1 | mtp top3 | mtp top3|miss | union top3|both miss")
    allv = []
    for j in sorted(dep):
        v = np.array(dep[j]); allv.append(v)
        dm = v[:, 0] == 0; mm = v[:, 2] == 0; bm = dm & mm
        if j < 6 or j == 14:
            print("  %5d | %5d | %8.3f | %8.3f | %13.3f | %8.3f | %8.3f | %13.3f | %12.3f" % (
                j, len(v), v[:, 0].mean(), v[:, 1].mean(), v[dm, 1].mean() if dm.any() else float('nan'), v[:, 2].mean(), v[:, 3].mean(),
                v[mm, 3].mean() if mm.any() else float('nan'), v[bm, 4].mean() if bm.any() else float('nan')))
    v = np.concatenate(allv); dm = v[:, 0] == 0; mm = v[:, 2] == 0; bm = dm & mm
    print("  all   | %5d | %8.3f | %8.3f | %13.3f | %8.3f | %8.3f | %13.3f | %12.3f" % (
        len(v), v[:, 0].mean(), v[:, 1].mean(), v[dm, 1].mean(), v[:, 2].mean(), v[:, 3].mean(), v[mm, 3].mean(), v[bm, 4].mean()))
    v0 = np.array(dep[0]); q0 = v0[:, 2].mean(); pm = 1 - q0; cov = v0[v0[:, 2] == 0, 3].mean()
    print("  sibling fan at depth 0 on the MTP chain: P(miss at 0) %.3f x P(true in MTP top-3 | miss) %.3f = covered %.3f per cycle;"
          " extra kept per cycle = covered x (a - q_0 %.3f): a 0.98 %+.3f, 0.95 %+.3f, 0.90 %+.3f" % (pm, cov, pm * cov, q0, pm * cov * (0.98 - q0), pm * cov * (0.95 - q0), pm * cov * (0.90 - q0)))
    # the MTP chain's teacher-forced kept at 16 rows from these anchors (first miss + 1), for calibration
    kept = []
    for t, m3, mr in zip(T, M3, MR_):
        k = 0
        for j in range(len(t)):
            if t[j] < 0 or mr[j] < 0 or m3[j, 0] != t[j]:
                break
            k += 1
        kept.append(min(k, 15) + 1)
    print("  MTP chain kept at 16 rows from every anchor (teacher-forced, first miss + 1, capped): %.2f" % np.mean(kept))
