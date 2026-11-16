"""THINK_1116c (Fable C, Mon 16 Nov 2026, 18:4x): is the head's DRIFT persistent within a sequence? Premise check for an
online per-sequence state correction (data assimilation). t30b cells on HE+ 24 / MBPP 100 (BigMoE ceiling dumps), all
anchors, marginal (teacher-forced), depths 3-8 = drift-dominated bands; non-tie rows only (the 125B's margin m >= 2 nats)
so ties do not masquerade as drift. Measures: (1) between-window spread of the drift gap (depth-0 accuracy minus depth
3-8 accuracy) against its binomial noise; (2) miss persistence: P(miss at depth d, anchor a | miss at depth d, anchor
a - lag) / P(miss at depth d), lags 1..32 (lag >= 8 = a different block). CPU, seconds, read-only.
usage: ~/BigMoE/.venv/bin/python -I -B drift_persist_1116c.py
"""
import os, json
import numpy as np
H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/fablec/drift_persist_1116c.out"
DEP = 31
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

for tag in ("heplus24", "mbppbank"):
    z = np.load(f"{CE}/cells_t30b_{tag}.npz")
    dep, right, m, fi, row = (z[k] for k in ("depth", "right", "m", "fi", "row"))
    say(f"== {tag}: {len(dep)} cells, {len(set(fi.tolist()))} windows")
    gaps = []; noise = []; pers = {lag: [0, 0] for lag in (1, 2, 4, 8, 16, 32)}; base = [0, 0]
    pers_all = {lag: [0, 0] for lag in (1, 2, 4, 8, 16, 32)}; base_all = [0, 0]
    for f_ in sorted(set(fi.tolist())):
        w = fi == f_
        d0 = (dep[w] == 0)
        anchors = row[w][d0] - 1
        order = np.argsort(anchors); anchors = anchors[order]
        n0 = len(anchors)
        # table [anchor, depth]: right (1/0/-1), margin
        R = np.full((n0, DEP), -1, np.int8); M = np.full((n0, DEP), np.nan, np.float32)
        aidx = {int(a): i for i, a in enumerate(anchors)}
        for d in range(DEP):
            md = w & (dep == d)
            rr = row[md] - 1 - d
            for a_, r_, m_ in zip(rr, right[md], m[md]):
                i = aidx.get(int(a_))
                if i is not None: R[i, d] = int(r_); M[i, d] = m_
        sure = M >= 2.0
        v0 = (R[:, 0] >= 0) & sure[:, 0]
        vd = (R[:, 3:9] >= 0) & sure[:, 3:9]
        if v0.sum() < 20 or vd.sum() < 60: continue
        a0 = (R[:, 0][v0] == 1).mean(); ad = (R[:, 3:9][vd] == 1).mean()
        gaps.append(a0 - ad); noise.append(a0 * (1 - a0) / v0.sum() + ad * (1 - ad) / vd.sum())
        miss = (R == 0) & sure; valid = (R >= 0) & sure
        for d in range(3, 9):
            base[0] += int(miss[:, d].sum()); base[1] += int(valid[:, d].sum())
            base_all[0] += int((R[:, d] == 0).sum()); base_all[1] += int((R[:, d] >= 0).sum())
            for lag in pers:
                if n0 <= lag: continue
                cur, prev = miss[lag:, d], miss[:-lag, d]
                vv = valid[lag:, d] & valid[:-lag, d]
                pers[lag][0] += int((cur & prev & vv).sum()); pers[lag][1] += int((prev & vv).sum())
                ca, pa = (R[lag:, d] == 0), (R[:-lag, d] == 0); va = (R[lag:, d] >= 0) & (R[:-lag, d] >= 0)
                pers_all[lag][0] += int((ca & pa & va).sum()); pers_all[lag][1] += int((pa & va).sum())
    gaps = np.array(gaps); noise = np.array(noise)
    var_obs = gaps.var(ddof=1); var_noise = noise.mean()
    say(f"  drift gap (depth-0 acc - depth 3-8 acc, non-tie rows), per window: mean {100 * gaps.mean():.2f} pt, sd {100 * gaps.std(ddof=1):.2f} pt, "
        f"binomial sd {100 * np.sqrt(var_noise):.2f} pt -> persistent (between-window) share of the variance {max(0.0, 1 - var_noise / var_obs):.2f} (n windows {len(gaps)})")
    pb = base[0] / max(base[1], 1); pba = base_all[0] / max(base_all[1], 1)
    say(f"  non-tie miss rate at depths 3-8: {100 * pb:.2f} % (all rows {100 * pba:.2f} %)")
    for lag in pers:
        p = pers[lag][0] / max(pers[lag][1], 1); pa_ = pers_all[lag][0] / max(pers_all[lag][1], 1)
        say(f"  P(non-tie miss at depth d, anchor a | non-tie miss at d, anchor a - {lag:2d}) = {100 * p:.2f} %  lift x{p / pb:.2f} (n {pers[lag][1]});  all rows lift x{pa_ / pba:.2f}")
open(OUT, "w").write("\n".join(lines) + "\n")
