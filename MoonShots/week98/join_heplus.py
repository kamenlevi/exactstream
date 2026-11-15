"""WEEK98 lever 1 v1: the HE+ CONFIRMATION of the frozen audit rule (MoonShots, Sun 15 Nov 2026 01:2x).
Frozen before this ran (LEVERS_A_MOONSHOTS.md): the 2B's token replaces the head's where they differ and the 2B's own top-1
margin >= tau(depth): 4 at depth 0, 2 at 1-2, 1 at 3-6, 0.5 at 7-30. LINE: CONFIRMED if HE+ depth 0 is within 0.3 pt of
t30b, AND 23-30 >= t30b + 3.0 pt, AND mean accepted at P = 1 >= t30b's x 1.05; else FAILS. No other schedule is tried.
Inputs (read-only): MTP Head's cells_t30b_heplus.npz (t30b's own chain on the 24 HE+ windows, bank context; explicit
fi / an / row), and rows2b_heplus.npz (the 2B's teacher-forced top-1 and margin per answer row).
usage: ~/BigMoE/.venv/bin/python -I join_heplus.py"""
import json, os
import numpy as np

H = os.path.expanduser
C = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus.npz"), allow_pickle=True)
R = np.load(H("~/MoonShots/week98/rows2b_heplus.npz"))
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
SCHED = {0: 4.0, 1: 2.0, 2: 2.0, **{j: 1.0 for j in range(3, 7)}, **{j: 0.5 for j in range(7, 31)}}

depth, right, mcell = C["depth"], C["right"].astype(bool), C["m"]
fi, an, row = C["fi"], C["an"], C["row"]
files = [os.path.basename(str(f)) for f in C["files"]]
key = {(os.path.basename(str(n)), int(r)): (bool(t == k), float(mg), float(m1))
       for n, r, t, k, mg, m1 in zip(R["fname"], R["row"], R["top1"], R["tk0"], R["margin2b"], R["margin125"])}
look = [key.get((files[int(f)], int(r))) for f, r in zip(fi, row)]
missing = sum(v is None for v in look)
assert missing == 0, f"{missing} cells have no 2B row"
r2b = np.array([v[0] for v in look]); m2b = np.array([v[1] for v in look], np.float32); m125 = np.array([v[2] for v in look], np.float32)
bad = int((~np.isclose(m125, mcell, atol=1e-4)).sum())
print(f"cells {len(depth)}; 2B rows joined; 125B margin fingerprint mismatches: {bad}")


def bands(ok):
    return {(f"{a}-{b}" if a != b else str(a)): round(100 * float(ok[(depth >= a) & (depth <= b)].mean()), 2) for a, b in BANDS}


def walk(ok_head, ok_comb, P):
    order = np.lexsort((depth, an, fi))
    f_, a_, h_, c_ = fi[order], an[order], ok_head[order], ok_comb[order]
    starts = np.r_[0, np.flatnonzero((np.diff(f_) != 0) | (np.diff(a_) != 0)) + 1, len(f_)]
    acc = []
    for s0, s1 in zip(starts[:-1], starts[1:]):
        n, used = 0, 0
        for j in range(s0, s1):
            if P is not None and used >= P:
                if h_[j]:
                    n += 1; continue
                break
            if h_[j] and c_[j]:
                n += 1
            elif not h_[j] and c_[j]:
                used += 1; n += 1
            else:
                break
        acc.append(n)
    return float(np.mean(acc))


tau = np.array([SCHED[int(d)] for d in depth], np.float32)
comb = np.where(m2b >= tau, r2b, right)
res = {"head": dict(bands=bands(right), accepted=walk(right, right, 0)),
       "frozen_schedule": dict(bands=bands(comb), acc_P1=walk(right, comb, 1), acc_P2=walk(right, comb, 2), acc_all=walk(right, comb, None))}
h, s = res["head"], res["frozen_schedule"]
d0 = s["bands"]["0"] - h["bands"]["0"]; d23 = s["bands"]["23-30"] - h["bands"]["23-30"]; ra = s["acc_P1"] / h["accepted"]
verdict = "CONFIRMED" if (abs(d0) <= 0.3 or d0 > 0) and d23 >= 3.0 and ra >= 1.05 else "FAILS"
res["line"] = dict(depth0_delta=round(d0, 2), d23_30_delta=round(d23, 2), accepted_ratio_P1=round(ra, 3), verdict=verdict)
print("head  :", h["bands"], f"accepted {h['accepted']:.2f}")
print("frozen:", s["bands"], f"accepted P1 {s['acc_P1']:.2f} P2 {s['acc_P2']:.2f} all {s['acc_all']:.2f}")
print("LINE:", res["line"])
json.dump(res, open(H("~/MoonShots/week98/join_heplus.json"), "w"), indent=1)
