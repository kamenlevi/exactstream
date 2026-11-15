"""WEEK98 S6 CSM (MTP Head, pre-registered in REPORT): is t30b's depth-0 dip at the walk's check starts hardness (the
125B itself less sure there) or missing information (the drafter misses more at equal 125B confidence)?
Input: cells_t30b_heplus.npz (sticky/w98_heplus_cells.py; per cell: depth, right, m = the 125B's top-1 margin at the
predicted row, fi, an = anchor row; depth j of anchor an predicts row an + j + 1). Walk per window as the engine does
(walk_diag.py: start at the second depth-0 anchor; accept depths to the first miss; next check k + 1 anchors later).
Usage: python3 -I w98_cs_margin.py CELLS.npz OUT.json"""
import sys, json
import numpy as np

CELLS, OUT = sys.argv[1], sys.argv[2]
z = np.load(CELLS)
dep, right, m, fi = (np.asarray(z[k]) for k in ("depth", "right", "m", "fi"))
an_rows = np.asarray(z["row"]) - dep - 1          # dd_ceiling schema: depth j of anchor s predicts row s + j + 1
if "an" in z.files:
    an = np.asarray(z["an"]); assert (an == an_rows).all(), "anchor field disagrees with row - depth - 1"
else:
    an = an_rows                                    # e.g. BigMoE's cells_t30b_mbppbank.npz has fi + row, no an
BINS = [0, 0.5, 1, 2, 4, 8, np.inf]
rows = []                                   # per depth-0 anchor: (is_start, right, m, bonus_m)
kept = []
for f in np.unique(fi):
    s = fi == f
    a, d, r, mm = an[s].astype(np.int64), dep[s].astype(np.int64), right[s].astype(np.int8), m[s].astype(np.float64)
    lo = a.min()
    T = -np.ones((a.max() - lo + 1, 31), np.int8)
    T[a - lo, d] = r
    M0 = np.full(T.shape[0], np.nan)
    z0 = d == 0
    M0[a[z0] - lo] = mm[z0]
    p0 = a[z0].min() - lo + 1               # walk_diag's r0 = gen - 1 - a0 = 1 with a0 = gen - 2
    starts = np.zeros(T.shape[0], bool)
    kind = np.zeros(T.shape[0], np.int8)    # exploratory: 1 first start, 2 after a full accept, 3 after a miss
    p = p0; prev = 1
    while p < T.shape[0]:
        starts[p] = True; kind[p] = prev
        k = 0
        while k < 31 and T[p, k] == 1:
            k += 1
        kept.append(k + 1)
        prev = 2 if k == 31 else 3
        p += k + 1
    for q in range(p0, T.shape[0]):
        if T[q, 0] >= 0:
            rows.append((starts[q], T[q, 0], M0[q], M0[q - 1], f, kind[q], T[q - 1, 0]))
R = np.array(rows, np.float64)
st, hit, mg, bm, win = R[:, 0] == 1, R[:, 1], R[:, 2], R[:, 3], R[:, 4].astype(np.int64)
kd, prev0 = R[:, 5].astype(np.int64), R[:, 6]   # prev0: was the bonus row right at depth 0 (teacher-forced)?
b = np.digitize(mg, BINS[1:-1])
out = {"n_starts": int(st.sum()), "n_other": int((~st).sum()), "kept_per_check": float(np.mean(kept)),
       "acc_starts": float(hit[st].mean()), "acc_other": float(hit[~st].mean()),
       "median_m_starts": float(np.median(mg[st])), "median_m_other": float(np.median(mg[~st])), "bins": []}
w = np.array([(b[st] == i).mean() for i in range(len(BINS) - 1)])
acc_o = np.array([hit[~st & (b == i)].mean() if (~st & (b == i)).any() else np.nan for i in range(len(BINS) - 1)])
acc_s = np.array([hit[st & (b == i)].mean() if (st & (b == i)).any() else np.nan for i in range(len(BINS) - 1)])
for i in range(len(BINS) - 1):
    out["bins"].append({"m": f"{BINS[i]}-{BINS[i + 1]}", "share_starts": float(w[i]),
                        "share_other": float((b[~st] == i).mean()), "n_starts": int((st & (b == i)).sum()),
                        "acc_starts": float(acc_s[i]), "acc_other": float(acc_o[i])})
matched = float(np.nansum(w * acc_o) / np.sum(w[~np.isnan(acc_o)]))
out["acc_matched"] = matched
out["share_explained"] = (out["acc_other"] - matched) / (out["acc_other"] - out["acc_starts"])
# second cut: was the miss that ended the previous check a close call for the 125B (bonus row margin < 1 nat)?
for name, sel in (("bonus_m_lt1", st & (bm < 1)), ("bonus_m_ge1", st & (bm >= 1))):
    out[name] = {"n": int(sel.sum()), "acc": float(hit[sel].mean()), "median_m": float(np.median(mg[sel]))}


def share(idx):                             # share explained and residual (matched - starts) on a row subset
    s_, h_, b_ = st[idx], hit[idx], b[idx]
    w_ = np.array([(b_[s_] == i).mean() for i in range(len(BINS) - 1)])
    ao = np.array([h_[~s_ & (b_ == i)].mean() if (~s_ & (b_ == i)).any() else np.nan for i in range(len(BINS) - 1)])
    mt = np.nansum(w_ * ao) / np.sum(w_[~np.isnan(ao)])
    a_s, a_o = h_[s_].mean(), h_[~s_].mean()
    return (a_o - mt) / (a_o - a_s), mt - a_s


rng = np.random.default_rng(1115)
W = np.unique(win); byw = {x: np.flatnonzero(win == x) for x in W}
bs = np.array([share(np.concatenate([byw[x] for x in rng.choice(W, len(W))])) for _ in range(2000)])
out["boot_windows"] = {"n": 2000, "share_ci95": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                       "residual_ci95_pt": (100 * np.percentile(bs[:, 1], [2.5, 97.5])).tolist()}
json.dump(out, open(OUT, "w"), indent=1)
print(f"checks {len(kept)}, kept/check {out['kept_per_check']:.3f}; depth 0: starts {100 * out['acc_starts']:.2f} % "
      f"(n {out['n_starts']}) vs other {100 * out['acc_other']:.2f} % (n {out['n_other']}); median m {out['median_m_starts']:.2f}"
      f" vs {out['median_m_other']:.2f}")
for e in out["bins"]:
    print(f"  m {e['m']:>9}: starts {100 * e['share_starts']:5.1f} % of starts, acc {100 * e['acc_starts']:6.2f} % "
          f"(n {e['n_starts']}) | other {100 * e['share_other']:5.1f} %, acc {100 * e['acc_other']:6.2f} %")
print(f"margin-matched other {100 * matched:.2f} %; share of the gap explained by the 125B's margin "
      f"{100 * out['share_explained']:.1f} %")
for name in ("bonus_m_lt1", "bonus_m_ge1"):
    e = out[name]
    print(f"  starts of any kind with bonus-row margin {name[8:]}: n {e['n']}, depth 0 {100 * e['acc']:.2f} %, "
          f"median m {e['median_m']:.2f}")
e = out["boot_windows"]
print(f"window bootstrap (2000): share explained 95 % CI {100 * e['share_ci95'][0]:.1f}-{100 * e['share_ci95'][1]:.1f} %; "
      f"residual at equal margin {e['residual_ci95_pt'][0]:.2f}-{e['residual_ci95_pt'][1]:.2f} pt")

# EXPLORATORY (not pre-registered): which starts carry the dip, margin-matched against the non-start anchors
acc_o_bins = np.array([hit[~st & (b == i)].mean() for i in range(len(BINS) - 1)])
miss_lt2 = float(((1 - hit) * st * (mg < 2)).sum() / ((1 - hit) * st).sum())
out["exploratory"] = {"start_misses_on_m_lt2": miss_lt2}
print(f"EXPLORATORY: {100 * miss_lt2:.1f} % of the depth-0 misses at starts are on rows with m < 2 nats")
for name, sel in (("first", st & (kd == 1)), ("after_full", st & (kd == 2)), ("after_miss", st & (kd == 3)),
                  ("after_miss_bonus_right_d0", st & (kd == 3) & (prev0 == 1)),
                  ("after_miss_bonus_wrong_d0", st & (kd == 3) & (prev0 == 0))):
    if not sel.any():
        continue
    wv = np.array([(b[sel] == i).mean() for i in range(len(BINS) - 1)])
    mt = float(np.sum(wv * acc_o_bins))
    a = float(hit[sel].mean())
    out["exploratory"][name] = {"n": int(sel.sum()), "acc": a, "matched_other": mt, "residual_pt": 100 * (mt - a),
                                "m_lt2_share": float((mg[sel] < 2).mean())}
    print(f"  {name:>26}: n {int(sel.sum()):5d}, depth 0 {100 * a:6.2f} %, margin-matched other {100 * mt:6.2f} % "
          f"(residual {100 * (mt - a):5.2f} pt), m < 2 on {100 * (mg[sel] < 2).mean():5.1f} %")
json.dump(out, open(OUT, "w"), indent=1)
