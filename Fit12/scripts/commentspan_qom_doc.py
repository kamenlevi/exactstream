"""commentspan_qom.py (Fit12, Mon 16 Nov 2026 13:3x; POST-HOC extension of the pre-registered COMMENT-EXACT price, labelled so):
(c'') measured per-row flips BY TAG for the map arms qom_0_live / qom_0.5_live (QOM at the 45 GB pool), t4_all (the
bit-plane path's own noise) and t3_all (MSB-3 everywhere), next to bt_1c; (b'') bytes / misses when comment + DOCSTRING rows run
QOM's cf-0 keep instead of exact (ranks 0-1 always read; ranks 2-9 read only if resident), with and without RES on the
other rows. Same inputs and tagging as commentspan_price.py (imported)."""
import json, os, sys
import numpy as np
sys.argv = ["x"]
here = "/Users/kamen/Fit12/scripts"
src = open(os.path.join(here, "commentspan_price.py")).read()
src = src[:src.index("# ---------------------------------------------------------------- (a) tagged shares")]
exec(compile(src, "commentspan_price_head", "exec"))
base = json.load(open(os.path.join(M, "baseline.json")))
TIE = {}
for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))):
    TIE[(r["nm"], int(r["i"]))] = r["cls"] == "tie"
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")
arms = {a_: json.load(open(os.path.join(M, a_ + ".json"))) for a_ in ("bt_1c", "qom_0_live", "qom_0.5_live", "t4_all", "t3_all")}
rows, acc = [], {R: {k: [] for k in ("qc_gb", "qc_m", "qcr_gb", "qcr_m")} for R in (16, 32)}
for f in code:
    z = np.load(os.path.join(M, "xcap", f)); ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
    m = ids.shape[0]; tok = np.array(base[f]["tk"])[:, 0]; tg = tags(tok)
    tie = np.array([TIE.get((f, a + j), False) for j in range(m)])
    fl = {a_: (np.array(d[f]["tk"])[:, 0] != tok) for a_, d in arms.items()}
    for j in range(m):
        rows.append((tg[j, 0], tg[j, 1], tie[j]) + tuple(bool(fl[a_][j]) for a_ in arms))
    res = np.load(os.path.join(M, "res_c256", f))["keep"].astype(bool)[a:a + m]
    t1 = np.load(os.path.join(M, "bt", "tiers_1c", f))
    tier32 = np.where(t1["keep"][a:a + m], 0, np.where(t1["keep3"][a:a + m], 1, 2))
    for R in (16, 32):
        rk = np.broadcast_to(np.arange(10), (R, 10))
        for bi in range(m // R):
            sl = slice(bi * R, (bi + 1) * R); X = np.broadcast_to((tg[sl, 0] | tg[sl, 1])[:, None], (R, 10))
            c = dict(qc_gb=0.0, qc_m=0, qcr_gb=0.0, qcr_m=0)
            for ly in range(NL):
                I, W, Rs = ids[sl, ly], w[sl, ly], res[sl, ly]
                T = tier32[sl, ly] if R == 32 else tiers_np(I, W)
                for e in np.unique(I):
                    se = I == e; nres = not Rs[se].all(); te = int(T[se].min())
                    req_x = bool((se & X & (rk < 2)).any())                 # a comment row's rank 0-1 use: required
                    opt_x = bool((se & X & (rk >= 2)).any())                # a comment row's rank 2-9 use: only if resident
                    used_nc = te != 2 and bool((se & ~X).any())
                    lsb = req_x or (te == 0 and bool((se & ~X & (rk < 2)).any()))
                    read = req_x or used_nc or (opt_x and not nres)
                    if read:
                        c["qc_gb"] += E4 if lsb else E3; c["qc_m"] += nres
                    # + RES on the other rows: their MSB-3-tier uses are optional too
                    read_r = req_x or (opt_x and not nres) or (used_nc and (te == 0 or not nres))
                    if read_r:
                        c["qcr_gb"] += E4 if lsb else E3; c["qcr_m"] += nres
            for k_ in c:
                acc[R][k_].append(c[k_])
rows = np.array(rows, bool); com, doc, tie = rows[:, 0], rows[:, 1], rows[:, 2]
L2 = ["# commentspan_qom (Fit12, post-hoc): measured map flips per row by tag, for each arm (12 code windows)",
      "| arm | all rows | '#' comment rows | docstring rows | other rows | non-tie: all / '#' comment |", "|---|---|---|---|---|---|"]
for i, a_ in enumerate(arms):
    fl = rows[:, 3 + i]
    q = lambda s: f"{fl[s].sum()} ({100 * fl[s].mean():.2f} %)"
    L2.append(f"| {a_} | {q(np.ones(len(rows), bool))} | {q(com)} | {q(doc)} | {q(~(com | doc))} | {fl[~tie].sum()} / {fl[~tie & com].sum()} |")
L2.append("")
L2.append("bytes / misses per check with '#' comment rows on QOM cf 0 (ranks 0-1 read, 2-9 only if resident) instead of exact:")
for R in (16, 32):
    A = {k: float(np.mean(v)) for k, v in acc[R].items()}
    L2.append(f"  {R} rows: 1c + comment-QOM {A['qc_gb']:.2f} GB / {A['qc_gb'] * MS_GB[R]:.1f} ms, misses {A['qc_m']:.1f} "
              f"({A['qc_m'] * MS_MISS:.1f} ms); + RES {A['qcr_gb']:.2f} GB / {A['qcr_gb'] * MS_GB[R]:.1f} ms, misses {A['qcr_m']:.1f} "
              f"({A['qcr_m'] * MS_MISS:.1f} ms)")
    acc[R] = A
FASTd = {"C3": (27.0, 28.6), "C5": (23.8, 26.6)}
for cn, (dl, dh) in FASTd.items():
    for lab, gk, mk in (("1c + comment-QOM", "qc_gb", "qc_m"), ("1c + comment-QOM + RES", "qcr_gb", "qcr_m")):
        e = acc[32][gk] * MS_GB[32]
        lo = max(dl + e + 1.8, 38.5) + acc[32][mk] * MS_MISS; hi = max(dh + e + 3.4, 38.5) + acc[32][mk] * MS_MISS
        L2.append(f"  {cn} {lab}: {lo:.1f}-{hi:.1f} ms per 32-row check -> {1e3 * KEPT / (hi + OVH):.0f}-{1e3 * KEPT / (lo + OVH):.0f} tok/s HE+ at K*")
print("\n".join(L2))
open("/Users/kamen/Fit12/data/lossy32/commentspan_qom_doc.txt", "w").write("\n".join(L2) + "\n")
