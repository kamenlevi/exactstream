"""SCREEN 7 READER (NewIdeas, Wed 4 Nov 2026 19:4x; CPU only, read-only on the map outputs). The reading pre-registered
in QUEUE.md (14:3x amendment + 14:5x amendment 2), mechanical, so s7_trl's verdict is one command when it lands.
Per domain, per row KL(REF2-T || arm) from map_kl.rows_kl; paired differences use the rows both arms have.
  gate: KL_cal - KL_ref >= 2 x its paired SE (per-row SE, as pre-registered; the window-cluster SE is shown beside it),
        else "no estimate" on that domain;
  scale = (KL_cal_all - KL_ref) / (KL_cal - KL_ref);
  est_abs_TRL = KL_ref + (KL_trl - KL_ref) x scale  (conservative; the verdict if s7_ref3 is missing);
  in-frame est = KL_ref + (est_abs_TRL - KL_ref3), the verdict;
  flips = the pre-registered fit a + b sqrt(KL); lines PASS <= 1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above.
Also prints E29 = KL_ref3 - KL_ref (the ranks 2-9 MSB-3 term at 239 slots) with its paired SEs.
Added 20:0x (MoonShots cd7e1f0; information, not pre-registered): the strict view, TRL as an independent quantizer
disagreeing with REF2-T's 3-bit ranks 2-9: strict = in-frame + 2 x E29 = est_abs + E29.
Usage: python s7_read.py   (writes ~/NewIdeas/data/ledger/s7_read.txt)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_kl as M  # noqa: E402
DOMS = ("code", "ifeval", "multi")
FIT = {"code": (-0.078, 15.68), "ifeval": (0.420, 39.04), "multi": (-0.173, 36.17)}
LINE = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}
DST = os.path.expanduser("~/NewIdeas/data/ledger/s7_read.txt")


def per_window(name):
    p = os.path.join(M.OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p))
    return {nm: (b["dom"], np.array(M.rows_kl(b, R[nm]), float)) for nm, b in M.BASE.items() if nm in R}


def paired(A, B, col=0):
    """mean of A - B per domain over rows both have, the per-row SE and the window-cluster SE."""
    out = {}
    for d in DOMS:
        ds, sums, ns = [], [], []
        for nm in A:
            if nm not in B or A[nm][0] != d: continue
            x = A[nm][1][:, col] - B[nm][1][:, col]; x = x[~np.isnan(x)]
            if len(x): ds.append(x); sums.append(x.sum()); ns.append(len(x))
        x = np.concatenate(ds); n = len(x); m = x.mean()
        se_row = x.std(ddof=1) / n ** 0.5
        g = len(ns); res = np.array(sums) - m * np.array(ns)
        se_win = (g / (g - 1) * (res ** 2).sum()) ** 0.5 / n
        out[d] = (m, se_row, se_win, n)
    return out


def mean_kl(A):
    return {d: float(np.nanmean(np.concatenate([v[1][:, 0] for v in A.values() if v[0] == d]))) for d in DOMS}


def flips(d, kl):
    a, b = FIT[d]; return a + b * max(kl, 0) ** 0.5


def verdict(d, f):
    return "PASS" if f <= LINE[d] else ("BETWEEN" if f <= 1.5 * LINE[d] else "KILL")


def main():
    A = {n: per_window(n) for n in ("s7_ref", "s7_ref3", "s7_cal", "s7_cal_all", "s7_trl")}
    L = ["SCREEN 7 READ (QUEUE 14:3x + amendment 2), KL(REF2-T || arm) mean per row; SE = paired per-row / per-window"]
    K = {n: mean_kl(v) for n, v in A.items() if v}
    L.append("| arm | " + " | ".join(DOMS) + " |"); L.append("|---|---|---|---|")
    for n in A:
        L.append(f"| {n} | " + (" | ".join(f"{K[n][d]:.4f}" for d in DOMS) if n in K else "missing | | ") + " |")
    P = {}
    for a, b, lab in (("s7_ref3", "s7_ref", "E29 = ref3 - ref (ranks 2-9 MSB-3 term, 239 slots)"),
                      ("s7_cal", "s7_ref", "cal - ref (S = 4 layers at MSB-2)"),
                      ("s7_cal_all", "s7_ref", "cal_all - ref (48 layers at MSB-2)"),
                      ("s7_trl", "s7_ref", "trl - ref (S = 4 layers from the TRL mix)")):
        if A[a] is None or A[b] is None: L.append(f"{lab}: missing"); continue
        P[a] = paired(A[a], A[b])
        L.append(f"{lab}: " + "; ".join(f"{d} {P[a][d][0]:+.5f} (SE {P[a][d][1]:.5f} / {P[a][d][2]:.5f}, "
                                          f"z {P[a][d][0] / P[a][d][1]:.1f} / {P[a][d][0] / P[a][d][2]:.1f})" for d in DOMS))
    if "s7_cal" in P and "s7_cal_all" in P:
        gate = {d: P["s7_cal"][d][0] >= 2 * P["s7_cal"][d][1] for d in DOMS}
        gate_w = {d: P["s7_cal"][d][0] >= 2 * P["s7_cal"][d][2] for d in DOMS}
        scale = {d: P["s7_cal_all"][d][0] / P["s7_cal"][d][0] for d in DOMS}
        L.append("gate (cal - ref >= 2 x paired SE): " + "; ".join(
            f"{d} {'OK' if gate[d] else 'NO ESTIMATE'} (window SE: {'OK' if gate_w[d] else 'fails'})" for d in DOMS))
        L.append("scale (cal_all - ref) / (cal - ref): " + " / ".join(f"{scale[d]:.1f}" for d in DOMS) + " (linear 48 / 4 = 12)")
        if "s7_trl" in P:
            L += ["| domain | trl - ref | est_abs_TRL KL | flips | in-frame est KL | flips | verdict (in-frame) | strict KL (= est_abs + E29) | flips | strict verdict | raw s7_trl flips |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for d in DOMS:
                if not gate[d]: L.append(f"| {d} | {P['s7_trl'][d][0]:+.5f} | no estimate | | | | | |"); continue
                ea = K["s7_ref"][d] + P["s7_trl"][d][0] * scale[d]
                ef = K["s7_ref"][d] + (ea - K["s7_ref3"][d]) if "s7_ref3" in K else ea
                raw = 100 * np.nanmean(np.concatenate([v[1][:, 2] for v in A["s7_trl"].values() if v[0] == d]))
                es = ea + (K["s7_ref3"][d] - K["s7_ref"][d]) if "s7_ref3" in K else ea
                L.append(f"| {d} | {P['s7_trl'][d][0]:+.5f} | {ea:.4f} | {flips(d, ea):.2f} ({verdict(d, flips(d, ea))}) | "
                         f"{ef:.4f} | {flips(d, ef):.2f} | {verdict(d, flips(d, ef))} | {es:.4f} | {flips(d, es):.2f} | "
                         f"{verdict(d, flips(d, es))} | {raw:.2f} |")
        else:
            L.append("s7_trl: not landed yet; the estimate table prints when it does")
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    open(DST, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
