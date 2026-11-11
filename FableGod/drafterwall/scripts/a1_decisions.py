#!/usr/bin/env python3
"""a1_decisions.py (drafterwall, DESCRIPTIVE, written after T1-T5; no PASS / KILL line). CPU.
(A) where does a check end? class of the 125B's row at each chain's first miss, and the 125B's close calls ('decisions':
    mid + tie rows) passed per check, on the 12 map windows per domain (tree3_map, official loop at 16 rows).
(B) what is lag worth, split by class: a chain whose SURE rows are drafted at depth-0 quality and whose decision rows
    stay at chain quality, and the reverse.
(C) the lag-2-everywhere drafter at 16 / 32 / 64 rows on the unseen d23h tables (MTP b5; 24 HumanEval+, 100 MBPP)."""
import numpy as np, os, glob
TD = os.path.expanduser("~/TrainLab/data"); out = []
def say(s=""):
    print(s); out.append(s)
def cls_of(p3):
    c = np.ones(p3.shape[:-1], dtype=np.int8); c[p3[..., 0] >= 0.99] = 0; c[(p3[..., 0] - p3[..., 1]) <= 0.1] = 2
    return c
say("## (A) + (B): the 12 map windows per domain, 16-row checks, the official loop on the teacher-forced tables")
for dom in ("code", "ifeval", "multi"):
    S = {r: dict(kept=[], dec=[], end=[0, 0, 0, 0]) for r in ("dfl", "mtp")}
    B = {k: [] for k in ("mtp chain", "sure rows at lag 2", "decision rows at lag 2", "all rows at lag 2")}
    share = [0, 0, 0]
    for f in sorted(glob.glob(f"{TD}/tree3_map/{dom}/*.npz")):
        z = np.load(f); true = z["true"]; n = true.shape[0]; mm = z["m"]; ok = true >= 0
        cl = cls_of(z["t125_p3"])                              # class of the row predicting (A, depth j)
        c0 = cl[:, 0]                                          # class of each position's own row
        for c in (0, 1, 2): share[c] += int((c0 == c).sum())
        R = {"dfl": (z["dfl_t3"][..., 0] == true) & ok, "mtp": (z["mtp_t3"][..., 0] == true) & ok}
        for r, tab in R.items():
            p = 0
            while p < n:
                m = min(15, int(mm[p])); k = 0
                while k < m and tab[p, k]: k += 1
                S[r]["kept"].append(k + 1)
                if k < m:
                    S[r]["end"][int(cl[p, k])] += 1
                    S[r]["dec"].append(int((cl[p, :k + 1] > 0).sum()))        # decisions among the kept tokens (the bonus row included)
                else:
                    S[r]["end"][3] += 1; S[r]["dec"].append(int((cl[p, :k] > 0).sum()))
                p += k + 1
        m_chain = R["mtp"]; m0 = R["mtp"][:, 0]
        def lag_tab(p, j, mode):
            q = p + j                                          # the anchor whose depth 0 predicts the same position
            if q >= n: return False
            is_dec = cl[p, j] > 0
            if mode == "all": return bool(m0[q])
            if mode == "sure": return bool(m0[q]) if not is_dec else bool(m_chain[p, j])
            if mode == "dec": return bool(m0[q]) if is_dec else bool(m_chain[p, j])
            return bool(m_chain[p, j])
        for name, mode in (("mtp chain", "chain"), ("sure rows at lag 2", "sure"), ("decision rows at lag 2", "dec"), ("all rows at lag 2", "all")):
            p = 0
            while p < n:
                m = min(15, int(mm[p])); k = 0
                while k < m and lag_tab(p, k, mode): k += 1
                B[name].append(k + 1); p += k + 1
    tot = sum(share)
    say(f"\n{dom}: rows sure / mid / tie {100 * share[0] / tot:.1f} / {100 * share[1] / tot:.1f} / {100 * share[2] / tot:.1f} %  (decision density rho = {(share[1] + share[2]) / tot:.3f})")
    for r in ("dfl", "mtp"):
        e = S[r]["end"]; ne = sum(e)
        say(f"   {r}: kept {np.mean(S[r]['kept']):.2f}; checks end at a sure / mid / tie row / block end: {100 * e[0] / ne:.0f} / {100 * e[1] / ne:.0f} / {100 * e[2] / ne:.0f} / {100 * e[3] / ne:.0f} %; "
            f"the 125B's close calls passed per check {np.mean(S[r]['dec']):.2f}  (kept x rho = {np.mean(S[r]['kept']) * (share[1] + share[2]) / tot:.2f})")
    say("   (B) 16-row checks, MTP: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in B.items()))
say("\n## (C) lag-2-everywhere drafter at wider checks (unseen d23h tables, MTP b5)")
for st in ("code", "mbpp"):
    z = np.load(f"{TD}/d23h/{st}_b5.npz"); res = {}
    for cap in (15, 31, 63, 127):
        real, lag = [], []
        for i in range(len(z["keys"])):
            rt = z[f"rt_{i}"].astype(bool); r0 = int(z["gen"][i]) - 1 - int(z["a0"][i]); sub = rt[r0:]; m = sub.shape[0]; d0 = sub[:, 0]
            p = 0
            while p < m:
                ok = d0[p:p + cap]; k = 0
                while k < len(ok) and ok[k]: k += 1
                lag.append(k + 1); p += k + 1
            if cap <= 31:
                p = 0
                while p < m:
                    ok = sub[p, :min(cap, m - p)]; k = 0
                    while k < len(ok) and ok[k]: k += 1
                    real.append(k + 1); p += k + 1
        res[cap] = (np.mean(real) if real else float("nan"), np.mean(lag))
    say(f"{st}: rows 16 / 32 / 64 / 128: real MTP chain {res[15][0]:.2f} / {res[31][0]:.2f} / - / -; lag-2 everywhere {res[15][1]:.2f} / {res[31][1]:.2f} / {res[63][1]:.2f} / {res[127][1]:.2f}")
open(os.path.join(os.path.dirname(__file__), "..", "out", "a1_decisions.txt"), "w").write("\n".join(out) + "\n")
