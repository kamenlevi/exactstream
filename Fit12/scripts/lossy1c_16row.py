"""lossy1c_16row.py (Fit12, Mon 16 Nov 2026 08:4x): what the 1c-tiers rule does at the 16-row checks EvalSuite gates.
Desk, CPU, python -I. The rule is a copy of BigMoE's bigmoe/tiers1c.py tiers_np (read-only source, c1c217b):
I_e = sum_d s^d w(d,e)^2, 4-bit if > 2.775 lam, MSB-3 if > 0.8457 lam, else drop (s 0.98, lam 0.016).
Inputs: the decision map's 12 code windows (~/FableGod/decision_map/out/xcap: answer rows' top-10 ids + normalised
weights). Cross-check: at 32-row blocks it must equal Breakthrough's tiers_1c masks bit for bit.
Output: per check-layer means (union, 4-bit with the LSB plane, MSB-3, dropped) and expert GB per check vs REF2-T, at 16
and 32 rows -> /Users/kamen/Fit12/data/lossy32/lossy1c_16row.txt (the counter sanity reference for BigMoE / EvalSuite).
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
T4_K, T3_K, S, LAM = 2.775, 0.8457, 0.98, 0.016
E4, E3, NL = 2.135e-3, 2.135e-3 * 0.778, 48


def tiers_np(ids, w, s=S, lam=LAM):
    ids = np.asarray(ids, np.int64); w = np.asarray(w, np.float64)
    rw = s ** np.arange(ids.shape[0])
    u, inv = np.unique(ids.ravel(), return_inverse=True)
    imp = np.bincount(inv, weights=(w ** 2 * rw[:, None]).ravel(), minlength=len(u))
    tier = np.where(imp > T4_K * lam, 0, np.where(imp > T3_K * lam, 1, 2)).astype(np.int8)
    return tier[inv].reshape(ids.shape)


base = json.load(open(os.path.join(M, "baseline.json")))
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")
out, mism, nslots = [], 0, 0
for R in (16, 32):
    acc = {k: [] for k in ("union", "ref_lsb", "lsb4", "e4", "e3", "edrop", "rows_dropped_slots")}
    for f in code:
        z = np.load(os.path.join(M, "xcap", f))
        ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
        if R == 32:
            t = np.load(os.path.join(M, "bt", "tiers_1c", f))
            k4 = t["keep"].astype(bool)[a:a + ids.shape[0]]
            k3 = t["keep3"].astype(bool)[a:a + ids.shape[0]]
        rank = np.arange(10)[None, :]
        for bi in range(ids.shape[0] // R):
            for ly in range(NL):
                I, W = ids[bi * R:(bi + 1) * R, ly], w[bi * R:(bi + 1) * R, ly]
                T = tiers_np(I, W)
                if R == 32:                                   # cross-check against Breakthrough's masks
                    ref = np.where(k4[bi * R:(bi + 1) * R, ly], 0, np.where(k3[bi * R:(bi + 1) * R, ly], 1, 2))
                    mism += int((ref != T).sum()); nslots += T.size
                rk = np.broadcast_to(rank, I.shape)
                u = np.unique(I)
                e_t = {}                                      # per distinct expert: its tier (same for every use)
                for e, tt in zip(I.ravel(), T.ravel()):
                    e_t[int(e)] = int(tt)
                acc["union"].append(len(u))
                acc["ref_lsb"].append(len(np.unique(I[rk < 2])))
                acc["lsb4"].append(len(np.unique(I[(T == 0) & (rk < 2)])))
                acc["e4"].append(sum(1 for v in e_t.values() if v == 0))
                acc["e3"].append(sum(1 for v in e_t.values() if v == 1))
                acc["edrop"].append(sum(1 for v in e_t.values() if v == 2))
                acc["rows_dropped_slots"].append(float((T == 2).mean()))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    kept = m["e4"] + m["e3"]
    ref_gb = NL * (m["ref_lsb"] * E4 + (m["union"] - m["ref_lsb"]) * E3)
    c1_gb = NL * (m["lsb4"] * E4 + (kept - m["lsb4"]) * E3)
    out.append(f"{R}-row checks (code, {len(acc['union']) // NL} blocks): per check-layer union {m['union']:.1f} = "
               f"e4 {m['e4']:.1f} + e3 {m['e3']:.1f} + edrop {m['edrop']:.1f} (dropped {100 * m['edrop'] / m['union']:.1f} % "
               f"of the union, {100 * m['rows_dropped_slots']:.1f} % of the row-rank slots); with the LSB plane: REF2-T "
               f"{m['ref_lsb']:.1f}, 1c {m['lsb4']:.1f}; expert GB per check REF2-T {ref_gb:.2f} -> 1c {c1_gb:.2f} "
               f"(x{c1_gb / ref_gb:.3f}); per check e3 + edrop = {NL * (m['e3'] + m['edrop']):.0f}")
out.append(f"cross-check at 32 rows: tiers_np vs Breakthrough's tiers_1c masks: {mism} of {nslots} slots differ "
           f"({'PASS' if mism == 0 else 'FAIL'})")
os.makedirs("/Users/kamen/Fit12/data/lossy32", exist_ok=True)
open("/Users/kamen/Fit12/data/lossy32/lossy1c_16row.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
