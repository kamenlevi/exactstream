"""res1c_16.py (Fit12, Mon 16 Nov 2026 13:1x; POST-HOC extension of the pre-registered RES-1c price, labelled so in PLAN).
Misses per 16-row check at the C 256 base pool on the map's 12 code windows, by keep rule (desk, CPU, python -I):
  exact      every union expert is read;
  QOM cf 0   ranks 0-1 always read, a non-resident expert used only at ranks 2-9 is dropped (QOM with every row sure:
             a LOWER bound on lossy-45's QOM misses; the engine gates QOM by the drafter's confidence and the bonus row);
  1c         every kept expert (4-bit or MSB-3 tier) is read (1c REPLACES QOM on the code sets);
  1c + RES   only the 4-bit tier may miss.
The rule is BigMoE's tiers1c.tiers_np (copied), on 16-row blocks; residency = Breakthrough's res_c256 masks (resident at
each 16-row live check's start). Measured anchor: lossy-45's real 16-row HE checks, median 35 misses (rlh_256).
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
T4_K, T3_K, S, LAM, NL = 2.775, 0.8457, 0.98, 0.016, 48


def tiers_np(ids, w, s=S, lam=LAM):
    rw = s ** np.arange(ids.shape[0])
    u, inv = np.unique(ids.ravel(), return_inverse=True)
    imp = np.bincount(inv, weights=(w ** 2 * rw[:, None]).ravel(), minlength=len(u))
    return np.where(imp > T4_K * lam, 0, np.where(imp > T3_K * lam, 1, 2))[inv].reshape(ids.shape)


base = json.load(open(os.path.join(M, "baseline.json")))
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")
L = []
for R in (16, 32):
    cnt = {k: [] for k in ("exact", "qom_cf0", "1c", "1c_res")}
    for f in code:
        z = np.load(os.path.join(M, "xcap", f))
        ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
        res = np.load(os.path.join(M, "res_c256", f))["keep"].astype(bool)[a:a + ids.shape[0]]
        rank = np.broadcast_to(np.arange(10), (R, 10))
        for bi in range(ids.shape[0] // R):
            sl = slice(bi * R, (bi + 1) * R)
            c = dict(exact=0, qom_cf0=0, c1=0, res=0)
            for ly in range(NL):
                I, W, Rs = ids[sl, ly], w[sl, ly], res[sl, ly]
                T = tiers_np(I, W)
                for e in np.unique(I):
                    sel = I == e
                    if Rs[sel].all():
                        continue                          # resident at every check start in the block: no miss
                    c["exact"] += 1
                    c["qom_cf0"] += bool((rank[sel] < 2).any())
                    te = int(T[sel].min())
                    c["c1"] += te in (0, 1)
                    c["res"] += te == 0
            cnt["exact"].append(c["exact"]); cnt["qom_cf0"].append(c["qom_cf0"]); cnt["1c"].append(c["c1"])
            cnt["1c_res"].append(c["res"])
    m = {k: (float(np.mean(v)), float(np.median(v))) for k, v in cnt.items()}
    L.append(f"{R}-row checks ({len(cnt['exact'])} blocks), misses per check (mean / median): exact {m['exact'][0]:.1f} / "
             f"{m['exact'][1]:.0f}; QOM cf 0 (lower bound of QOM) {m['qom_cf0'][0]:.1f} / {m['qom_cf0'][1]:.0f}; 1c "
             f"{m['1c'][0]:.1f} / {m['1c'][1]:.0f}; 1c + RES {m['1c_res'][0]:.1f} / {m['1c_res'][1]:.0f}")
L.append("anchor [K]: lossy-45 (QOM, real decoding, C 256) 16-row HE checks: median 35 misses (rlh_256)")
os.makedirs("/Users/kamen/Fit12/data/lossy32", exist_ok=True)
open("/Users/kamen/Fit12/data/lossy32/res1c_16.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
