"""res1c_masks.py (Fit12, Mon 16 Nov 2026 13:2x): the map-arm masks for "1c + RES" (1c-tiers + the resident-only MSB-3 rule),
in Breakthrough's tiers_1c format, so the decision map can run it as an arm (Breakthrough's harness; it copies them).
Desk, CPU, python -I.
- Start from ~/FableGod/decision_map/out/bt/tiers_1c/<w>.npz (keep = 4-bit tier, keep3 = MSB-3 tier; dropped = neither).
- In each 32-row answer block and layer, an MSB-3-tier expert with any slot non-resident in Breakthrough's res_c256
  masks (the C 256 base pool at its check's start) is dropped: its keep3 slots -> False.
- Prompt rows and the 4-bit tier are unchanged.
- Self-check: the masks' reach-weighted code E equals res1c_price.py's 1c + RES E (0.02088).
Output: /Users/kamen/Fit12/data/lossy32/masks_1c_res/<w>.npz (keep, keep3), for the 36 map windows (RES is meant for the
code sets; the chat windows get the same rule for completeness and are not part of the code-only claim).
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
OUT = "/Users/kamen/Fit12/data/lossy32/masks_1c_res"
ROWS, NL, S = 32, 48, 0.98
ERR = np.array([0.0, 0.08, 1.0])
os.makedirs(OUT, exist_ok=True)
base = json.load(open(os.path.join(M, "baseline.json")))
Ecode, RWc, ndrop, n3 = [], [], 0, 0
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f))
    ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
    m = ids.shape[0]
    t = np.load(os.path.join(M, "bt", "tiers_1c", f))
    keep, keep3 = t["keep"].astype(bool).copy(), t["keep3"].astype(bool).copy()
    res = np.load(os.path.join(M, "res_c256", f))["keep"].astype(bool)[a:a + m]
    for bi in range(m // ROWS):
        r0 = bi * ROWS
        for ly in range(NL):
            I = ids[r0:r0 + ROWS, ly]
            K3 = keep3[a + r0:a + r0 + ROWS, ly]
            Rs = res[r0:r0 + ROWS, ly]
            for e in np.unique(I[K3]):
                sel = I == e
                n3 += 1
                if not Rs[sel].all():
                    K3[sel] = False; ndrop += 1
            keep3[a + r0:a + r0 + ROWS, ly] = K3
        if base[f]["dom"] == "code":
            T = np.where(keep[a + r0:a + r0 + ROWS], 0, np.where(keep3[a + r0:a + r0 + ROWS], 1, 2))
            Ecode.extend(((ERR[T] * w[r0:r0 + ROWS] ** 2).sum((1, 2)) / NL).tolist())
            RWc.extend((S ** np.arange(ROWS)).tolist())
    np.savez_compressed(os.path.join(OUT, f), keep=keep, keep3=keep3)
Ecode, RWc = np.asarray(Ecode), np.asarray(RWc)
Er = float((Ecode * RWc).sum() / RWc.sum())
ok = abs(Er / 0.02088 - 1) <= 0.005
msg = (f"masks_1c_res: 36 windows written to {OUT}; MSB-3-tier experts dropped as non-resident {ndrop} of {n3} "
       f"({100 * ndrop / n3:.2f} %, all domains); SELF-CHECK code reach-weighted E {Er:.5f} vs res1c_price's 0.02088: "
       f"{'PASS' if ok else 'FAIL'}")
print(msg)
open("/Users/kamen/Fit12/data/lossy32/masks_1c_res.txt", "w").write(msg + "\n")
