#!/usr/bin/env python3
"""a4_lag2_dflash_all.py (drafterwall, DESCRIPTIVE; no line). DFlash2 L1-20 at EVERY position of the D-17 sets
(~/TrainLab/data/d02b_walk_d17e_<set>.npz: one 16-row check per answer position, k accepted): its own chain in the
official loop, and the lag-2-everywhere drafter (right at a position exactly when DFlash2 is right there at depth 0),
for the domains the MTP tables do not cover (agent, GSM, MGSM)."""
import numpy as np, os
TD = os.path.expanduser("~/TrainLab/data"); out = []
for st in ("agent", "code", "gsm", "mgsm", "ifeval", "multi"):
    z = np.load(f"{TD}/d02b_walk_d17e_{st}.npz"); names = list(z["names"])
    chain = []; lag = {15: [], 31: []}; d0 = [0, 0]; L = []
    for i in range(len(names)):
        ch = z[f"{i}_checks"]; K = {int(A): int(k) for k, m, A in ch if int(m) >= 1}
        A0 = min(K); An = max(K); L.append(An - A0 + 1)
        ok = np.array([K.get(A, 0) >= 1 for A in range(A0, An + 1)])
        d0[0] += int(ok.sum()); d0[1] += len(ok)
        A = A0
        while A in K:
            chain.append(K[A] + 1); A += K[A] + 1
        for cap in lag:
            p = 0; n = len(ok)
            while p < n:
                k = 0
                while k < cap and p + k < n and ok[p + k]: k += 1
                lag[cap].append(k + 1); p += k + 1
    s = (f"{st}: {len(names)} windows, mean answer length {np.mean(L):.0f} tokens; DFlash2 right at depth 0 {100 * d0[0] / d0[1]:.1f} %; "
         f"its chain keeps {np.mean(chain):.2f} per 16-row check; lag-2 everywhere {np.mean(lag[15]):.2f} (16 rows) / {np.mean(lag[31]):.2f} (32 rows)")
    print(s); out.append(s)
open(os.path.join(os.path.dirname(__file__), "..", "out", "a4_lag2_dflash_all.txt"), "w").write("\n".join(out) + "\n")
