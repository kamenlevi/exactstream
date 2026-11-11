"""Exploratory pricing (not pre-registered; MTP Head's SAMELAYER-T was BETWEEN, so its exposed3 pricing did not run):
forecast-missed demand at C 256 (M-25b joint sizes, held out) with the untrained vs the ridge-trained same-layer forecast.
Recalls (MTP Head 09:48): untrained 0.706 code / 0.821 chat; trained 0.823 / 0.851, layers 40-47 0.854 / 0.892 (layers
1-39 take the overall value; layer 0 has no forecast). Exposed ms = MB x kept per check / 12.2 GB/s x 0.56 (napkin)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import fcastsize as F
import laysz45 as LZ
import sidecars45 as SC

J = np.load(os.path.join(LZ.OUT, "joint45_curves.npz"))
names = [str(x) for x in J["names"]]
dom = np.array([w.split("_")[1] for w in names])
CAPm, CAPl, Kc, H = J["cap_m"], J["cap_l"], J["cap_kept"], list(LZ.H)
Wm, Wl, Wk = J["win_m"], J["win_l"], J["win_kept"]
folds = ([], [])
for d in ("code", "ifeval", "multi"):
    ix = [i for i in range(len(names)) if dom[i] == d]
    folds[0].extend(ix[0::2]); folds[1].extend(ix[1::2])
um, ul = SC.UNI[256]
budget = LZ.NL * (um * LZ.MR + ul * LZ.LR)


def fit(win):
    k = Kc[H].sum() + Wk[win].sum()
    cm = (CAPm[:, :, H].sum(2) + Wm[:, :, win].sum(2)) * LZ.MR / k
    cl = (CAPl[:, :, H].sum(2) + Wl[:, :, win].sum(2)) * LZ.LR / k
    return tuple(SC.alloc([cm, cl], [SC.GM, SC.GL], [LZ.MR, LZ.LR], budget, [160, 32]))


fs = [fit(folds[f]) for f in (0, 1)]


def rvec(all_, late):
    r = np.full(LZ.NL, all_); r[0] = 0.0; r[40:] = late
    return r


R = {"untrained": {"code": rvec(0.706, 0.706), "chat": rvec(0.821, 0.821)},
     "trained": {"code": rvec(0.823, 0.854), "chat": rvec(0.851, 0.892)}}
out = {}
for name, rr in R.items():
    for d in ("code", "ifeval", "multi"):
        w = 1 - rr["code" if d == "code" else "chat"]
        tot = kp = 0.0
        for f in (0, 1):
            te = [i for i in folds[1 - f] if dom[i] == d]
            im, jl = fs[f]
            tot += sum(w[l] * (Wm[l, im[l], te].sum() * LZ.MR + Wl[l, jl[l], te].sum() * LZ.LR) for l in range(LZ.NL))
            kp += Wk[te].sum()
        out[(name, d)] = tot / kp / 1e6
lines = ["== SAMELAYER-T exploratory pricing (Wed; MoonShots; not pre-registered): forecast-missed demand MB per kept token at "
         "C 256 (M-25b joint sizes, held out), untrained -> trained same-layer forecast"]
for d in ("code", "ifeval", "multi"):
    a, b = out[("untrained", d)], out[("trained", d)]
    ms = (a - b) * F.KPC[d] / F.GBS * F.EXPOSED
    lines.append(f"  {d:6s}: {a:5.2f} -> {b:5.2f} MB per kept token ({100 * (b / a - 1):+.0f} %); exposed ~-{ms:.2f} ms per check "
                 f"if the forecast's reads hide fully (kept / check {F.KPC[d]})")
print("\n".join(lines))
open(os.path.expanduser("~/MoonShots/data/fcastsize.txt"), "a").write("\n".join(lines) + "\n")
