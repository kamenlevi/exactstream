"""R-25 DYNK (MoonShots Tue 20:1x; NON-exact, bytes side only; quality would be TrainLab's): instead of fixed top-8 (S9,
fails strict), drop a rank-8 / rank-9 expert (0-indexed, of the top-10) only when its top-10-renormalized gate weight is
below tau. wall24 LFU replay (16-row checks accepting 10, tokens 400+) at the exact stack (C_m 159 / C_l 44, W24-ec + 11-bit
at load) and the v3 split; per tau: share of rank 8-9 uses dropped, mean dropped gate mass per token-layer, bytes/token.
PASS (written first): some tau keeps >= 60 % of fixed top-8's byte cut while dropping <= 40 % of top-8's gate mass (then a
TrainLab strict / KLD screen at that tau); KILL: every tau reaching 50 % of the cut drops >= 70 % of the mass.
usage: [BigMoE venv] taskpolicy -b python dynk24.py -> stdout + data/dynk24.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

IDS0 = {k: v.copy() for k, v in W.IDS.items()}
WT = {}
for k, d in W.SETS.items():
    g = np.stack([d[f"G{l}"] for l in range(W.NL)], 1)[:, :, :10].astype(np.float64)
    WT[k] = g / g.sum(-1, keepdims=True)


def needs(ids, R, A):                                    # wall24.needs, skipping dropped (-1) entries
    n, out, pos = ids.shape[0], [], 0
    while pos + R <= n:
        lay = []
        for l in range(W.NL):
            best = {}
            for r in range(R):
                for k, e in enumerate(ids[pos + r, l]):
                    e = int(e)
                    if e >= 0 and (e not in best or k < best[e]):
                        best[e] = k
            lay.append((frozenset(best), frozenset(e for e, k in best.items() if k < 2)))
        out.append((W.MEAS[0] <= pos < W.MEAS[1], lay))
        pos += A
    return out


W.needs = needs
lines = []
for nm, cm, cl, msb, lsb in (("exact stack", 159, 44, 1900544, 720896),):
    W.MSB, W.LSB = msb, lsb
    res = {}
    for tau in (None, 0.04, 0.05, 0.06, 0.07, 0.08, 1.0):  # None = top-10, 1.0 = fixed top-8
        drop_n, drop_m, tot = 0, 0.0, 0
        for k in W.IDS:
            ids = IDS0[k].copy()
            if tau is not None:
                d = np.zeros(ids.shape, bool)
                d[:, :, 8:] = WT[k][:, :, 8:] < tau
                ids[d] = -1
                drop_n += int(d.sum()); drop_m += float((WT[k] * d).sum()); tot += ids.shape[0] * ids.shape[1]
            W.IDS[k] = ids
        W._NEEDS.clear()
        b = W.bytes_per_token("lfu", Cm=cm, Cl=cl)[0]
        res[tau] = (b, drop_n / max(tot, 1) / 2, drop_m / max(tot, 1))
    b10 = res[None][0]
    cut8, mass8 = b10 - res[1.0][0], res[1.0][2]
    for tau, (b, share, mass) in res.items():
        s = (f"{nm}: {'top-10' if tau is None else ('fixed top-8' if tau == 1.0 else f'tau {tau:.2f}')}: {b / 1e6:.1f} MB / token "
             f"({100 * (b / b10 - 1):+.1f} %); rank 8-9 uses dropped {100 * share:.1f} %, gate mass dropped {100 * mass:.2f} % per "
             f"token-layer" + ("" if tau in (None, 1.0) else f" | {100 * (b10 - b) / cut8:.0f} % of top-8's cut for {100 * mass / mass8:.0f} % of its mass"))
        print(s, flush=True); lines.append(s)
open(os.path.expanduser("~/MoonShots/data/dynk24.txt"), "w").write("\n".join(lines) + "\n")
