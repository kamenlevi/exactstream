#!/usr/bin/env python3
"""POST-HOC, information only (not part of the pre-registered verdict of rd_sim_sub1.py, which is KILL at the Q4-like
line). Breakthrough, Wed 11 Nov 2026 18:2x. The same model (rd_sim.py / rd_sim_sub1.py, C = 51, energy law), but the
other way round: at a fixed read budget of 150 MB/token (the 90 tok/s line), which predicted KLD does each mix reach, and
how much does the 0.5-bit copy lower it? This matters if the 12 GB route is gated on benchmarks (the map) at a looser
line than KLD 0.0096. lam by bisection so that MB/token = 150 (+-0.3 %).
Output: ~/Breakthrough/data/m1/rd_sim_sub1_at150.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B rd_sim_sub1_at150.py
"""
import glob
import os
from collections import OrderedDict

import numpy as np

NL, C = 48, 51
BASE = [("drop", 1.0, 0.0), ("msb2", 0.10, 1.536), ("msb3", 0.02, 2.150), ("full", 0.0, 2.765)]
TREL = [("tr2", 0.071, 1.30), ("tr1", 0.27, 0.70)]
BUDGET = 150.0
files = sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz")))
sets = {f.split("/")[-1].split("_")[0]: np.load(f) for f in files}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
W64 = {k: (g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True)).astype(np.float64) for k, g in gp.items()}
IDL = {k: [ids[k][:, l, :10].tolist() for l in range(NL)] for k in ids}
NTOK = sum(v.shape[0] for v in ids.values())
out = []


def say(s=""):
    print(s, flush=True)
    out.append(s)


def kld(E):
    return 0.0043 + 0.0051 * (E / 0.0032) ** 1.5


def run(opts, lam):      # = rd_sim_sub1.run (ranks 0-1 at 4 bits)
    err = np.array([o[1] for o in opts]); size = np.array([o[2] for o in opts])
    full = [o[0] for o in opts].index("full")
    E = mb = 0.0
    cnt = np.zeros(len(opts))
    for k in ids:
        Wk = W64[k]
        opt = np.stack([(e * Wk) * Wk + lam * s for e, s in zip(err, size)], -1).argmin(-1)
        opt[:, :, :2] = full
        isfull = opt == full
        miss = np.empty(opt.shape, bool)
        T = opt.shape[0]
        for l in range(NL):
            c = OrderedDict(); mte, pop = c.move_to_end, c.popitem
            rows, frows = IDL[k][l], isfull[:, l, :].tolist()
            ml = []; app = ml.append
            for t in range(T):
                row, frow = rows[t], frows[t]
                for r in range(10):
                    e = row[r]
                    if e in c:
                        mte(e); app(False)
                    else:
                        app(True)
                        if frow[r]:
                            c[e] = 1
                while len(c) > C:
                    pop(last=False)
            miss[:, l, :] = np.array(ml, bool).reshape(T, 10)
        o = opt[miss]
        E += float(((err[o] * Wk[miss]) * Wk[miss]).sum())
        mb += float(size[o].sum())
        cnt += np.bincount(o, minlength=len(opts))
    return E / (NTOK * NL), mb / NTOK, cnt / NTOK


def at_budget(opts):
    lo, hi = 1e-4, 5e-2                      # MB falls as lam grows
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        E, mb, cnt = run(opts, mid)
        if abs(mb / BUDGET - 1) < 0.003:
            break
        if mb > BUDGET:
            lo = mid
        else:
            hi = mid
    return mid, E, mb, cnt


say(f"POST-HOC (information only): predicted KLD at a fixed {BUDGET:.0f} MB/token, C = {C}, the energy law of 21 Oct")
say("| mix | MB/token | lam | E | predicted KLD | reads per token by option |")
say("|---|---|---|---|---|---|")
MIXES = [("A: drop / MSB-2 / MSB-3 / 4-bit", BASE),
         ("B: A + trellis", BASE + TREL),
         ("C: A + q05 NMSE 0.50", BASE + [("q05", 0.50, 0.39)]),
         ("D: B + q05 NMSE 0.54", BASE + TREL + [("q05", 0.54, 0.39)]),
         ("D: B + q05 NMSE 0.50", BASE + TREL + [("q05", 0.50, 0.39)]),
         ("D: B + q05 NMSE 0.50 at 0.31 MB", BASE + TREL + [("q05", 0.50, 0.31)])]
res = {}
for name, opts in MIXES:
    lam, E, mb, cnt = at_budget(opts)
    res[name] = kld(E)
    say(f"| {name} | {mb:.1f} | {lam:.2e} | {E:.5f} | {kld(E):.4f} | "
        + " ".join(f"{n} {c:.1f}" for (n, _, _), c in zip(opts, cnt) if c >= 0.05) + " |")
b, d = res["B: A + trellis"], res["D: B + q05 NMSE 0.50"]
say(f"At {BUDGET:.0f} MB/token the 0.5-bit copy (NMSE 0.50) moves the predicted KLD {b:.4f} -> {d:.4f} ({d / b - 1:+.1%}; "
    f"excess over the exact floor 0.0043: {(d - 0.0043) / (b - 0.0043) - 1:+.1%}). The Q4-like line is 0.0096.")
open(os.path.expanduser("~/Breakthrough/data/m1/rd_sim_sub1_at150.out"), "w").write("\n".join(out) + "\n")
