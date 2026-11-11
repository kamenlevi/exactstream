"""Missed gate ENERGY under frugal policies (Breakthrough, 2026-10-21 night). The 125B's routed experts give mutually
orthogonal outputs (sim_table.py: best |cos| median 0.024) with similar norms within a layer (CV 0.15), so the error a
missing expert leaves is ~ w_r^2 ||y||^2 and errors of different experts add: the natural predictor of KLD is the missed
gate energy E = sum over missing ranks of w_r^2 (top-10 renormalised weights), and for an expert read at b bits instead
of dropped, NMSE_b * w_r^2. This replays the 125B traces (data/expA/traces125/*_800_g.npz) with an LRU cache of C
experts per layer for the probe policies and prints mean missed mass, missed energy and loads/token, plus per 8-layer
band. Fit KLD ~ a * energy on the probes afterwards (fit_energy in this file's tail).   usage: energy_sim.py [C]"""
import glob, os, sys
import numpy as np
from collections import OrderedDict

NL = 48
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz")))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}


def run(C, nw_rule, band=None):
    """nw_rule(wt_row) -> ranks waited; band=(a, b): policy only there, exact elsewhere. Returns per-token means."""
    mass = np.zeros(NL); en = np.zeros(NL); loads = 0; ntok = 0
    for k in ids:
        tr, wt = ids[k], w[k]
        ntok += tr.shape[0]
        for l in range(NL):
            c = OrderedDict()
            inband = band is None or band[0] <= l < band[1]
            for t in range(tr.shape[0]):
                row, ww = tr[t, l], wt[t, l]
                nw = nw_rule(ww) if inband else 10
                for r in range(10):
                    e = int(row[r])
                    if e in c:
                        c.move_to_end(e)
                    elif r < nw:
                        c[e] = 1; loads += 1
                    else:
                        mass[l] += ww[r]; en[l] += ww[r] ** 2
                while len(c) > C:
                    c.popitem(last=False)
    return mass / ntok, en / ntok, loads / ntok


def theta(th):
    return lambda ww: int(np.argmax(np.cumsum(ww) >= th - 1e-6)) + 1


C = int(sys.argv[1]) if len(sys.argv) > 1 else 51
pol = {"r2 (base/noren)": lambda ww: 2, "w4": lambda ww: 4, "m50": theta(0.5), "m70": theta(0.7), "m85": theta(0.85)}
print(f"C={C}: per token (48 layers): missed mass / layer-mean, missed energy sum w^2 / layer-mean, loads")
for name, f in pol.items():
    m, e, ld = run(C, f)
    print(f"  {name:16s} mass {m.mean():.3f}  energy {e.mean():.4f}  loads {ld:6.1f}   per band energy: "
          + " ".join(f"{e[a:a + 8].mean():.4f}" for a in range(0, 48, 8)), flush=True)
