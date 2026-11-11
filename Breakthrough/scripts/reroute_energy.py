"""Gate-energy prediction of near-tie re-routing (NewIdeas' bNN: +delta on resident experts' router logits before the
top-10, true ranks 0-1 pinned, weights from the original probabilities over the true top-10 sum, every chosen expert read)
before its GPU probe (Breakthrough 2026-10-21 15:55). Experts are orthogonal with similar norms (energy_sim.py), so the
error of a swap is the displaced expert's w^2 plus the added expert's w^2. Replays the 125B traces (top-20 ids and gate
probabilities, data/expA/traces125/*_800_g.npz) with an LRU cache of C per layer, prints reads/token, swapped mass and
energy, and predicts teacher-forced KLD with the energy law fitted per kind of text on the measured mass-rule probes
(ifeval+multi: frugal/m50/m70/m85 rows; gsm+code: the _Sgc rows).   usage: reroute_energy.py [C ...]"""
import glob, os, sys
import numpy as np
from collections import OrderedDict

NL = 48
T = {f.split("/")[-1].split("_")[0]: np.load(f)
     for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz")))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in T.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64) for k, d in T.items()}


def replay(C, sets, delta=0.0, nw_rule=None):
    """delta >= 0: re-routing policy (all chosen read). nw_rule: mass-rule policy instead (wait ranks < nw, others used
    only if resident, no renorm). Returns reads/token, swapped-or-missed mass and energy (per layer, mean)."""
    reads = mass = en = 0.0; ntok = 0
    for k in sets:
        tr, g = ids[k], gp[k]
        ntok += tr.shape[0]
        for l in range(NL):
            c = OrderedDict()
            for t in range(tr.shape[0]):
                row, p = tr[t, l], g[t, l]
                w = p / p[:10].sum()
                res = np.array([int(e) in c for e in row])
                if nw_rule is not None:
                    nw = nw_rule(w[:10])
                    use = [r for r in range(10) if r < nw or res[r]]
                    miss = [r for r in range(10) if r >= nw and not res[r]]
                    mass += w[miss].sum(); en += (w[miss] ** 2).sum()
                else:
                    z = np.log(np.maximum(p, 1e-30)) + delta * res
                    z[:2] = np.inf                                  # true ranks 0-1 pinned
                    use = list(np.argsort(-z, kind="stable")[:10])
                    disp = [r for r in range(10) if r not in use]
                    add = [r for r in use if r >= 10]
                    mass += w[disp].sum(); en += (w[disp] ** 2).sum() + (w[add] ** 2).sum()
                for r in use:
                    e = int(row[r])
                    if e in c:
                        c.move_to_end(e)
                    else:
                        c[e] = 1; reads += 1
                while len(c) > C:
                    c.popitem(last=False)
    n = ntok
    return reads / n, mass / (n * NL), en / (n * NL)


def theta(th):
    return lambda ww: int(np.argmax(np.cumsum(ww) >= th - 1e-6)) + 1


HARD, EASY = ("chat", "bulgarian"), ("code", "math")
# measured teacher-forced KLD at C=51 (data/tfvar/results.txt, every 8th): hard = ifeval+multi, easy = gsm+code
MEAS = {"hard": {"exact": 0.0041, "r2": 0.1001, "m50": None, "m70": None, "m85": 0.0094},
        "easy": {"exact": None, "r2": 0.0273, "m50": 0.0114, "m70": 0.0045, "m85": 0.0021}}
RULES = {"r2": lambda ww: 2, "m50": theta(0.5), "m70": theta(0.7), "m85": theta(0.85), "exact": lambda ww: 10}
Cs = [int(a) for a in sys.argv[1:]] or [51, 77, 97]
print("fit at C=51: KLD = k0 + a * E^1.5 per kind of text (E = mean per-layer missed/swapped gate energy)")
law = {}
for kind, sets in (("hard", HARD), ("easy", EASY)):
    pts = []
    for pol, kl in MEAS[kind].items():
        if kl is None:
            continue
        r, m, e = replay(51, sets, nw_rule=RULES[pol])
        pts.append((pol, e, kl, r))
        print(f"  {kind:4s} {pol:5s} E {e:.5f}  mass {m:.4f}  reads {r:6.1f}  measured KLD {kl:.4f}")
    X = np.array([[1.0, p[1] ** 1.5] for p in pts]); y = np.array([p[2] for p in pts])
    (k0, a), *_ = np.linalg.lstsq(X, y, rcond=None)
    law[kind] = (k0, a)
    print(f"  {kind}: k0 {k0:.4f}  a {a:.2f}  residuals " + " ".join(f"{p[0]} {p[2] - (k0 + a * p[1] ** 1.5):+.4f}" for p in pts))
print("\nnear-tie re-routing (all chosen experts read, ranks 0-1 pinned):")
for C in Cs:
    for d in (0.0, 0.25, 0.5):
        out = []
        for kind, sets in (("hard", HARD), ("easy", EASY)):
            r, m, e = replay(C, sets, delta=d)
            k0, a = law[kind]
            out.append(f"{kind} reads {r:6.1f} swapped mass {m:.4f} E {e:.5f} -> KLD {k0 + a * e ** 1.5:.4f}")
        print(f"  C={C:3d} delta {d:4.2f}: " + " | ".join(out), flush=True)
