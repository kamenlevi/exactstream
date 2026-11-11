"""Rate-distortion-optimal coverage for frugal routing (Breakthrough, 2026-10-21 06:15). 125B traces with router probs,
LRU cache of C whole 4-bit experts per layer. Per token-layer, ranks 0-1 are always computed at 4 bits (read if
missing); each missing rank r >= 2 (top-10 weight w) takes the option minimising  err + lam * MB:
    drop        err w^2              0 MB
    MSB-2 read  err NMSE2 * w^2      1.536 MB   (not cached)
    MSB-3 read  err NMSE3 * w^2      2.150 MB   (not cached)
    4-bit read  err 0                2.765 MB   (enters the cache)
    [optional]  trellis 2-bit / 1-bit copies (NMSE 0.071 / 0.27 at ~1.30 / ~0.70 MB incl. scales), --trellis
Missed energy E = mean over token-layers of the summed err; the probes fit KLD_excess ~ 0.0051 * (E / 0.0032)^1.5 on
the hard subset (ifeval+multi; exact floor 0.0043), validated by m85_nr_t2 (predicted 0.0152, measured 0.0151).
Prints the frontier (MB/token vs predicted KLD) for each lam, plus the simple theta rules for comparison.
usage: rd_sim.py [C] [--trellis]"""
import glob, os, sys
import numpy as np
from collections import OrderedDict

NL = 48
NMSE2, NMSE3 = 0.10, 0.02
OPTS = [("drop", 1.0, 0.0), ("msb2", NMSE2, 1.536), ("msb3", NMSE3, 2.150), ("full", 0.0, 2.765)]
if "--trellis" in sys.argv:
    OPTS += [("tr2", 0.071, 1.30), ("tr1", 0.27, 0.70)]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
C = int(args[0]) if args else 51
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz")))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}


def kld(E):
    return 0.0043 + 0.0051 * (E / 0.0032) ** 1.5


def run(choose):
    """choose(r, w) -> option index; returns E per token-layer, MB per token, option counts per token"""
    E = mb = 0.0; ntl = ntok = 0; cnt = np.zeros(len(OPTS))
    for k in ids:
        tr, wt = ids[k], w[k]
        ntok += tr.shape[0]
        for l in range(NL):
            c = OrderedDict()
            for t in range(tr.shape[0]):
                ntl += 1
                for r in range(10):
                    e, ww = int(tr[t, l, r]), float(wt[t, l, r])
                    if e in c:
                        c.move_to_end(e); continue
                    o = 3 if r < 2 else choose(r, ww)
                    name, err, size = OPTS[o]
                    E += err * ww * ww; mb += size; cnt[o] += 1
                    if name == "full":
                        c[e] = 1
                while len(c) > C:
                    c.popitem(last=False)
    return E / ntl, mb / ntok, cnt / ntok


print(f"C={C}: options {[o[0] for o in OPTS]}; predicted KLD on the hard subset from E")
for lam in (1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 8e-3):
    def choose(r, ww, lam=lam):
        return int(np.argmin([err * ww * ww + lam * size for _, err, size in OPTS]))
    E, mb, cnt = run(choose)
    print(f"  RD lam {lam:.0e}: {mb:6.1f} MB/token, E {E:.4f} -> KLD ~{kld(E):.4f} | per token " +
          " ".join(f"{OPTS[i][0]} {cnt[i]:.1f}" for i in range(len(OPTS))), flush=True)
for th in (0.5, 0.7, 0.85):            # the theta rule with full reads, for comparison (cumulative mass needs the row)
    E = mb = 0.0; ntl = ntok = 0
    for k in ids:
        tr, wt = ids[k], w[k]; ntok += tr.shape[0]
        cw = np.cumsum(wt, -1)
        for l in range(NL):
            c = OrderedDict()
            for t in range(tr.shape[0]):
                ntl += 1
                nw = int(np.argmax(cw[t, l] >= th - 1e-6)) + 1
                for r in range(10):
                    e, ww = int(tr[t, l, r]), float(wt[t, l, r])
                    if e in c: c.move_to_end(e)
                    elif r < nw: c[e] = 1; mb += 2.765
                    else: E += ww * ww
                while len(c) > C: c.popitem(last=False)
    print(f"  theta {th:.2f} (4-bit reads): {mb / ntok:6.1f} MB/token, E {E / ntl:.4f} -> KLD ~{kld(E / ntl):.4f}", flush=True)
