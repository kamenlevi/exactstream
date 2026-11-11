"""Paired comparison of frugal KLD probes (data/tfvar/<VAR>_e<EVERY>.npz from tf_frugal_var.py): every variant scores the
same positions in the same order, so differences vs a baseline are paired per position. Prints mean KLD per set and
overall, top-1 agreement, and the bootstrap 95 % interval (over sequences' positions, 2,000 resamples of positions)
of the KLD difference to the baseline.   usage: tfvar_compare.py [EVERY] [BASELINE]   (default 8 base)"""
import glob, os, sys
import numpy as np

EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = sys.argv[2] if len(sys.argv) > 2 else "base"
D = os.path.expanduser("~/Breakthrough/data/tfvar")
runs = {}
for f in sorted(glob.glob(f"{D}/*_e{EVERY}.npz")):
    v = os.path.basename(f)[: -len(f"_e{EVERY}.npz")]
    z = np.load(f)
    sets = sorted(k[4:] for k in z.files if k.startswith("kld_"))
    runs[v] = {s: (z[f"kld_{s}"], z[f"agree_{s}"]) for s in sets}
if BASE not in runs:
    sys.exit(f"no {BASE}_e{EVERY}.npz yet; have {sorted(runs)}")
rng = np.random.default_rng(0)
b = np.concatenate([runs[BASE][s][0] for s in sorted(runs[BASE])])
print(f"every {EVERY}: {len(b)} positions; baseline {BASE} KLD {b.mean():.4f}")
for v, r in sorted(runs.items(), key=lambda kv: np.concatenate([kv[1][s][0] for s in kv[1]]).mean()):
    k = np.concatenate([r[s][0] for s in sorted(r)])
    a = np.concatenate([r[s][1] for s in sorted(r)])
    if len(k) != len(b):
        print(f"  {v:12s} position count differs ({len(k)} vs {len(b)}): skipped"); continue
    d = k - b
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per = " ".join(f"{s.replace('tf_', '')} {r[s][0].mean():.4f}" for s in sorted(r))
    print(f"  {v:12s} KLD {k.mean():.4f} ({per}) top-1 {a.mean():.3f} | vs {BASE} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
