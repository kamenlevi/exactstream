"""A1 on the MTP head: greedy vs lossless-sampling acceptance at depth 1, turn17's exact arm (top-16 target + head)."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
from a1_ceiling import proc, overlap, CONFIGS
z = np.load(sys.argv[1])
ti, tl = z["exact__tgt_top_ids"], z["exact__tgt_top_lp"]
mi, ml = z["exact__mtp_top_ids"], z["exact__mtp_top_lp"]
ok = np.isfinite(tl[:, 0].astype(np.float32)) & np.isfinite(ml[:, 0].astype(np.float32))
ti, tl, mi, ml = ti[ok], tl[ok], mi[ok], ml[ok]
p1 = np.exp(tl[:, 0].astype(np.float32)); fork = p1 < 0.7
print(f"n {len(ti)}, forks {fork.mean():.3f}, head p1 mean {np.exp(ml[:, 0].astype(np.float32)).mean():.3f}")
for name, cfg in CONFIGS:
    if cfg is None:
        a = (mi[:, 0] == ti[:, 0]).astype(float)
    else:
        T, k, tp = cfg
        k = min(k, 16)
        a = overlap(ti[:, :k], proc(tl, T, k, tp), mi[:, :k], proc(ml, T, k, tp))
    print(f"  {name:36s} {a.mean():.4f}  forks {a[fork].mean() if fork.any() else float('nan'):.3f}  non-forks {a[~fork].mean():.4f}")
