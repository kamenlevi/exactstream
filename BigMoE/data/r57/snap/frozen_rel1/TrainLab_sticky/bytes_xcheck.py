"""CPU: cross-check of the driver's 3B / cold-only byte replays (budget125_tf._observe_3b / cold_observe) against
MoonShots' budget24.py on the same traces (Breakthrough's traces125: bulgarian / code 800 tokens, chat 559, math 767 despite
the file names): the first half warms the pools (one prefill call), the second half is measured in passes of R tokens
(R = 1: MoonShots' per-token replay). usage: bytes_xcheck.py [R]"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget125_tf as B  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
T = os.path.expanduser("~/Breakthrough/data/expA/traces125/")
out = [f"bytes_xcheck R={R}: MB per token (v3 147 + 46 reference -> arm)"]
tot = {}
for f in sorted(glob.glob(T + "*_800_g.npz")):
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(48)], 1)[:, :, :10].astype(np.int32)
    name = os.path.basename(f).split("_")[0]
    n, h = ids.shape[0], ids.shape[0] // 2
    for arm, upd in (("3b140t32", dict(mode="3b", C=140, C2=32, r2=4, wt=None)),
                     ("3b163t0", dict(mode="3b", C=163, C2=0, r2=99, wt=None))):
        acc = dict(v3_bytes=0.0, b2_bytes=0.0, cold_reads=0, two_reads=0, rows=0)
        B.COLD.update(on=True, lru={}, lru2={}, lv3={}, lsd={}, cur={}, cold={}, two={}, acc=acc, **upd)
        for L in range(48):
            B.cold_observe(L, ids[:h, L])
        nt = 0
        for t in range(h, n - R + 1, R):
            for L in range(48):
                B.cold_observe(L, ids[t:t + R, L])
            nt += R
        v3, a = acc["v3_bytes"] / nt / 1e6, acc["b2_bytes"] / nt / 1e6
        tot.setdefault(arm, []).append((v3, a))
        out.append(f"  {name:9s} {arm:8s}: v3 {v3:6.1f} -> {a:6.1f} MB ({1 - a / v3:+.1%} saved), 2-bit reads {acc['two_reads'] / nt:.1f}/tok ({nt} tok)")
for arm, xs in tot.items():
    v3, a = np.mean([x[0] for x in xs]), np.mean([x[1] for x in xs])
    out.append(f"  mean of sets {arm:8s}: v3 {v3:.1f} -> {a:.1f} MB ({1 - a / v3:.1%} saved)")
print("\n".join(out))
open(os.path.expanduser(f"~/TrainLab/data/bytes_xcheck_R{R}.txt"), "w").write("\n".join(out) + "\n")
