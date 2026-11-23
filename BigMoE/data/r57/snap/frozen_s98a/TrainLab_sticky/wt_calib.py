"""CPU: calibrate the HOBBIT-style weight threshold (arm cC_wT) against the rank rule (cC_rR) on Breakthrough's 125B
routing traces (~/Breakthrough/data/expA/traces125/<set>_800_g.npz: per layer the rank-ordered top-20 ids + softmax
gate probs; bulgarian / code 800 tokens, chat 559, math 767 despite the file names). The driver's own replay
(budget125_tf.cold_observe) runs on them: the first half seeds the LRU as one prefill call, then 4-token passes over the
second half (union per pass, as the verify passes of turn18c128). Per config: cold reads and 2-bit reads per token, SSD bytes vs v3, and a damage proxy, the
summed w^2 of the 2-bit uses per token (w = the row's top-10 weight normalised over its top-10, the budget's energy).
usage: wt_calib.py [C] [R] [T ...]   (T in thousandths)"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget125_tf as B  # noqa: E402

C = int(sys.argv[1]) if len(sys.argv) > 1 else 128
R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TS = [int(t) for t in sys.argv[3:]] or [60, 70, 80, 90, 100, 110]
T = os.path.expanduser("~/Breakthrough/data/expA/traces125/")
SETS = {os.path.basename(f).split("_")[0]: np.load(f) for f in sorted(glob.glob(T + "*_800_g.npz"))}


def run(ids, w, wt, r2):
    acc = dict(v3_bytes=0.0, b2_bytes=0.0, cold_reads=0, two_reads=0, rows=0)
    B.COLD.update(on=True, mode="single", C=C, r2=r2, wt=wt, lru={}, cur={}, cold={}, two={}, acc=acc)
    dmg, n, h = 0.0, 0, ids.shape[0] // 2
    for L in range(ids.shape[1]):
        B.cold_observe(L, ids[:h, L], w[:h, L])
    for t in range(h, ids.shape[0] - 3, 4):
        for L in range(ids.shape[1]):
            i_, w_ = ids[t:t + 4, L], w[t:t + 4, L]
            B.cold_observe(L, i_, w_)
            two = B.COLD["two"][L]
            dmg += sum(float(x) ** 2 for row, wr in zip(i_, w_) for e, x in zip(row, wr) if int(e) in two)
        n += 4
    return dict(cold=acc["cold_reads"] / n, two=acc["two_reads"] / n, v3=acc["v3_bytes"] / n / 1e6,
                arm=acc["b2_bytes"] / n / 1e6, dmg=dmg / n)


out = [f"wt_calib C={C}: rank rule r{R} vs weight thresholds (2-bit if cold and max top-10 weight in the pass < T)"]
for name, d in SETS.items():
    ids = np.stack([d[f"L{l}"] for l in range(48)], 1)[:, :, :10].astype(np.int32)
    g = np.stack([d[f"G{l}"] for l in range(48)], 1)[:, :, :10].astype(np.float32)
    w = g / g.sum(-1, keepdims=True)
    rows = [("r%d" % R, run(ids, w, None, R))] + [("w%d" % t, run(ids, w, t / 1000, 99)) for t in TS]
    for lab, x in rows:
        out.append(f"  {name:9s} {lab:5s} cold {x['cold']:5.1f}/tok, 2-bit {x['two']:5.1f} ({x['two'] / x['cold']:.0%}), "
                   f"SSD {x['v3']:.1f} -> {x['arm']:.1f} MB ({1 - x['arm'] / x['v3']:.1%} saved), 2-bit w^2 {x['dmg']:.3f}/tok")
print("\n".join(out))
open(os.path.expanduser(f"~/TrainLab/data/wt_calib_C{C}.txt"), "w").write("\n".join(out) + "\n")
