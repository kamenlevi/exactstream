"""earlydeath.py (Fable B, THINK_1115 idea 1, re-priced on Leap's MEASURED unions): bytes and byte-floor ms of a K-node
exact check with dead subtrees dropped at layer L, against the one-pass check. CPU, tiny.
Unions per layer [K Leap treek_price.md]: 73.2 / 109.7 / 135.5 / 156.5 / 187.4 / 211.1 at 16 / 32 / 48 / 64 / 96 / 128
consecutive rows (a tree of K nodes routes like K consecutive rows; siblings may widen it a little). Expert 2.14 MB,
dense 3.42 GB, 280 GB/s [P from the 16 / 32-row floors]. Survivors after the drop: S rows (the kept path + the siblings
that die only at their own depth), union u(S) by interpolation. kept per check [K S11, HE+]."""
import numpy as np
K_, U_ = [16, 32, 48, 64, 96, 128], [73.2, 109.7, 135.5, 156.5, 187.4, 211.1]
u = lambda k: float(np.interp(k, K_, U_))
E_, DENSE, BW = 2.14e-3, 3.42, 280.0
kept = {16: 11.26, 32: 16.26, 48: 17.18, 64: 17.91, 96: 18.86, 128: 19.14}
def gb(K, L, S):
    return DENSE + (L * u(K) + (48 - L) * u(S)) * E_
print("one-pass byte floor: " + ", ".join("K%d %.1f GB %.0f ms (%.3f kept/ms)" % (K, gb(K, 48, K), gb(K, 48, K) / BW * 1e3, kept[K] / (gb(K, 48, K) / BW * 1e3)) for K in K_))
print("\nearly death: dead subtrees leave the batch after layer L; S survivors; repair pass = a check of S rows (%.0f ms at S 24)" % (gb(24, 48, 24) / BW * 1e3))
print("   K    L    S   GB   byte ms  kept/ms  vs K32 one-pass  break-even false-drop rate")
for K in (64, 128):
    for L in (8, 16, 24):
        for S in (24, 32):
            g = gb(K, L, S); ms = g / BW * 1e3; kpm = kept[K] / ms
            rep = gb(S, 48, S) / BW * 1e3
            ref = gb(32, 48, 32) / BW * 1e3 / kept[32]              # ms per kept token, K32 one-pass
            f = (ref * kept[K] - ms) / rep                          # repairs per check the gain can absorb
            print("%4d  %3d  %3d  %5.1f  %7.1f  %7.3f  %14s  %s" % (K, L, S, g, ms, kpm, "x%.2f" % (kpm / (kept[32] / (gb(32, 48, 32) / BW * 1e3))), "%.0f %%" % (100 * f) if f > 0 else "none"))
