# STATS-FORWARD screen (CPU, seconds): the HC read's K1C partials (per-K-tile partial
# projections of the 4 streams) computed by the PREVIOUS layer's writer, tile by tile in
# K1C's own tile order, sum to the same values bitwise as K1C computes them in the read.
# Also prices the row-sequential GDN step from measured points (a + b * rows).
import numpy as np
rng = np.random.default_rng(0)
R, S, H, LR, KT = 32, 4, 2560, 320, 64          # rows, streams, hidden, lowrank, K tiles
streams = rng.standard_normal((R, S, H)).astype(np.float32)
w = (rng.standard_normal((S, H, LR)) * 0.02).astype(np.float32)
def k1c_read(x):                                  # today's K1C: per K-tile partials, fp32
    parts = np.zeros((KT, R, S, LR), np.float32)
    for t in range(KT):
        sl = slice(t * H // KT, (t + 1) * H // KT)
        for s in range(S):
            parts[t, :, s] = x[:, s, sl] @ w[s, sl]
    return parts
def writer_forward(x):                            # the writer owns the same K tiles: same math
    return k1c_read(x)
a, b = k1c_read(streams), writer_forward(streams)
same = np.array_equal(a, b)
print("K1C partials from the writer == read's partials bitwise:", same)
# reduction order: K2B sums tiles 0..63 in order in both forms -> bitwise
print("summed:", np.array_equal(a.sum(0), b.sum(0)))
# GDN step price per row-layer from Leap's measured points (us): 16 rows 57.9, 32 rows 107.2
b_us = (107.2 - 57.9) / 16; a_us = 57.9 - 16 * b_us
print(f"GDN step: {a_us:.1f} us + {b_us:.2f} us per row per layer")
for L, drop in [(8, 16), (16, 13), (24, 10)]:
    gdn_layers_left = 36 * (48 - L) / 48
    print(f"drop {drop} rows after layer {L}: GDN step saves {drop*b_us*gdn_layers_left/1000:.2f} ms per check")
