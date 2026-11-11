"""ksplan.py (Fable B, DENSE_REBUILD v2): the ks / tile plan per narrow shape from Leap's part B numbers. CPU, no MLX.
Fill model [P, fitted to part B]: a code-MMA simdgroup takes TAU us per 64-K quant group, and the GPU keeps about C
simdgroups in flight. A call of `tiles` 32-column tiles, KG quant groups, split ks ways along K costs
    t = t_sync + TAU * max(KG / ks, tiles * KG / C)        (chain-bound below C simdgroups, fill-bound above)
TAU: out_proj 112 us / 96 groups = 1.17, a|b 45 / 40 = 1.13, k / v 48 / 40 = 1.20, shared gate|up 47 / 40 = 1.18.
C:   in_proj 512 x 40 x 1.15 / 113 us = 208; attention q 384 x 40 x 1.15 / 75 = 235.
Limits: ks divides KG; 32 ks threads per threadgroup; the split's partials take (ks - 1) x 2 KB of threadgroup memory
(keep <= 20 KB: residency holds to 20 KB [K NAX-OCC]), so ks <= 11 by memory, <= 16 hard."""
TAU, TSYNC = 1.15, 1.6
SH = [  # name, tiles, KG, MB, us today [K], calls per check, note
    ("out_proj / o_proj", 80, 96, 8.85, 112.0, 48, ""),
    ("GDN a|b (alone)", 3, 40, 0.22, 45.5, 36, ""),
    ("attention k|v merged (alone)", 32, 40, 1.6, 97.0, 12, "today = 2 calls"),
    ("shared gate|up + gate", 41, 40, 2.0, 47.0, 48, ""),
    ("shared down", 80, 10, 1.0, 13.7, 48, ""),
    ("router 8-bit (qmm16c8)", 16, 40, 1.46, 18.6, 48, "today = MLX qmm; routes may flip"),
    ("attention 8-bit 640 x 2,560", 20, 40, 1.8, 21.9, 12, "today = MLX qmm"),
]
def t(tiles, KG, ks, C):
    return TSYNC + TAU * max(KG / ks, tiles * KG / C)
print("%-30s %5s %4s | %-22s | %9s %9s | %8s | %7s %7s | %s" % ("shape", "tiles", "KG", "ks: us at C 208 / 235", "plan ks", "simdgrps", "us today", "us plan", "ms/check", "GB/s; floor us at 250"))
tot = 0.0
for name, tiles, KG, mb, today, calls, note in SH:
    cands = [k for k in range(1, 17) if KG % k == 0 and (k - 1) * 2 <= 20]
    best = min(cands, key=lambda k: (round(t(tiles, KG, k, 208), 1), k))
    row = "  ".join("%d:%.0f/%.0f" % (k, t(tiles, KG, k, 208), t(tiles, KG, k, 235)) for k in cands if k in (1, 2, 4, 5, 8, 10))
    tp = t(tiles, KG, best, 208)
    sv = (today - tp) * calls / 1e3
    tot += sv
    print("%-30s %5d %4d | %-46s | %7d %9d | %8.1f | %7.1f %7.2f | %3.0f; %.1f  %s" % (name, tiles, KG, row, best, tiles * best, today, tp, -sv, mb / tp * 1e3, mb / 250 * 1e3, note))
print("\nfolds (no K split, no new kernel: the narrow rows ride as extra tiles of the wide call that reads the same x):")
for name, wide_t, add_t, wide_us, alone_us, calls in (("a|b into in_proj qkv|z (512 + 3 = 515 tiles)", 512, 3, 113.0, 45.5, 36),
                                                      ("k|v into attention q (384 + 32 = 416 tiles)", 384, 32, 75.0, 97.0, 12)):
    extra = wide_us * add_t / wide_t
    sv = (alone_us - extra) * calls / 1e3
    tot += sv - (0 if "k|v" not in name else 0)
    print("  %-48s +%.1f us on the wide call instead of %.1f us alone: %.2f ms per check" % (name, extra, alone_us, -sv))
print("\nsum with both folds and the alone rows of a|b and k|v left out: see DENSE_REBUILD.md section 2")
