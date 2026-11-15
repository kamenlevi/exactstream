"""treecost.py (Fable B, THINK_1115 problem 2): what a K-node exact tree check costs, from the records. CPU, tiny.
Records [K]: exact floor 38.8 ms at 16 rows, 52.0 ms at 32 (14.56 GB); union 72.5 / 108.5 experts per layer at 16 / 32
rows; matrix-unit peak 14.0 T MAC/s; dense set 3.6 G weights (4-bit incl. HC8 / router); expert = 3 x 640 x 2560.
Fit [P]: union(K) = A (1 - exp(-K / tau)) through the two points; bytes = dense + union x 48 x e; MACs grow with rows."""
import math
u16, u32, ms16, ms32, GB32 = 72.5, 108.5, 38.8, 52.0, 14.56
r = u32 / u16; tau = -16 / math.log(r - 1); A = u16 / (1 - math.exp(-16 / tau))
BW = GB32 / ms32 * 1e3                              # GB/s implied by the 32-row floor
e = (ms32 - ms16) / 1e3 * BW / ((u32 - u16) * 48)   # GB per expert from the two floors
dense = GB32 - u32 * 48 * e
PEAK = 14.0e12
print("union fit: A %.0f experts per layer, tau %.1f rows; expert %.2f MB; dense %.2f GB; floor rate %.0f GB/s" % (A, tau, e * 1e3, dense, BW))
print("  K   union  bytes GB  byte ms   MAC ms(peak)   max(ms)  kept HE+  kept/ms   kept MBPP  kept/ms")
kept = {16: (11.26, 7.49), 32: (16.26, 9.03), 48: (17.18, 9.73), 64: (17.91, 10.16), 96: (18.86, 10.88), 128: (19.14, 11.24)}
for K in (16, 32, 48, 64, 96, 128):
    u = A * (1 - math.exp(-K / tau))
    gb = dense + u * 48 * e
    mac = (3.6e9 + 10 * 48 * 3 * 640 * 2560) * K           # dense + this row's 10 experts, per row
    byte_ms, mac_ms = gb / BW * 1e3, mac / PEAK * 1e3
    t = max(byte_ms, mac_ms)
    h, m = kept[K]
    print("%4d  %6.1f  %8.2f  %7.1f  %12.1f  %9.1f  %8.2f  %7.3f  %9.2f  %7.3f" % (K, u, gb, byte_ms, mac_ms, t, h, h / t, m, m / t))
print("\nnote: the MAC column is the matrix-unit floor at 100 %; today's expert kernel runs at ~57 % of it, dense at ~40-60 %.")
