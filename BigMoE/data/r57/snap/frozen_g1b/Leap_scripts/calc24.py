"""24 GB section of TABLE_350 (Leap, Sat 2026-10-24; Kamen ~20:00 via the orchestrator: 350 tok/s on the 125B in <= 24 GB,
45 GB as a waypoint). What tokens per check, SSD bytes per check and kernel efficiency give 350 at C_m 192?
Model of one verify check of R rows (all 48 layers):
  union per layer U(R) = u16 x (R/16)^0.72 (u16 73.6 top-10, 62.4 top-8; topk17.py)
  expert bytes per instance s = 1.845 MB (3.0 b; Fit12's C_m slope: (51.7 - 45.5) GB / (70 x 48))
  misses per check M(R) = M4 x (R/4)^x, M4 = 155-283 (Fit12 RES1, C_m 192, 4-row MTP checks); x = 0.72 (misses grow like
    the union) or 1.0 (tail experts grow per row); SSD bytes = M x s, at B_ssd GB/s
  RAM bytes = dense 2.1 GB (diet) + (U x 48 - M) x s, at 280 GB/s x kernel efficiency e
  pass = RAM time + exposed SSD time + drafting 3.5 ms; exposed = SSD time (no overlap, today: Fit12's stall ~ its read
    time) or max(0, SSD - RAM) (perfect prefetch)
tok/s = T / pass. Prints the grid and the SSD-bytes cut needed for 350.
Usage: python3 calc24.py"""
import itertools

S_MB, DENSE, BW = 1.845, 2.1, 280.0

def check(T, R, e, m4, x, bssd, overlap, u16=73.6, cut=1.0):
    U = u16 * (R / 16) ** 0.72 * 48
    M = min(U, m4 * (R / 4) ** x / cut)
    ram = (DENSE + (U - M) * S_MB / 1000) / (BW * e) * 1000
    ssd = M * S_MB / 1000 / bssd * 1000
    exp_ = max(0.0, ssd - ram) if overlap else ssd
    p = ram + exp_ + 3.5
    return T / p * 1000, p, ram, ssd, M

def need_cut(T, R, e, m4, x, bssd, overlap, u16=73.6):
    """factor by which misses (SSD bytes) must fall for 350"""
    lo, hi = 1.0, 1000.0
    if check(T, R, e, m4, x, bssd, overlap, u16, 1.0)[0] >= 350:
        return 1.0
    if check(T, R, e, m4, x, bssd, overlap, u16, hi)[0] < 350:
        return float("inf")
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if check(T, R, e, m4, x, bssd, overlap, u16, mid)[0] >= 350:
            hi = mid
        else:
            lo = mid
    return hi

print("Today's measured frame, check: MTP 4 rows, ~3.4 tokens, C_m 192 (Fit12 RES1: 35.8 tok/s code; verify 63-82 ms,"
      " stall 31-49 ms)")
for m4 in (155, 283):
    r = check(3.4, 4, 0.45, m4, 0.72, 10.6, False)
    print(f"  m4 {m4}: model {r[0]:5.1f} tok/s, pass {r[1]:5.1f} ms (RAM {r[2]:4.1f} at 45 % kernels, SSD {r[3]:5.1f})")
print()
print("tok/s at C_m 192 (3.0-bit experts, dense diet), R = 16 rows; misses from RES1 scaled by rows^x")
print("T    e     x     prefetch | m4 155: tok/s  pass  RAM  SSD | m4 283: tok/s  pass  SSD | SSD cut for 350 (155 / 283)")
for T, e, x, ov in itertools.product((9.5, 12.4, 15.0), (0.8, 0.95), (0.72, 1.0), (False, True)):
    a = check(T, 16, e, 155, x, 10.6, ov); b = check(T, 16, e, 283, x, 10.6, ov)
    ca, cb = need_cut(T, 16, e, 155, x, 10.6, ov), need_cut(T, 16, e, 283, x, 10.6, ov)
    print(f"{T:4.1f} {e:.2f} {x:.2f} {'yes' if ov else 'no ':3s}      | {a[0]:6.0f} {a[1]:6.1f} {a[2]:4.1f} {a[3]:5.1f} | "
          f"{b[0]:6.0f} {b[1]:6.1f} {b[3]:5.1f} | x{ca:4.1f} / x{cb:4.1f}")
print()
print("Per generated token: 350 tok/s at B_ssd 10.6 GB/s allows <= 30.3 MB of SSD reads per token (16.4 misses of 1.845 MB)"
      " when SSD time is fully hidden; today (R 16, T 9.5, m4 155-283, x 0.72): "
      + " / ".join(f"{check(9.5, 16, 0.8, m, 0.72, 10.6, False)[4] / 9.5:.0f}" for m in (155, 283)) + " misses per token")
print()
print("Block 32 (oracle-like drafter, T 20 at R 32) and truncation (T 9.2 at R 11): misses grow with rows, tokens with T")
for T, R in ((9.2, 11), (9.5, 16), (15.0, 16), (20.0, 32)):
    for ov in (False, True):
        a = check(T, R, 0.95, 155, 0.72, 10.6, ov); b = check(T, R, 0.95, 283, 0.72, 10.6, ov)
        print(f"  T {T:4.1f} R {R:2d} e 0.95 prefetch {'yes' if ov else 'no '}: {a[0]:4.0f}-{b[0]:4.0f} tok/s; misses per token "
              f"{a[4] / T:4.0f}-{b[4] / T:4.0f}; SSD cut for 350 x{need_cut(T, R, 0.95, 155, 0.72, 10.6, ov):.1f}"
              f"-x{need_cut(T, R, 0.95, 283, 0.72, 10.6, ov):.1f}")
