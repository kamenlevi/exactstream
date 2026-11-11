# LUNATIC round 2: the 1000 tok/s budget on the M5 Pro, priced on the union per pass (CPU arithmetic only).
# Inputs are measured numbers from lu_union.out / lu_alpha.out / AUDIT_350 / SSD_WALL / PHYSICS; tags in LUNACY2.md.
import math

BW = 270e9                       # RAM bytes/s [K PHYSICS 260-290]
SSD = 12e9                       # SSD bytes/s [K PHYSICS 11-13]
EXP_US = 10.15                   # expert kernel us per (expert, layer) group [K AUDIT fit]
LAYER_MS = 0.094                 # fixed ms per layer of the expert phase [K AUDIT fit]
DENSE_GB = 3.44                  # trunk bytes per pass [K]
DENSE_MAC = 2.7e9                # trunk MACs per row [G: 2.0-3.4 G]
EXP_MAC = 2.35e9                 # routed expert MACs per row: 10 x 48 x 4.9 M [P]
PEAK = 14e12                     # bf16 MAC/s [K PHYSICS]
EXP_B = 2.15e6                   # bytes per expert record [K]
U_CODE = {1: 10.0, 2: 17.3, 4: 29.2, 8: 48.3, 16: 76.6, 32: 115.1, 64: 165.6, 128: 228.0, 256: 283.0}  # [K lu_union A]

def U(rows, mixed=False, q1=1.0):
    """union per layer for `rows` rows of one domain (cross-stream ~ consecutive, lu_union B: x1.04);
    beyond 256 rows log-log extrapolation, capped at 512; mixed domains x1.20 (B8: 283.4 vs 236.9)."""
    ks = sorted(U_CODE)
    if rows in U_CODE:
        u = U_CODE[rows]
    elif rows > 256:
        u = U_CODE[256] * (rows / 256) ** (math.log(283.0 / 228.0) / math.log(2))
    else:
        lo = max(k for k in ks if k <= rows); hi = min(k for k in ks if k >= rows)
        t = math.log(rows / lo) / math.log(hi / lo)
        u = math.exp(math.log(U_CODE[lo]) * (1 - t) + math.log(U_CODE[hi]) * t)
    return min(512.0, u * (1.20 if mixed else 1.0) * q1)

def kept(a, n):
    return (1 - a ** (n + 1)) / (1 - a) if a < 1 else n + 1

def dense_ms(rows, int8=False):
    # 12.7 ms of trunk bytes + per-row compute at ~60 % of peak + ~3.6 ms fixed: 19.3 ms at 16 rows [K 17-23]
    return DENSE_GB * 1e9 / BW * 1e3 + 3.6 + rows * DENSE_MAC / (PEAK * (1.6 if int8 else 1.0) * 0.6) * 1e3

def gpass(rows, u, int8=False):
    """GPU ms per pass: expert phase (group fit) + trunk; returns (ms, groups, ram_GB, compute_util_at_this_rate)."""
    g = u * 48
    e_ms = 48 * LAYER_MS + g * EXP_US / 1e3
    d_ms = dense_ms(rows, int8)
    ram = DENSE_GB + g * EXP_B / 1e9
    return e_ms + d_ms, g, ram

def line(name, rows, k, u, draft_ms=0.0, ssd_mb_per_kept=0.0, int8=False):
    ms, g, ram = gpass(rows, u, int8)
    cyc = ms + draft_ms
    tps_gpu = k / cyc * 1e3
    tps_ram = BW * k / (ram * 1e9)
    tps_ssd = SSD / (ssd_mb_per_kept * 1e6) if ssd_mb_per_kept else float('inf')
    mac = rows * (EXP_MAC + DENSE_MAC)
    util = mac / (cyc / 1e3) / PEAK
    tps = min(tps_gpu, tps_ssd)
    print(f"{name:44s} rows {rows:4d} kept {k:6.1f} U {u:5.1f} groups/kept {g / k:6.0f} | GPU {cyc:6.1f} ms -> {tps_gpu:6.0f} | "
          f"RAM roof {tps_ram:6.0f} | SSD cap {tps_ssd:6.0f} | compute {util * 100:4.0f} % bf16 | => {tps:5.0f} tok/s")
    return tps

print("== 1000 tok/s budget per kept token: RAM %.0f MB, SSD %.1f MB, compute %.1f G MAC (bf16 peak), time 1.0 ms"
      % (BW / 1e9 * 1e3 / 1000 * 1e3 / 1e3, SSD / 1e6 / 1000, PEAK / 1e9 / 1000))
print("   touched per kept token: experts %.2f GB + trunk %.2f GB = %.2f GB -> reuse needed %.1fx"
      % (480 * EXP_B / 1e9, DENSE_GB, 480 * EXP_B / 1e9 + DENSE_GB, (480 * EXP_B / 1e9 + DENSE_GB) * 1000 / (BW / 1e9)))

print("\n== A. today's shape (single stream, 16-row checks, serial 12 ms draft)")
line("EXACT today a 0.909, C 256 (SSD 62 MB/kept)", 16, kept(0.909, 16) * 10.5 / 8.8, U(16), 12, 62.4)
line("EXACT today a 0.909, C 320 (SSD 37.9)", 16, 10.5, U(16), 12, 37.9)
line("EXACT today, no SSD at all", 16, 10.5, U(16), 12, 0)

print("\n== B. single stream, everything RAM-resident, draft overlapped (best case per alpha)")
for nm, a, q1 in [("EXACT a 0.909 (today's drafter)", 0.909, 1), ("EXACT a 0.9485 (sure rows perfect)", 0.9485, 1),
                  ("EXACT a 0.982 (ideal drafter bound [G])", 0.982, 1), ("FAST a 0.974 R2 relaxed + sure perfect", 0.974, 1),
                  ("FAST a 0.974 + Q1-UNION groups x0.6 [G]", 0.974, 0.6), ("FAST a 0.99 + Q1 x0.6 (beyond data)", 0.99, 0.6)]:
    best = max(((line(f"  {nm} N{n}", n, kept(a, n), U(n, q1=q1)), n) for n in (16, 32, 64, 128)), key=lambda t: t[0])
    print(f"  -> best {best[0]:.0f} tok/s at N {best[1]}")

print("\n== C. copy spans (agentic edits), single stream, EXACT, RAM-resident")
for a in (0.995, 0.999):
    for n in (128, 256, 512):
        line(f"  copy a {a} N{n}", n, kept(a, n), U(n))

print("\n== D. one user's B concurrent streams (aggregate), 32 rows each, draft batched + overlapped")
for nm, a, q1, mixed, ssd in [("EXACT a 0.909 code, SSD C320 x1.11", 0.909, 1, False, 37.9 * 1.11),
                              ("EXACT a 0.909 code, store in RAM", 0.909, 1, False, 0),
                              ("FAST a 0.974 + Q1 code, sidecar drop C320", 0.974, 0.6, False, 8.1 * 1.11),
                              ("FAST a 0.974 + Q1 mixed domains", 0.974, 0.6, True, 8.1 * 1.11),
                              ("FAST a 0.974 + Q1 mixed, int8 trunk", 0.974, 0.6, True, 8.1 * 1.11)]:
    for b in (4, 8, 12, 16):
        line(f"  {nm} B{b}", 32 * b, b * kept(a, 32), U(32 * b, mixed, q1), 0, ssd, int8=("int8" in nm))

print("\n== E. Parareal decoding: one exact sweep (N_main rows, drafts = 125B lookahead text, accept rate rho)"
      "\n      + L lookahead streams x 16 rows (DFlash2 drafts, kept 8.8 each) that must supply >= kept_main tokens per pass")
for rho in (0.95, 0.98, 0.99, 0.995, 0.999):
    for nm in (64, 128):
        km = kept(rho, nm)
        L = max(1, math.ceil(km / 8.8))
        rows = nm + 16 * L
        line(f"  rho {rho} N_main {nm} + L {L:2d} lookahead", rows, km, U(rows))
for rho in (0.99, 0.995):
    km = kept(rho, 128); L = math.ceil(km / 8.8); rows = 128 + 16 * L
    line(f"  FAST: rho {rho} N_main 128 + L {L} + Q1 x0.6", rows, km, U(rows, q1=0.6), int8=True)

print("\n== F. EXACT aggregate if the EXACT store were RAM-resident (re-based all-3-bit, 52.8 GB) + horizon fix, int8 trunk")
for a, nm in ((0.909, "today's drafter"), (0.9485, "sure rows perfect")):
    for b in (8, 12, 16, 24):
        line(f"  EXACT a {a} ({nm}) B{b} x32, int8", 32 * b, b * kept(a, 32), U(32 * b, True), 0, 0, int8=True)
print("\n== G. prefill (agent tool output): compute per token %.2f G MAC -> %.0f tok/s at 60 %% of bf16, %.0f with int8"
      % ((EXP_MAC + DENSE_MAC) / 1e9, 0.6 * PEAK / (EXP_MAC + DENSE_MAC), 0.6 * 1.6 * PEAK / (EXP_MAC + DENSE_MAC)))
print("   store sizes: EXACT 3-bit + sidecar %.1f GB, all-3-bit %.1f GB (24,576 experts)" % (24576 * 2.92e-3, 24576 * 2.15e-3))
