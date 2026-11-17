#!/usr/bin/env python3
# IDEAS_1117c pricing (fable-zero, Tue 17 Nov 2026): the SLC as a second memory, and path-p byte tiers.
# Measured parts [K]: Leap tue1 per-class table (32 rows), check25_bound_real.txt, hc_bw_probe.txt / l3_slc_bench.txt,
# MoonShots' sync-free pop (0.55 fixed + 0.56 bytes), PLAN900's per-level bytes, Breakthrough's quiet rewrite (0.816-0.843).
BW = 280.0          # GB/s, DRAM, the GPU's measured rate on distinct bytes
SLC_X = 2.5         # x faster re-read from the cache (S4 6.6 MB: 38.9 -> 13.0..18.7 us) [K]
EXP_GB, EXP_MS = 4.60, 26.3     # C6 experts per 32-row check, tue1 class time
DENSE_GB, DENSE_MS = 2.13, 8.3 + 2.6 + 2.2 + 1.7   # gdn_proj + att_proj + moe_shared + router (tue1)
DENSE_FLOOR = DENSE_GB / BW * 1e3
LAYERS = 48

print("== DRAM slack inside the expert phase (the window a CPU prefetch can use) ==")
for name, ms in [("tue1 one-call", EXP_MS), ("after the rewrite x0.82", EXP_MS * 0.82), ("at the byte floor", EXP_GB / BW * 1e3)]:
    rate = EXP_GB / ms * 1e3
    idle = max(0.0, 1 - rate / BW)
    win_mb = idle * ms / 1e3 * BW * 1e3 / LAYERS
    print(f"  {name:26s}: {rate:5.0f} GB/s, DRAM idle {idle*100:4.0f} % = {idle*ms:4.1f} ms = {win_mb:4.0f} MB per layer of free reads")

print("\n== Use (a): the next layer's dense projections prefetched into the SLC by the CPU ==")
print(f"  dense today {DENSE_MS:.1f} ms for {DENSE_GB} GB ({DENSE_GB/DENSE_MS*1e3:.0f} GB/s); byte floor {DENSE_FLOOR:.1f} ms")
for x_mb in [8, 16, 24, 44]:
    gb = min(x_mb, 44) * LAYERS / 1e3
    saved_floor = gb * (1 / BW - 1 / (BW * SLC_X)) * 1e3
    eff = DENSE_FLOOR / DENSE_MS          # today's dense runs at this share of the byte rate
    print(f"  SLC window {x_mb:2d} MB per layer: {gb:.2f} GB per check from the cache; -{saved_floor:.1f} ms at the floor, "
          f"~-{saved_floor*eff:.1f} ms at today's dense efficiency [P]")

print("\n== Use (b): the draft head's re-read part served from the cache ==")
FIXED, BYTES = 0.55, 0.56          # ms per pop [K]
PER_POP_MB = 73 + 11.8 + 30.7       # Q8H32: dense + head + 10 routed experts (PLAN900 / MoonShots)
REREAD_MB = 73 + 11.8               # the same bytes every pop; the experts change
rate = PER_POP_MB / BYTES / 1e3 * 1e3
print(f"  pop {FIXED+BYTES:.2f} ms = {FIXED} fixed + {BYTES} bytes for ~{PER_POP_MB:.0f} MB ({rate:.0f} GB/s); re-read part {REREAD_MB:.0f} MB")
for fit_mb in [16, 24, 48, 85]:
    f = min(fit_mb, REREAD_MB) / PER_POP_MB
    bytes_ms = BYTES * (1 - f) + BYTES * f / SLC_X
    print(f"  cache holds {fit_mb:3d} MB of it: bytes {bytes_ms:.2f} ms, pop {FIXED+bytes_ms:.2f} ms, "
          f"tree-16 (15 pops) -{15*(BYTES-bytes_ms):.1f} ms per block [P]")

print("\n== Cycle (kept 17, HE+ lossy) ==")
for name, chk, drf, ovh in [("tue1 today", 56.1, 16.6, 4.0), ("Sat realistic", 48.0, 8.8, 1.5)]:
    base = 17 / (chk + drf + ovh) * 1e3
    for dc, dd, tag in [(1.0, 0.9, "SLC ~16 MB"), (4.6, 4.4, "SLC >= ~85 MB")]:
        new = 17 / (chk - dc + drf - dd + ovh) * 1e3
        print(f"  {name:13s} {base:4.0f} tok/s -> {new:4.0f} with {tag} (check -{dc}, draft -{dd} ms): x{new/base:.2f}")

print("\n== Idea 2: expert bytes by the row's survival (path_p) ==")
# a tree-32's rows by band of path_p (Lunatic's 127-node dumps give the real split; this is the shape [G])
for low_share in [0.30, 0.40]:
    for cut in [0.25, 1.0]:   # MSB-3 -> MSB-2 reads 2/3 of the bytes (cut 0.25 of 4.6 GB-eq), or drop the expert (cut 1.0)
        gb = EXP_GB * low_share * (0.33 if cut < 1 else 1.0) * (1 if cut < 1 else 0.6)  # drop only single-use experts (~60 % of low-row experts [G])
        print(f"  experts used only by rows with path_p < 0.2 = {low_share*100:.0f} % of union bytes; "
              f"{'MSB-2' if cut < 1 else 'drop single-use'}: -{gb:.2f} GB = -{gb/BW*1e3:.1f} ms at the floor, -{gb/EXP_GB*EXP_MS:.1f} ms at tue1's rate [P/G]")
