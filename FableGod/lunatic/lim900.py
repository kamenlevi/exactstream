# IDEAS_1116_limits: the 900 tok/s budget in bytes on this Mac (single stream, code, lossy). Inputs [K THINK_1116 /
# WEEK98]: lossy 32-row code check 6.12 GB (byte floor 21.9 ms); t30b head 177.6 MB per draft level (+0.31 ms fixed per
# level); tree-16 12.6 levels, tree-32 20.8, K128 39.6; DFlash2 block 0.52 GB.
BW = {'273': 273e9, '280': 280e9, '300': 300e9}
for k, b in BW.items():
    print(f"BW {k} GB/s: 900 tok/s allows {b / 900 / 1e6:5.0f} MB of DRAM traffic per kept token")
B = 280e9; C = 6.12e9; L = 177.6e6
def tps(kept, draft_bytes, check_ms=None, draft_ms=None):
    cm = check_ms if check_ms is not None else C / B * 1e3
    dm = draft_ms if draft_ms is not None else draft_bytes / B * 1e3
    return kept / (cm + dm) * 1e3, (C + draft_bytes) / kept / 1e6
print("\n-- at the byte floor (check 21.9 ms; draft = its bytes at 280 GB/s), MB per kept token in brackets")
for name, kept, db in [('today lossy K* tree-32, t30b', 16.91, 20.8 * L), ('flat 97.8 %, tree-32 t30b', 23.15, 20.8 * L),
                       ('flat 97.8 %, chain-30 t30b', 23.15, 30 * L), ('perfect (33), tree-32 t30b', 33.0, 20.8 * L),
                       ('flat 97.8 %, 6 t30b levels', 23.15, 6 * L), ('flat 97.8 %, 6 levels + 0.52 GB block', 23.15, 6 * L + 0.52e9),
                       ('flat 97.8 %, free drafting', 23.15, 0.0), ('kept 28, tree-32 t30b', 28.0, 20.8 * L),
                       ('kept 28, 6 levels + 0.52 GB block', 28.0, 6 * L + 0.52e9)]:
    t, mb = tps(kept, db)
    print(f"  {name:38s} kept {kept:5.2f} draft {db / 1e9:4.2f} GB -> {t:5.0f} tok/s  [{mb:4.0f} MB/kept]")
print("\n-- the spec's triplet: kept for 900 = 0.9 x (check ms + draft ms); flat accuracy a giving that kept/32")
import math
def kept_flat(a): return 1 + a * (1 - a ** 31) / (1 - a)
def a_for(k):
    lo, hi = 0.9, 0.99999
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kept_flat(mid) < k else (lo, mid)
    return lo
for cm, dm in [(25, 0.7), (25, 3), (25, 5), (21.9, 3.8), (21.9, 5), (30, 4), (41, 8.3), (45, 13.8)]:
    k = 0.9 * (cm + dm)
    print(f"  check {cm:4.1f} + draft {dm:4.1f} ms -> needs kept {k:5.2f} = flat {a_for(k) * 100:5.2f} % per guess"
          + ("  (beyond 32 rows)" if k > 33 else ""))
print(f"  flat 97.8 % gives kept {kept_flat(0.978):5.2f}: at 25 + 3..5 ms = {kept_flat(0.978) / 28 * 1e3:4.0f}..{kept_flat(0.978) / 30 * 1e3:4.0f} tok/s")
print("\n-- width (aggregate, one user's B code streams in one pass): check bytes per kept x groups ratio [K WALL_L T2],"
      "\n   head levels batched across streams (one weight read per level for all B)")
for Bn, g in [(1, 1.0), (2, 0.78), (4, 0.55), (8, 0.35)]:
    kept = 16.91 * Bn
    cb = C * Bn * g                     # expert-dominated; the dense part is shared too, so this is conservative
    db = 20.8 * L                       # tree-32 levels, read once for all streams
    t = kept / ((cb + db) / B * 1e3) * 1e3
    print(f"  B{Bn}: kept {kept:5.1f} per pass, check {cb / 1e9:5.2f} GB + draft {db / 1e9:4.2f} GB -> {t:5.0f} tok/s aggregate ({t / Bn:4.0f} per stream) at the byte floor")
