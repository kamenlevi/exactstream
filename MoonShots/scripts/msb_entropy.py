"""24 GB wall, lossless lever: empirical entropy of the MSB-3 part of real experts (records3), to bound how many SSD bytes
an exact (lossless) code could save. Per expert, per projection (gate / up / down: w3 = MLX 3-bit packing of m, 8 values in
3 bytes little-endian; s3, b3 bf16 per group of 64). Checks the unpacking (every group of 64 has m min 0 and max 7, since
the 4-bit code spans 0..15). Reports bits per weight: order-0 entropy of m (per projection type, pooled), the same per
group-position-free context (previous weight's m in the row), and order-0 byte entropy of s3 / b3; vs today's 3.5.
usage: msb_entropy.py [n_experts=48]   (CPU only)"""
import json, sys
import numpy as np

D = "/Users/kamen/BigMoE/models/bare/records3/"
IX = json.load(open(D + "index.json"))["layers"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
REC = 2932736
rng = np.random.default_rng(7)
picks = [(int(L), int(e)) for L, e in zip(rng.integers(0, 48, N), rng.integers(0, 512, N))]
SH = {0: (640, 2560), 1: (640, 2560), 2: (2560, 640)}      # gate, up [640, 2560]; down [2560, 640] (out, K)


def unpack3(b, n):
    t = b.reshape(-1, 3).astype(np.uint32)
    v = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
    return ((v[:, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).reshape(-1)[:n].astype(np.uint8)


def H(c):
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


hist = np.zeros((3, 8)); cond = np.zeros((3, 8, 8)); sb = [np.zeros(65536), np.zeros(65536)]
bad = 0
for L, e in picks:
    ly = IX[str(L)]
    with open(D + ly["file"], "rb") as f:
        f.seek(e * REC); rec = np.frombuffer(f.read(ly["msb_sz"]), np.uint8)
    off, sz = ly["msb_piece_off"], ly["msb_piece_size"]
    for p in range(3):
        o, K = SH[p]
        m = unpack3(rec[off[3 * p]:off[3 * p] + sz[3 * p]], o * K).reshape(o, K)
        g = m.reshape(o, K // 64, 64)
        bad += int((g.min(-1) != 0).sum() + (g.max(-1) != 7).sum())
        hist[p] += np.bincount(m.reshape(-1), minlength=8)
        np.add.at(cond[p], (m[:, :-1].reshape(-1), m[:, 1:].reshape(-1)), 1)
        for j, q in ((0, 1), (1, 2)):
            v = np.frombuffer(rec[off[3 * p + q]:off[3 * p + q] + sz[3 * p + q]].tobytes(), np.uint16)
            sb[j] += np.bincount(v, minlength=65536)
print(f"{N} experts; groups whose m does not span 0..7 (unpacking check, want 0): {bad}")
names = ("gate", "up", "down")
h0 = [H(hist[p]) for p in range(3)]
h1 = [sum(cond[p][a].sum() / cond[p].sum() * H(cond[p][a]) for a in range(8)) for p in range(3)]
for p in range(3):
    print(f"{names[p]:4s} m histogram {np.round(hist[p] / hist[p].sum(), 4).tolist()}  H0 {h0[p]:.3f}  H(m | previous m) {h1[p]:.3f} bits")
hs, hb = H(sb[0]), H(sb[1])
print(f"s3 bf16 order-0 entropy {hs:.2f} of 16 bits, b3 {hb:.2f} of 16 bits (one pair per 64 weights)")
cur = 3 + 32 / 64
best = np.mean(h1) + (hs + hb) / 64
print(f"bits per weight: today {cur:.3f}; lossless floor (order-1 codes + order-0 scales) ~{best:.3f} -> "
      f"{100 * (1 - best / cur):.1f} % fewer MSB bytes per miss")
