"""24 GB wall, exact scale/bias packing: per projection of real experts (records3 MSB part s3, b3; LSB part b), the bf16
exponent span (max - min over the projection's 25,600 groups), sign use, and the bits a lossless fixed-width packing needs:
sign (if used) + 7 mantissa + ceil(log2(span + 1)) exponent offset from a per-projection base (+ zero handling).
usage: scale_exp.py [n_experts=48]   (CPU only)"""
import json, math, sys
import numpy as np

D = "/Users/kamen/BigMoE/models/bare/records3/"
IX = json.load(open(D + "index.json"))["layers"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
REC = 2932736
rng = np.random.default_rng(7)
picks = [(int(L), int(e)) for L, e in zip(rng.integers(0, 48, N), rng.integers(0, 512, N))]
stats = {"s3": [], "b3": [], "b": []}
for L, e in picks:
    ly = IX[str(L)]
    with open(D + ly["file"], "rb") as f:
        f.seek(e * REC); rec = np.frombuffer(f.read(REC), np.uint8)
    mo, ms, lo, ls = ly["msb_piece_off"], ly["msb_piece_size"], ly["lsb_piece_off"], ly["lsb_piece_size"]
    for p in range(3):
        for name, o, sz in (("s3", mo[3 * p + 1], ms[3 * p + 1]), ("b3", mo[3 * p + 2], ms[3 * p + 2]), ("b", lo[2 * p + 1], ls[2 * p + 1])):
            v = np.frombuffer(rec[o:o + sz].tobytes(), np.uint16)
            ex = (v >> 7) & 0xFF; sg = v >> 15; z = (v & 0x7FFF) == 0
            exn = ex[~z]
            span = int(exn.max() - exn.min()) if exn.size else 0
            stats[name].append((span, int(sg.any()), int(z.sum()), int(exn.min()) if exn.size else 0))
for name, st in stats.items():
    sp = np.array([s[0] for s in st]); sg = np.array([s[1] for s in st]); zz = np.array([s[2] for s in st])
    bits = [(1 if s[1] else 0) + 7 + math.ceil(math.log2(s[0] + 2)) for s in st]   # +2: one code reserved for zero
    print(f"{name:2s}: exponent span per projection median {int(np.median(sp))} max {sp.max()}; sign used in "
          f"{100 * sg.mean():.0f} % of projections; zeros per projection median {int(np.median(zz))}; lossless fixed-width "
          f"bits median {int(np.median(bits))} max {max(bits)} (vs 16)")
