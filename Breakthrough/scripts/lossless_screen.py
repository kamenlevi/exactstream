"""LOSSLESS expert records screen (Breakthrough Sun 1 Nov 04:2x; CPU only; lines sent to NewIdeas before the run: PASS >= 10 %
smaller records, KILL < 5 %): how much would a lossless entropy coder (DFloat11 / ZipNN / Huff-LLM principle) shrink the
4-bit expert records the 12 GB engine streams? Sample: 8 experts in each of layers 0 / 12 / 24 / 36 / 47 of records3 (the
same codes records3m holds; its scales / biases are already re-packed narrower). Per piece type: the 3-bit MSB code plane
(values 0-7 unpacked from MLX's bits=3 stream), the 1-bit LSB plane, bf16 scales / biases: order-0 entropy (an ideal
static coder's bound) and zlib-9 / lzma ratios on the raw bytes (a real codec, no layout knowledge). Then the bound on a
records3m record (MSB 2,048,000 B: 3 x 614,400 code bytes; LSB 720,896 B: 3 x 204,800 bit bytes; the rest kept as is).
Usage: [BigMoE venv] python lossless_screen.py"""
import json, lzma, os, zlib
import numpy as np

D = os.path.expanduser("~/BigMoE/models/bare/records3")
ix = json.load(open(os.path.join(D, "index.json")))
rng = np.random.default_rng(1)


def H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


c3, c1 = np.zeros(8), np.zeros(2)
sb, raw, zl, xz = [], {"msb": 0, "lsb": 0}, {"msb": 0, "lsb": 0}, {"msb": 0, "lsb": 0}
for L in (0, 12, 24, 36, 47):
    m = ix["layers"][str(L)]
    lo, ls = m.get("lsb_piece_off"), m.get("lsb_piece_size")
    with open(os.path.join(D, m["file"]), "rb") as f:
        for e in sorted(rng.choice(m["E"], 8, replace=False)):
            f.seek(int(e) * m["recsz"])
            rec = f.read(m["recsz"])
            msb, lsb = rec[:m["msb_sz"]], rec[m["msb_sz"]:m["msb_sz"] + m["lsb_sz"]]
            for k, part in (("msb", msb), ("lsb", lsb)):
                raw[k] += len(part); zl[k] += len(zlib.compress(part, 9)); xz[k] += len(lzma.compress(part, preset=6))
            for i, (o, s) in enumerate(zip(m["msb_piece_off"], m["msb_piece_size"])):
                b = np.frombuffer(msb[o:o + s], np.uint8)
                if i % 3 == 0:                                   # w3: 3-bit codes, value i at bits 3i.. (little-endian)
                    v = np.unpackbits(b, bitorder="little").reshape(-1, 3) @ np.array([1, 2, 4])
                    c3 += np.bincount(v, minlength=8)
                else:                                            # s3 / b3 (bf16): byte stream
                    sb.append(bytes(b))
            if lo:
                for o, s in zip(lo, ls):
                    b = np.frombuffer(lsb[o:o + s], np.uint8)
                    if s == 204800:                              # l1: 1 bit per weight
                        c1 += np.bincount(np.unpackbits(b), minlength=2)
h3, h1 = H(c3), H(c1)
sbb = b"".join(sb)
h_sb = H(np.bincount(np.frombuffer(sbb, np.uint8), minlength=256)) / 8
print(f"3-bit codes: {int(c3.sum()):,} values, histogram {np.round(c3 / c3.sum(), 4).tolist()}, entropy {h3:.3f} bits (of 3): "
      f"bound {100 * (1 - h3 / 3):.1f} % smaller")
print(f"1-bit plane: {int(c1.sum()):,} bits, P(1) {c1[1] / c1.sum():.4f}, entropy {h1:.4f} bits: bound {100 * (1 - h1):.2f} % smaller")
print(f"bf16 scales / biases: order-0 byte entropy {h_sb:.3f} of 1 (zlib-9 {len(zlib.compress(sbb, 9)) / len(sbb):.3f})")
for k in ("msb", "lsb"):
    print(f"{k} part (records3), real codecs: zlib-9 {zl[k] / raw[k]:.3f}, lzma-6 {xz[k] / raw[k]:.3f} of raw")
rec3m = 2048000 + 720896
codes, bits = 3 * 614400, 3 * 204800
ideal = codes * h3 / 3 + bits * h1 + (rec3m - codes - bits)
print(f"records3m record bound with ideal order-0 coding of the code planes (rest unchanged): {ideal:,.0f} of {rec3m:,} B = "
      f"{100 * (1 - ideal / rec3m):.1f} % smaller -> line: " + ("PASS" if ideal <= 0.90 * rec3m else "KILL" if ideal > 0.95 * rec3m else "BETWEEN"))
