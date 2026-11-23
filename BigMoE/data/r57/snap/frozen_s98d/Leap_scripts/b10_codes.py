"""B10(b) side check (Leap, Sun 2026-10-25, CPU only): any computed code f(state) of a 12-bit bitshift trellis can be
tabulated into the 4,096-entry LUT that Fit12's / Breakthrough's encoder takes, so a GPU decode that COMPUTES f(s)
(no table gather) changes nothing in the format, only the reconstruction values. Is its error the same?
Codes (each normalised to unit RMS over its 4,096 states; the row scale absorbs the constant):
  lut    D2's code: 4,096 Gaussian quantiles in a fixed random order (make_lut(12, 0))
  3inst  QTIP's 3INST: x = 89226354 s + 64248484 (mod 2^32); m = (x & 0x8FFF8FFF) ^ 0x3B603B60; f = half(m_lo) + half(m_hi)
  1mad   QTIP's 1MAD: x = 34038481 s + 76625530 (mod 2^32); f = sum of x's 4 bytes (a CLT Gaussian)
NMSE of the exact Viterbi (encoder copy in scripts/enc) on iid N(0,1) tiles of 256 (Fit12's B3 test: lut k=2 0.0711),
and on a Laplace source (heavier tails, like weights after an imperfect RHT).
Usage: taskpolicy -b python3 b10_codes.py [threads=2] -> ~/Leap/data/b10_codes.txt"""
import ctypes, os, sys, time
from statistics import NormalDist
import numpy as np

NTHR = int(sys.argv[1]) if len(sys.argv) > 1 else 2
L, T = 12, 256
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "enc", "libb10trellis.dylib"))
lib.viterbi_tiles_n.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int]
s = np.arange(1 << L, dtype=np.uint64)
q = np.array([NormalDist().inv_cdf((i + 0.5) / 4096) for i in range(4096)], np.float32)
codes = {"lut": q[np.random.default_rng(0).permutation(4096)]}
x3 = (np.uint64(89226354) * s + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)
m3 = ((x3 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
codes["3inst"] = (m3 & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32) + \
    (m3 >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
x1 = ((np.uint64(34038481) * s + np.uint64(76625530)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
codes["1mad"] = sum(((x1 >> (8 * i)) & 0xFF).astype(np.float32) for i in range(4))
for k in codes:
    c = codes[k] - (codes[k].mean() if k != "lut" else 0.0)
    codes[k] = np.ascontiguousarray((c / np.sqrt((c * c).mean())).astype(np.float32))


def nmse(x, lut, k):
    out = np.empty_like(x)
    lib.viterbi_tiles_n(x.ctypes.data, out.ctypes.data, x.shape[0], T, L, k, lut.ctypes.data, NTHR)
    return float(((out - x) ** 2).mean() / (x * x).mean())


rng = np.random.default_rng(1)
src = {"gauss": rng.standard_normal((1024, T)).astype(np.float32),
       "laplace": (rng.laplace(size=(1024, T)) / np.sqrt(2)).astype(np.float32)}
lines = [f"B10(b) code check (CPU, {NTHR} threads, {time.strftime('%a %H:%M')}): L = {L}, tiles of {T}, 1,024 tiles per cell"]
for k in codes:
    c = codes[k]
    lines.append(f"  {k:5s}: distinct values {len(np.unique(c))}, kurtosis {float((c ** 4).mean()):.2f} (Gaussian 3.00)")
for kb in (2, 3):
    for nm, x in src.items():
        t0 = time.time()
        row = {k: nmse(np.ascontiguousarray(x), codes[k], kb) for k in codes}
        lines.append(f"  k={kb} {nm:7s}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items()) +
                     f"  (vs lut: " + ", ".join(f"{k} {row[k] / row['lut']:.3f}x" for k in codes if k != "lut") + f"; {time.time() - t0:.0f} s)")
        print(lines[-1], flush=True)
open(os.path.expanduser("~/Leap/data/b10_codes.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:4]))
