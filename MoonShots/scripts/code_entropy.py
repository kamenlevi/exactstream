"""Higher-order entropy of MLX 4-bit weight codes. SpecDecode measured order-0 at 3.6–3.7 of 4 bits. MLX issue #3043
claims 2.17 bits. If neighbouring codes (same row, next column; same column, next row) predict each other,
a fused rANS/table decoder could cut *all* weight bytes losslessly. CPU, mmap, one tensor at a time.
Usage: code_entropy.py [model_dir] [max_tensors]"""
import glob, json, os, struct, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def cond_H(a, b, k=16):
    """H(b | a) for code arrays a, b of equal shape."""
    joint = np.bincount((a.astype(np.int64) * k + b).ravel(), minlength=k * k).reshape(k, k)
    return H(joint.ravel()) - H(joint.sum(1))


done = 0
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, m in hdr.items():
        if not name.endswith(".weight") or m["dtype"] != "U32" or len(m["shape"]) < 2:
            continue
        if done >= MAXT or (done % 3 and "experts" not in name and "switch" not in name):
            pass
        a, b = m["data_offsets"]
        shp = m["shape"]
        rows = shp[-2] if len(shp) == 2 else shp[-2]
        words = np.memmap(f, np.uint32, "r", offset=8 + n + a, shape=((b - a) // 4,))
        take = min(len(words), 4_000_000)                     # ≤ 16 MB per tensor
        w = np.asarray(words[:take])
        codes = np.stack([(w >> (4 * i)) & 0xF for i in range(8)], 1).reshape(-1).astype(np.uint8)
        cols = shp[-1] * 8
        nrow = len(codes) // cols
        c = codes[: nrow * cols].reshape(nrow, cols)
        h0 = H(np.bincount(c.ravel(), minlength=16))
        h_row = cond_H(c[:, :-1], c[:, 1:])                   # next column, same row
        h_col = cond_H(c[:-1, :], c[1:, :]) if nrow > 1 else float("nan")   # next row, same column
        print(f"{name[-60:]:60s} H0 {h0:.3f}  H(next col|prev) {h_row:.3f}  H(next row|prev) {h_col:.3f} bits", flush=True)
        done += 1
        if done >= MAXT:
            sys.exit()
