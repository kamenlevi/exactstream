"""Print summary stats of 1-D (norm) tensors in safetensors files without loading them (header + byte ranges).
usage: st_peek.py FILE [REGEX]   (bf16/f32/f16; prints mean, min, max, first 6 values)"""
import json, re, struct, sys
import numpy as np
f, pat = sys.argv[1], re.compile(sys.argv[2] if len(sys.argv) > 2 else r".")
with open(f, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    for k in sorted(hdr):
        m = hdr[k]
        if k == "__metadata__" or len(m["shape"]) != 1 or not pat.search(k):
            continue
        a, b = m["data_offsets"]
        fh.seek(8 + n + a)
        raw = fh.read(b - a)
        dt = m["dtype"]
        if dt == "BF16":
            v = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            v = np.frombuffer(raw, np.float16).astype(np.float32)
        elif dt == "F32":
            v = np.frombuffer(raw, np.float32)
        else:
            continue
        print(f"{k:70s} {dt} n={v.size:6d} mean {v.mean():+.4f} min {v.min():+.4f} max {v.max():+.4f}  {np.round(v[:6], 3).tolist()}")
