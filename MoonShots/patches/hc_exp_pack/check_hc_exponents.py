"""Can Qwen3.8-Flash-Next's bf16 hyper-connection (gated-residual) mixer weights be stored LOSSLESSLY in ~12 bits?
MTPLX's exact decode path reads them in bf16 every token: 13.2 MB x 96 reads = 1.27 GB/token, about a third of
the 3.78 GB a token reads. Its fused v3 kernel halves that, but with 8-bit affine weights, so it isn't exact.

Exact fixed-rate scheme ("exponent offset pack"): per group of 64 weights store one base exponent (8 bits),
then per weight: sign + 7 mantissa bits (8 bits) + a b-bit exponent offset from the base. A group whose exponents
don't fit in [base, base + 2^b - 1] is stored raw (escape). Decode = one add and a shift, so it is free in a
bandwidth-bound kernel. Reconstruction is bit-identical by construction.
CPU only, reads ~1.2 GB of tensors via mmap (one at a time). Usage: check_hc_exponents.py [model_dir]"""
import json, os, struct, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
names = [n for n in idx if "hyper_connection" in n and n.endswith(".weight")]


def tensor(name):
    f = f"{D}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    m = hdr[name]
    assert m["dtype"] == "BF16", m["dtype"]
    a, b = m["data_offsets"]
    return np.memmap(f, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,))


G = 64
tot = 0
fit = {2: 0, 3: 0, 4: 0}
ngroups = 0
ent_num = 0.0
hist = np.zeros(256, np.int64)
for nm in names:
    w = np.asarray(tensor(nm))
    w = w[: len(w) // G * G]
    e = ((w >> 7) & 0xFF).astype(np.int16).reshape(-1, G)
    hist += np.bincount(e.ravel(), minlength=256)
    spread = e.max(1) - e.min(1)                 # conservative: an exact zero (exponent 0) widens the spread
    for b in fit:
        fit[b] += int((spread < (1 << b)).sum())
    ngroups += len(e)
    tot += w.size
p = hist / hist.sum()
H = -(p[p > 0] * np.log2(p[p > 0])).sum()
print(f"{len(names)} hyper-connection tensors, {tot/1e6:.0f}M bf16 weights ({tot*2/1e9:.2f} GB)")
print(f"exponent entropy {H:.2f} bits (of 8); Huffman-style bound ≈ {8 + H:.2f} bits/weight ({(8+H)/16:.0%} of bf16)")
for b, n in fit.items():
    f = n / ngroups
    bits = f * (8 + b + 8 / G) + (1 - f) * (16 + 8 / G) + 1 / G   # +1 flag bit per group
    print(f"{b}-bit exponent offset: {f:.1%} of groups fit; {bits:.2f} bits/weight = {bits/16:.0%} of bf16 "
          f"-> saves {1.27*(1-bits/16):.2f} GB of the 1.27 GB/token HC read")
