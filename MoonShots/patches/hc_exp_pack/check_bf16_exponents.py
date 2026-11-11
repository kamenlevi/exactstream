"""Lossless exponent-offset pack for every bf16 tensor a decode token reads (generalises check_hc_exponents.py).
Scans all safetensors headers in the model dir, takes BF16 tensors except the embedding and n-gram tables (one
row per token) and vision tensors, and per tensor family reports GB and the share of 64-weight groups whose
exponents fit a 3- or 4-bit offset from the group's minimum. CPU, mmap, read-only.
Usage: check_bf16_exponents.py [model_dir]"""
import glob, json, os, re, struct, sys
from collections import defaultdict
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/models/bare")
G = 64
fam = defaultdict(lambda: [0, 0, 0, 0])          # weights, groups, fit3, fit4
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    if "ngram" in f:
        continue
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, m in hdr.items():
        if name == "__metadata__" or m["dtype"] != "BF16":
            continue
        if re.search(r"embed_tokens|ngram|visual|vision", name):
            continue
        a, b = m["data_offsets"]
        cnt = (b - a) // 2 // G * G
        if cnt == 0:
            continue
        w = np.memmap(f, np.uint16, "r", offset=8 + n + a, shape=(cnt,))
        e = ((np.asarray(w) >> 7) & 0xFF).astype(np.int16).reshape(-1, G)
        spread = e.max(1) - e.min(1)
        key = re.sub(r"\.\d+\.", ".N.", name.replace("language_model.model.", ""))
        key = re.sub(r"^(mtp\.)?layers\.N\.", r"\1", key)
        r = fam[key]
        r[0] += cnt; r[1] += len(e); r[2] += int((spread < 8).sum()); r[3] += int((spread < 16).sum())

tot = sum(r[0] for r in fam.values())
print(f"bf16 tensors read per token (excl. embed/ngram/vision): {tot*2/1e9:.2f} GB")
saved = 0.0
for k, (wn, g, f3, f4) in sorted(fam.items(), key=lambda kv: -kv[1][0])[:15]:
    best = 16
    for bits, fit in ((3, f3), (4, f4)):
        fr = fit / g
        best = min(best, fr * (8 + bits) + (1 - fr) * 16 + 9 / G)
    saved += wn * 2 * (1 - best / 16)
    print(f"  {k[:60]:60s} {wn*2/1e9:6.3f} GB  fit3 {f3/g:5.1%}  fit4 {f4/g:5.1%}  -> {best:5.2f} bits")
print(f"top families: lossless pack saves {saved/1e9:.2f} GB/token (bf16 total {tot*2/1e9:.2f} GB)")
