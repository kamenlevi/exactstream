"""Fetch small 1-D (norm) tensors of the official Qwen/Qwen3.8-Flash-Next checkpoint by HTTP range and print the same
stats as st_peek.py, to compare the raw (HF) norm convention with BigMoE's converted bare model.
usage: official_peek.py REGEX   (only 1-D tensors under 1 MB are fetched)"""
import json, re, struct, subprocess, sys, urllib.request
import numpy as np
REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"
pat = re.compile(sys.argv[1])
idx = json.load(urllib.request.urlopen(f"{REPO}/raw/main/model.safetensors.index.json"))["weight_map"]


def rng(fname, a, b):
    r = subprocess.run(["curl", "-sfL", "-r", f"{a}-{b - 1}", f"{REPO}/resolve/main/{fname}"], capture_output=True, check=True)
    assert len(r.stdout) == b - a
    return r.stdout


want = {}
for k, f in idx.items():
    if pat.search(k):
        want.setdefault(f, []).append(k)
for f, keys in sorted(want.items()):
    n = struct.unpack("<Q", rng(f, 0, 8))[0]
    hdr = json.loads(rng(f, 8, 8 + n))
    for k in sorted(keys):
        m = hdr[k]
        a, b = m["data_offsets"]
        if len(m["shape"]) != 1 or b - a > 1 << 20 or m["dtype"] != "BF16":
            continue
        v = (np.frombuffer(rng(f, 8 + n + a, 8 + n + b), np.uint16).astype(np.uint32) << 16).view(np.float32)
        print(f"{k:70s} BF16 n={v.size:6d} mean {v.mean():+.4f} min {v.min():+.4f} max {v.max():+.4f}  {np.round(v[:6], 3).tolist()}", flush=True)
