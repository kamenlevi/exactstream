#!/usr/bin/env python3
"""Fetch ONLY the native MTP head tensors (mtp.*) of Qwen/Qwen3.5-2B from its single safetensors file, by HTTP range
requests (header first, then each tensor's byte range). ~121 MB instead of the 4.5 GB file. Writes
dl/qwen35_2b_mtp/mtp.safetensors (bf16, original names) + fetch.json (sizes, sha256 of each tensor's bytes)."""
import hashlib, json, os, struct, subprocess, sys
URL = "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors-00001-of-00001.safetensors"
OUT = os.path.expanduser("~/Sandbox98/dl/qwen35_2b_mtp")


def rng(a, b):                      # inclusive byte range
    r = subprocess.run(["curl", "-sSL", "--fail", "-m", "600", "-r", f"{a}-{b}", URL], capture_output=True, check=True)
    assert len(r.stdout) == b - a + 1, (a, b, len(r.stdout))
    return r.stdout


n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1))
base = 8 + n
mtp = {k: v for k, v in hdr.items() if k.startswith("mtp.")}
tot = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in mtp.values())
print(f"header {n} bytes, {len(hdr) - 1} tensors; mtp: {len(mtp)} tensors, {tot / 1e6:.1f} MB", flush=True)
new_hdr, blobs, off, info = {}, [], 0, {}
for k in sorted(mtp):
    v = mtp[k]
    a, b = v["data_offsets"]
    data = rng(base + a, base + b - 1)
    new_hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + len(data)]}
    blobs.append(data); off += len(data)
    info[k] = {"dtype": v["dtype"], "shape": v["shape"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    print(f"  {k} {v['dtype']} {v['shape']} {len(data)} B", flush=True)
hj = json.dumps(new_hdr).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
with open(os.path.join(OUT, "mtp.safetensors.tmp"), "wb") as f:
    f.write(struct.pack("<Q", len(hj))); f.write(hj)
    for d in blobs:
        f.write(d)
os.replace(os.path.join(OUT, "mtp.safetensors.tmp"), os.path.join(OUT, "mtp.safetensors"))
json.dump({"source": URL, "total_bytes": off, "tensors": info}, open(os.path.join(OUT, "fetch.json"), "w"), indent=1)
print(f"wrote {OUT}/mtp.safetensors ({off / 1e6:.1f} MB)")
