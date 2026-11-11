"""Fetch only the MTP module (31 bf16 tensors, ~8 GB) of the official Qwen/Qwen3.8-Flash-Next checkpoint with HTTP range
requests, for the MTP-precision test (diagnosis (c), orchestrator OK 13:25: official repo, MTP module only, keep >= ~100 GB
free). The tensors are spread over 28 of the 131 shards (360 GB total), so whole-shard downloads (~77 GB) are avoided:
each shard's safetensors header is read (8 + n bytes), then only the MTP tensors' byte ranges are fetched and streamed into
one local safetensors file with the official names.   usage: fetch_mtp_bf16.py OUT.safetensors"""
import json, os, struct, subprocess, sys, time, urllib.request
REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"
OUT = sys.argv[1]
idx = json.load(urllib.request.urlopen(f"{REPO}/raw/main/model.safetensors.index.json"))["weight_map"]
mtp = {k: f for k, f in idx.items() if k.startswith("mtp.")}
print(f"{len(mtp)} mtp tensors in {len(set(mtp.values()))} shards", flush=True)


def rng(fname, a, b):
    """bytes [a, b) of a shard (curl keeps Range across the CDN redirect)."""
    r = subprocess.run(["curl", "-sfL", "-r", f"{a}-{b - 1}", f"{REPO}/resolve/main/{fname}"], capture_output=True, check=True)
    assert len(r.stdout) == b - a, (fname, a, b, len(r.stdout))
    return r.stdout


plan = []                                            # (name, dtype, shape, shard, abs_start, abs_end)
for fname in sorted(set(mtp.values())):
    n = struct.unpack("<Q", rng(fname, 0, 8))[0]
    hdr = json.loads(rng(fname, 8, 8 + n))
    for name, m in hdr.items():
        if name.startswith("mtp."):
            a, b = m["data_offsets"]
            plan.append((name, m["dtype"], m["shape"], fname, 8 + n + a, 8 + n + b))
plan.sort()
total = sum(e - s for *_, s, e in plan)
st = os.statvfs(os.path.dirname(os.path.abspath(OUT)))
free = st.f_bavail * st.f_frsize / 1e9
print(f"to fetch: {total / 1e9:.2f} GB in {len(plan)} tensors; free {free:.0f} GB", flush=True)
assert free - total / 1e9 >= 110, "would go below the 110 GB margin"
header, off = {}, 0
for name, dt, shape, _, s, e in plan:
    header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + (e - s)]}; off += e - s
header["__metadata__"] = {"source": "Qwen/Qwen3.8-Flash-Next (official), MTP tensors only, fetched by range"}
hb = json.dumps(header).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
t0 = time.time()
with open(OUT + ".part", "wb") as fh:
    fh.write(struct.pack("<Q", len(hb))); fh.write(hb)
    for i, (name, dt, shape, fname, s, e) in enumerate(plan):
        CH = 256 << 20                               # 256 MB range pieces for the big expert tensors
        for a in range(s, e, CH):
            fh.write(rng(fname, a, min(e, a + CH)))
        print(f"  {i + 1}/{len(plan)} {name} {dt} {shape} ({(e - s) / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
os.replace(OUT + ".part", OUT)
print(f"done: {OUT} ({os.path.getsize(OUT) / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
