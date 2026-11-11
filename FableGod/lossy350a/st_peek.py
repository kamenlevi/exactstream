import json, os, struct, sys
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
keys = [k for k in idx if ".layers.24.mlp." in k]
print(len(idx), "tensors;", len(keys), "under layers.24.mlp")
hdr = {}
for k in sorted(keys):
    f = idx[k]
    if f not in hdr:
        with open(os.path.join(MD, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr[f] = (n, json.loads(fh.read(n)))
    n, h = hdr[f]
    t = h[k]
    print(k, f, t["dtype"], t["shape"], t["data_offsets"][1] - t["data_offsets"][0])
