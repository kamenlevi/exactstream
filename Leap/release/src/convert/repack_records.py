"""One-record expert pack (DESIGN_record.md): per layer, models/bare/records/L{L:02d}.bin holds E records, each the 9
pieces of one expert (gate/up/down x weight/scales/biases, loader order) back to back, the record padded to a 16 KB
multiple (2,768,896 B for the 125B). Same bytes as the safetensors pieces (pure relayout). F_NOCACHE on both ends so the
70 GB pass does not flood the page cache. Stops before the disk falls under 110 GB free. Resumable (skips complete files).
Usage: repack_records.py [model_dir] [--layers L ...] [--min-free-gb 110] [--out DIR]   (CPU / SSD only)
Package copy: the research tool with three options added (--layers, --min-free-gb, --out; defaults = the original
behaviour); the relayout loop is unchanged."""
import fcntl, json, os, shutil, struct, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("model_dir", nargs="?", default="~/BigMoE/models/bare")
_ap.add_argument("--layers", type=int, nargs="*", help="only these layers (default: all)")
_ap.add_argument("--min-free-gb", type=float, default=110.0, help="stop before the free disk falls under this")
_ap.add_argument("--out", help="the records directory (default: <model_dir>/records)")
_A = _ap.parse_args()
MD = os.path.expanduser(_A.model_dir)
OUT = os.path.expanduser(_A.out) if _A.out else os.path.join(MD, "records"); os.makedirs(OUT, exist_ok=True)
PIECES = [(p, f) for p in ("gate_proj", "up_proj", "down_proj") for f in ("weight", "scales", "biases")]
DTS = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1}
ALIGN = 16384
F_NOCACHE = 48


def headers():
    idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(idx.values())):
        p = os.path.join(MD, f)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (p, 8 + n + v["data_offsets"][0], v["dtype"], v["shape"])
    return out


hdr = headers()
fds = {}
layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in hdr if ".mlp.switch_mlp.gate_proj.weight" in k})
index = {"align": ALIGN, "layers": {}}
for L in [l for l in layers if not _A.layers or l in _A.layers]:
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp"
    info = []
    for proj, field in PIECES:
        path, off, dt, shape = hdr[f"{pre}.{proj}.{field}"]
        per = int(np.prod(shape[1:])) * DTS[dt]
        info.append((path, off, per))
    E = hdr[f"{pre}.gate_proj.weight"][3][0]
    body = sum(p for _, _, p in info)
    recsz = (body + ALIGN - 1) // ALIGN * ALIGN
    inrec = list(np.cumsum([0] + [p for _, _, p in info])[:-1].astype(int))
    dst = os.path.join(OUT, f"L{L:02d}.bin")
    index["layers"][str(L)] = {"file": os.path.basename(dst), "E": int(E), "recsz": int(recsz),
                               "piece_off": [int(x) for x in inrec], "piece_size": [int(p) for _, _, p in info]}
    if os.path.exists(dst) and os.path.getsize(dst) == E * recsz:
        continue
    if shutil.disk_usage(OUT).free < _A.min_free_gb * 1e9 + E * recsz:
        print(f"stop at layer {L}: disk would fall under {_A.min_free_gb:.0f} GB free", flush=True); break
    for path, _, _ in info:
        if path not in fds:
            fds[path] = os.open(path, os.O_RDONLY); fcntl.fcntl(fds[path], F_NOCACHE, 1)
    out = os.open(dst + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644); fcntl.fcntl(out, F_NOCACHE, 1)
    pad = bytes(recsz - body)
    for e0 in range(0, E, 16):                          # 16 experts (~44 MB) per write
        chunk = bytearray()
        for e in range(e0, min(E, e0 + 16)):
            for path, off, per in info:
                b = os.pread(fds[path], per, off + e * per)
                assert len(b) == per
                chunk += b
            chunk += pad
        os.write(out, chunk)
    os.close(out); os.replace(dst + ".tmp", dst)
    print(f"layer {L}: {E} records x {recsz} B -> {dst}", flush=True)
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print("done; free", round(shutil.disk_usage("/").free / 1e9), "GB", flush=True)
