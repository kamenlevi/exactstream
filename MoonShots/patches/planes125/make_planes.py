"""Write MSB/LSB plane files for the 125B's routed experts (plane-granular streaming prototype, DESIGN.md). CPU only (numpy),
reads ~/BigMoE/models/bare read-only. Per layer two files, piece-major (as safetensors), fixed bytes per expert:
  L{L}.msb : for gate, up, down: codes m = q>>2 packed 2-bit (uint32 [rows, K/16]), scales 4s (bf16 [rows, K/64]),
             bias b + s*off (bf16), off = per-group mean of l = q&3                         -> 1.54 MB per expert
  L{L}.lsb : for gate, up, down: codes l packed 2-bit, bias b - bias_msb (bf16)              -> 1.38 MB per expert
The LSB term runs with scale s = (4s)/4 taken from the resident MSB plane; MSB term + LSB term = the 4-bit weight up to
bf16 rounding of the two biases. ~72 GB for 48 layers: needs Kamen's OK (the disk floor is 200 GB free).
layout.json records piece names, dtypes, shapes and byte offsets within an expert's record.
usage: make_planes.py OUTDIR [first_layer last_layer]"""
import json, os, struct, sys, time
import numpy as np
OUT = sys.argv[1]
L0, L1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 47)
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
os.makedirs(OUT, exist_ok=True)
st = os.statvfs(OUT)
free_gb = st.f_bavail * st.f_frsize / 1e9
need_gb = (L1 - L0 + 1) * 512 * 2.92e6 / 1e9
assert free_gb - need_gb >= float(os.environ.get("PLANES_MIN_FREE_GB", "200")), (free_gb, need_gb)


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16}[m["dtype"]]
    return np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,)).reshape(m["shape"]), m["dtype"]


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):                     # round to nearest even
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def planes(w, s, b):
    """w uint32 [E, rows, K/8] (4-bit codes), s/b bf16-as-uint16 [E, rows, K/64] -> MSB and LSB pieces."""
    m16 = np.zeros(w.shape, np.uint32); l16 = np.zeros(w.shape, np.uint32); lsum = np.zeros(w.shape, np.uint32)
    for j in range(8):
        nib = (w >> np.uint32(4 * j)) & np.uint32(15)
        m16 |= (nib >> np.uint32(2)) << np.uint32(2 * j)
        l16 |= (nib & np.uint32(3)) << np.uint32(2 * j)
        lsum += nib & np.uint32(3)
    pack = lambda v: v[..., 0::2] | (v[..., 1::2] << np.uint32(16))           # 4-bit words 2k, 2k+1 -> 2-bit word k
    off = lsum.reshape(*s.shape, 8).sum(-1).astype(np.float32) / 64
    sf, bfl = bf16_to_f32(s), bf16_to_f32(b)
    s4 = f32_to_bf16(4 * sf)
    bm = f32_to_bf16(bfl + sf * off)
    bl = f32_to_bf16(bfl - bf16_to_f32(bm))
    return (pack(m16), s4, bm), (pack(l16), bl)


layout, t0 = None, time.time()
for L in range(L0, L1 + 1):
    msb, lsb = [], []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{proj}."
        w, _ = tensor(pre + "weight"); s, sdt = tensor(pre + "scales"); b, _ = tensor(pre + "biases")
        assert sdt == "BF16", sdt
        (mc, ms, mb), (lc, lb) = planes(np.array(w), np.array(s), np.array(b))
        msb += [(f"{proj}.codes", mc), (f"{proj}.scales4", ms), (f"{proj}.bias", mb)]
        lsb += [(f"{proj}.codes", lc), (f"{proj}.bias", lb)]
    for kind, pieces in (("msb", msb), ("lsb", lsb)):
        # piece-major like safetensors: piece i of expert e sits at offset_i + e * bytes_i (BigMoE ext.add_layer addressing)
        layout = layout or {}
        if kind not in layout:
            off, items = 0, []
            for n, a in pieces:
                items.append({"name": n, "dtype": str(a.dtype), "shape": list(a.shape[1:]), "offset": off,
                              "bytes": int(a[0].nbytes)}); off += int(a.nbytes)
            layout[kind] = {"pieces": items, "expert_bytes": sum(it["bytes"] for it in items), "file_bytes": off}
        with open(os.path.join(OUT, f"L{L}.{kind}"), "wb") as fh:
            for _, a in pieces:
                fh.write(np.ascontiguousarray(a).tobytes())
    print(f"L{L}: done ({time.time() - t0:.0f}s)", flush=True)
json.dump(layout, open(os.path.join(OUT, "layout.json"), "w"), indent=1)
print(f"layers {L0}-{L1} written to {OUT}: MSB {layout['msb']['expert_bytes'] / 1e6:.2f} MB, LSB "
      f"{layout['lsb']['expert_bytes'] / 1e6:.2f} MB per expert")
