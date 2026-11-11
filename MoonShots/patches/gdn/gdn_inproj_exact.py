"""Is MTPLX_FUSED_GDN_INPROJ exact on our M? Real 125B GDN in_proj weights (qkv 8192, z 6144, b 48, a 48 rows; K=2560,
4-bit g64), separate quantized_matmuls vs one row-concatenated matmul split back, for M = 1..8 rows (S=1 runs on [x; x]
under dup-row, verify on S rows). Bitwise per part, plus time per GDN layer x 36 layers. usage: gdn_inproj_exact.py [L]"""
import json, os, struct, sys, time
import numpy as np
import mlx.core as mx
L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]; dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


pre = f"language_model.model.layers.{L}.linear_attn."
parts = [tuple(tensor(pre + f"{p}.{f}") for f in ("weight", "scales", "biases")) for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
rows = [w.shape[0] for w, _, _ in parts]
fw, fs, fb = (mx.concatenate([p[i] for p in parts], axis=0) for i in range(3))
mx.eval(fw, fs, fb, parts)
splits = list(np.cumsum(rows)[:-1])
qm = lambda x, w, s, b: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
print(f"layer {L}: rows {rows}, fused {fw.shape[0]}", flush=True)
for M in (1, 2, 3, 4, 5, 6, 8):
    x = (mx.random.normal((1, M, 2560)) * 0.5).astype(mx.bfloat16)
    sep = [qm(x, *p) for p in parts]
    fus = mx.split(qm(x, fw, fs, fb), splits, axis=-1)
    same = [mx.array_equal(a, b).item() for a, b in zip(sep, fus)]

    def chain(fused, n=30):                         # dependent chain: the next x depends on every output part
        y = x
        for _ in range(n):
            outs = mx.split(qm(y, fw, fs, fb), splits, axis=-1) if fused else [qm(y, *p) for p in parts]
            y = (y + sum(o.astype(mx.float32).sum() for o in outs) * 0).astype(mx.bfloat16)
        mx.eval(y)

    def bench(fused):
        for _ in range(3): chain(fused)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); chain(fused); ts.append((time.perf_counter() - t0) / 30)
        return sorted(ts)[3]
    a, b = bench(False), bench(True)
    print(f"M={M}: bit-identical per part {same} | separate {a*1e3:.3f} ms, fused {b*1e3:.3f} ms per layer (dependent chain) "
          f"-> x36 {((a - b) * 36e3):+.2f} ms/pass", flush=True)
