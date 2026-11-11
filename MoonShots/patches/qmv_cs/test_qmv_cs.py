"""Round 31 check: qmv_cs (chunk-streaming, v2b) vs mx.quantized_matmul on the 125B's real 4-bit projections (read from the bare model's shards):
bitwise equality at M = 2..4, then ms per call and GB/s with all layers' instances of one projection in one command
buffer, dependent calls (x_next = x + 0 * y.sum()), as module_bench125.py.   usage: test_qmv_hb.py   (GPU, quiet lock)"""
import json, os, struct, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qmv_hb"))
from qmv_hb import qmv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qmv_cs"))
import qmv_cs
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
_h = {}


def tensor(name):
    f = f"{MD}/{idx[name]}"
    if f not in _h:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _h[f] = (n, json.loads(fh.read(n)))
    n, h = _h[f]
    m = h[name]; a, b = m["data_offsets"]
    dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


def lin(prefix):
    return tuple(tensor(f"{prefix}.{f}") for f in ("weight", "scales", "biases"))


cfg = json.load(open(f"{MD}/config.json")); cfg = cfg.get("text_config", cfg)
types = cfg["layer_types"]
gdn = [L for L, t in enumerate(types) if t == "linear_attention"]
qsa = [L for L, t in enumerate(types) if t != "linear_attention"]
P = "language_model.model.layers"
SETS = {
    "gdn.in_proj_qkv": [lin(f"{P}.{L}.linear_attn.in_proj_qkv") for L in gdn],
    "gdn.in_proj_z": [lin(f"{P}.{L}.linear_attn.in_proj_z") for L in gdn],
    "gdn.out_proj": [lin(f"{P}.{L}.linear_attn.out_proj") for L in gdn],
    "qsa.q_proj": [lin(f"{P}.{L}.self_attn.q_proj") for L in qsa],
    "qsa.o_proj": [lin(f"{P}.{L}.self_attn.o_proj") for L in qsa],
    "lm_head": [lin("language_model.lm_head")],
}
for v in SETS.values():
    mx.eval(*[t for l in v for t in l])


def ref(x, w, s, b):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)


def hb(x, w, s, b):
    return qmv(x, w, s, b, impl="hb")


def mlxw(x, w, s, b):
    return qmv(x, w, s, b, impl="mlx")


def cs4(x, w, s, b):
    return qmv_cs.qmv(x, w, s, b, rows_per_sg=4, sgs=8)


def cs8(x, w, s, b):
    return qmv_cs.qmv(x, w, s, b, rows_per_sg=8, sgs=4)


mx.random.seed(0)
print("relative error vs mx.quantized_matmul (max |diff| / max |ref|, M = 1..4):", flush=True)
for name, mats in SETS.items():
    w, s, b = mats[0]
    K = w.shape[1] * 8
    worst = {}
    for M in (1, 2, 3, 4):
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
        r = ref(x, w, s, b)
        for nm, fn in (("cs4", cs4), ("cs8", cs8)):
            d = mx.abs(fn(x, w, s, b).astype(mx.float32) - r.astype(mx.float32)).max().item()
            worst[nm] = max(worst.get(nm, 0.0), d / mx.abs(r.astype(mx.float32)).max().item())
    print(f"  {name:16s} {tuple(w.shape)}  cs4 {worst['cs4']:.2e}  cs8 {worst['cs8']:.2e}", flush=True)


def bench(fn, mats, M, iters=6):
    K = mats[0][0].shape[1] * 8
    x0 = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(x0)
    reps = max(1, 48 // len(mats))

    def once():
        x = x0
        for _ in range(reps):
            for w, s, b in mats:
                y = fn(x, w, s, b)
                x = x0 + (y.astype(mx.float32).sum() * 0).astype(x0.dtype)
        mx.eval(x)
    once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    n = reps * len(mats)
    return (time.perf_counter() - t0) / iters / n * 1e3


print("\nms per call (GB/s), dependent calls in one command buffer:", flush=True)
for name, mats in SETS.items():
    by = sum(t.nbytes for t in mats[0])
    for M in (1, 2, 4):
        tr, t2, t4 = bench(ref, mats, M), bench(cs4, mats, M), bench(cs8, mats, M)
        print(f"  {name:16s} M={M}  {by / 1e6:7.2f} MB  mlx {tr:.4f} ({by / 1e6 / tr:4.0f})  cs4 {t2:.4f} ({by / 1e6 / t2:4.0f}, {tr / t2 - 1:+.0%})  "
              f"cs8 {t4:.4f} ({by / 1e6 / t4:4.0f}, {tr / t4 - 1:+.0%})", flush=True)
