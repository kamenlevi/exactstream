"""Round 28: the hyper-connection read with bf16 vs 8-bit mixers (hc8.py), as MTPLX's eager read computes it (grouped
RMSNorm over the 4 substreams, input_mix_weight_down [320, 10240], silu, input_mix_weight_up [10240, 320], sigmoid, mix,
mean over substreams), real weights of 48 layers' mlp hyper-connections, 48 dependent reads in one command buffer, M = 1
and 4 rows.   usage: hc8_op_bench.py   (GPU ~2 GB, quiet lock)"""
import json, os, struct, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
_h = {}


def tensor(name):
    f = f"{MD}/{idx[name]}"
    if f not in _h:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _h[f] = (n, json.loads(fh.read(n)))
    n, h = _h[f]; m = h[name]; a, b = m["data_offsets"]
    return mx.array(np.array(np.memmap(f, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,))).reshape(m["shape"])).view(mx.bfloat16)


HC, D = 4, 2560
layers = []
for L in range(48):
    p = f"language_model.model.layers.{L}.mlp_hyper_connection"
    down, up, norm = nn.Linear(HC * D, 320, bias=False), nn.Linear(320, HC * D, bias=False), tensor(f"{p}.hc_norm.weight")
    down.weight, up.weight = tensor(f"{p}.input_mix_weight_down.weight"), tensor(f"{p}.input_mix_weight_up.weight")
    layers.append((down, up, norm))
q8 = [(nn.QuantizedLinear.from_linear(d, 64, 8), nn.QuantizedLinear.from_linear(u, 64, 8), n) for d, u, n in layers]
mx.eval([[l.parameters() for l in t[:2]] + [t[2]] for t in layers + q8])


def read(h, down, up, norm):
    g = h.reshape(*h.shape[:-1], HC, D)
    normed = (mx.fast.rms_norm(g, None, 1e-6).reshape(h.shape) * norm)
    mix = mx.sigmoid(up(nn.silu(down(normed) / HC)))
    return mx.mean(mix.reshape(*mix.shape[:-1], HC, D) * normed.reshape(*normed.shape[:-1], HC, D), axis=-2)


def bench(mods, M, iters=8):
    h0 = (mx.random.normal((1, M, HC * D)) * 0.5).astype(mx.bfloat16)
    mx.eval(h0)

    def once():
        h = h0
        for d, u, n in mods:
            y = read(h, d, u, n)
            h = h0 + (y.astype(mx.float32).sum() * 0).astype(h0.dtype)
        mx.eval(h)
    once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    return (time.perf_counter() - t0) / iters / len(mods) * 1e3


b16 = sum(d.weight.nbytes + u.weight.nbytes for d, u, _ in layers) / 48
b8 = sum(sum(a.nbytes for a in (d.weight, d.scales, d.biases, u.weight, u.scales, u.biases)) for d, u, _ in q8) / 48
for M in (1, 4):
    t16, t8 = bench(layers, M), bench(q8, M)
    print(f"M={M}: HC read bf16 {t16:.4f} ms ({b16 / 1e6 / t16:.0f} GB/s of mixer bytes) | 8-bit {t8:.4f} ms "
          f"({b8 / 1e6 / t8:.0f} GB/s) | {t16 / t8 - 1:+.0%}", flush=True)
