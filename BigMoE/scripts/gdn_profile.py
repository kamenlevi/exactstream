"""GDN sub-op profile (outside review R3): each step of a 125B GDN layer, timed over 36 layers with distinct weights
(so nothing stays in cache), at M rows (2 = decode under dup-row, 4 = verify S=4), vs its memory floor at 200 GB/s.
Shapes: hidden 2560; q/k 16 heads x 128, v/z 48 x 128 (conv_dim 10240); Q4 g64 projections; fp32 state 48x128x128."""
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
AL = mx.zeros((48,), mx.float32); DB = mx.zeros((48,), mx.float32)   # per-head decay params (A_log, dt_bias)
H, Hk, Hv, D, L = 2560, 16, 48, 128, 36
CONV = 2 * Hk * D + Hv * D                         # 10240
def ql(i, o):
    m = nn.QuantizedLinear(i, o, bias=False, group_size=64, bits=4); return m
layers = []
for _ in range(L):
    layers.append(dict(qkv=ql(H, CONV), z=ql(H, Hv * D), a=ql(H, Hv), b=ql(H, Hv), out=ql(Hv * D, H),
                       conv=(mx.random.normal((CONV, 4, 1)) * 0.1).astype(mx.bfloat16),
                       norm=mx.ones((D,), mx.bfloat16), state=mx.zeros((1, Hv, D, D), mx.float32)))
mx.eval([[v if isinstance(v, mx.array) else v.parameters() for v in d.values()] for d in layers])
def t(fn, reps=15):
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - a)
    return np.median(ts) * 1e3
for M in (2, 4):
    x = (mx.random.normal((1, M, H)) * 0.1).astype(mx.bfloat16)
    qkv = [d["qkv"](x) for d in layers]; mx.eval(qkv)
    tail = mx.zeros((1, 3, CONV), mx.bfloat16)
    def conv_norm():
        out = []
        for d, y in zip(layers, qkv):
            c = nn.silu(mx.conv1d(mx.concatenate([tail, y], 1), d["conv"], groups=CONV)[:, -M:])
            q, k, v = mx.split(c, [Hk * D, 2 * Hk * D], -1)
            q = q.reshape(1, M, Hk, D); k = k.reshape(1, M, Hk, D)
            q = q * mx.rsqrt((q.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-6).astype(q.dtype)
            k = k * mx.rsqrt((k.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-6).astype(k.dtype)
            out.append((q, k, v.reshape(1, M, Hv, D)))
        return out
    cn = conv_norm(); mx.eval(cn)
    g = mx.random.uniform(shape=(1, M, Hv)).astype(mx.bfloat16); bt = mx.random.uniform(shape=(1, M, Hv)).astype(mx.bfloat16)
    rec = [gated_delta_update(q, k, v, g, bt, AL, DB, state=d["state"]) for d, (q, k, v) in zip(layers, cn)]; mx.eval(rec)
    yv = [r[0] for r in rec]
    zz = [d["z"](x) for d in layers]; mx.eval(zz)
    def gate_norm():
        return [(mx.fast.rms_norm(y.reshape(1, M, Hv, D), d["norm"], 1e-6) * mx.sigmoid(z.reshape(1, M, Hv, D).astype(mx.float32)).astype(mx.bfloat16)).reshape(1, M, Hv * D)
                for d, y, z in zip(layers, yv, zz)]
    gn = gate_norm(); mx.eval(gn)
    W = lambda o, i: o * i / 2 * 1.125                # Q4 g64 bytes incl. scales/biases
    rows = [
        ("in_proj qkv", lambda: [d["qkv"](x) for d in layers], W(CONV, H)),
        ("in_proj z", lambda: [d["z"](x) for d in layers], W(Hv * D, H)),
        ("in_proj a+b", lambda: [d["a"](x) + d["b"](x) for d in layers], 2 * W(Hv, H)),
        ("conv+silu+l2norm", conv_norm, M * CONV * 2 * 5),
        ("recurrence", lambda: [gated_delta_update(q, k, v, g, bt, AL, DB, state=d["state"])[1] for d, (q, k, v) in zip(layers, cn)], 2 * Hv * D * D * 4),
        ("gated rms norm", gate_norm, M * Hv * D * 2 * 3),
        ("out_proj", lambda: [d["out"](y) for d, y in zip(layers, gn)], W(H, Hv * D)),
    ]
    print(f"M={M} (36 layers):")
    for name, fn, by in rows:
        ms = t(fn); fl = L * by / 200e9 * 1e3
        print(f"  {name:18s} {ms:6.2f} ms   floor {fl:5.2f} ms   x{ms / max(fl, 1e-3):5.1f}", flush=True)
