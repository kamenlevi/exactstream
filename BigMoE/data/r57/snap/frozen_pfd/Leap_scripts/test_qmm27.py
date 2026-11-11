"""L1-27 test (Leap, Sat 2026-10-24): 4-bit operands fed straight to matmul2d (qmm27.py).
1. Dense (random 4-bit weights, per-check shapes / counts, cold): max |err| vs an fp32 reference (dequantized weights) for
   MLX, qmm16c and qmm16i; batch invariance (rows 1 / 2 / 5 / 8 / 12 alone vs inside the 16-row call, bitwise); ms per
   check vs qmm16c and qmm16n (L1-26) and the floor.
2. Routed experts (real 4-bit weights of 4 layers): error vs fp32 (experts_nax vs experts_int4), invariance, ms per layer
   at 4 / 8 / 16 rows vs experts_nax / experts_native and the floor.
Usage: [BigMoE venv] test_qmm27.py -> ~/Leap/data/qmm27.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts"))
from qmm16 import qmm16c
from qmm26 import repack, qmm16n, experts_native
from qmmg25 import experts_nax
from qmm27 import qmm16i, experts_int4

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

mx.random.seed(27)
say("L1-27: 4-bit operands straight into matmul2d (bfloat x uint4b_format -> float), vs qmm16c / qmm16n / MLX")
X = (mx.random.normal((32, 2560)) * 0.5).astype(mx.bfloat16)
for name, N, K, cnt, cfg in (("in_proj_qkv", 10240, 2560, 36, (2, 2)), ("o_proj+out_proj", 2560, 6144, 48, (2, 2)),
                             ("in_proj_z", 6144, 2560, 36, (1, 1))):
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(cnt)]
    mx.eval(Ws)
    x = (mx.random.normal((32, K)) * 0.5).astype(mx.bfloat16)
    ref = x[:16].astype(mx.float32) @ mx.dequantize(*Ws[0], group_size=64, bits=4).astype(mx.float32).T
    err = lambda y: float(mx.abs(y.astype(mx.float32) - ref).max())
    e_mlx = err(mx.quantized_matmul(x[:16], *Ws[0], transpose=True, group_size=64, bits=4))
    e_c = err(qmm16c(x[:16], *Ws[0], ks=cfg[0], nt=cfg[1]))
    y16 = qmm16i(x[:16], *Ws[0])
    e_i = err(y16)
    full = bits(y16)
    inv = all(np.array_equal(bits(qmm16i(x[:M], *Ws[0])), full[:M]) for M in (1, 2, 5, 8, 12))
    floor = cnt * sum(a.nbytes for a in Ws[0]) / 280e9 * 1e3
    t_c = timeit(lambda: [qmm16c(x[:16], *W, ks=cfg[0], nt=cfg[1]) for W in Ws])
    Wn = [repack(*W) for W in Ws]; mx.eval(Wn)
    t_n = timeit(lambda: [qmm16n(x[:16], *W, N, *cfg) for W in Wn])
    t_i = timeit(lambda: [qmm16i(x[:16], *W) for W in Ws])
    say(f"  {name:16s} {N}x{K} x{cnt}: max |err| vs fp32 MLX {e_mlx:.4g}, qmm16c {e_c:.4g}, int4 {e_i:.4g}; invariant {inv}; "
        f"floor {floor:.2f} ms; qmm16c {t_c:.2f} ({floor / t_c:.0%}), native {t_n:.2f} ({floor / t_n:.0%}), "
        f"int4 {t_i:.2f} ({floor / t_i:.0%})")
    del Ws, Wn

MD = os.path.expanduser("~/BigMoE/models/bare")
WMAP = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def load_layer(L):
    p = f"language_model.model.layers.{L}.mlp.switch_mlp."
    keys = [p + f"{n}.{s}" for n in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "scales", "biases")]
    got = {}
    for f in sorted({WMAP[k] for k in keys}):
        d = mx.load(os.path.join(MD, f))
        got.update({k: d[k] for k in keys if k in d})
        del d
    gu = [mx.concatenate([got[p + f"gate_proj.{s}"], got[p + f"up_proj.{s}"]], axis=1) for s in ("weight", "scales", "biases")]
    dn = [got[p + f"down_proj.{s}"] for s in ("weight", "scales", "biases")]
    mx.eval(gu, dn)
    return gu, dn
LAYERS = (12, 20, 28, 36)
W = [load_layer(L) for L in LAYERS]
WN = [(repack(*gu), repack(*dn)) for gu, dn in W]
mx.eval(WN)
E = 512
EB = sum(a[0].nbytes for a in W[0][0]) + sum(a[0].nbytes for a in W[0][1])
rng = np.random.default_rng(27)
def routing(rows, k=10):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))
def ref32(x, gu, dn, ids):
    out = []
    xf = x.astype(mx.float32)
    for r in range(ids.shape[0]):
        row = []
        for e in np.array(ids[r]).tolist():
            Wg = mx.dequantize(gu[0][e], gu[1][e], gu[2][e], group_size=64, bits=4).astype(mx.float32)
            Wd = mx.dequantize(dn[0][e], dn[1][e], dn[2][e], group_size=64, bits=4).astype(mx.float32)
            g = xf[r] @ Wg.T
            gate, up = g[:g.shape[0] // 2], g[g.shape[0] // 2:]
            h = (gate * mx.sigmoid(gate) * up).astype(mx.bfloat16).astype(mx.float32)
            row.append(h @ Wd.T)
        out.append(mx.stack(row))
    return mx.stack(out)
x = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
ids4 = routing(4)
ref = ref32(x[:4], *W[2], ids4)
e_n = float(mx.abs(experts_nax(x[:4], *W[2], ids4, (4, 1), (1, 2)).astype(mx.float32) - ref).max())
e_i = float(mx.abs(experts_int4(x[:4], *W[2], ids4).astype(mx.float32) - ref).max())
ids16 = routing(16)
full = bits(experts_int4(x, *W[1], ids16))
inv = " ".join(f"{M}:{'=' if np.array_equal(bits(experts_int4(x[:M], *W[1], ids16[:M])), full[:M]) else 'DIFF'}"
               for M in (1, 2, 4, 7, 8, 12))
say(f"  experts: max |err| vs fp32 (4 rows, layer {LAYERS[2]}): experts_nax {e_n:.4g}, int4 {e_i:.4g}; invariance {inv}")
for rows in (4, 8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    floor = U * EB / 280e9 * 1e3
    t_n = timeit(lambda: [experts_nax(x[:rows], *W[i], R[i], (4, 1), (1, 2)) for i in range(len(LAYERS))]) / len(LAYERS)
    t_v = timeit(lambda: [experts_native(x[:rows], *WN[i], R[i], (4, 1), (1, 2)) for i in range(len(LAYERS))]) / len(LAYERS)
    t_i = timeit(lambda: [experts_int4(x[:rows], *W[i], R[i]) for i in range(len(LAYERS))]) / len(LAYERS)
    say(f"  {rows:2d} rows ({U:.0f} distinct experts): floor {floor:.3f} ms per layer; experts_nax {t_n:.3f} ({floor / t_n:.0%}), "
        f"native {t_v:.3f} ({floor / t_v:.0%}), int4 {t_i:.3f} ({floor / t_i:.0%}); per 48-layer check: nax {48 * t_n:.1f}, "
        f"native {48 * t_v:.1f}, int4 {48 * t_i:.1f} ms")
open(os.path.expanduser("~/Leap/data/qmm27.txt"), "w").write("\n".join(lines) + "\n")
