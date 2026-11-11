"""Prefill-width 4-bit GEMM throughput on the M5 Pro (applegpu_g17s; MLX 0.32.x has NAX kernels incl. gather_qmm_rhs_nax and
qmm_t_nax, dispatched automatically): real 125B layer-0 expert weights, sorted gather_qmm (SwitchGLU's prefill path) at
rows/expert 40..320 for gate (K=2560 -> 640) and down (K=640 -> 2560), and a dense qmm of a GDN in_proj shape
(K=2560 -> 16480) at M=2048/8192. TFLOPS (2*M*K*N / t) and determinism.   usage: nax_gemm.py   (GPU, quiet lock, ~1 min)"""
import json, os, struct, time
import numpy as np
import mlx.core as mx
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]; dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


pre = "language_model.model.layers.0."
W = {p: tuple(tensor(pre + f"mlp.switch_mlp.{p}.{f}") for f in ("weight", "scales", "biases")) for p in ("gate_proj", "down_proj")}
QKV = tuple(tensor(pre + f"linear_attn.in_proj_qkv.{f}") for f in ("weight", "scales", "biases"))
mx.eval(W, QKV)
E = 512


def t(f, n=7):
    for _ in range(2): mx.eval(f())
    ts = []
    for _ in range(n):
        a = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - a)
    return sorted(ts)[n // 2]


print("device", mx.device_info()["architecture"], flush=True)
for rpe in (40, 80, 160, 320):
    ids = mx.array(np.repeat(np.arange(E), rpe).astype(np.uint32)); P = E * rpe
    for name, K, N in (("gate", 2560, 640), ("down", 640, 2560)):
        x = (mx.random.normal((P, 1, K)) * 0.5).astype(mx.bfloat16)
        f = lambda: mx.gather_qmm(x, *W[name + "_proj"], rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)
        dt = t(f)
        det = mx.array_equal(f(), f()).item()
        print(f"gather_qmm sorted {name} K={K} N={N}: {rpe:3d} rows/expert, {P} rows: {dt*1e3:7.2f} ms -> "
              f"{2 * P * K * N / dt / 1e12:5.1f} TFLOPS, deterministic {det}", flush=True)
for M in (2048, 8192):
    x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
    f = lambda: mx.quantized_matmul(x, *QKV, transpose=True, group_size=64, bits=4)
    dt = t(f); N = QKV[0].shape[0]
    print(f"dense qmm in_proj_qkv K=2560 N={N} M={M}: {dt*1e3:6.2f} ms -> {2 * M * 2560 * N / dt / 1e12:5.1f} TFLOPS", flush=True)
