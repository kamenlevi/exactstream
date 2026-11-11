"""D2 step 1: what error does a real code reach per bit on real expert weights? (Breakthrough, 2026-10-20)
The noise sweep (data/noise35/results.txt) maps NMSE -> KLD per tensor class; this measures the NMSE side.
Codes, all on the same weights:
  affine g64 at 4 / 3 bits (MLX quantize: today's format and its 3-bit sibling),
  RHT + Lloyd-Max-like scalar (nearest of 2^k Gaussian quantiles, per-row RMS) at 3 / 4 bits,
  RHT + bitshift trellis (scripts/trellis.py, Viterbi, L = 8 / 10 / 12) at k = 2 / 3 / 4.
Sources: iid Gaussian; real 35B routed experts (8-bit unsloth pack, layers 0/20/39, experts 0-3, gate/up/down);
real 125B routed experts (4-bit bare pack, dequantized, so its NMSE is relative to the 4-bit values).
GPU, a few minutes. usage: trellis_nmse.py
"""
import glob, json, math, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from trellis import quantize, make_lut, T


def nmse(a, b):
    return float(mx.mean((a - b) ** 2).item() / mx.mean(b * b).item())


def affine(w, bits):
    q = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=bits)
    return mx.dequantize(*q, group_size=64, bits=bits).astype(mx.float32)


def rht_scalar(w, k, seed=0):
    d = w.shape[-1]
    signs = mx.array(np.random.default_rng(seed + 1).choice([-1.0, 1.0], d).astype(np.float32))
    x = mx.hadamard_transform(w.reshape(-1, d) * signs, scale=1.0 / math.sqrt(d))
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True)) + 1e-12
    from statistics import NormalDist
    lv = mx.array(np.array([NormalDist().inv_cdf((i + 0.5) / (1 << k)) for i in range(1 << k)], np.float32))
    z = x / rms
    idx = mx.argmin(mx.abs(z[..., None] - lv), axis=-1)
    xh = lv[idx] * rms
    return (mx.hadamard_transform(xh, scale=1.0 / math.sqrt(d)) * signs).reshape(w.shape)


def load_tensor(path, name):
    return mx.load(path)[name]


def experts35():
    d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MLX-8bit/snapshots/*/"))[0]
    idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
    out = {}
    for l in (0, 20, 39):
        for p in ("gate_proj", "up_proj", "down_proj"):
            base = f"language_model.model.layers.{l}.mlp.switch_mlp.{p}"
            f = mx.load(d + idx[base + ".weight"])
            w = mx.dequantize(f[base + ".weight"][:4], f[base + ".scales"][:4], f[base + ".biases"][:4], group_size=64, bits=8)
            out[f"35B L{l} {p}"] = w.astype(mx.float32)
    return out


def experts125():
    d = os.path.expanduser("~/BigMoE/models/bare/")
    idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
    out = {}
    for l in (0, 24, 47):
        for p in ("gate_proj", "up_proj", "down_proj"):
            base = f"language_model.model.layers.{l}.mlp.switch_mlp.{p}"
            if base + ".weight" not in idx:
                continue
            f = mx.load(d + idx[base + ".weight"])
            w = mx.dequantize(f[base + ".weight"][:2], f[base + ".scales"][:2], f[base + ".biases"][:2], group_size=64, bits=4)
            out[f"125B L{l} {p}"] = w.astype(mx.float32)
    return out


if __name__ == "__main__":
    t0 = time.time()
    srcs = {"gaussian": mx.random.normal((32, 4096), key=mx.random.key(0))}
    srcs.update(experts35())
    srcs.update(experts125())
    print(f"loaded {len(srcs)} sources in {time.time() - t0:.0f}s", flush=True)
    cols = ["aff4", "aff3", "rht-s4", "rht-s3", "tr12k4", "tr12k3", "tr12k2", "tr10k3", "tr8k3"]
    print("source               " + " ".join(f"{c:>8s}" for c in cols), flush=True)
    agg = {c: [] for c in cols}
    for name, w in srcs.items():
        w = w.reshape(-1, w.shape[-1])[:4096]            # <= 4096 rows per source
        res = {"aff4": affine(w, 4), "aff3": affine(w, 3), "rht-s4": rht_scalar(w, 4), "rht-s3": rht_scalar(w, 3),
               "tr12k4": quantize(w, 4, L=12), "tr12k3": quantize(w, 3, L=12), "tr12k2": quantize(w, 2, L=12),
               "tr10k3": quantize(w, 3, L=10), "tr8k3": quantize(w, 3, L=8)}
        vals = {c: nmse(res[c], w) for c in cols}
        for c in cols:
            if name != "gaussian":
                agg[c].append(vals[c])
        print(f"{name:20s} " + " ".join(f"{vals[c]:8.4f}" for c in cols), flush=True)
    print("mean real experts    " + " ".join(f"{np.mean(agg[c]):8.4f}" for c in cols))
    print(f"bits/weight: aff4 4.5, aff3 3.5 (g64 bf16 scale+bias); rht-s k + ~0.01; trellis k + L/{T} + ~0.01. "
          f"{time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB")
