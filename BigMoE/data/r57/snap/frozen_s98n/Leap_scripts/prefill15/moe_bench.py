"""Prefill 1.5K goal: routed-expert matmul speed at prompt widths, PF-A (1,024-token calls) vs PF-B (one call per layer
over the whole prompt, sorted rows padded to a multiple of 64: MLX 0.32.2's BUG1 workaround, exact).
Real layer-0 routed-expert weights of Qwen3.8-Flash-Next 4-bit (~/BigMoE/models/bare), skewed synthetic routing
(top-10 of 512), mlx-lm's sorted SwitchGLU path (as PrefillKernels' nax_probe.py part A). Dense 4-bit qmm at the same
width (GDN in_proj 2560 -> 16480, layer 0) as the dense reference. Median of 5 after 2 warm-ups, synchronised.
Prints ms per call, TFLOPS and ms per prompt token over 48 layers. usage: moe_bench.py T1,T2,...  (GPU, ~6 GB at 30K)"""
import json, os, struct, sys, time
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
E, K, H, TOPK = 512, 2560, 640, 10


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]; dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = mx.array(np.array(np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,))).reshape(m["shape"]))
    return arr.view(mx.bfloat16) if m["dtype"] == "BF16" else arr


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


pre = "language_model.model.layers.0.mlp.switch_mlp."
W = {p: tuple(tensor(pre + f"{p}.{f}") for f in ("weight", "scales", "biases")) for p in ("gate_proj", "up_proj", "down_proj")}
dname = next((n for n in idx if n.startswith("language_model.model.layers.0.") and "in_proj" in n and n.endswith(".weight")), None)
DW = tuple(tensor(dname[:-len("weight")] + f) for f in ("weight", "scales", "biases")) if dname else None
mx.eval(W, DW)
rng = np.random.default_rng(0)
pop = rng.lognormal(0.0, 0.8, E); pop /= pop.sum()


def route(T, seed):
    r = np.random.default_rng(seed)
    g = np.log(pop)[None, :] + r.gumbel(size=(T, E))
    return mx.array(np.argsort(-g, axis=1)[:, :TOPK].astype(np.uint32))


def switch_glu_padded(x, inds):
    """mlx-lm's sorted SwitchGLU, with the sorted rows padded to a multiple of 64 (copies of the last row, same expert)"""
    fl = inds.flatten(); order = mx.argsort(fl); inv = mx.argsort(order)
    xs = x[order // TOPK]; ids = fl[order]
    n = xs.shape[0]; pad = (-n) % 64
    if pad:
        xs = mx.concatenate([xs, mx.repeat(xs[-1:], pad, axis=0)]); ids = mx.concatenate([ids, mx.repeat(ids[-1:], pad)])
    xs = mx.expand_dims(xs, -2)
    g = mx.gather_qmm(xs, *W["gate_proj"], rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)
    u = mx.gather_qmm(xs, *W["up_proj"], rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)
    y = mx.gather_qmm(nn_silu(g) * u, *W["down_proj"], rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)
    return y[:n].squeeze(-2)[inv].reshape(x.shape[0], TOPK, K)


def nn_silu(a):
    return a * mx.sigmoid(a)


print(f"MLX {mx.__version__}; layer-0 experts E={E}, K={K}, H={H}, top-{TOPK}; dense ref {dname}", flush=True)
print("tokens/call  rows/expert   SwitchGLU ms   TF    | dense in_proj ms   TF  | per prompt token x48 layers: MoE  dense(in_proj)", flush=True)
for T in [int(x) for x in sys.argv[1].split(",")]:
    x = mx.random.normal((T, K), key=mx.random.key(T)).astype(mx.bfloat16)
    inds = route(T, T)
    mx.eval(x, inds)
    t = timeit(lambda: switch_glu_padded(x, inds))
    fl = T * TOPK * 3 * 2 * K * H
    td = timeit(lambda: mx.quantized_matmul(x, *DW, transpose=True, group_size=64, bits=4)) if DW else float("nan")
    fld = 2 * T * K * (DW[0].shape[0] if DW else 0)
    print(f"{T:10d}  {T * TOPK / E:10.1f}   {t:12.1f} {fl / t / 1e9:6.1f} | {td:16.1f} {fld / td / 1e9:5.1f} | "
          f"{t * 48 / T:27.3f}  {td * 48 / T:.3f}", flush=True)
    del x, inds
    mx.clear_cache()
print(f"peak MLX memory {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
