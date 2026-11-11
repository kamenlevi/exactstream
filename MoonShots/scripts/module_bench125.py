"""Round 26c: which weight-bearing ops of a 125B verify pass are slow? Every Linear / QuantizedLinear / GatedResidual of
the real model's 48 layers (real weights, BigMoE's load; routed experts excluded: round 26 measured them), timed the way
a pass runs them: all 48 (or 36 / 12) layers' instances of one op in one command buffer, each call depending on the
previous one (x_next = x + 0 * y.sum()), M = 1 and 4 rows (decode, 4-row verify). Bytes = the module's parameters, read
once per call. Plus the non-layer ops (lm_head, embedding gather, final mixer).
usage: module_bench125.py [FRAC]   (GPU; BigMoE-size memory at FRAC 0.125 = 9 GB of expert pools; quiet lock)"""
import os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from bigmoe.loader import load_streamed
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
import mtplx.models.qwen4_exp as Q
TM = rt.model.language_model
LAYERS = TM.model.layers
CEIL = 318.0


def nbytes(mod):
    return sum(a.nbytes for _, a in tree_flatten(mod.parameters()))


# collect op instances by name across layers
ops = {}
for L, layer in enumerate(LAYERS):
    for name, mod in layer.named_modules():
        if ".switch_mlp" in name or name.endswith("switch_mlp") or name == "":
            continue
        if isinstance(mod, Q.GatedResidual):
            ops.setdefault(("hc", name), []).append(mod)
        elif isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            if any(name.startswith(p + ".") for p in ("attn_hyper_connection", "mlp_hyper_connection")):
                continue                                             # timed as part of the GatedResidual read
            ops.setdefault(("lin", name), []).append(mod)
        elif name.endswith("shared_expert") and hasattr(mod, "gate_proj"):
            ops.setdefault(("mlp", name), []).append(mod)


def in_dim(mod):
    if isinstance(mod, nn.QuantizedLinear):
        return mod.weight.shape[1] * 32 // mod.bits
    if isinstance(mod, nn.Linear):
        return mod.weight.shape[1]
    if hasattr(mod, "gate_proj"):
        return in_dim(mod.gate_proj)
    return mod.hc_count * mod.hidden_size


def call(kind, mod, x):
    if kind == "hc":
        out = mod(x)
        return out[0] if isinstance(out, tuple) else out
    return mod(x)


def bench(kind, mods, M, iters=5):
    d = in_dim(mods[0])
    x0 = (mx.random.normal((1, M, d)) * 0.5).astype(mx.bfloat16)
    mx.eval(x0)

    def once():
        x = x0
        for m in mods:
            y = call(kind, m, x)
            x = x0 + (y.astype(mx.float32).sum() * 0).astype(x0.dtype)
        mx.eval(x)
    once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    return (time.perf_counter() - t0) / iters / len(mods) * 1e3


rows = []
for (kind, name), mods in sorted(ops.items(), key=lambda kv: -nbytes(kv[1][0])):
    b = nbytes(mods[0])
    dt = {M: bench(kind, mods, M) for M in (1, 4)}
    dtype = str(getattr(mods[0], "weight", mx.zeros(1)).dtype).replace("mlx.core.", "") if kind == "lin" else kind
    bits = getattr(mods[0], "bits", None)
    rows.append((name, kind, len(mods), b, dt, bits, dtype))
    print(f"{name:52s} x{len(mods):2d} {b / 1e6:7.2f} MB {('q%d' % bits) if bits else dtype:6s}  M=1 {dt[1]:.4f} ms "
          f"({b / 1e6 / dt[1]:5.0f} GB/s)  M=4 {dt[4]:.4f} ms ({b / 1e6 / dt[4]:5.0f} GB/s, {b / 1e6 / dt[4] / CEIL:4.0%})", flush=True)
tot = {M: sum(n * dt[M] for _, _, n, _, dt, _, _ in rows) for M in (1, 4)}
byt = sum(n * b for _, _, n, b, _, _, _ in rows)
print(f"\nall per-layer weight-bearing ops (excl. routed experts): {byt / 1e9:.2f} GB per pass; M=1 {tot[1]:.1f} ms "
      f"({byt / 1e6 / tot[1]:.0f} GB/s), M=4 {tot[4]:.1f} ms ({byt / 1e6 / tot[4]:.0f} GB/s)", flush=True)
# non-layer: lm_head over 1 and 4 rows
lm = TM.lm_head if hasattr(TM, "lm_head") else None
if lm is not None:
    b = nbytes(lm)
    for M in (1, 4):
        x = (mx.random.normal((1, M, in_dim(lm))) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        for _ in range(3):
            mx.eval(lm(x))
        t0 = time.perf_counter()
        for _ in range(10):
            mx.eval(lm(x))
        dt = (time.perf_counter() - t0) / 10 * 1e3
        print(f"lm_head ({b / 1e6:.0f} MB) M={M}: {dt:.3f} ms ({b / 1e6 / dt:.0f} GB/s)", flush=True)
