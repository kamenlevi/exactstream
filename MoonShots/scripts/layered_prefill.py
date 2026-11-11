"""#18 layer-major MoE prefill: does one MoE layer get cheaper per token when it sees more tokens at once?
Today a 2048-token chunk runs every layer, so each of 256 experts gets ~64 rows. Layer-major prefill would
run a layer's MoE once over the whole prompt (8K-32K tokens -> 256-1024 rows per expert).
One Qwen3.6-35B-A3B MoE layer (256 experts, top-8, hidden 2048, expert intermediate 512) at Q8 or 4-bit,
sorted gather_qmm as mlx-lm's SwitchGLU does it. Prints µs per token and TFLOPS vs tokens per call.
Usage: layered_prefill.py [bits]   (GPU, needs a quiet lock; ~1 GB of weights)"""
import sys, time
import mlx.core as mx

BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SHAPE = sys.argv[2] if len(sys.argv) > 2 else "35b"
NE, K, D, F = {"35b": (256, 8, 2048, 512), "125b": (512, 10, 2560, 640)}[SHAPE]
G = 64


def qexperts(o, i):
    w = (mx.random.normal((NE, o, i)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=G, bits=BITS)


gate, up, down = qexperts(F, D), qexperts(F, D), qexperts(D, F)
mx.eval(gate, up, down)


def moe(x, idx):
    """x: (T, D); idx: (T, K) expert ids. Sorted path, like SwitchGLU with sort."""
    T = x.shape[0]
    flat = idx.reshape(-1)
    order = mx.argsort(flat)
    xs = x[order // K][:, None, :]                       # (T*K, 1, D)
    ids = flat[order]
    kw = dict(transpose=True, group_size=G, bits=BITS, sorted_indices=True)
    g = mx.gather_qmm(xs, *gate, rhs_indices=ids, **kw)
    u = mx.gather_qmm(xs, *up, rhs_indices=ids, **kw)
    h = mx.gather_qmm(nn_silu(g) * u, *down, rhs_indices=ids, **kw)
    return h


def nn_silu(a):
    return a * mx.sigmoid(a)


print(f"{SHAPE}: {BITS}-bit MoE layer, {NE} experts top-{K}, D={D}, F={F}")
base = None
for T in (512, 2048, 4096, 8192, 16384, 32768):
    x = mx.random.normal((T, D)).astype(mx.bfloat16)
    logits = mx.random.normal((T, NE))
    idx = mx.argpartition(-logits, K, axis=-1)[:, :K]
    for _ in range(2):
        mx.eval(moe(x, idx))
    ts = []
    for _ in range(7):
        t = time.perf_counter(); mx.eval(moe(x, idx)); ts.append(time.perf_counter() - t)
    t = sorted(ts)[3]
    us = t / T * 1e6
    base = base or us
    tflops = 2 * T * K * 3 * D * F / t / 1e12
    print(f"T={T:>6} ({T*K//NE:>5} rows/expert): {us:6.2f} µs/token ({base/us:4.2f}x vs T=512), {tflops:5.1f} TFLOPS")
