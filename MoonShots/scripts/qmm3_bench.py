"""Round 25 check before a 3-bit tier prototype: MLX gather_qmm speed at 3 bits vs 4 bits for the 125B's routed-expert
shapes (gate/up [640, 2560], down [2560, 640], group 64) over a 128-slot pool, 1-4 rows x top-10 (decode and MTP verify),
and the plan-4433333333 split (top-2 of each row at 4 bits, 8 at 3 bits: two calls per projection) vs all 4-bit.
GPU-light (~1 GB), ~1 min; timing: quiet lock.   usage: qmm3_bench.py"""
import time
import mlx.core as mx
SLOTS, H, I, G = 128, 2560, 640, 64
mx.random.seed(0)


def qpool(bits, n_out, n_in):
    w = mx.random.normal((SLOTS, n_out, n_in)).astype(mx.bfloat16) * 0.02
    q = mx.quantize(w, group_size=G, bits=bits)
    mx.eval(q)
    return q


P = {b: {"gate": qpool(b, I, H), "up": qpool(b, I, H), "down": qpool(b, H, I)} for b in (3, 4)}


def glu(bits, x, idx):
    """x [rows, 1, H] (one token per row), idx [rows, k] slot ids -> [rows, k, H]"""
    p = P[bits]
    xg = mx.expand_dims(x, -2)                                           # [rows, 1, 1, H]
    g = mx.gather_qmm(xg, *p["gate"], rhs_indices=idx, transpose=True, group_size=G, bits=bits)
    u = mx.gather_qmm(xg, *p["up"], rhs_indices=idx, transpose=True, group_size=G, bits=bits)
    a = (g * mx.sigmoid(g)) * u                                          # [rows, k, 1, I]
    return mx.gather_qmm(a, *p["down"], rhs_indices=idx, transpose=True, group_size=G, bits=bits)


def bench(fn, iters=200):
    for _ in range(10):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters * 1e3


print("ms per layer's routed-expert GLU (gate, up, down), 128-slot pools, top-10 per row:")
for rows in (1, 2, 4):
    x = mx.random.normal((rows, 1, H)).astype(mx.bfloat16)
    idx = mx.stack([mx.random.permutation(SLOTS)[:10] for _ in range(rows)]).astype(mx.uint32)
    mx.eval(x, idx)
    t4 = bench(lambda: glu(4, x, idx))
    t3 = bench(lambda: glu(3, x, idx))
    t_split = bench(lambda: (glu(4, x, idx[:, :2]), glu(3, x, idx[:, 2:])))
    mb4 = rows * 10 * 2.77
    print(f"rows {rows}: 4-bit {t4:.3f} ms ({mb4 / t4:.0f} GB/s of expert bytes), 3-bit {t3:.3f} ms, "
          f"plan split (2 @ 4-bit + 8 @ 3-bit) {t_split:.3f} ms ({t_split / t4 - 1:+.0%} vs 4-bit)", flush=True)
