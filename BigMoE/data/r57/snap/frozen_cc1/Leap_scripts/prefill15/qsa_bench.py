"""Prefill 1.5K goal (Kamen 11-11 12:2x): how big is the exact QSA-attention lever at long contexts? Per 1,024-row prefill
chunk (PF-A's chunk) of one QSA layer, at context T, real shapes (GQA 24 query / 2 KV heads, head dim 256), a QSA-like
mask (512 selected 4-token blocks before each query + the causal tail; as PrefillKernels' qsa_exact_probe.py):
  stock  : mx.fast.scaled_dot_product_attention with the bool mask (D 256 -> MLX's unfused fallback: the score plane)
  band   : MTPLX's dense-band SDPA (exact, the engine's lane: mask folded into the score GEMM)
  sol    : a speed-of-light proxy for a fused exact kernel that never writes the plane: unmasked SDPA over only the
           visible-key count (as many keys as the densest row sees), same q; NOT the same arithmetic, timing only
Median of 5 after 2 warm-ups, each eval synchronised. Prints ms per chunk and the implied ms per prompt token over the
12 QSA layers (chunk ms x 12 / 1,024). usage: qsa_bench.py T1,T2,...   (GPU, ~5 GB at T = 30K, ~1-2 min)"""
import sys, time
import numpy as np
import mlx.core as mx

S, D, HKV, REP = 1024, 256, 2, 12
scale = D ** -0.5
try:
    from mtplx.kernels.qsa_dense_band_sdpa import dense_band_sdpa
except Exception as ex:                                     # the engine's venv has it
    dense_band_sdpa = None
    print("dense band not available:", ex, flush=True)


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


print(f"MLX {mx.__version__}; S={S} rows per chunk, D={D}, heads {HKV * REP}/{HKV}", flush=True)
print("T       visible(max row)  stock ms  band ms  sol ms | per prompt token over 12 QSA layers: stock  band  sol (ms)", flush=True)
for T in [int(x) for x in sys.argv[1].split(",")]:
    key = mx.random.split(mx.random.key(0), 4)
    q = mx.random.normal((1, HKV * REP, S, D), key=key[0]).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, T, D), key=key[1]).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, T, D), key=key[2]).astype(mx.bfloat16)
    rng = np.random.default_rng(1)
    pos0 = T - S
    m = np.zeros((S, T), dtype=bool)
    for s in range(S):
        qp = pos0 + s
        full = (qp + 1) // 4
        if full:
            sel = rng.choice(full, size=min(512, full), replace=False)
            m[s, (4 * sel[:, None] + np.arange(4)).ravel()] = True
        m[s, 4 * full:qp + 1] = True
    vis = int(m.sum(axis=1).max())
    mask = mx.array(m)[None, None]
    kv, vvv = k[:, :, :vis], v[:, :, :vis]
    mx.eval(q, k, v, mask, kv, vvv)
    t_stock = timeit(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask))
    t_band = timeit(lambda: dense_band_sdpa(q, k, v, scale=scale, mask=mask)) if dense_band_sdpa else float("nan")
    t_sol = timeit(lambda: mx.fast.scaled_dot_product_attention(q, kv, vvv, scale=scale))
    per = lambda t: t * 12 / S
    print(f"{T:6d}  {vis:6d}            {t_stock:8.1f} {t_band:8.1f} {t_sol:7.1f} | {per(t_stock):24.3f} {per(t_band):5.3f} {per(t_sol):5.3f}",
          flush=True)
    del q, k, v, mask, kv, vvv, m
    mx.clear_cache()
print(f"peak MLX memory {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
