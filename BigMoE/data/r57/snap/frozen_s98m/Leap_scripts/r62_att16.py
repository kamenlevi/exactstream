"""R-62 ATT16 (Leap, Sat 2026-10-31 20:0x; pre-registered Sat 02:4x in ~/Leap/TABLE_350.md 'R-61 / R-62'; shared GPU, rough):
one SDPA call per QSA layer for a 16-row check vs MTPLX's 12 head-chunked calls (Q._sdpa_head_chunked, 2 q heads each).
MLX 0.32.2 sends q_len > 8 to its full (NAX) kernel, which takes GQA natively, so the chunking may only add dispatches.
Synthetic bf16 at the model's shapes: 24 q heads, 2 kv heads, head_dim 256, 16 rows, causal bool mask; key lengths 383 /
1024 / 2048 (REF2b runs calls past 512 keys at 512-multiples). Per N: bitwise equal? ms per layer call (median of 50),
x 12 QSA layers per check. Synthetic data can miss rare data-dependent flips (3o vs 3m): a real-data check follows a PASS.
Usage: [BigMoE venv] r62_att16.py -> ~/Leap/data/r62_att16.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from mtplx.models import qwen4_exp as Q

D = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))

def timeit(f, n=50):
    for _ in range(5):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return med(ts)

eq = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())
mx.random.seed(0)
sc = 256 ** -0.5
say(f"R-62 ATT16: MLX {mx.__version__}; q [1, 24, 16, 256], k / v [1, 2, N, 256] bf16, causal bool mask")
for N in (383, 1024, 2048):
    q = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    m = (mx.arange(N)[None, :] <= mx.arange(N - 16, N)[:, None])[None, None]
    mx.eval(q, k, v, m)
    A = lambda: Q._sdpa_head_chunked(q, k, v, scale=sc, mask=m, heads_per_chunk=2)
    B = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=m)
    a, b = A(), B()
    mx.eval(a, b)
    ta, tb = timeit(A), timeit(B)
    say(f"  N {N}: bitwise equal {eq(a, b)}; chunked (12 calls) {1e3 * ta:.3f} ms, one call {1e3 * tb:.3f} ms per layer; "
        f"x 12 layers: {12e3 * ta:.1f} -> {12e3 * tb:.1f} ms per check")
say("R-62 done")
open(os.path.join(D, "r62_att16.txt"), "w").write("\n".join(lines) + "\n")
