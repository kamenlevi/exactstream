"""Rollback cost (outside review R2): MTPLX's commit_verified_window replays each GDN layer's recurrence over the kept
rows from the pre-verify snapshot. Time 36 chained gated_delta_update replays at the 125B's GDN shapes (48 value heads,
16 key heads, dk = dv = 128, fp32 state) for kept = 1..4 rows."""
import time
import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
AL = mx.zeros((48,), mx.float32); DB = mx.zeros((48,), mx.float32)   # per-head decay params (A_log, dt_bias)
Hv, Hk, D = 48, 16, 128
st0 = mx.zeros((1, Hv, D, D), dtype=mx.float32)
for n in (1, 2, 3, 4):
    q = mx.random.normal((1, n, Hk, D)).astype(mx.bfloat16); k = mx.random.normal((1, n, Hk, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, n, Hv, D)).astype(mx.bfloat16)
    g = mx.random.uniform(shape=(1, n, Hv)).astype(mx.bfloat16); b = mx.random.uniform(shape=(1, n, Hv)).astype(mx.bfloat16)
    states = [st0 + 0.0 for _ in range(36)]; mx.eval(states)
    ts = []
    for _ in range(15):
        t = time.perf_counter()
        outs = [gated_delta_update(q, k, v, g, b, AL, DB, state=s)[1] for s in states]
        mx.eval(outs); ts.append(time.perf_counter() - t)
    print(f"kept {n}: 36-layer replay {np.median(ts) * 1e3:.2f} ms", flush=True)
