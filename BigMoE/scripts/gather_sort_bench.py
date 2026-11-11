"""v2-8 sizing: a 125B verify-shaped expert MLP (gate/up/down, hidden 2560, inter 640, Q4 g64) for 4 rows x top-10 with
~27 distinct experts (row_share.py), through mx.gather_qmm unsorted (MLX default below 64 ids) vs expert-sorted
(sorted_indices=True, as SwitchGLU does at >= 64 ids). 48 layers chained, median of 20 runs."""
import time
import mlx.core as mx
import numpy as np
E, H, I, R, K = 256, 2560, 640, 4, 10
rng = np.random.default_rng(0)
def q(shape):
    w = mx.random.normal(shape) * 0.02
    return mx.quantize(w, group_size=64, bits=4)
Wg, Wu, Wd = q((E, I, H)), q((E, I, H)), q((E, H, I))
mx.eval(Wg, Wu, Wd)
def ids():
    pool = rng.choice(E, 27, replace=False)
    rows = [rng.choice(pool, K, replace=False) for _ in range(R)]
    return mx.array(np.stack(rows).astype(np.uint32))
IDS = [ids() for _ in range(48)]
def mlp(x, idx, srt):
    if srt:
        flat = idx.reshape(-1); order = mx.argsort(flat); inv = mx.argsort(order)
        xs = mx.repeat(x, K, axis=0)[order]               # [R*K, 1, H] in expert order
        e = flat[order]
        g = mx.gather_qmm(xs, *Wg, rhs_indices=e, transpose=True, group_size=64, bits=4, sorted_indices=True)
        u = mx.gather_qmm(xs, *Wu, rhs_indices=e, transpose=True, group_size=64, bits=4, sorted_indices=True)
        y = mx.gather_qmm(mx.sigmoid(g) * g * u, *Wd, rhs_indices=e, transpose=True, group_size=64, bits=4, sorted_indices=True)
        return y[inv].reshape(R, K, 1, H).sum(axis=1)
    xe = x[:, None]                                       # [R, 1, 1, H]
    g = mx.gather_qmm(xe, *Wg, rhs_indices=idx, transpose=True, group_size=64, bits=4)
    u = mx.gather_qmm(xe, *Wu, rhs_indices=idx, transpose=True, group_size=64, bits=4)
    y = mx.gather_qmm(mx.sigmoid(g) * g * u, *Wd, rhs_indices=idx, transpose=True, group_size=64, bits=4)
    return y.sum(axis=1)
x0 = (mx.random.normal((R, 1, H)) * 0.1).astype(mx.bfloat16)
for srt in (False, True, False, True):
    ts = []
    for _ in range(20):
        x = x0; t = time.perf_counter()
        for L in range(48):
            x = (x + mlp(x, IDS[L], srt).astype(mx.bfloat16)) * 0.5
        mx.eval(x); ts.append(time.perf_counter() - t)
    print(f"{'sorted  ' if srt else 'unsorted'}: {np.median(ts) * 1e3:.2f} ms per 48 layers", flush=True)
