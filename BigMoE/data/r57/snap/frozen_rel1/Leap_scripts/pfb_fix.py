"""PF-B FIX (Leap, Sun 08:3x): MLX 0.32.2's affine_gather_qmm_rhs_nax computes sgp_sm = min(SM, short(M - (y_row + tm))) when
M % BM != 0: the short overflows when > 32,767 rows follow, and those simdgroups store zeros. Workaround: pad the sorted
rows to a multiple of 64 (copies of the last row, same expert) so the kernel takes its aligned path. Check: the padded
SwitchGLU one-call is bitwise == 1,024-token calls (which never exceed 10,240 rows) and has no all-zero rows; re-time.
switch_glu_padded(m, x, ids) = SwitchGLU's forward (sort, up / gate / down sorted gather, SwiGLU, unsort) with the pad."""
import time
import numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear, _gather_sort, _scatter_unsort

E, TOPK, D, F = 512, 10, 2560, 640
rng = np.random.default_rng(11)


def switch_glu_padded(m, x, indices, align=64):
    xe = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(xe, indices)
    R = idx.shape[0]
    pad = (-R) % align
    if pad:
        xs = mx.concatenate([xs, mx.repeat(xs[-1:], pad, axis=0)], axis=0)
        idx = mx.concatenate([idx, mx.repeat(idx[-1:], pad, axis=0)], axis=0)
    h = m.activation(m.up_proj(xs, idx, sorted_indices=True), m.gate_proj(xs, idx, sorted_indices=True))
    y = m.down_proj(h, idx, sorted_indices=True)
    if pad:
        y = y[:R]
    return _scatter_unsort(y, inv, indices.shape).squeeze(-2)


m = SwitchGLU(D, F, E)
for p, (o, i) in (("gate_proj", (F, D)), ("up_proj", (F, D)), ("down_proj", (D, F))):
    q = QuantizedSwitchLinear(i, o, E, bias=False, group_size=64, bits=4)
    q.weight = mx.array(rng.integers(0, 2 ** 32, (E, o, i // 8), dtype=np.uint32))
    q.scales = mx.random.uniform(0.002, 0.02, (E, o, i // 64)).astype(mx.bfloat16)
    q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
    mx.eval(q.weight, q.scales, q.biases)
    setattr(m, p, q)
pop = 1.0 / np.arange(1, E + 1) ** 0.6; pop = pop[rng.permutation(E)]; pop /= pop.sum()


def tmin(fn, reps=3):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


print(f"PF-B FIX {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; GPU shared with EvalSuite's gate (timings indicative)", flush=True)
for T in (1565, 4730, 7932):
    ids = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32)
    x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    ch = lambda c: mx.concatenate([m(x[a:a + c], mx.array(ids[a:a + c])) for a in range(0, T, c)], 0)
    y1 = ch(1024); yp = switch_glu_padded(m, x, mx.array(ids)); yu = m(x, mx.array(ids))
    mx.eval(y1, yp, yu)
    zr = lambda y: int((~mx.any(y != 0, axis=-1)).sum().item())
    t1 = tmin(lambda: mx.eval(ch(1024)))
    tp = tmin(lambda: mx.eval(switch_glu_padded(m, x, mx.array(ids))))
    gfl = T * TOPK * 3 * D * F * 2 / 1e9
    print(f"  T {T}: padded one-call == 1,024-token calls: {bool(mx.array_equal(yp, y1).item())}; all-zero rows: 1,024 {zr(y1)}, "
          f"padded {zr(yp)}, unpadded one-call {zr(yu)} (rows {T * TOPK}, % 64 = {T * TOPK % 64}); MoE 1,024-token calls "
          f"{t1 * 1e3:.1f} ms ({gfl / t1 / 1e3:.1f} TF), padded one call {tp * 1e3:.1f} ms ({gfl / tp / 1e3:.1f} TF), x{t1 / tp:.2f}",
          flush=True)
