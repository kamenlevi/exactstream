"""Chunked GDN vs mlx_lm's sequential ops (values, final state, gradients). CPU, small shapes."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/TrainLab"))
import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_ops, compute_g
from gdn_chunk import gated_delta_chunked

mx.set_default_device(mx.cpu)
mx.random.seed(0)
B, T, Hk, Hv, Dk, Dv = 2, 100, 2, 4, 32, 16
def inputs():
    q = mx.random.normal((B, T, Hk, Dk)); k = mx.random.normal((B, T, Hk, Dk)); v = mx.random.normal((B, T, Hv, Dv))
    q = (Dk ** -1) * mx.fast.rms_norm(q, None, 1e-6); k = (Dk ** -0.5) * mx.fast.rms_norm(k, None, 1e-6)
    a = mx.random.normal((B, T, Hv)); b = mx.random.normal((B, T, Hv))
    A_log = mx.log(mx.random.uniform(0, 16, (Hv,))); dt_bias = mx.ones(Hv)
    s0 = 0.1 * mx.random.normal((B, Hv, Dv, Dk))
    return q, k, v, a, b, A_log, dt_bias, s0
q, k, v, a, b, A_log, dt_bias, s0 = inputs()
def ref(q, k, v, a, b, s0):
    g = compute_g(A_log, a, dt_bias); beta = mx.sigmoid(b)
    return gated_delta_ops(q, k, v, g, beta, s0)
def chk(q, k, v, a, b, s0, chunk):
    log_g = -mx.exp(A_log.astype(mx.float32)) * mx.logaddexp(a + dt_bias, 0.0)   # log of compute_g
    return gated_delta_chunked(q, k, v, log_g, mx.sigmoid(b), s0, chunk=chunk)
y0, S0 = ref(q, k, v, a, b, s0)
for c in (16, 32, 64):
    y1, S1 = chk(q, k, v, a, b, s0, c)
    print(f"chunk {c}: max|dy| {mx.abs(y1 - y0).max().item():.2e} (|y| {mx.abs(y0).max().item():.2f})  max|dS| {mx.abs(S1 - S0).max().item():.2e}")
# gradients of a random projection of y and S
wy = mx.random.normal(y0.shape); wS = mx.random.normal(S0.shape)
def loss_ref(q, k, v, a, b, s0):
    y, S = ref(q, k, v, a, b, s0); return (y * wy).sum() + (S * wS).sum()
def loss_chk(q, k, v, a, b, s0):
    y, S = chk(q, k, v, a, b, s0, 32); return (y * wy).sum() + (S * wS).sum()
gr = mx.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, a, b, s0)
gc = mx.grad(loss_chk, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, a, b, s0)
for name, x, y in zip("q k v a b s0".split(), gr, gc):
    print(f"grad {name}: rel err {(mx.abs(x - y).max() / (mx.abs(x).max() + 1e-12)).item():.2e}")
# adversarial (repetitive text): identical keys, beta ~ 1, no decay; and strongly correlated keys
k1 = mx.broadcast_to(k[:, :1], k.shape)
cases = [("identical keys, beta~1, no decay", k1, mx.full(b.shape, 8.0), mx.full(a.shape, -30.0)),
         ("correlated keys, beta~0.9", 0.2 * k + 0.98 * k1, mx.full(b.shape, 2.2), a)]
for name, kk, bb, aa in cases:
    kk = (Dk ** -0.5) * mx.fast.rms_norm(kk, None, 1e-6)
    y0, S0 = ref(q, kk, v, aa, bb, s0)
    y1, S1 = chk(q, kk, v, aa, bb, s0, 64)
    g0 = mx.grad(lambda q_: (ref(q_, kk, v, aa, bb, s0)[0] * wy).sum())(q)
    g1 = mx.grad(lambda q_: (chk(q_, kk, v, aa, bb, s0, 64)[0] * wy).sum())(q)
    print(f"{name}: max|dy| {mx.abs(y1 - y0).max().item():.2e} (|y| {mx.abs(y0).max().item():.2f}), "
          f"max|dS| {mx.abs(S1 - S0).max().item():.2e}, grad q rel err "
          f"{(mx.abs(g1 - g0).max() / mx.abs(g0).max()).item():.2e}, finite {bool(mx.isfinite(y1).all().item())}")
