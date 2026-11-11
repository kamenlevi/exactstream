"""R-66 FATT16 screen (Leap, Sun 2026-11-01; shared GPU, rough): fatt16 vs MLX 0.32's 16-row SDPA at the 125B's QSA shape
(24 q heads, 2 kv heads, head_dim 256, bf16, causal bool mask over N keys; the fallback MLX runs for head_dim 256).
1. accuracy vs an fp64 reference: fatt16 and MLX's single call, max and mean |err| / RMS
2. INV (bitwise): a probe row (row 5) with (a) other rows' queries changed, (b) the same query at row 11 with row 5's mask
   row, (c) N grown by 37 / 300 keys that the probe row cannot see (they are masked for it), (d) a 1-row AR-style call: the
   query alone in a 16-row call (other rows zero) at N + 7 keys, its visible keys the same
3. ms per call at N 383 / 1024 / 2048: MTPLX's 12 head chunks, one MLX call (R-62 att_single), fatt16; x 12 layers
Usage: [BigMoE venv] r66_fatt_test.py -> ~/Leap/data/r66_fatt.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from mtplx.models import qwen4_exp as Q
from fatt16 import fatt16

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
SC = 256 ** -0.5
bits = lambda a: np.array(a.view(mx.uint16)).astype(np.int64)
mx.random.seed(5)
def causal(N, rows=16):
    return (mx.arange(N)[None, :] <= mx.arange(N - rows, N)[:, None])
# 1. accuracy at N 1024
N = 1024
q = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
m = causal(N)
mx.eval(q, k, v, m)
yf = fatt16(q, k, v, SC, m)
ym = mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m[None, None])
mx.eval(yf, ym)
qn, kn, vn, mn = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v, m.astype(mx.float32)))
ref = np.zeros((24, 16, 256))
for h in range(24):
    s_ = qn[0, h] @ kn[0, h // 12].T * SC
    s_ = np.where(mn > 0, s_, -np.inf)
    p_ = np.exp(s_ - s_.max(-1, keepdims=True)); p_ /= p_.sum(-1, keepdims=True)
    ref[h] = p_ @ vn[0, h // 12]
rms = float(np.sqrt((ref ** 2).mean()))
for nm, y in (("fatt16", yf), ("MLX one call", ym)):
    e = np.abs(np.array(y.astype(mx.float32))[0].astype(np.float64) - ref)
    say(f"1. {nm}: |err| vs fp64 max {e.max() / rms:.2e} mean {e.mean() / rms:.2e} (RMS {rms:.3f})")
# 2. INV
P = 5
base = bits(fatt16(q, k, v, SC, m)[0, :, P])
q2 = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
qa = mx.concatenate([q2[:, :, :P], q[:, :, P:P + 1], q2[:, :, P + 1:]], axis=2)
a_ok = (bits(fatt16(qa, k, v, SC, m)[0, :, P]) == base).all()
qb = mx.concatenate([q2[:, :, :11], q[:, :, P:P + 1], q2[:, :, 12:]], axis=2)
mb = mx.concatenate([m[:11], m[P:P + 1], m[12:]], axis=0)
b_ok = (bits(fatt16(qb, k, v, SC, mb)[0, :, 11]) == base).all()
c_ok = True
for extra in (37, 300):
    k3 = mx.concatenate([k, (mx.random.normal((1, 2, extra, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
    v3 = mx.concatenate([v, mx.random.normal((1, 2, extra, 256)).astype(mx.bfloat16)], axis=2)
    m3 = mx.concatenate([m, mx.zeros((16, extra), mx.bool_)], axis=1)
    c_ok &= bool((bits(fatt16(q, k3, v3, SC, m3)[0, :, P]) == base).all())
qd = mx.concatenate([mx.zeros((1, 24, P, 256), mx.bfloat16), q[:, :, P:P + 1], mx.zeros((1, 24, 15 - P, 256), mx.bfloat16)], axis=2)
k4 = mx.concatenate([k, (mx.random.normal((1, 2, 7, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
v4 = mx.concatenate([v, mx.random.normal((1, 2, 7, 256)).astype(mx.bfloat16)], axis=2)
m4 = mx.concatenate([m, mx.zeros((16, 7), mx.bool_)], axis=1)
d_ok = bool((bits(fatt16(qd, k4, v4, SC, m4)[0, :, P]) == base).all())
say(f"2. INV: (a) other rows changed {'PASS' if a_ok else 'FAIL'}; (b) same query at row 11 {'PASS' if b_ok else 'FAIL'}; "
    f"(c) N + 37 / + 300 masked keys {'PASS' if c_ok else 'FAIL'}; (d) AR-style call (query alone, N + 7) {'PASS' if d_ok else 'FAIL'}")
# 3. timing
def bench(f, chain=8, n=30):
    for _ in range(4):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain
for N in (383, 1024, 2048):
    q = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    m = causal(N); m4d = m[None, None]
    mx.eval(q, k, v, m, m4d)
    t = [bench(lambda: Q._sdpa_head_chunked(q, k, v, scale=SC, mask=m4d, heads_per_chunk=2)),
         bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m4d)),
         bench(lambda: fatt16(q, k, v, SC, m))]
    say(f"3. N {N}: 12 chunks {t[0]:.3f}, one call {t[1]:.3f}, fatt16 {t[2]:.3f} ms per layer call (fatt16 {t[1] / t[2]:.2f}x vs one "
        f"call); x 12 layers: {12 * t[0]:.1f} / {12 * t[1]:.1f} / {12 * t[2]:.1f} ms per check")
open(os.path.expanduser("~/Leap/data/r66_fatt.txt"), "w").write("\n".join(lines) + "\n")
