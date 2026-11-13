"""R-67 GATES screen (Leap, Sun 2026-11-01; shared GPU, rough): gdn_gates.gated_delta_update_g vs mlx_lm's stock
gated_delta_update at the 125B's GDN shapes (Hk 16, Hv 48, Dk = Dv = 128, bf16 q / k / v / a / b, fp32 state).
1. difference vs stock (the gate rounding moves from bf16 to fp32): max |dy| / RMS(y), max |dstate| / RMS(state)
2. INV: 16 one-row calls in sequence == one 16-row call (y rows and the final state, bitwise)
3. ms per call at T 16 and T 1 (stock: compute_g + sigmoid + kernel = 3 dispatches; fused: 1), 36 GDN layers per check
Usage: [BigMoE venv] r67_gates_test.py -> ~/Leap/data/r67_gates.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
from mlx_lm.models import gated_delta as GD
from gdn_gates import gated_delta_update_g

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
mx.random.seed(0)
B, Hk, Hv, Dk, Dv = 1, 16, 48, 128, 128
bits = lambda a: np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a.view(mx.uint32))
def mk(T):
    q = (mx.random.normal((B, T, Hk, Dk)) / Dk ** 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((B, T, Hk, Dk)) / Dk ** 0.5).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    a = (mx.random.normal((B, T, Hv)) * 2).astype(mx.bfloat16)
    b = (mx.random.normal((B, T, Hv)) * 2).astype(mx.bfloat16)
    return q, k, v, a, b
for adt in (mx.float32, mx.bfloat16):
    A_log = (mx.random.uniform(shape=(Hv,)) * 3 - 1).astype(adt)
    dt_bias = (mx.random.normal((Hv,)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.1).astype(mx.float32)
    q, k, v, a, b = mk(16)
    ys, ss = GD.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state)
    yf, sf = gated_delta_update_g(q, k, v, a, b, A_log, dt_bias, state)
    mx.eval(ys, ss, yf, sf)
    rms = lambda x: float(mx.sqrt(mx.mean(x.astype(mx.float32) ** 2)).item())
    dy = float(mx.max(mx.abs(ys.astype(mx.float32) - yf.astype(mx.float32))).item()) / rms(ys)
    ds = float(mx.max(mx.abs(ss - sf)).item()) / rms(ss)
    # INV: 16 one-row calls
    st = state
    yr = []
    for t in range(16):
        y1, st = gated_delta_update_g(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], a[:, t:t + 1], b[:, t:t + 1], A_log, dt_bias, st)
        yr.append(y1)
    y16 = mx.concatenate(yr, axis=1)
    mx.eval(y16, st)
    inv = bool((bits(y16) == bits(yf)).all()) and bool((bits(st) == bits(sf)).all())
    say(f"A_log {str(adt).split('.')[-1]}: vs stock max |dy| / RMS {dy:.2e}, max |dstate| / RMS {ds:.2e}; INV (16 x T1 == T16, "
        f"y and state bitwise): {'PASS' if inv else 'FAIL'}")
def bench(f, n=50):
    for _ in range(5):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
for T in (16, 1):
    q, k, v, a, b = mk(T)
    chain = 8
    fs = lambda: [GD.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state)[0] for _ in range(chain)]
    ff = lambda: [gated_delta_update_g(q, k, v, a, b, A_log, dt_bias, state)[0] for _ in range(chain)]
    t_s, t_f = [], []
    for _ in range(3):
        t_s.append(bench(fs)); t_f.append(bench(ff))
    s_, f_ = 1e3 * min(t_s) / chain, 1e3 * min(t_f) / chain
    say(f"T {T}: stock {s_:.3f} ms, fused gates {f_:.3f} ms per call ({s_ / f_:.2f}x); x 36 GDN layers: {36 * (s_ - f_):+.2f} ms "
        f"per check saved (8 chained calls per eval, best of 3 medians)")
open(os.path.expanduser("~/Leap/data/r67_gates.txt"), "w").write("\n".join(lines) + "\n")
