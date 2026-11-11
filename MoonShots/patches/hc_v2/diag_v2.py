"""HCReadV2 diagnostics: relative error vs the reference (ulp counts blow up near zero), and per-kernel time (A norm,
B down+inject, C up+gate+mean) over 16 real modules. Usage: diag_v2.py   (small GPU job)"""
import os, sys, time
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_exp_pack"))
sys.argv = [sys.argv[0], "0"]
import test_packed_hc as T            # noqa: E402
import hc_read_v2 as H                # noqa: E402

hc, r0, _ = T.build("attn_hyper_connection", False)
v2 = H.HCReadV2(hc, T.args.rms_norm_eps)
h = (mx.random.normal((1, 1, 10240)) * 0.5).astype(mx.bfloat16)
a, b = r0(h), v2(h)
for name, x, y in zip(("mixed", "written", "inject"), a, b):
    xa = np.array(x.astype(mx.float32)).ravel(); ya = np.array(y.astype(mx.float32)).ravel()
    print(f"{name}: max|diff| {np.abs(xa-ya).max():.3e}  max|ref| {np.abs(xa).max():.3e}  "
          f"rel {np.abs(xa-ya).max()/np.abs(xa).max():.2e}  mean|diff|/mean|ref| {np.abs(xa-ya).mean()/np.abs(xa).mean():.2e}")

mods = []
for L in range(8):
    T.LAYER = L
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        m, _, _ = T.build(part, False)
        mods.append(H.HCReadV2(m, T.args.rms_norm_eps))
S = 1
x = (mx.random.normal((S, 10240)) * 0.5).astype(mx.bfloat16)
normed0, _ = H._KA(inputs=[x, mods[0].gamma, mods[0].dummy_bo, mods[0].dummy_inj, mods[0].epsv],
                   template=[("M", S), ("HC", 4), ("D", 2560), ("PENDING", 0)], grid=(S * 4 * 256, 1, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(S, 10240)] * 2, output_dtypes=[mx.bfloat16] * 2)
mix0 = mx.zeros((S, 320), mx.bfloat16)
mx.eval(normed0, mix0)


def t(fn):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(21):
        a_ = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - a_)
    return sorted(ts)[10] / len(mods) * 1e6


ta = t(lambda: [H._KA(inputs=[x, m.gamma, m.dummy_bo, m.dummy_inj, m.epsv], template=[("M", S), ("HC", 4), ("D", 2560),
            ("PENDING", 0)], grid=(S * 4 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, 10240)] * 2,
            output_dtypes=[mx.bfloat16] * 2)[0] for m in mods])
tb = t(lambda: [H._KB(inputs=[normed0, *m.pd], template=[("M", S), ("K", 10240), ("R", 320), ("HC", 4)],
            grid=(324 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, 320), (S, 4)],
            output_dtypes=[mx.bfloat16] * 2)[0] for m in mods])
tc = t(lambda: [H._KC(inputs=[mix0, normed0, *m.pu], template=[("M", S), ("D", 2560), ("R", 320), ("HC", 4)],
            grid=(320 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, 2560)],
            output_dtypes=[mx.bfloat16])[0] for m in mods])
pb = sum(a_.nbytes for a_ in mods[0].pd) / 1e6; pc = sum(a_.nbytes for a_ in mods[0].pu) / 1e6
print(f"independent launches, 16 modules: A norm {ta:.1f} µs | B down+inject {tb:.1f} µs ({pb:.1f} MB, {pb/tb*1e3:.0f} GB/s)"
      f" | C up {tc:.1f} µs ({pc:.1f} MB, {pc/tc*1e3:.0f} GB/s)")
