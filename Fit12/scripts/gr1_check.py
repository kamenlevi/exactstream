"""Fit12 GR1 pre-check: Leap's recurrence_r4 vs mlx_lm's gated_delta_kernel at the 125B's GDN shapes (Hk 16, Hv 48,
Dk = Dv = 128, bf16 in, fp32 state), T = 1, 3, 4, 16, 512: outputs and states must be bitwise equal."""
import importlib.util, os
import mlx.core as mx
import mlx_lm.models.gated_delta as GD
spec = importlib.util.spec_from_file_location("leap_gdn35_exact", os.path.expanduser("~/Leap/scripts/gdn35_exact.py"))
LG = importlib.util.module_from_spec(spec); spec.loader.exec_module(LG)
mx.random.seed(0)
ok = True
for T in (1, 3, 4, 16, 512):
    q = (mx.random.normal((1, T, 16, 128)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((1, T, 16, 128)) * 0.1).astype(mx.bfloat16)
    v = mx.random.normal((1, T, 48, 128)).astype(mx.bfloat16)
    g = mx.random.uniform(0.5, 1.0, (1, T, 48)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (1, T, 48)).astype(mx.bfloat16)
    st = (mx.random.normal((1, 48, 128, 128)) * 0.05).astype(mx.float32)
    y0, s0 = GD.gated_delta_kernel(q, k, v, g, beta, st)
    y1, s1 = LG.recurrence_r4(q, k, v, g, beta, st)
    mx.eval(y0, s0, y1, s1)
    eq = bool(mx.array_equal(y0, y1).item()) and bool(mx.array_equal(s0, s1).item())
    ok &= eq
    print(f"T={T}: y equal {bool(mx.array_equal(y0, y1).item())}, state equal {bool(mx.array_equal(s0, s1).item())}")
print("GR1 check:", "BITWISE" if ok else "DIFFERS")
