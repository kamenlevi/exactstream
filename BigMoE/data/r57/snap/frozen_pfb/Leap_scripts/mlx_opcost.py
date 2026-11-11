"""L1-15 follow-up (Leap, Sat 2026-10-24): MLX's per-op cost when the work per op is tiny (the 16-row verify regime).
Chains / fans of N small ops on [16, 2048] bf16, eager and under mx.compile; also custom metal_kernel calls. Gives the
CPU-side floor per dispatched op (graph build + encode + launch), to set against the ~41 ms "target build" phase of the
35B DFlash loop (small35_prof.txt).
Usage: [SpecDecode venv] mlx_opcost.py  -> ~/Leap/data/mlx_opcost.txt"""
import os, time
import mlx.core as mx
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
def timeit(fn, reps=11):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); o = fn(); t1 = time.perf_counter(); mx.eval(o); t2 = time.perf_counter()
        ts.append((t1 - t, t2 - t1))
    ts.sort(key=lambda x: x[0] + x[1])
    return ts[reps // 2]
x = mx.random.normal((16, 2048)).astype(mx.bfloat16); w = mx.random.normal((2048,)).astype(mx.bfloat16)
mx.eval(x, w)
N = 2000
def chain_add():
    y = x
    for _ in range(N):
        y = y + w
    return y
def chain_mix():                     # alternate ops so nothing fuses trivially
    y = x
    for i in range(N // 4):
        y = mx.fast.rms_norm(y, w, 1e-6); y = y * w; y = mx.sigmoid(y); y = y + x
    return y
def fan_add():
    return [x + w for _ in range(N)]
k = mx.fast.metal_kernel(name="leap_opcost_copy", input_names=["a"], output_names=["b"],
                         source="uint i = thread_position_in_grid.x; b[i] = a[i] + T(1);")
def chain_custom():
    y = x.reshape(-1)
    for _ in range(N):
        (y,) = k(inputs=[y], template=[("T", mx.bfloat16)], grid=(y.size, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[y.shape], output_dtypes=[y.dtype])
    return y
for nm, f in (("chain of adds", chain_add), ("chain of rms_norm / mul / sigmoid / add", chain_mix),
              ("fan of independent adds", fan_add), ("chain of custom metal_kernel calls", chain_custom)):
    b, e = timeit(f)
    say(f"{nm:42s} N {N}: build {b * 1e6 / N:5.1f} us/op + eval {e * 1e6 / N:5.1f} us/op = {(b + e) * 1e6 / N:5.1f} us/op")
cf = mx.compile(chain_mix)
b, e = timeit(cf)
say(f"{'chain_mix under mx.compile':42s} N {N}: {(b + e) * 1e6 / N:5.1f} us per original op")
open(os.path.expanduser("~/Leap/data/mlx_opcost.txt"), "w").write("\n".join(lines) + "\n")
