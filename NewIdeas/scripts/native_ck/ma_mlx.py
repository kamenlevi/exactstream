"""NATIVE-CK M-A (NewIdeas Sun 8 Nov; pre-registered IDEAS.md 15:44, dd4c28f): MLX's host cost per op on a dependent
chain shaped like the chat check's small ops. Two chains of 2,000 ops each, 20 timed reps after 3 warm-ups, medians:
  stock: (4-bit quantized matvec 1 x 2560 -> 512, quantized matvec 512 -> 2560, add, rms_norm) x 500
  custom: a trivial mx.fast.metal_kernel (fused multiply-add over 2560 floats) x 2000 (the engine runs many custom kernels)
Per rep: Python build time (graph construction), thread CPU inside mx.eval (encode + submit), eval wall.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I ma_mlx.py"""
import time, statistics as st
import mlx.core as mx

H, O, N = 2560, 512, 2000
mx.random.seed(0)
w1 = mx.random.normal((O, H)).astype(mx.bfloat16) * 0.02
w2 = mx.random.normal((H, O)).astype(mx.bfloat16) * 0.02
q1 = mx.quantize(w1, group_size=64, bits=4)
q2 = mx.quantize(w2, group_size=64, bits=4)
g = mx.ones((H,), mx.bfloat16)
a = mx.random.normal((H,)).astype(mx.float32) * 0.01
mx.eval(*q1, *q2, g, a)

K = mx.fast.metal_kernel(name="ck_fma", input_names=["x", "a"], output_names=["y"],
                         source="uint i = thread_position_in_grid.x; y[i] = x[i] * a[i] + 1.0f;")


def stock(x):
    for _ in range(N // 4):
        y = mx.quantized_matmul(x, *q1, transpose=True, group_size=64, bits=4)
        y = mx.quantized_matmul(y, *q2, transpose=True, group_size=64, bits=4)
        x = mx.fast.rms_norm(x + y, g, 1e-6)
    return x


def custom(x):
    for _ in range(N):
        x = K(inputs=[x, a], grid=(H, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(H,)],
              output_dtypes=[mx.float32])[0]
    return x


def run(name, fn, x0):
    b_, e_, w_ = [], [], []
    for rep in range(23):
        x = x0 + 0
        mx.eval(x)
        t0 = time.perf_counter()
        y = fn(x)
        t1 = time.perf_counter()
        c0 = time.thread_time()
        mx.eval(y)
        c1 = time.thread_time(); t2 = time.perf_counter()
        if rep >= 3:
            b_.append(t1 - t0); e_.append(c1 - c0); w_.append(t2 - t1)
    b, e, w = st.median(b_) * 1e6 / N, st.median(e_) * 1e6 / N, st.median(w_) * 1e6 / N
    print(f"M-A {name:6s}: per op build {b:6.2f} us + eval thread CPU {e:6.2f} us = host {b + e:6.2f} us; "
          f"eval wall {w:6.2f} us per op ({st.median(w_) * 1e3:.2f} ms per {N}-op chain)", flush=True)
    return b + e


hs = run("stock", stock, mx.random.normal((1, H)).astype(mx.bfloat16))
hc = run("custom", custom, mx.random.normal((H,)).astype(mx.float32))
print(f"M-A host CPU per op: stock {hs:.2f} us, custom {hc:.2f} us (the engine: ~17 us per op, Leap S3-PYFLOOR)")
