"""When does a CPU thread see an MLX kernel's write? Kernel A writes v into a >4 KB buffer
(const_cast); kernel B (depends on A) burns ~N ms. CPU polls from another thread."""
import os, sys, time, threading
import numpy as np, mlx.core as mx
buf = mx.zeros((4096,), mx.int32); mx.eval(buf)
B = np.frombuffer(memoryview(buf), np.int32)
wr = mx.fast.metal_kernel(name="vis_write", input_names=["buf", "v"], output_names=["out"], source="""
if (thread_position_in_grid.x == 0) { atomic_store_explicit((device atomic_int*)const_cast<device int*>(buf), v[0], memory_order_relaxed); out[0] = v[0]; }""")
burn = mx.fast.metal_kernel(name="vis_burn", input_names=["dep"], output_names=["out"], source="""
float x = (float)dep[0]; for (int j = 0; j < 20000000; j++) x = x * 1.0000001f + 1e-7f; out[0] = x;""")
for rep in range(3):
    v = 100 + rep
    a = wr(inputs=[buf, mx.array([v], mx.int32)], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.int32])[0]
    b = burn(inputs=[a], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])[0]
    seen = [None]
    def poll():
        t0 = time.perf_counter()
        while B[0] != v and time.perf_counter() - t0 < 5: pass
        seen[0] = time.perf_counter() - t0
    th = threading.Thread(target=poll); t0 = time.perf_counter(); th.start(); mx.eval(b); t_done = time.perf_counter() - t0; th.join()
    print(f"rep {rep}: CPU saw kernel A's write at {seen[0]*1e3:.1f} ms; whole eval {t_done*1e3:.1f} ms", flush=True)
