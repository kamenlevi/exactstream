"""GPU load for the slot-reuse stress run: bf16 2048^3 matmuls in batches of 8, at a duty cycle, for DUR seconds or until
killed. It slows the model's GPU phases (the GPU then reads a released slot later), with the loader's timing unchanged.
usage: gpuload.py DUR [DUTY=0.5]"""
import sys, time
import mlx.core as mx
dur = float(sys.argv[1])
duty = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
a = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
b = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
mx.eval(a, b)
t_end = time.time() + dur
while time.time() < t_end:
    t0 = time.perf_counter()
    mx.eval([a @ b for _ in range(8)])
    busy = time.perf_counter() - t0
    if duty < 1:
        time.sleep(busy * (1 - duty) / duty)
