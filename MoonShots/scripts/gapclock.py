"""Does the GPU (or the memory fabric) slow down after short idle gaps?
A fake decode step = ~1.9 GB of 4-bit matvecs in one graph. We time the step itself and vary the
CPU-side idle gap before it (0 .. 20 ms), as happens between speculative passes or in Python glue.
If step time grows with the gap, clocks are ramping down between tokens."""
import time, random
import mlx.core as mx

N, IN, OUT = 200, 4096, 4096
mats = []
for _ in range(N):
    q, s, b = mx.quantize((mx.random.normal((OUT, IN)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
    mats.append((q, s, b))
mx.eval(mats)
gb = sum(a.nbytes for p in mats for a in p) / 1e9
x = mx.random.normal((1, IN)).astype(mx.bfloat16)


def step():
    mx.eval([mx.quantized_matmul(x, *p, transpose=True, group_size=64, bits=4) for p in mats])


def spin(ms):                      # busy-wait, so the CPU side stays awake like a real decode loop
    t = time.perf_counter() + ms / 1e3
    while time.perf_counter() < t:
        pass


for _ in range(10):
    step()
gaps = [0, 0.5, 2, 5, 10, 20]
res = {g: [] for g in gaps}
order = gaps * 40
random.Random(0).shuffle(order)
for g in order:
    spin(g) if g else None
    t = time.perf_counter(); step(); res[g].append(time.perf_counter() - t)
base = sorted(res[0])[20]
print(f"step = {gb:.2f} GB of 4-bit matvecs; back-to-back {base*1e3:.2f} ms = {gb/base:.0f} GB/s")
for g in gaps:
    v = sorted(res[g]); med = v[len(v) // 2]
    print(f"gap {g:>4} ms: step {med*1e3:6.2f} ms ({med/base:5.3f}x), p10 {v[4]*1e3:6.2f} p90 {v[36]*1e3:6.2f}")
