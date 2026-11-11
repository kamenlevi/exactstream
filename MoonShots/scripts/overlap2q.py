"""Decoupled runahead, measurement 1: do two MLX GPU streams (two Metal command queues) overlap on the M5 Pro?
A = main stream, B = second stream (mx.new_stream(mx.gpu)). Three cases, medians of 9:
  1. spin: A runs a busy-loop kernel on a few threadgroups (the BigMoE gate's spin while the CPU loads experts, ~1 ms);
     B runs a slice-sized chain of 4-bit matvecs (a GDN in_proj-sized matrix, ~24 MB/step). Does B finish while A spins?
  2. busy: A runs a bandwidth-heavy qmv chain (a layer's worth); B the same slice chain. Total vs A alone + B alone.
  3. B alone, A alone: references.
usage: overlap2q.py   (GPU, quiet lock, <1 min)"""
import time
import mlx.core as mx

SA, SB = mx.default_stream(mx.gpu), mx.new_stream(mx.gpu)
SPIN = mx.fast.metal_kernel(name="spin", input_names=["n"], output_names=["o"], source="""
    uint acc = thread_position_in_grid.x;
    for (uint i = 0; i < (uint)n[0]; i++) acc = acc * 1664525u + 1013904223u;   // ALU-only busy loop
    o[thread_position_in_grid.x] = acc;
""")


def spin(iters, tgs=2):
    return SPIN(inputs=[mx.array([iters], mx.int32)], grid=(32 * tgs, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(32 * tgs,)], output_dtypes=[mx.uint32], stream=SA)[0]


def qmat(rows, cols=2560):
    w = mx.random.randint(0, 2**30, (rows, cols // 8), dtype=mx.int32).view(mx.uint32)
    s = (mx.random.normal((rows, cols // 64)) * 0.01).astype(mx.bfloat16)
    b = (mx.random.normal((rows, cols // 64)) * 0.01).astype(mx.bfloat16)
    mx.eval(w, s, b)
    return w, s, b


WB = qmat(16384)          # ~22 MB: a GDN in_proj (qkv+z) of the 125B -> the slice's main cost per layer
WA = qmat(40000)          # ~55 MB: a main layer's worth of weight reads
x0 = (mx.random.normal((1, 2560)) * 0.1).astype(mx.bfloat16)


def chain(W, n, stream):
    y = x0
    with mx.stream(stream):
        for _ in range(n):
            o = mx.quantized_matmul(y, *W, transpose=True, group_size=64, bits=4)
            y = (o[:, :2560] * 0.001).astype(mx.bfloat16)
    return y


def t(f, reps=9):
    for _ in range(3): f()
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
    return sorted(ts)[reps // 2] * 1e3


# calibrate the spin to ~1 ms
it = 200000
for _ in range(6):
    ms = t(lambda: mx.eval(spin(it)), 5)
    it = int(it * 1.0 / max(ms, 0.05))
ms_spin = t(lambda: mx.eval(spin(it)))
ms_b = t(lambda: mx.eval(chain(WB, 4, SB)))
ms_a = t(lambda: mx.eval(chain(WA, 8, SA)))
print(f"alone: spin {ms_spin:.3f} ms (2 threadgroups) | B slice chain 4 x 22 MB {ms_b:.3f} ms | A main chain 8 x 55 MB {ms_a:.3f} ms", flush=True)


def both_spin():
    a = spin(it); b = chain(WB, 4, SB)
    t0 = time.perf_counter(); mx.async_eval(a); mx.eval(b); tb = time.perf_counter() - t0; mx.eval(a)
    return tb


def both_busy():
    a = chain(WA, 8, SA); b = chain(WB, 4, SB)
    t0 = time.perf_counter(); mx.async_eval(a); mx.eval(b); tb = time.perf_counter() - t0; mx.eval(a)
    return tb, time.perf_counter() - t0


for _ in range(3): both_spin()
tbs = sorted(both_spin() for _ in range(9))
print(f"spin on A + slice on B: B done after {tbs[4]*1e3:.3f} ms (alone {ms_b:.3f}; serialized would be ~{ms_spin + ms_b:.3f})", flush=True)
for _ in range(3): both_busy()
r = sorted(both_busy() for _ in range(9))
print(f"main chain on A + slice on B: B done after {r[4][0]*1e3:.3f} ms, both after {r[4][1]*1e3:.3f} ms "
      f"(A alone {ms_a:.3f}, A+B serial {ms_a + ms_b:.3f})", flush=True)
