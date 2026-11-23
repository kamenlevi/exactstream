"""ENC2 stream (Leap, Sun 08:4x): is enc2's two-thread gain on the GPU-light toy from the thread or from the fresh stream?
Serial SB-2 check on the default stream vs on mx.new_stream(mx.gpu) (same thread), narrow matmuls, bitwise compare."""
import time, mlx.core as mx
import enc2_mod as E
E.W[:] = [w[:, :1280] for w in E.W]; mx.eval(E.W)
def layer(h, i):
    for j in range(6):
        a = mx.fast.rms_norm(h, E.g, 1e-6)
        b = a @ E.W[(i + j) % 4]
        c = mx.sigmoid(b[:, :640]) * b[:, 640:1280]
        d = mx.concatenate([c, c, c, c], axis=-1)
        h = h + 0.1 * d.astype(h.dtype) - 0.05 * mx.pad(mx.tanh(b), ((0, 0), (0, E.D - 1280))) * 0.1
        h = h * (1.0 / (1.0 + mx.abs(h).mean(axis=-1, keepdims=True)))
        e = mx.softmax(b[:, :256].astype(mx.float32), axis=-1).astype(h.dtype)
        h = h + mx.pad(e, ((0, 0), (0, E.D - 256))) * 0.01
    return h
E.layer = layer
x = (mx.random.normal((E.ROWS, E.D)) * 0.5).astype(mx.bfloat16); mx.eval(x)
s = mx.new_stream(mx.gpu)
md = lambda v: sorted(v)[len(v) // 2] * 1e3
ref = E.check_serial(x)
for rnd in range(2):
    a, b = [], []
    for k in range(30):
        t0 = time.perf_counter(); E.check_serial(x); a.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        with mx.stream(s):
            y = E.check_serial(x)
        b.append(time.perf_counter() - t0)
    print(f"  round {rnd}: default stream {md(a):.1f} ms, new stream (same thread) {md(b):.1f} ms (x{md(a) / md(b):.2f}); "
          f"bitwise {bool(mx.array_equal(y, ref).item())}", flush=True)
