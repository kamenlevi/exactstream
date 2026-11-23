"""ENC2 split (Leap, Sun 08:3x): the toy check's build (Python graph construction), encode (async_eval's return) and GPU
tail, with the full-width matmuls (as enc2) and with narrow ones (GPU-light, same op count), to see when two threads can pay."""
import time, mlx.core as mx
import enc2_mod as E
x = (mx.random.normal((E.ROWS, E.D)) * 0.5).astype(mx.bfloat16)
mx.eval(x)
for tag, wide in (("full-width matmuls (enc2 toy)", True), ("narrow matmuls (GPU-light)", False)):
    if not wide:
        E.W[:] = [w[:, :1280] for w in E.W]; mx.eval(E.W)
        def layer(h, i):
            for j in range(6):
                a = mx.fast.rms_norm(h, E.g, 1e-6)
                b = a @ E.W[(i + j) % 4]                       # [3, 1280]
                c = mx.sigmoid(b[:, :640]) * b[:, 640:1280]
                d = mx.concatenate([c, c, c, c], axis=-1)
                h = h + 0.1 * d.astype(h.dtype) - 0.05 * mx.pad(mx.tanh(b), ((0, 0), (0, E.D - 1280))) * 0.1
                h = h * (1.0 / (1.0 + mx.abs(h).mean(axis=-1, keepdims=True)))
                e = mx.softmax(b[:, :256].astype(mx.float32), axis=-1).astype(h.dtype)
                h = h + mx.pad(e, ((0, 0), (0, E.D - 256))) * 0.01
            return h
        E.layer = layer
    bs, es, ws, ser = [], [], [], []
    for k in range(25):
        t0 = time.perf_counter(); h = x
        for i in range(E.L):
            h = E.layer(h, i)
        t1 = time.perf_counter(); mx.async_eval(h); t2 = time.perf_counter(); mx.eval(h); t3 = time.perf_counter()
        bs.append(t1 - t0); es.append(t2 - t1); ws.append(t3 - t2)
        t0 = time.perf_counter(); E.check_serial(x); ser.append(time.perf_counter() - t0)
    md = lambda v: sorted(v)[len(v) // 2] * 1e3
    if E.ENC is None:
        E.ENC = E.Enc()
    two = []
    for k in range(40):
        t0 = time.perf_counter(); E.check_two(x); two.append(time.perf_counter() - t0)
    print(f"  {tag}: build {md(bs):.1f} ms, encode (async_eval) {md(es):.1f} ms, GPU tail {md(ws):.1f} ms | serial SB-2 check "
          f"{md(ser):.1f} ms, two-thread {md(two):.1f} ms (x{md(ser) / md(two):.2f})", flush=True)
