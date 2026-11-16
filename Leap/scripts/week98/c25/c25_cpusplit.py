"""c25_cpusplit.py (Leap, Mon 16 Nov 2026 18:1x; CHECK25, the orchestrator's 18:0x ask: is the lossy check CPU- or
GPU-paced?): splits the forward call's wall (c25's 'build + submit') into Python graph build, MLX encode and blocking.

MLX encodes GPU work on the calling thread inside mx.async_eval / mx.eval and commits command buffers there. When too
many command buffers are in flight, or active memory is over the limit, it waits inside the same calls.

So per check:
- MLX encode = the thread CPU spent inside those calls;
- blocked = their wall minus that CPU;
- Python build = the forward's thread CPU minus the CPU inside the calls;
- other waits = the forward's wall minus its thread CPU minus blocked (syncs outside the wrapped calls, e.g. .item() or
  np.array(...) on an unevaluated array).

install() wraps mx.async_eval and mx.eval. Callers look them up as module attributes at call time, so the wrappers see
ref2's seams, MTPLX's mid-loop submits and any eval inside the forward. A caller that saved the function earlier is not
seen (c25_engine saves S["ae0"] before install; pass K is not run with this).
snap() -> the accumulators; delta(a, b) -> b - a. Off unless C25_CPUSPLIT=1 (c25_engine.setup).
usage: python c25_cpusplit.py   (self-test on a toy graph: the split's parts add up; < 1 s, < 50 MB)"""
import time

import mlx.core as mx

A = dict(n_ae=0, cpu_ae=0.0, wall_ae=0.0, n_ev=0, cpu_ev=0.0, wall_ev=0.0)
_ORIG = {}
_DEPTH = [0]


def _wrap(name, kn, kc, kw):
    f0 = getattr(mx, name)
    _ORIG[name] = f0

    def w(*a, **k):
        if _DEPTH[0]:                                    # nested (an eval inside a wrapped call): counted by the outer one
            return f0(*a, **k)
        _DEPTH[0] += 1
        c0, t0 = time.thread_time(), time.perf_counter()
        try:
            return f0(*a, **k)
        finally:
            A[kc] += time.thread_time() - c0
            A[kw] += time.perf_counter() - t0
            A[kn] += 1
            _DEPTH[0] -= 1
    w.__name__ = name
    w.__doc__ = f0.__doc__
    setattr(mx, name, w)


def install():
    if not _ORIG:
        _wrap("async_eval", "n_ae", "cpu_ae", "wall_ae")
        _wrap("eval", "n_ev", "cpu_ev", "wall_ev")


def uninstall():
    for k, f in _ORIG.items():
        setattr(mx, k, f)
    _ORIG.clear()


def snap():
    return dict(A)


def delta(a, b):
    return {k: b[k] - a[k] for k in a}


def parts(d, fwd_wall, fwd_cpu):
    """ms: Python build, MLX encode, blocked in eval, other waits (the four add up to the forward's wall)"""
    enc = d["cpu_ae"] + d["cpu_ev"]
    blk = max(0.0, d["wall_ae"] + d["wall_ev"] - enc)
    py = fwd_cpu - enc
    return dict(py=1e3 * py, enc=1e3 * enc, blk=1e3 * blk, other=1e3 * (fwd_wall - fwd_cpu - blk),
                n_ae=d["n_ae"], n_ev=d["n_ev"])


if __name__ == "__main__":
    f_ae0, f_ev0 = mx.async_eval, mx.eval
    install()
    assert mx.async_eval is not f_ae0 and mx.eval is not f_ev0
    x = mx.random.normal((512, 512))
    a0, w0, c0 = snap(), time.perf_counter(), time.thread_time()
    y = x
    for i in range(40):
        y = (y @ x) * 1e-3 + 1.0
        if i % 10 == 9:
            mx.async_eval(y)
    mx.eval(y)
    w1, c1 = time.perf_counter(), time.thread_time()
    p = parts(delta(a0, snap()), w1 - w0, c1 - c0)
    tot = p["py"] + p["enc"] + p["blk"] + p["other"]
    ok = p["n_ae"] == 4 and p["n_ev"] == 1 and abs(tot - 1e3 * (w1 - w0)) < 1e-6 and p["enc"] >= 0 and p["py"] >= 0
    uninstall()
    ok = ok and mx.async_eval is f_ae0 and mx.eval is f_ev0
    print(f"c25_cpusplit self-test: {'PASS' if ok else 'FAIL'}; {p}; wall {1e3 * (w1 - w0):.3f} ms")
