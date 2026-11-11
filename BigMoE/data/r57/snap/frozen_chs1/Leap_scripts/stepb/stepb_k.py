"""STEP 0 PART B, kernel level (Leap, Fri 2026-11-06 07:4x; TABLE_350 'FLAT-CUT PLAN' amendment; the orchestrator 07:2x:
Kamen made the dense-kernel rebuild the top priority).

The generated FLAT-2 harness imports this module right after `import evalsuite.env`, before anything imports mlx.
patch() wraps the mx calls that launch kernels: the mx ops below, mx.fast's norms / rope / sdpa, every kernel made by
mx.fast.metal_kernel and every function made by mx.compile. Off (K["on"] False) a wrapper only passes the call through,
so the engine's lazy graph and its results are unchanged.

In the probe's kernel-level synced forwards (K["on"] True), each wrapped call is timed alone:
  pre   eval of its inputs: the unwrapped glue (operators, slices, astype, ...) it depends on, flushed first
  op    the call itself, then eval of its outputs
Each record: (piece, layer, name, signature, bytes, flags, pre_ms, build_ms, eval_ms). A call nested in another wrapped
call (a compiled function's trace, the ops inside the experts unit) passes through and is only counted.
bytes = distinct inputs + outputs; an input above POOL bytes (an expert store, a table) is not counted ('P' flag);
gathers count indices + 2 x outputs.

calibrate() measures, in the same process: the eval round trip (taken off every record), the dependent-dispatch cost
of a chain of tiny kernels, and the size-matched read floor (a vectorised streaming-read kernel at 16 KB-256 MB, cold:
each read is a different window of a 1 GB buffer, and warm: the same window again)."""
import threading
import time

import numpy as np

K = {"on": False, "rec": [], "ctx": [], "L": None, "depth": 0, "skip": 0, "cnt": {}, "cap": False, "caps": []}
POOL = 256 << 20
_P = {}
_MAIN = threading.get_ident()

OPS = ("quantized_matmul gather_qmm gather_mm matmul addmm conv1d conv_general einsum softmax logsumexp argpartition "
       "argsort topk take_along_axis put_along_axis take concatenate stack where sum mean max min cumsum sigmoid exp log "
       "log1p logaddexp sqrt rsqrt square tanh erf maximum minimum clip pad repeat tile dequantize contiguous abs").split()
FAST = ("rms_norm", "layer_norm", "rope", "scaled_dot_product_attention")
GATHER = ("mx.take", "mx.take_along_axis")


def _arrs(o, out, mx):
    if isinstance(o, mx.array):
        out.append(o)
    elif isinstance(o, (list, tuple)):
        for x in o:
            _arrs(x, out, mx)
    elif isinstance(o, dict):
        for x in o.values():
            _arrs(x, out, mx)
    return out


def _dt(a):
    return str(a.dtype).rsplit(".", 1)[-1]


def _bytes(name, ins, outs, a, kw):
    ob = sum(int(o.nbytes) for o in outs)
    if name in GATHER:
        return sum(int(x.nbytes) for x in ins[1:]) + 2 * ob, "G"
    if name == "mx.gather_qmm":                          # x, w, scales, biases, lhs_indices, rhs_indices
        try:
            w_ = ins[1]
            ri = kw.get("rhs_indices", a[5] if len(a) > 5 else None)
            u = len(np.unique(np.array(ri))) if ri is not None else w_.shape[0]
            per = sum(int(x.nbytes) for x in ins[1:4]) / max(1, w_.shape[0])
            return int(u * per) + int(ins[0].nbytes) + ob, "X"
        except Exception:
            pass
    seen, b, fl = set(), 0, ""
    for x in ins:
        if id(x) in seen:
            continue
        seen.add(id(x))
        n = int(x.nbytes)
        if n > POOL:
            fl = "P"
            continue
        b += n
    return b + ob, fl


class _W:
    """a pass-through callable; not a descriptor, like the nanobind function or kernel object it stands for"""
    __slots__ = ("name", "fn")

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __getattr__(self, k):                            # anything else (__name__, __doc__, ...) from the wrapped object
        if k in ("name", "fn"):
            raise AttributeError(k)
        return getattr(self.fn, k)

    def __call__(self, *a, **kw):
        if not K["on"] or threading.get_ident() != _MAIN:
            return self.fn(*a, **kw)
        if K["depth"]:
            K["cnt"][self.name] = K["cnt"].get(self.name, 0) + 1
            return self.fn(*a, **kw)
        return _timed(self, a, kw)


def _timed(w, a, kw):
    mx, clock = _P["mx"], time.perf_counter
    ins = _arrs(list(a), [], mx)
    _arrs(kw, ins, mx)
    try:
        t0 = clock(); mx.eval(ins); t1 = clock()
    except Exception:                                    # inside a trace, or an input that cannot be evaluated
        K["skip"] += 1
        return w.fn(*a, **kw)
    K["depth"] += 1
    try:
        out = w.fn(*a, **kw)
    finally:
        K["depth"] -= 1
    t2 = clock()
    outs = _arrs(out, [], mx)
    mx.eval(outs)
    t3 = clock()
    b, fl = _bytes(w.name, ins, outs, a, kw)
    sig = ";".join("x".join(str(d) for d in x.shape) + ":" + _dt(x) for x in ins[:4]) + "->" + \
        ";".join("x".join(str(d) for d in x.shape) + ":" + _dt(x) for x in outs[:2])
    K["rec"].append((K["ctx"][-1] if K["ctx"] else "-", K["L"], w.name, sig, b, fl,
                     1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2)))
    if K.get("cap"):                                     # kept for replay(): the call, its args (evaluated) and bytes
        K["caps"].append((w, a, kw, K["ctx"][-1] if K["ctx"] else "-", K["L"], sig, b, fl))
    return out


def patch():
    """wrap before anything imports mlx (the generated harness calls this right after `import evalsuite.env`)"""
    if _P:
        return
    import mlx.core as mx
    _P["mx"] = mx
    n_ = 0
    for n in OPS:
        f = getattr(mx, n, None)
        if f is not None and not isinstance(f, _W):
            setattr(mx, n, _W("mx." + n, f)); n_ += 1
    for n in FAST:
        f = getattr(mx.fast, n, None)
        if f is not None and not isinstance(f, _W):
            setattr(mx.fast, n, _W("fast." + n, f)); n_ += 1
    mk, cp = mx.fast.metal_kernel, mx.compile

    def metal_kernel(*a, **kw):
        k = mk(*a, **kw)
        return _W("mk." + str(kw.get("name", a[0] if a else "?")), k)

    def compile(fun, *a, **kw):
        c = cp(fun, *a, **kw)
        nm = getattr(fun, "__qualname__", None) or getattr(fun, "__name__", None) or type(fun).__name__
        md = (getattr(fun, "__module__", None) or "?").rsplit(".", 1)[-1]
        return _W(f"cc.{md}.{nm}", c)

    mx.fast.metal_kernel, mx.compile = metal_kernel, compile
    _P["orig"] = dict(metal_kernel=mk, compile=cp)
    _P["n"] = n_ + 2
    return n_ + 2


def experts_rec(L, ids, idx, pre_ms, build_ms, eval_ms):
    """the experts unit of one layer (its inner ops pass through): groups from the routed ids. base = distinct experts
    used (MSB-3 records), lsb = distinct experts at ranks 0-1 (the LSB sidecar). QOM / split drops are not taken off,
    so the bytes are an upper bound of what the kernels read."""
    try:
        a = np.array(ids).reshape(-1, np.array(ids).shape[-1])
        base, lsb = len(np.unique(a)), len(np.unique(a[:, :2]))
        r = (idx or {}).get(str(L)) or {}
        msb, rec = int(r.get("msb_sz", 0)), int(r.get("recsz", 0))
        b = base * msb + lsb * max(0, rec - msb)
        info = f"base {base} lsb {lsb} rows {a.shape[0]} msb_sz {msb} lsb_sz {rec - msb}"
    except Exception as e:
        b, info = 0, f"groups failed: {e!r}"
    K["rec"].append(("experts", L, "experts_unit", info, b, "U", pre_ms, build_ms, eval_ms))
    if K["cnt"]:
        K["rec"].append(("experts", L, "experts_inner_calls", repr(sorted(K["cnt"].items())), 0, "C", 0.0, 0.0, 0.0))
    K["cnt"] = {}


_RD = """
    uint gid = thread_position_in_grid.x;
    uint nt = threads_per_grid.x;
    uint n = n4[0];
    uint o = off4[0];
    const device uint4* p = (const device uint4*)a;
    uint4 acc = uint4(0);
    for (uint i = gid; i < n; i += nt) acc ^= p[o + i];
    uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (v == 0x9E3779B9u) out[0] = v;
"""


def calibrate(clock=time.perf_counter, big_mb=1024, reps=24):
    """round trip, dependent-dispatch cost and the read floor (cold and warm), all in ms"""
    mx = _P.get("mx")
    if mx is None:
        import mlx.core as mx
    on = K["on"]
    K["on"] = False
    res = {}
    try:
        a = mx.array([1.0]); mx.eval(a)
        ts = []
        for _ in range(400):
            t0 = clock(); b = a + 1; mx.eval(b); ts.append(clock() - t0)
        res["sync_ms"] = 1e3 * float(np.median(ts))
        res["sync_p10_p90"] = [1e3 * float(np.percentile(ts, 10)), 1e3 * float(np.percentile(ts, 90))]
        ch = []
        for n in (1, 64, 512):
            v = []
            for _ in range(15):
                x = mx.array([0.0]); mx.eval(x)
                t0 = clock()
                for _ in range(n):
                    x = x + 1.0
                mx.eval(x); v.append(clock() - t0)
            ch.append((n, 1e3 * float(np.median(v))))
        res["chain_ms"] = ch
        res["dispatch_us"] = 1e3 * (ch[-1][1] - ch[0][1]) / (ch[-1][0] - ch[0][0])
        mkf = (_P.get("orig") or {}).get("metal_kernel", mx.fast.metal_kernel)
        kern = mkf(name="stepb_read", input_names=["a", "n4", "off4"], output_names=["out"], source=_RD)
        N = (big_mb << 20) // 4
        big = mx.arange(N, dtype=mx.uint32); mx.eval(big)
        rd = []
        for kb in (16, 64, 256, 1024, 2048, 4096, 8192, 16384, 65536, 262144):
            nb = kb << 10
            n4 = nb // 16
            th = min(n4, 1 << 17)
            span = (N * 4 - nb) // 16
            cold, warm = [], []
            for j in range(reps):
                off = (j * 7919 * 4096 // 16) % max(1, span)
                o = kern(inputs=[big, mx.array([n4], dtype=mx.uint32), mx.array([off], dtype=mx.uint32)],
                         grid=(th, 1, 1), threadgroup=(min(256, th), 1, 1), output_shapes=[(1,)],
                         output_dtypes=[mx.uint32], init_value=0)[0]
                t0 = clock(); mx.eval(o); cold.append(clock() - t0)
            for j in range(reps):
                o = kern(inputs=[big, mx.array([n4], dtype=mx.uint32), mx.array([0], dtype=mx.uint32)],
                         grid=(th, 1, 1), threadgroup=(min(256, th), 1, 1), output_shapes=[(1,)],
                         output_dtypes=[mx.uint32], init_value=0)[0]
                t0 = clock(); mx.eval(o); warm.append(clock() - t0)
            c_, w_ = 1e3 * float(np.median(cold)), 1e3 * float(np.median(warm[2:]))
            rd.append(dict(kb=kb, cold_ms=c_, warm_ms=w_,
                           cold_gbs=nb / max(1e-9, (c_ - res["sync_ms"]) * 1e-3) / 1e9,
                           warm_gbs=nb / max(1e-9, (w_ - res["sync_ms"]) * 1e-3) / 1e9))
        res["read"] = rd
        del big
    finally:
        K["on"] = on
    return res


def gpu_ms(ms, sync):
    """a synced record's GPU part: the eval minus one round trip; an eval that found nothing to run (< half a round
    trip) is 0"""
    return 0.0 if ms < 0.5 * sync else max(0.0, ms - sync)


def brief(kr, sync, key, check, j):
    """one kernel-level forward in a line: per piece, wrapped calls, their GPU ms, the glue flushed before them / at the
    piece's end, and the GB/s of the wrapped calls"""
    rec = kr.get("krec") or []
    agg = {}
    for ctx, L, name, sig, b, fl, pre, bld, ev in rec:
        a = agg.setdefault(ctx, [0, 0.0, 0.0, 0, 0])
        if name == "glue_end":
            a[2] += gpu_ms(pre, sync)
            continue
        if name == "experts_inner_calls":
            continue
        a[0] += 1
        a[1] += gpu_ms(ev, sync)
        a[2] += gpu_ms(pre, sync)
        if "P" not in fl:
            a[3] += b
    tot = sum(v[1] + v[2] for v in agg.values())
    parts = "; ".join(f"{k} {v[0]} calls {v[1]:.2f} + glue {v[2]:.2f} ms ({v[3] / 1e6:.1f} MB, "
                      f"{v[3] / 1e6 / max(1e-6, v[1]):.0f} GB/s)" for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1] - kv[1][2]))
    return (f"step B {key} check {check} #{j + 1}: rows {kr.get('rows')}, wall {kr.get('wall', 0):.0f} ms, "
            f"{len(rec)} records, skipped {kr.get('kskip')}, logits == live {kr.get('equal')}; GPU sum {tot:.2f} ms: {parts}")


def _carrier(a, kw, mx):
    """the call's smallest floating input: the one a replay link perturbs (by + 0 x the last output's first value)"""
    best = None
    for x in _arrs(list(a), [], mx) + _arrs(kw, [], mx):
        if x.dtype in (mx.float16, mx.bfloat16, mx.float32) and x.size > 0 and (best is None or x.nbytes < best.nbytes):
            best = x
    return best


def _swap(o, old, new, mx):
    if o is old:
        return new
    if isinstance(o, list):
        return [_swap(x, old, new, mx) for x in o]
    if isinstance(o, tuple):
        return tuple(_swap(x, old, new, mx) for x in o)
    if isinstance(o, dict):
        return {k: _swap(v, old, new, mx) for k, v in o.items()}
    return o


def replay(sync, min_n=3, reps=3, clock=time.perf_counter):
    """Dependent chains of the captured calls, one chain per kernel type (same name and signature), its instances in
    layer order (different weights, so cold reads, as in the check). Link i: the call's smallest float input + 0 x the
    first value of call i-1's first output (3 tiny kernels: multiply, astype, add), so call i waits for call i-1 and
    nothing else changes. The same links without the calls are the null chain; per call = (chain - null) / n.
    Returns [dict(ctx, name, sig, n, bytes, flags, us_chain, us_null)] and frees the captures."""
    mx = _P["mx"]
    on, K["on"] = K["on"], False
    groups = {}
    for w, a, kw, ctx, L, sig, b, fl in K["caps"]:
        groups.setdefault((w.name, sig), []).append((w, a, kw, ctx, L, b, fl))
    K["caps"], K["cap"] = [], False
    out = []
    try:
        for (name, sig), ins in groups.items():
            if len(ins) < min_n or name.startswith("mx.take"):
                continue
            ins.sort(key=lambda t: -1 if t[4] is None else t[4])
            seed = mx.zeros((1,), mx.float32); mx.eval(seed)

            def build(with_calls):
                prev = seed
                for w, a, kw, ctx, L, b, fl in ins:
                    c = _carrier(a, kw, mx)
                    if c is None:
                        return None
                    c2 = c + (prev * 0).astype(c.dtype)
                    if not with_calls:
                        prev = c2.reshape(-1)[:1].astype(mx.float32)
                        continue
                    o = w.fn(*_swap(list(a), c, c2, mx), **_swap(kw, c, c2, mx))
                    oa = _arrs(o, [], mx)
                    if not oa:
                        return None
                    prev = oa[0].reshape(-1)[:1].astype(mx.float32)
                return prev

            try:
                tn, tc = [], []
                for _ in range(reps):
                    for with_calls, acc in ((False, tn), (True, tc)):
                        p_ = build(with_calls)
                        if p_ is None:
                            raise ValueError("no float carrier / no output")
                        t0 = clock(); mx.eval(p_); acc.append(clock() - t0)
                n = len(ins)
                out.append(dict(ctx=ins[0][3], name=name, sig=sig, n=n, bytes=ins[0][5], flags=ins[0][6],
                                us_chain=1e6 * float(np.median(tc)) / n, us_null=1e6 * float(np.median(tn)) / n,
                                layers=[t[4] for t in ins][:4]))
            except Exception as e:
                out.append(dict(ctx=ins[0][3], name=name, sig=sig, n=len(ins), error=repr(e)[:200]))
    finally:
        K["on"] = on
    return out


_TRIV = "uint i = thread_position_in_grid.x; if (i < n[0]) out[i] = inp[i] + 1.0f;"


def sync_chains(clock=time.perf_counter, n=200, reps=7):
    """t_sync: a dependent chain of a kernel with almost no bytes, at grid sizes from one simdgroup to 2,560 x 256
    threads (each kernel reads its predecessor's output): per kernel = (T(n) - T(1)) / (n - 1), in us"""
    mx = _P.get("mx")
    if mx is None:
        import mlx.core as mx
    on, K["on"] = K["on"], False
    mkf = (_P.get("orig") or {}).get("metal_kernel", mx.fast.metal_kernel)
    kern = mkf(name="stepb_triv", input_names=["inp", "n"], output_names=["out"], source=_TRIV)
    res = []
    try:
        for th, tg in ((32, 32), (256, 256), (1024, 1024), (16 * 1024, 1024), (20 * 1024, 1024), (80 * 256, 256),
                       (512 * 256, 256), (2560 * 256, 256)):
            x0 = mx.zeros((th,), mx.float32); nn_ = mx.array([th], dtype=mx.uint32); mx.eval(x0, nn_)
            tt = {}
            for m in (1, n):
                v = []
                for _ in range(reps):
                    x = x0
                    for _ in range(m):
                        x = kern(inputs=[x, nn_], grid=(th, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(th,)],
                                 output_dtypes=[mx.float32])[0]
                    t0 = clock(); mx.eval(x); v.append(clock() - t0)
                tt[m] = float(np.median(v))
            res.append(dict(threads=th, tg=tg, us=1e6 * (tt[n] - tt[1]) / (n - 1), one_ms=1e3 * tt[1]))
    finally:
        K["on"] = on
    return res


def preload(name, path):
    """put a candidate module (a file not installed in /Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts) into sys.modules under the live name, so the
    harness's `import <name>` gets it; nothing when path is empty"""
    if not path:
        return None
    import importlib.util, os, sys
    p = os.path.expanduser(path)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    print(f"[stepb] preloaded {name} from {p}", flush=True)
    return m
