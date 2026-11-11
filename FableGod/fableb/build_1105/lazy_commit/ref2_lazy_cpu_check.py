"""ref2_lazy_cpu_check.py (Fable B, build_1105): the lazy-commit INTEGRATION against the real library code, on the CPU.
What is real here: the patched ref2's own functions (_lz_form, _lz_true, gdn_lazy_flush, gdn_lazy_state, _lz_commit_pre,
_lz_commit_post, the patched _commit and _gdn_call), MTPLX's commit_verified_window, MTPLX's lazy snapshot and its
rollback (views in, clones out), mlx_lm's ArraysCache and its gated_delta_update (the ops path, which is what MLX runs
off the GPU). What is a stand-in: the Metal kernel. The lazy forward's double leaves in the cache exactly what
_gdn_body_call's lazy branch leaves (state = [conv, committed, pack], the window mark, MTPLX's stash), computing the
committed state and the outputs with gated_delta_update. So this proves the bookkeeping on the real objects, not the
kernel's bits (gdnb_lazy_check.py, on the GPU, does that).
Two caches run the same random event stream: today's protocol (ref2's lazy switch off: the library alone) and the lazy
one. After every event, for every layer: outputs, conv state and TRUE recurrent state equal bit for bit.
MLX is put on its CPU device before anything else; nothing here touches the GPU. ~1-2 min.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I ref2_lazy_cpu_check.py [events] [streams]"""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
import mlx.core as mx
mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.join(HERE, "patched"))            # the patched ref2.py and gdnbody.py
sys.path.append(os.path.expanduser(os.environ.get("FTA_LEAP", "~/Leap") + "/scripts"))   # the rest of Leap's modules
import importlib.util
import numpy as np


def _load(name):                                             # by path: Leap's modules put their own folder first on sys.path
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "patched", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GB = _load("gdnbody")                                        # registered first, so ref2's _gbmod() gets the patched one
ref2 = _load("ref2")
from mtplx import cache_state as CS
from mtplx.models import qwen4_exp as Q
from mlx_lm.models.gated_delta import gated_delta_update

assert os.path.dirname(os.path.abspath(ref2.__file__)) == os.path.join(HERE, "patched"), ref2.__file__
assert os.path.dirname(os.path.abspath(GB.__file__)) == os.path.join(HERE, "patched"), GB.__file__
assert mx.default_device() == mx.cpu
N_EV = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_ST = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NL, CC = 2, 32                                               # layers per cache; conv channels (any width works for the commit)
BF = mx.bfloat16
ref2._ORIG["commit"] = Q.Qwen4ExpTextModel.commit_verified_window   # what apply() would install under the patched _commit


def bits(x):
    x = x.astype(mx.float32) if x.dtype == BF else x
    mx.eval(x)
    return np.array(x).tobytes(), tuple(x.shape)


eq = lambda a, b: bits(a) == bits(b)


def mk_model(rng):
    lays = []
    for _ in range(NL):
        gdn = types.SimpleNamespace(A_log=mx.array(rng.standard_normal(48).astype(np.float32) * 0.3).astype(BF),
                                    dt_bias=mx.array(rng.standard_normal(48).astype(np.float32) * 0.3).astype(BF),
                                    conv_kernel_size=4, training=False)
        lays.append(types.SimpleNamespace(linear_attn=gdn, ple=None))
    def refuse(i, why):
        raise AssertionError("MTPLX refused the commit at layer %d: %s" % (i, why))
    return types.SimpleNamespace(layers=lays, _refuse_commit=refuse)


def mk_cache(rng):
    out = []
    for _ in range(NL):
        e = Q.ArraysCache(size=2)
        e[0] = mx.array(rng.standard_normal((1, 3, CC)).astype(np.float32)).astype(BF)
        e[1] = mx.array(rng.standard_normal((1, 48, 128, 128)).astype(np.float32) * 0.1)
        out.append(e)
    return out


def clone_cache(cache):
    out = []
    for e in cache:
        c = Q.ArraysCache(size=2)
        c[0], c[1] = mx.array(np.array(e[0].astype(mx.float32))).astype(BF), mx.array(np.array(e[1]))
        out.append(c)
    return out


def mk_rows(rng, S):
    f = lambda *sh: mx.array(rng.standard_normal(sh).astype(np.float32)).astype(BF)
    nrm = lambda x: (x / mx.sqrt((x.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-6)).astype(BF)
    return [(f(1, S, CC), nrm(f(1, S, 16, 128)), nrm(f(1, S, 16, 128)), f(1, S, 48, 128), f(1, S, 48), f(1, S, 48)) for _ in range(NL)]


def conv_after(conv, qkv):
    return mx.contiguous(mx.concatenate([conv, qkv], axis=1)[:, -3:, :])


def fwd_stock(gdn, e, rows):
    """what today's <= 16-row forward (and any stock-path forward) leaves in the cache"""
    qkv, q, k, v, a, b = rows
    y, end = gated_delta_update(q, k, v, a, b, gdn.A_log, gdn.dt_bias, e[1], None, use_kernel=False)
    e._mtplx_verify_rows = rows
    e[0], e[1] = conv_after(e[0], qkv), end
    e.advance(qkv.shape[1])
    return y


def fwd_lazy(gdn, e, rows):
    """the double of ref2._gdn_body_call's lazy branch + gdnbody.gdn_body_lazy (the kernel's contract, on the CPU)"""
    qkv, q, k, v, a, b = rows
    S = qkv.shape[1]
    lz = ref2._lz_form(e)
    assert lz in (2, 3), lz
    pend = e.state[2] if lz == 3 else None
    committed = ref2._lz_true(gdn, e[1], pend) if pend is not None else e[1][...]
    y, _ = gated_delta_update(q, k, v, a, b, gdn.A_log, gdn.dt_bias, committed, None, use_kernel=False)
    pack = mx.concatenate([q.reshape(1, S, -1), k.reshape(1, S, -1), v.reshape(1, S, -1), a, b], axis=-1)
    assert pack.shape[-1] == GB.PW
    e._mtplx_verify_rows = rows
    e.state = [conv_after(e[0], qkv), committed, pack]
    e._ref2_lz_c = (committed, S)
    e.advance(S)
    ref2.COUNTS["gdn_lazy"] += 1
    return y


def forward(model, cache, rows, lazy, body=True):
    ys = []
    for layer, e, r in zip(model.layers, cache, rows):
        gdn = layer.linear_attn
        if lazy and body:
            if getattr(e, "_ref2_lz_c", None) is not None:   # _gdn_call's first lines
                e._ref2_lz_c = None
            ys.append(fwd_lazy(gdn, e, r))
        elif lazy:                                           # the REAL patched _gdn_call: clears the mark, flushes, then
            ref2._ORIG["gdn_call"] = lambda self, inputs, mask, cache_, r_=r: fwd_stock(self, cache_, r_)   # "MTPLX's forward"
            ys.append(ref2._gdn_call(gdn, r[0], None, e))
        else:
            ys.append(fwd_stock(gdn, e, r))
    mx.eval(ys, [e.state for e in cache])
    return ys


def commit(model, cache, snap, keep, verified, lazy):
    if lazy:
        ok = ref2._commit(model, cache, snap.states, keep_tokens=keep, verified_tokens=verified)   # the patched hook
    else:
        ok = Q.Qwen4ExpTextModel.commit_verified_window(model, cache, snap.states, keep_tokens=keep, verified_tokens=verified)
    assert ok
    mx.eval([e.state for e in cache])


def same(model, A, Bc, tag, ya=None, yb=None):
    tb = ref2.gdn_lazy_state(model, Bc)
    for i in range(NL):
        ok = eq(A[i].state[0], Bc[i].state[0]) and eq(A[i].state[1], tb[i]) and (ya is None or eq(ya[i], yb[i]))
        assert ok, "%s: layer %d differs (forms %s)" % (tag, i, [ref2._lz_form(e) for e in Bc])


tot = dict(full=0, partial=0, rollback=0, abandon=0, ar=0, prefill=0, stockpath=0, lazy3=0)
for st in range(N_ST):
    rng = np.random.default_rng(1105 + st)
    model = mk_model(rng)
    A = mk_cache(rng)
    Bc = clone_cache(A)
    ref2.CFG["gdn_lazy_commit"] = True
    ref2._ON[0] = st % 2 == 0                                # with and without the commit's phase scopes
    for ev in range(N_EV):
        r = rng.random()
        if r < 0.08:
            rows = mk_rows(rng, int(rng.integers(17, 25)))
            ya = forward(model, A, rows, False)
            yb = forward(model, Bc, rows, True, body=False)
            tot["prefill"] += 1
            same(model, A, Bc, "prefill", ya, yb)
            continue
        ar = r < 0.2
        S = 1 if ar else int(rng.integers(1, 17))
        rows = mk_rows(rng, S)
        sa, sb = CS.snapshot_untrimmable_cache_lazy(A), CS.snapshot_untrimmable_cache_lazy(Bc)
        body = rng.random() < 0.85
        tot["stockpath"] += 0 if body else 1
        ya = forward(model, A, rows, False)
        yb = forward(model, Bc, rows, True, body=body)
        same_y = all(eq(x, y) for x, y in zip(ya, yb))
        r2 = rng.random()
        if not ar and r2 < 0.10:
            CS.rollback_after_verify(A, sa, S); CS.rollback_after_verify(Bc, sb, S)
            tot["abandon"] += 1
        else:
            if not ar and r2 < 0.25:
                CS.rollback_after_verify(A, sa, S); CS.rollback_after_verify(Bc, sb, S)
                sa, sb = CS.snapshot_untrimmable_cache_lazy(A), CS.snapshot_untrimmable_cache_lazy(Bc)
                body = rng.random() < 0.85
                ya2 = forward(model, A, rows, False)
                yb2 = forward(model, Bc, rows, True, body=body)
                same_y = same_y and all(eq(x, y) for x, y in zip(ya2, yb2))
                tot["rollback"] += 1
            keep = S if (ar or rng.random() < 0.25) else int(rng.integers(1, S + 1))
            tot["ar" if ar else ("full" if keep == S else "partial")] += 1
            commit(model, A, sa, keep, S, False)
            commit(model, Bc, sb, keep, S, True)
        assert same_y, "stream %d event %d: outputs differ" % (st, ev)
        tot["lazy3"] += sum(1 for e in Bc if ref2._lz_form(e) == 3)
        same(model, A, Bc, "stream %d event %d" % (st, ev))
    # the switch goes off with rows pending: the next call must flush and run today's path
    ref2.CFG["gdn_lazy_commit"] = False
    for e, layer in zip(Bc, model.layers):
        ref2.gdn_lazy_flush(layer.linear_attn, e)
    assert all(ref2._lz_form(e) == 2 for e in Bc)
    same(model, A, Bc, "after the flush")
print("%d streams x %d events on %d layers, real ref2 / MTPLX / mlx_lm code on the CPU device: every output, conv state and "
      "true recurrent state equal bit for bit" % (N_ST, N_EV, NL))
print("   events: %s" % tot)
print("   ref2 counts: gdn_lazy %d, gdn_lazy_flush %d" % (ref2.COUNTS["gdn_lazy"], ref2.COUNTS["gdn_lazy_flush"]))
print("PASS")
