"""D8 BITS (Leap, Sun 15 Nov 2026 17:5x; v2 adds V6 (V32_MOE1 off) and per-component captures in V0; diagnosing C8's (b1)): on the 12-layer slice, one 32-row check's rows 16-31 differ
from two 16-row checks (full-accept commit between) under REF2-T's step-1 forms AND the contract (16 / 32 rows equal),
while the 4-layer slice (C3, C3-ONE, C8 smoke) is bitwise. Which factor? b1 under variants, one factor at a time:
  V0 as C8 (32-row: v32one W1, seams 32; 16-row: seams 16)   V1 seams 16 for both   V2 seams 32 for both
  V3 V0 without the mid-forward submits (SB off)              V4 v32one off (v32's split)   V5 v32one + C7's forms (W1 off)
and in V0 the first layer whose output rows 16-31 differ (a hook on the decoder layer class).
usage: [BigMoE venv] d8_bits.py   (MEM ~21 GB; share lock; bits only)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("V32_MOE1", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
assert os.environ.get("REF2_DENSE_NATIVE", "0") != "1" and os.environ.get("V32_MR2", "0") != "1", "C7 runs without DENSE-N"
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/week98"))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SBN = 4
t00 = time.time()


def say(s):
    print(s, flush=True)


MD = os.path.expanduser("~/BigMoE/models/bare")
NL = 4 if SMOKE else 12
orig = mu.load_model


def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a:
        a[0] = True
    else:
        kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        inner = model.language_model.model
        inner.layers = inner.layers[:NL]
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (SBN if 2 <= int(rows) <= 32 else 0)
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body"):
    ref2.CFG[k_] = True
import v32, v32one
v32.install(ref2)
v32one.install(v32, ref2)
import refc
refc.install(ref2)
say(f"D8 BITS {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; SB {SBN}; dense_native {ref2.CFG['dense_native']}")
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P = 1024


@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield


def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]


def prefill():
    cache = model.make_cache()
    lg = model(mx.array([ids[0:P]]), cache=cache)
    mx.eval(lg, states(cache))
    return cache


def check(cache, pos, n):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    t0 = time.perf_counter()
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    t1 = time.perf_counter()
    mx.argmax(lg[0], axis=-1).tolist()
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    mx.eval(states(cache))
    mx.synchronize()
    return t1 - t0, time.perf_counter() - t0


import mtplx.models.qwen4_exp as Q_


def prefill_off():
    on, refc.ON[0] = refc.ON[0], False
    try:
        return prefill()
    finally:
        refc.ON[0] = on


def logits_check(cache, pos, n):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)[0]
    mx.eval(lg)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    mx.eval(states(cache))
    return lg


def rows_eq(x, y):
    return [bool(v) for v in mx.all(x == y, axis=-1).tolist()]


REC = []
LC = type(inner.layers[0])
_lcall = LC.__call__


def _rec_call(self, *a, **kw):
    out = _lcall(self, *a, **kw)
    if REC is not None and REC_ON[0]:
        o = out[0] if isinstance(out, tuple) else out
        REC.append(o)
    return out


REC_ON = [False]
LC.__call__ = _rec_call
COMP = {}


def _hook(cls, tag):
    f0 = cls.__call__

    def g(self, *a, **kw):
        out = f0(self, *a, **kw)
        if REC_ON[0]:
            o = out[0] if isinstance(out, tuple) else out
            COMP.setdefault(tag, []).append(o)
        return out
    cls.__call__ = g


_l8 = inner.layers[8]
_hook(type(_l8.mlp), "moe_block")
if hasattr(_l8.mlp, "switch_mlp"):
    _hook(type(_l8.mlp.switch_mlp), "experts")
if hasattr(_l8.mlp, "shared_expert"):
    _hook(type(_l8.mlp.shared_expert), "shared")
_mid0 = Q._prefill_midloop_eval_layers


def b1(seam32, seam16, sb=True, one=True, w1=True, capture=False, moe1=True):
    v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0] = one, w1, 4
    v32.ST["moe1"] = moe1
    refc.ON[0] = False
    Q._prefill_midloop_eval_layers = (lambda rows: _mid(rows) or (SBN if 2 <= int(rows) <= 32 else 0)) if sb else _mid
    ref2.CFG["seam_hi"] = seam32
    c1 = prefill_off()
    REC.clear(); COMP.clear(); REC_ON[0] = capture
    l32 = logits_check(c1, P, 32)
    r32 = list(REC); c32 = {k: list(v) for k, v in COMP.items()}
    ref2.CFG["seam_hi"] = seam16
    c2 = prefill_off()
    REC.clear(); COMP.clear()
    la = logits_check(c2, P, 16)
    ra = list(REC); ca = {k: list(v) for k, v in COMP.items()}
    REC.clear(); COMP.clear()
    lb = logits_check(c2, P + 16, 16)
    rb = list(REC); cb = {k: list(v) for k, v in COMP.items()}
    REC_ON[0] = False
    eq = rows_eq(l32, mx.concatenate([la, lb]))
    first = None
    if capture:
        for i, (x32, xa, xb) in enumerate(zip(r32, ra, rb)):
            x16 = mx.concatenate([xa.reshape((-1,) + xa.shape[2:]), xb.reshape((-1,) + xb.shape[2:])])
            e = rows_eq(x32.reshape((-1,) + x32.shape[2:]).reshape(32, -1), x16.reshape(32, -1))
            if not all(e):
                first = (i, [j for j, v in enumerate(e) if not v][:8])
                break
    comp = {}
    if capture:
        for k in c32:
            for i, (x32, xa, xb) in enumerate(zip(c32[k], ca.get(k, []), cb.get(k, []))):
                n32 = x32.size // 32
                if xa.size // 16 != n32 or xb.size // 16 != n32:
                    comp[k] = f"call {i}: shapes {x32.shape} / {xa.shape} / {xb.shape}"
                    break
                e = rows_eq(x32.reshape(32, n32), mx.concatenate([xa.reshape(16, n32), xb.reshape(16, n32)]))
                if not all(e):
                    comp[k] = f"first differing call {i} of {len(c32[k])}, rows {[j for j, v in enumerate(e) if not v][:8]}"
                    break
            else:
                comp[k] = f"all {len(c32[k])} calls bitwise"
        first = (first, comp)
    return sum(eq), [j for j, v in enumerate(eq) if not v][:6], first, (len(r32), len(ra), len(rb))


V = {"V0 (C8: 32 seams 32, 16 seams 16)": dict(seam32=32, seam16=16, capture=True),
     "V6 V0 with V32_MOE1 off (MoE block + experts as 2 x 16)": dict(seam32=32, seam16=16, moe1=False)}
for name, kw in V.items():
    n, bad, first, nrec = b1(**kw)
    say(f"  {name}: b1 {n} / 32 rows bitwise; first differing rows {bad}" +
        (f"; first differing layer output {first} (records {nrec})" if kw.get("capture") else ""))
    mx.clear_cache()
Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (SBN if 2 <= int(rows) <= 32 else 0)
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
