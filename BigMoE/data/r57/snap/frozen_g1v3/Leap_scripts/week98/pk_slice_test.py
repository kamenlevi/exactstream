"""pk_slice_test.py (Leap, Mon 16 Nov 2026 14:4x; CHECK25): c25_passk.py (the per-class GPU pass ported from CHECK55)
on the 4-layer slice before it runs in the engine: the FAST stack + v32 + f32one as f32one_test.py loads it, libmtlprof loaded
before mlx; a timing pass (3 rounds, no flushes) gives the logits and GPU busy of F16 / F32s / F32o; then pass K (classes
flushed into their own tagged command buffers, 2 rounds after a discarded one). PASS = every pass-K logit row == the timing
pass's bit for bit, and >= 5 classes with GPU time per arm. MEM ~9 GB; share lock; ~1 min."""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("REF2_HC8_FAST", "1"), ("REF2_SMALL_NATIVE", "1"), ("REF2_DN_KS", "1"), ("REF2_OOO_1C", "1"),
               ("REF2_NAX_1CQ", "1"), ("REF2_NAX_COH", "1")):
    os.environ.setdefault(k_, v_)                       # lossy-45's REF2_* (release configs/lossy-45.env) minus GDN_LAZY
os.environ.pop("REF2_GDN_LAZY", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
os.environ.setdefault("V32_MOE1", "1")
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_third_party/mtlprof"))
import mtlprof as MP
MP.load(2)                                               # before mlx (names pipelines as MLX creates them)
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98"))
t00 = time.time()
OUT = os.path.expanduser("~/Leap/data/week98/check25/pk_slice_test.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


MD = os.path.expanduser("~/BigMoE/models/bare")
NL = int(os.environ.get("C3_NL", "4"))
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
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "hc_b", "att_single"):
    ref2.CFG[k_] = True
_dn = ref2.dense_native_convert(model)
import hc8r
_h8 = hc8r.hc8_install(model)
ref2.CFG["hc8_fast"] = True
_svh = ref2.screened_head_install(model.language_model.lm_head)
ref2.CFG["screened_head"] = True
import v32
v32.install(ref2)
import f32one
import int8d
say(f"PASS-K SLICE TEST {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}; DENSE-N-RAM {_dn[0]} weights "
    f"({_dn[1]:.0f} MB); HC8 {_h8[0]} mixers; SVH base {_svh}; ref2 forms on: "
    f"{sorted(k for k, v in ref2.CFG.items() if v is True)}; {mx.get_active_memory() / 1e9:.1f} GB active")


sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98/c25"))
import types
import c25_passk
os.environ["C25_PROF"] = "1"
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P0 = 1024


@contextlib.contextmanager
def _vs(m):
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), m.verify_capture_scope():
        yield


ref2.verify_scope = getattr(ref2, "verify_scope", None) or _vs


def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]


cache = model.make_cache()
lg0 = model(mx.array([ids[:P0]]), cache=cache)
mx.eval(lg0, states(cache))
ref2.screened_head_reset()
ref2.screened_head_add(ids[:P0 + 64])
S0 = CS.snapshot_untrimmable_cache_lazy(cache)
mx.eval(states(cache))
f32one.install("qmm32n")
f32one.ON[0] = False
G = dict(model=model, ref2=ref2, CS=CS, states=states, say=say)
REC = []
E = types.SimpleNamespace(ARMS=["F16", "F32s", "F32o"], S=dict(vocab=int(model.language_model.lm_head.weight.shape[0]),
                          ae0=mx.async_eval))


def knobs(G_, a, n):
    f32one.ON[0] = a == "F32o"


def unknob(G_):
    f32one.ON[0] = False


def back(G_, P_, S0_, n):
    CS.rollback_after_verify(P_, S0_, n)
    mx.eval(states(P_))


def cmp(a, b):
    eq = mx.all(a.view(mx.uint32) == b.view(mx.uint32), axis=-1)
    am = mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)
    mx.eval(eq, am)
    return int(np.array(eq).sum()), int(np.array(am).sum()), int(eq.size)


def _walls():
    W = {}
    for r in REC:
        W.setdefault((r["key"], r["round"]), {})[r["arm"]] = r
    return W


E.knobs, E.unknob, E.back, E.cmp, E._walls, E._say = knobs, unknob, back, cmp, _walls, (lambda G_, s: say(s))
LG = {}
for rnd in range(3):                                     # the timing pass (no flushes): logits + GPU busy per arm
    for a in E.ARMS:
        n = 16 if a == "F16" else 32
        knobs(G, a, n)
        sn = CS.snapshot_untrimmable_cache_lazy(cache)
        mx.synchronize()
        c0, m0 = MP.counts(), MP.now()
        with ref2.verify_scope(model):
            lg = model(mx.array([ids[P0:P0 + n]]), cache=cache)
        lg = lg[0].astype(mx.float32)
        mx.eval(lg)
        assert model.commit_verified_window(cache, sn.states, keep_tokens=n, verified_tokens=n)
        mx.eval(states(cache))
        mx.synchronize()
        m1, c1 = MP.now(), MP.counts()
        unknob(G)
        back(G, cache, S0, n)
        cb = MP.cbs(c0["ncb"], c1["ncb"])
        if rnd > 0:
            REC.append(dict(arm=a, key="slice", round=rnd, prof=dict(busy=MP.busy(cb, m0, m1))))
        if rnd == 1:
            LG[a] = lg
PK = c25_passk.run(E, G, cache, S0, ids, P0, "slice", LG, arms=tuple(E.ARMS), rounds=2)
c25_passk.report(E, G, PK)
bad = [a for a, rows in PK.items() for r in rows if r["bits"] is None or r["bits"][0] != r["bits"][2]]
nz = {a: sum(1 for g in np.mean([r["gpu"] for r in rows], axis=0) if g > 0) for a, rows in PK.items()}
say(f"classes with GPU time per arm: {nz}")
ok = not bad and all(v >= 5 for v in nz.values())
say(f"VERDICT: {'PASS' if ok else 'FAIL'} (pass-K logits == the timing pass bit for bit in every row: {not bad}; >= 5 classes "
    f"seen per arm); done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(OUT, "a").write("\n".join(L_) + "\n")
