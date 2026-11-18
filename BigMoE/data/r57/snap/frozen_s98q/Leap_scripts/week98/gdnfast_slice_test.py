"""gdnfast_slice_test.py (Leap, Mon 16 Nov 2026 14:1x; CHECK25): f32one.FASTGDN (gdn_fast.py, == mlx_lm's gated delta
kernel bit for bit in isolation, 182 / 182) through the real GDN path on the 4-layer slice (GGGA; loaded and stacked as
f32one_test.py: FAST stack + v32 + f32one qmm32n): O + FASTGDN vs O at M 32 / 24 / 17 (every logit and every cache state
bit for bit), and through the commit replay (keep k, then a 4-row probe). Bits only, no timing.
usage: [BigMoE venv] gdnfast_slice_test.py   (MEM ~9 GB; share lock)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("REF2_HC8_FAST", "1"), ("REF2_SMALL_NATIVE", "1"), ("REF2_DN_KS", "1"), ("REF2_OOO_1C", "1"),
               ("REF2_NAX_1CQ", "1"), ("REF2_NAX_COH", "1")):
    os.environ.setdefault(k_, v_)                       # lossy-45's REF2_* (release configs/lossy-45.env) minus GDN_LAZY
os.environ.pop("REF2_GDN_LAZY", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
os.environ.setdefault("V32_MOE1", "1")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/week98"))
t00 = time.time()
OUT = os.path.expanduser("~/Leap/data/week98/check25/gdnfast_slice_test.txt")
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
say(f"GDNFAST SLICE TEST {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}; DENSE-N-RAM {_dn[0]} weights "
    f"({_dn[1]:.0f} MB); HC8 {_h8[0]} mixers; SVH base {_svh}; ref2 forms on: "
    f"{sorted(k for k, v in ref2.CFG.items() if v is True)}; {mx.get_active_memory() / 1e9:.1f} GB active")


files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/*.py")))
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
    lg = model(mx.array([ids[:P]]), cache=cache)
    mx.eval(lg, states(cache))
    ref2.screened_head_reset()
    ref2.screened_head_add(ids[:P + 64])
    return cache


def verify(cache, pos, n, keep=None):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg)
    assert model.commit_verified_window(cache, snap.states, keep_tokens=n if keep is None else keep, verified_tokens=n)
    mx.eval(states(cache))
    return lg


def bits(a, b):
    """every logit, bit for bit (non-finite ones included), and the state arrays"""
    return bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())


def arm(fg):
    f32one.install("qmm32n")
    f32one.FASTGDN[0] = fg
    f32one.COUNTS.update({k: 0 for k in f32one.COUNTS})


bad = []
for M in (32, 24, 17):
    out, st, cnt = {}, {}, {}
    for nm, fg in (("O", False), ("OG", True)):
        c = prefill()
        arm(fg)
        out[nm] = verify(c, P, M)
        st[nm] = [a for s_ in states(c) for a in (s_ if isinstance(s_, (list, tuple)) else [s_]) if isinstance(a, mx.array)]
        cnt[nm] = f32one.COUNTS["gdn_fast"]
        f32one.remove()
        f32one.FASTGDN[0] = False
        del c
    ok_l = bits(out["OG"], out["O"])
    ok_s = len(st["OG"]) == len(st["O"]) and all(
        a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b, equal_nan=True).item()) for a, b in zip(st["OG"], st["O"]))
    say(f"M {M}: O + FASTGDN vs O: logits bitwise {ok_l}, cache states bitwise {ok_s} ({len(st['O'])} arrays); gdn_fast calls "
        f"{cnt['OG']} (O: {cnt['O']})")
    if not (ok_l and ok_s and cnt["OG"] > 0 and cnt["O"] == 0):
        bad.append(f"M {M}")
for k in (1, 9, 16, 17, 24, 31):                       # through the commit replay: keep k, then a 4-row probe
    pv = {}
    for nm, fg in (("O", False), ("OG", True)):
        c = prefill()
        arm(fg)
        verify(c, P, 32, keep=k)
        pv[nm] = verify(c, P + k, 4)
        f32one.remove()
        f32one.FASTGDN[0] = False
        del c
    ok = bits(pv["OG"], pv["O"])
    say(f"keep {k}: the probe after the commit, O + FASTGDN vs O: bitwise {ok}")
    if not ok:
        bad.append(f"keep {k}")
say(f"VERDICT: {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}; done in {time.time() - t00:.0f} s; peak "
    f"{mx.get_peak_memory() / 1e9:.1f} GB")
open(OUT, "a").write("\n".join(L_) + "\n")
