"""hc4_slice_test.py (Leap, Mon 16 Nov 2026 13:4x; V3 = R57_HC4 for EvalSuite's G2: Breakthrough's switch hc4.py 210d95d
over Leap's hc4i router): on the 4-layer slice (layers 0-3, GGGA, layer 1 carries the PLE; loaded as f32one_test.py
loads it, the FAST stack: DENSE-N-RAM, HC8, att_single, SVH), Breakthrough's install sequence:
  dense_native_convert -> hc4.prepare(model) (4-bit copies from the bf16 originals, the map's bt_mix4) -> hc8_install ->
  v32 + f32one -> hc4i.install(); hc4.install() -> per verify, after the prefill: hc4.engage(on)
Tests (bits / numbers only, no timing):
  (1) PER CALL: every GatedResidual read hc4i serves (<= 16 rows: hc4r.read on 16 zero-padded rows; 17-32 rows in the
      verify phase: hc4r.read32) is also run through the chain below hc4i (MTPLX's stock eager read of the same 4-bit
      modules, the reference Breakthrough named) on the SAME inputs: max rel error of mixed / written / inject
      (max |a - b| / max |b|) vs hc4r's tolerance (mixed rel <= 1.8e-3).
  (2) WHOLE SLICE, engaged: verify logits with hc4i ON vs OFF at 16 rows, at 32 rows over v32 (f32one off) and over f32one
      (dn32 native): max |logit diff| over shared finite logits, argmax agreement.
  (3) engage(False) after an engaged item: a 16-row verify == the never-engaged 16-row verify bit for bit (the originals
      come back), and hc4i passes every 8-bit HC8 read through (hc4 counts 0).
  (4) the counts: hc4.COUNTS, hc4i.COUNTS, the copies made (prepare's counts, MB).
usage: [BigMoE venv] hc4_slice_test.py   (MEM ~9 GB; share lock; ~1 min)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("REF2_HC8_FAST", "1"), ("REF2_SMALL_NATIVE", "1"), ("REF2_DN_KS", "1"), ("REF2_OOO_1C", "1"),
               ("REF2_NAX_1CQ", "1"), ("REF2_NAX_COH", "1")):
    os.environ.setdefault(k_, v_)                       # lossy-45's REF2_* minus GDN_LAZY (as f32one_test.py)
os.environ.pop("REF2_GDN_LAZY", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
os.environ.setdefault("V32_MOE1", "1")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
t00 = time.time()
OUT = os.path.expanduser("~/Leap/data/week98/check25/hc4_slice_test.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


MD = os.path.expanduser("~/BigMoE/models/bare")
NL = 4
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
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts/m1/g2"))
import hc4                                               # Breakthrough's switch (read-only import)
_c4 = hc4.prepare(model)                                 # BEFORE hc8_install (from the bf16 originals)
import hc8r
_h8 = hc8r.hc8_install(model)
ref2.CFG["hc8_fast"] = True
_svh = ref2.screened_head_install(model.language_model.lm_head)
ref2.CFG["screened_head"] = True
import v32
v32.install(ref2)
import f32one
f32one.install("dn32")
f32one.ON[0] = False                                     # off until the f32one arm
import hc4i
hc4i.install()                                           # after v32 / f32one: hc4i is the outermost GatedResidual read
hc4.install()
say(f"HC4 SLICE TEST {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}; DENSE-N-RAM {_dn[0]} "
    f"weights; hc4.prepare -> {_c4}; HC8 {_h8[0]} mixers; SVH base {_svh}; hc4 {os.path.getmtime(hc4.__file__):.0f} "
    f"({hc4.__file__}); {mx.get_active_memory() / 1e9:.1f} GB active")

# ---- (1) the per-call comparison: wrap the outermost read (hc4i._gr4)
GR = ref2.Q.GatedResidual
_outer = GR.__call__
CALLS = []                                               # (rows, rel mixed, rel written, rel inject)
CMP_ON = [False]


def _rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float((mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-30)).item())


def gr_cmp(self, hyper_input, pending=None):
    if not CMP_ON[0]:
        return _outer(self, hyper_input, pending)
    n0 = hc4i.COUNTS["hc4_16"] + hc4i.COUNTS["hc4_32"]
    out = _outer(self, hyper_input, pending)
    if hc4i.COUNTS["hc4_16"] + hc4i.COUNTS["hc4_32"] > n0:   # hc4i served it: the eager 4-bit read on the same inputs
        hc4i.ON[0] = False
        try:
            ref = _outer(self, hyper_input, pending)
        finally:
            hc4i.ON[0] = True
        mx.eval(out, ref)
        rows = ref2._rows(hyper_input)[1]                 # as hc4i counts them
        CALLS.append((rows,) + tuple(_rel(o, r) for o, r in zip(out, ref)))
    return out


GR.__call__ = gr_cmp

files = sorted(glob.glob(os.path.expanduser("~/Leap/scripts/*.py")))
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
    hc4.engage(False)                                   # every prefill on the original objects
    cache = model.make_cache()
    lg = model(mx.array([ids[:P]]), cache=cache)
    mx.eval(lg, states(cache))
    ref2.screened_head_reset()
    ref2.screened_head_add(ids[:P + 64])
    return cache


def verify(n, engaged, h4, one=False):
    cache = prefill()
    hc4.engage(engaged)
    hc4i.ON[0] = h4
    f32one.ON[0] = one
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[P:P + n]]), cache=cache)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg)
    assert model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    mx.eval(states(cache))
    f32one.ON[0] = False
    hc4i.ON[0] = True
    return lg


def cmp(a, b):
    fin = mx.isfinite(a) & mx.isfinite(b)
    eq = mx.all(mx.where(fin, a == b, a.view(mx.uint32) == b.view(mx.uint32)), axis=-1)
    d = mx.max(mx.where(fin, mx.abs(a - b), 0))
    am = mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)
    mx.eval(eq, d, am)
    return f"rows bitwise {int(np.array(eq).sum())} / {eq.size}, max |logit diff| {float(d.item()):.4g}, argmax agrees {int(np.array(am).sum())} / {am.size}"


TOL = 1.8e-3
bad = []
# (3) first: the never-engaged reference, and hc4i's pass-through of 8-bit HC8 reads
for k in hc4i.COUNTS:
    hc4i.COUNTS[k] = 0
base16 = verify(16, False, True)
say(f"(3) never engaged, 16 rows: hc4i {dict(hc4i.COUNTS)} (hc4 reads must be 0)")
if hc4i.COUNTS["hc4_16"] or hc4i.COUNTS["hc4_32"]:
    bad.append("hc4i served an 8-bit read")
# (1) + (2): engaged
for n, one, tag in ((16, False, "16 rows"), (32, False, "32 rows, v32"), (32, True, "32 rows, f32one dn32")):
    CALLS.clear()
    for k in hc4i.COUNTS:
        hc4i.COUNTS[k] = 0
    CMP_ON[0] = True
    on = verify(n, True, True, one)
    CMP_ON[0] = False
    cnt = dict(hc4i.COUNTS)
    off = verify(n, True, False, one)
    if CALLS:
        R = np.array(CALLS)
        say(f"(1) {tag}: {len(CALLS)} reads served by hc4i at rows {sorted(set(int(r) for r in R[:, 0]))}; max rel vs the eager "
            f"4-bit read: mixed {R[:, 1].max():.3g}, written {R[:, 2].max():.3g}, inject {R[:, 3].max():.3g} (tolerance mixed "
            f"{TOL:.1e}); hc4i {cnt}")
        if R[:, 1].max() > TOL:
            bad.append(f"{tag}: mixed rel {R[:, 1].max():.3g} > {TOL:.1e}")
    else:
        say(f"(1) {tag}: NO read served by hc4i (hc4i {cnt})")
        bad.append(f"{tag}: hc4i served nothing")
    say(f"(2) {tag}, engaged, hc4i ON vs OFF (eager 4-bit read): {cmp(on, off)}")
    say(f"    vs the never-engaged 8-bit run (first 16 rows): {cmp(on[:16], base16)}")
back16 = verify(16, False, True)
say(f"(3) engage(False) after engaged items, 16 rows vs never engaged: {cmp(back16, base16)}")
if not bool(mx.all((back16 == base16) | (mx.isnan(back16) & mx.isnan(base16))).item()):
    bad.append("engage(False) did not restore the originals bit for bit")
say(f"(4) hc4.COUNTS {dict(hc4.COUNTS)}")
say(f"VERDICT: {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}; done in {time.time() - t00:.0f} s; peak "
    f"{mx.get_peak_memory() / 1e9:.1f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "a").write("\n".join(L_) + "\n")
