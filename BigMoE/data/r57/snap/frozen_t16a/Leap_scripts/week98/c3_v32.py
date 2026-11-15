"""C3 CORES 32 (WEEK98 pillar 2; Leap, Sun 15 Nov 2026 01:4x; data/week98/plan_1115.md 'the 17-32-row path'): is v32's
17-32-row verify forward bitwise equal to <= 16-row verify forwards of the same rows, and does its commit at keep k leave
the state of k verified rows? Bits only, no timing (the orchestrator 01:0x).
Slice: layers 0-3 of the 125B (G G G A: three GDN layers incl. layer 1 with the PLE, one QSA attention layer), resident,
loaded as ca_anat.py loads its slice. ref2 applied with the exact-45 env (REF2_SB_FIRST / PLE_LATE / COMMIT_SUB 9 /
KEEP_CACHES) and its exact forms switched on as the exact-45 loop does (K-73: att_rope, idx_prep, hc_write; K-83: hc_gate,
gdn_inproj_x; GDNB: gdn_body; HCB: hc_b), then v32.install(). The MoE block is MTPLX's own with resident experts (ref2's
experts_nax at <= 16 rows); the loop's K-73 fused MoE block and BigMoE's streamed path are not in this test.
Real code tokens after a 1,024-token prefill (Leap's scripts, as ca_anat).
  T1 forward: one M-row verify (v32; M = 32, 17, 24, 31) vs a 16-row verify, a full-accept commit, then an (M - 16)-row
     verify of the next rows. Logits per row, bitwise.
  T2 commit: a 32-row verify (v32) then commit keep k (k = 1, 9, 16, 17, 24, 31), then a 4-row probe verify of the next
     4 tokens; vs: verify exactly those k rows as <= 16-row checks with full accepts, then the same probe. Probe logits,
     bitwise.
Pre-registered (plan_1115.md C3): all bitwise -> the non-expert 32-row path works (experts as 2 x 16) and goes to BigMoE
with the gate; any row differs -> find the op by bisection before anything else.
usage: [venv] c3_v32.py (MEM ~8 GB)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1")):
    os.environ.setdefault(k_, v_)                       # exact-45's env for ref2 (configs/exact-45.env)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16a/Leap_scripts/week98"))
t00 = time.time()


def say(s):
    print(s, flush=True)


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
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
PLE_L = [i for i, l in enumerate(inner.layers) if getattr(l, "ple", None) is not None]
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body", "hc_b"):
    ref2.CFG[k_] = True
import v32
v32.install(ref2)
if os.environ.get("V32_GDN32") == "1":                  # build step 2: GDN projections at full width (v32mr2.py)
    import v32mr2
    v32mr2.install(v32, ref2)
if os.environ.get("V32_HC32") == "1":                   # build step 4: the HC read at 32 rows in one pass (hc32.py)
    import hc32
    hc32.install(v32, ref2)
if os.environ.get("V32_ONE") == "1":                    # CHECK55 step 1: one dispatch per op at 17-32 rows (v32one.py)
    import v32one
    v32one.install(v32, ref2)
say(f"C3 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS} (PLE layer {PLE_L}), "
    f"{mx.get_active_memory() / 1e9:.1f} GB active, loaded in {time.time() - t00:.0f} s; ref2 forms on: "
    f"{sorted(k for k, v in ref2.CFG.items() if v is True)}")

files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16a/Leap_scripts/*.py")))
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
    for a in range(0, P, 1024):
        lg = model(mx.array([ids[a:min(P, a + 1024)]]), cache=cache)
        mx.eval(lg, states(cache))
    return cache


def verify(cache, pos, n, keep=None):
    """one n-row verify of ids[pos:pos+n], then commit keep (default n = full accept); returns logits [n, V] (fp32)"""
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n if keep is None else keep, verified_tokens=n)
    if not ok:
        raise RuntimeError(f"commit refused (n {n}, keep {keep})")
    mx.eval(states(cache))
    return lg


def chunks(n):
    out, a = [], 0
    while a < n:
        out.append(min(16, n - a)); a += out[-1]
    return out


def cmp(a, b):
    eq = mx.all(a == b, axis=-1)
    d = mx.max(mx.abs(a - b))
    mx.eval(eq, d)
    rows = np.array(eq).astype(bool)
    return int(rows.sum()), len(rows), float(d.item())


allok = True
if os.environ.get("V32_MR2") == "1":                    # DENSE-N's native copies are made by the first <= 16-row call of
    c0 = prefill()                                       # each weight: one 16-row verify first, so T1's M 32 runs mr2 too
    verify(c0, P, 16)
    del c0
say("T1 forward: one M-row verify (v32) vs <= 16-row verifies of the same rows (full accepts between)")
for M in (32, 17, 24, 31):
    c1 = prefill()
    v32.CNT.update({k: 0 for k in v32.CNT})
    if "v32one" in sys.modules:
        sys.modules["v32one"].CNT.update({k: 0 for k in sys.modules["v32one"].CNT})
    lv = verify(c1, P, M)
    cnt = dict(v32.CNT)
    c2 = prefill()
    ref_parts, pos = [], P
    for n in chunks(M):
        ref_parts.append(verify(c2, pos, n)); pos += n
    lr = mx.concatenate(ref_parts, axis=0)
    eq, tot, d = cmp(lv, lr)
    allok &= eq == tot
    extra = (f"; v32mr2 {dict(sys.modules['v32mr2'].CNT)}" if "v32mr2" in sys.modules else "") + \
        (f"; hc32 {dict(sys.modules['hc32'].CNT)}" if "hc32" in sys.modules else "") + \
        (f"; v32one {dict(sys.modules['v32one'].CNT)}" if "v32one" in sys.modules else "")
    say(f"  M {M}: rows bitwise {eq} / {tot} (max |diff| {d:.3g}); v32 wrappers fired {cnt}{extra}")
    del c1, c2

say("T2 commit: a 32-row verify (v32), commit keep k, then a 4-row probe; vs k rows verified as <= 16-row checks, same probe")
for k in (1, 9, 16, 17, 24, 31):
    c1 = prefill()
    verify(c1, P, 32, keep=k)
    pv = verify(c1, P + k, 4)
    c2 = prefill()
    pos = P
    for n in chunks(k):
        verify(c2, pos, n); pos += n
    pr = verify(c2, P + k, 4)
    eq, tot, d = cmp(pv, pr)
    allok &= eq == tot
    say(f"  keep {k}: probe rows bitwise {eq} / {tot} (max |diff| {d:.3g})")
    del c1, c2

rc = {k: v for k, v in ref2.counts().items() if v}
say(f"ref2 forms that ran (counts): {rc}")
say(f"C3 verdict: {'ALL BITWISE' if allok else 'NOT BITWISE -> bisect the op first'} (V32_MOE1={os.environ.get('V32_MOE1', '')}, "
    f"V32_MR2={os.environ.get('V32_MR2', '')}, V32_GDN32={os.environ.get('V32_GDN32', '')}, V32_ONE={os.environ.get('V32_ONE', '')}, "
    f"REF2_DENSE_NATIVE={os.environ.get('REF2_DENSE_NATIVE', '')}); v32 counts at the end {dict(v32.CNT)}")
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
