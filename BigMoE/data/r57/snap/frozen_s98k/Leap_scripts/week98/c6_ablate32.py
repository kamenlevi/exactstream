"""C6 ABLATE32 (WEEK98 pillar 2; Leap, Sun 15 Nov 2026 02:2x): where does the exact 32-row check's extra time go?
C5's fair read: one exact 32-row check ~1.6x a 16-row check (12-layer resident slice), against the deep drafter's
1.35-1.4 line. This runs C5 v2's setup (each arm on its own cache, all advancing 32 tokens per round, the engine's SB
every 4 layers for 2-32-row forwards unless C6_SB=0) with the 32-row arm in five forms, each one build step off:
  C32     everything on (v32 + V32_MOE1 + v32mr2 GDN at full width + hc32 + mr2 + DENSE-N);
  noMOE1  the experts as 2 x 16 (V32_MOE1 off: v32 splits the MoE block);
  noGDN32 the GDN layer chunked whole (v32's _gdn: projections 2 x 16 too);
  noHC32  the HC read as 2 x 16;
  noMR2   no mr2 (projections at full width call the 2 x 16 split per weight instead);
plus A16 (a 16-row check), B2x16 (two), and the 17-24-row grid the orchestrator asked for (02:4x): C20 and C24, one
exact 20- / 24-row check with everything on (rows 17-24 pad to the 32-row forms where those pad). The graph build is timed apart: the forward call's Python time (no eval) vs
the eval's wait, per arm.
Pre-registered (before any run): a step's worth = (its arm - C32) / A16. Any step worth < 0.03 x A16 is not paying
at 32 rows; the build share (Python time / check time) >= 0.5 at 32 rows means the 32-row check is CPU-bound and the next
lever is fewer dispatches (one-pass forms for the remaining 2 x 16 pieces), not kernels.
usage: [venv] c6_ablate32.py [smoke] (MEM ~23 GB at 12 layers; quiet ticket)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("REF2_DENSE_NATIVE", "1"), ("V32_MOE1", "1"), ("V32_MR2", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98k/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98k/Leap_scripts/week98"))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SBN = int(os.environ.get("C6_SB", "4"))
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
if SBN:                                                  # the engine's SB (r57 ref_on), extended to 2-32 rows: scheduling only
    _mid = Q._prefill_midloop_eval_layers
    Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (SBN if 2 <= int(rows) <= 32 else 0)
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body"):
    ref2.CFG[k_] = True
import v32, v32mr2, hc32
v32.install(ref2)
v32mr2.install(v32, ref2)
hc32.install(v32, ref2)
say(f"C6 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB active; "
    f"SB {SBN or 'off'}")
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98k/Leap_scripts/*.py")))
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
    """returns (python build s, wall s): the forward call (graph build, with SB's mid-loop submits) then the read"""
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


GDN_FULL, GDN_CHUNK = v32mr2._gdn32, v32._gdn
HC_ONE, HC_SPLIT = hc32._gr32, v32._gr


def setv(on=True, moe1=True, gdn32=True, hc=True, mr2=True):
    v32.ST.update(on=on, moe1=moe1, mr2=mr2)
    Q.GatedDeltaNet.__call__ = GDN_FULL if gdn32 else GDN_CHUNK
    Q.GatedResidual.__call__ = HC_ONE if hc else HC_SPLIT


ARMS = {"A16": (16, dict(on=False)), "B2x16": (16, dict(on=False)), "C20": (20, {}), "C24": (24, {}), "C32": (32, {}),
        "noMOE1": (32, dict(moe1=False)),
        "noGDN32": (32, dict(gdn32=False)), "noHC32": (32, dict(hc=False)), "noMR2": (32, dict(mr2=False))}
C = {a: prefill() for a in ARMS}
POS = {a: P for a in ARMS}


def run_arm(a):
    n, kw = ARMS[a]
    setv(**kw)
    c, p = C[a], POS[a]
    mx.synchronize()
    if a == "A16":
        b, w = check(c, p, 16)
        setv(on=False)
        check(c, p + 16, 16)                              # untimed: keep pace (32 tokens per round)
    elif a == "B2x16":
        b1, w1 = check(c, p, 16)
        b2, w2 = check(c, p + 16, 16)
        b, w = b1 + b2, w1 + w2
    else:
        b, w = check(c, p, n)
        if n < 32:                                       # untimed: keep pace (32 tokens per round)
            check(c, p + n, 32 - n)
    POS[a] += 32
    return b, w


for i in range(2):
    for a in ARMS:
        run_arm(a)
rounds = 3 if SMOKE else 12
T = {a: [] for a in ARMS}
Bt = {a: [] for a in ARMS}
names = list(ARMS)
for r in range(rounds):
    order = names[r % len(names):] + names[:r % len(names)]
    for a in order:
        b, w = run_arm(a)
        T[a].append(w); Bt[a].append(b)
setv()
m = lambda xs: 1e3 * float(np.median(xs))
A = m(T["A16"])
say(f"{NL} layers, {rounds} rounds, SB {SBN or 'off'} (contexts {P} -> {POS['C32']}); ms per check (python build inside):")
for a in names:
    say(f"  {a:8s} {m(T[a]):7.2f} ms (build {m(Bt[a]):6.2f} ms, {np.median(Bt[a]) / np.median(T[a]):.2f} of it) = "
        f"{m(T[a]) / A:.3f} x A16" + ("" if a in ("A16", "B2x16", "C20", "C24", "C32") else f"; step worth {(m(T[a]) - m(T['C32'])) / A:+.3f} x A16"))
say(f"  R32 = C32 / A16 {m(T['C32']) / A:.3f}; B2x16 / A16 {m(T['B2x16']) / A:.3f}")
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
