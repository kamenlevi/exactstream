"""C7 ONE32 (CHECK55 step 1; Leap, Sun 15 Nov 2026 11:1x): what does the one-dispatch 17-32-row path (v32one, C3-ONE
all bitwise) save against v32's 2 x 16 split, on the 12-layer resident slice (layers 0-11: 9 GDN incl. the PLE layer,
3 QSA attention) with the engine's SB (a submit every 4 layers, extended to 2-32 rows)?
Arms, each on its own cache, every cache advancing 32 tokens per round (C5 v2 / C6's harness), rounds rotated:
  A16    one 16-row check (+ an untimed 16-row check to keep pace);
  B2x16  two 16-row checks;
  C32s   one 32-row check, v32's split (experts one call, V32_MOE1; everything else 2 x 16): today's engine path;
  C32o   one 32-row check, v32one on (one dispatch per op; the head mixer and a few small matmuls still split).
REF2's exact-45 forms (att_rope, idx_prep, hc_write, hc_gate, gdn_inproj_x) plus gdn_body; no DENSE-N (as exact-45).
Each check: snapshot, verify forward (Python build timed apart, SB submits inside), the acceptance read, the commit.
PRE-REGISTERED (before any run; Leap 11:1x):
  primary: R32o = C32o / A16 and the saving (C32s - C32o) / A16, medians of per-round ratios.
  prediction: C32s ~1.55-1.65 x A16 (C6's C32 1.579 had mr2 + hc32 on); C32o ~1.25-1.35; saving ~0.25-0.35 x A16.
  lines: R32o <= 1.35 -> step 1 meets its target on the slice (BigMoE's engine arm next);
         1.35 < R32o <= 1.50 -> partial: find what is still split (counts printed) before the engine arm;
         R32o > 1.50 -> the one-dispatch forms do not cut the step on their own: profile the new kernels' GPU time.
usage: [BigMoE venv] c7_one32.py [smoke]   (MEM ~23 GB at 12 layers; quiet ticket via qscreen.sh)"""
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_ng1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_ng1/Leap_scripts/week98"))
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
say(f"C7 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; SB {SBN}; dense_native {ref2.CFG['dense_native']}")
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_ng1/Leap_scripts/*.py")))
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


ARMS = {"A16": (16, False), "B2x16": (16, False), "C32s": (32, False), "C32o": (32, True)}
C = {a: prefill() for a in ARMS}
POS = {a: P for a in ARMS}


def run_arm(a):
    n, one = ARMS[a]
    v32one.ON[0] = one
    c, p = C[a], POS[a]
    mx.synchronize()
    if a == "A16":
        b, w = check(c, p, 16)
        check(c, p + 16, 16)                              # untimed: keep pace (32 tokens per round)
    elif a == "B2x16":
        b1, w1 = check(c, p, 16)
        b2, w2 = check(c, p + 16, 16)
        b, w = b1 + b2, w1 + w2
    else:
        b, w = check(c, p, 32)
    POS[a] += 32
    return b, w


for i in range(2):
    for a in ARMS:
        run_arm(a)
v32one.CNT.update({k: 0 for k in v32one.CNT})
v32one.MISS.clear()
v32.CNT.update({k: 0 for k in v32.CNT})
rounds = 3 if SMOKE else 14
T = {a: [] for a in ARMS}
Bt = {a: [] for a in ARMS}
RS, RO, SAV = [], [], []
names = list(ARMS)
for r in range(rounds):
    order = names[r % len(names):] + names[:r % len(names)]
    got = {}
    for a in order:
        b, w = run_arm(a)
        T[a].append(w); Bt[a].append(b); got[a] = w
    RS.append(got["C32s"] / got["A16"]); RO.append(got["C32o"] / got["A16"])
    SAV.append((got["C32s"] - got["C32o"]) / got["A16"])
v32one.ON[0] = True
m = lambda xs: 1e3 * float(np.median(xs))
say(f"{NL} layers, {rounds} rounds, SB {SBN} (contexts {P} -> {POS['A16']}); ms per check (python build inside):")
for a in names:
    say(f"  {a:6s} {m(T[a]):7.2f} ms (build {m(Bt[a]):6.2f} ms, {np.median(Bt[a]) / np.median(T[a]):.2f} of it) = "
        f"{m(T[a]) / m(T['A16']):.3f} x A16")
q = lambda xs: f"{np.median(xs):.3f} [{np.percentile(xs, 25):.3f}-{np.percentile(xs, 75):.3f}]"
say(f"  per-round medians [IQR]: R32s {q(RS)}; R32o {q(RO)}; saving (C32s - C32o) / A16 {q(SAV)}")
say(f"  v32one counts over the timed C32o checks: {dict(v32one.CNT)}; matmuls still split (shape: calls): {v32one.MISS}; "
    f"v32 wrappers: {dict(v32.CNT)}")
r = float(np.median(RO))
say(f"C7 verdict (pre-registered): R32o {r:.3f} -> " + ("MEETS step 1's target on the slice (<= 1.35)" if r <= 1.35 else
    "PARTIAL (1.35-1.50): find what is still split" if r <= 1.50 else "the forms alone do not cut the step (> 1.50)"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
