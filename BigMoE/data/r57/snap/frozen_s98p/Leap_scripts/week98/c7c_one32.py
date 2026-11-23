"""C7c ONE32 (CHECK55 step 1 with the per-shape forms; Leap, Sun 15 Nov 2026 12:5x): C7b (d692d4b) lost because every
small kernel ran qmv_wide at 32 vectors per threadgroup; one32_bench (caf8db4, cold weights, engine counts) then picked
each shape's fastest bitwise form: qmm16c2a (one weight pass) for weights >= 4 MB, qmv_wide at 2 / 8 vectors (k / v on
MLX's split), the interleaved HC S4 (these calls 36.8 -> 24.1 ms per 32-row check). v32one's W1 now selects that table.
Same slice and harness as C7 / C7b (12-layer resident slice, SB every 4 layers to 32 rows, own caches, rotated).
Arms, each on its own cache, every cache advancing 32 tokens per round, rounds rotated:
  A16     one 16-row check (+ an untimed 16-row check to keep pace);
  C32s    v32's split (experts one call), seams at <= 16 rows: today's engine path;
  C32o1   v32one with C7's forms (qmm16c2, qmv_wide 4 vectors, the tile-on-y S4), seams <= 16 (C7's C32o);
  C32b    v32one with the per-shape best forms (W1), seams <= 16;
  C32o    C32b + seams to 32 rows: step 1 complete.
REF2's exact-45 forms (att_rope, idx_prep, hc_write, hc_gate, gdn_inproj_x) plus gdn_body; no DENSE-N (as exact-45).
Each check: snapshot, verify forward (Python build timed apart, SB submits inside), the acceptance read, the commit.
PRE-REGISTERED (before any run; Leap 12:5x, during the lid pause):
  primary: R32o = C32o / A16 (median of per-round ratios); attribution by paired differences / A16: the best forms
  (C32o1 - C32b), the seams (C32b - C32o), all of step 1 (C32s - C32o).
  prediction: C32o1 ~1.43-1.45 (C7, C7b); the best forms ~0.03-0.05 x A16 (the bench's ~3 ms per engine check scaled to
  12 layers + the head); the seams ~0.05 (C7b); R32o ~1.33-1.40.
  lines: R32o <= 1.35 -> step 1 meets its target on the slice (Monday's E2 engine arm as booked);
         1.35 < R32o <= 1.45 -> partial: E2 still runs (the engine has 4x the per-layer share), the gap goes on the list;
         R32o > 1.45 -> the bench's picks do not carry to the slice: profile with Breakthrough's hook before E2.
usage: [BigMoE venv] c7c_one32.py [smoke]   (MEM ~23 GB at 12 layers; quiet ticket via qscreen.sh)"""
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts/week98"))
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
say(f"C7c {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; SB {SBN}; dense_native {ref2.CFG['dense_native']}")
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts/*.py")))
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


ARMS = {"A16": (16, None), "C32s": (32, (False, False, 4, 16)), "C32o1": (32, (True, False, 4, 16)),
        "C32b": (32, (True, True, 4, 16)), "C32o": (32, (True, True, 4, 32))}   # (v32one on, W1 best forms, C7 vpt, seam_hi)
C = {a: prefill() for a in ARMS}
POS = {a: P for a in ARMS}


def run_arm(a):
    n, cfg = ARMS[a]
    if cfg is not None:
        v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0], ref2.CFG["seam_hi"] = cfg
    else:
        v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0], ref2.CFG["seam_hi"] = True, True, 4, 16
    c, p = C[a], POS[a]
    mx.synchronize()
    if a == "A16":
        b, w = check(c, p, 16)
        check(c, p + 16, 16)                              # untimed: keep pace (32 tokens per round)
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
RAT = {a: [] for a in ARMS}
PW, PS, PA = [], [], []
names = list(ARMS)
for r in range(rounds):
    order = names[r % len(names):] + names[:r % len(names)]
    got = {}
    for a in order:
        b, w = run_arm(a)
        T[a].append(w); Bt[a].append(b); got[a] = w
    for a in names:
        RAT[a].append(got[a] / got["A16"])
    PW.append((got["C32o1"] - got["C32b"]) / got["A16"]); PS.append((got["C32b"] - got["C32o"]) / got["A16"])
    PA.append((got["C32s"] - got["C32o"]) / got["A16"])
v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0], ref2.CFG["seam_hi"] = True, True, 4, 16
m = lambda xs: 1e3 * float(np.median(xs))
say(f"{NL} layers, {rounds} rounds, SB {SBN} (contexts {P} -> {POS['A16']}); ms per check (python build inside):")
for a in names:
    say(f"  {a:6s} {m(T[a]):7.2f} ms (build {m(Bt[a]):6.2f} ms, {np.median(Bt[a]) / np.median(T[a]):.2f} of it) = "
        f"{m(T[a]) / m(T['A16']):.3f} x A16")
q = lambda xs: f"{np.median(xs):.3f} [{np.percentile(xs, 25):.3f}-{np.percentile(xs, 75):.3f}]"
say("  per-round medians [IQR] x A16: " + "; ".join(f"{a} {q(RAT[a])}" for a in names if a != "A16"))
say(f"  paired attribution / A16: best forms (C32o1 - C32b) {q(PW)}; seams to 32 (C32b - C32o) {q(PS)}; "
    f"all of step 1 (C32s - C32o) {q(PA)}")
say(f"  v32one counts over the timed C32o checks: {dict(v32one.CNT)}; matmuls still split (shape: calls): {v32one.MISS}; "
    f"v32 wrappers: {dict(v32.CNT)}")
r = float(np.median(RAT["C32o"]))
say(f"C7c verdict (pre-registered): R32o {r:.3f} -> " + ("MEETS step 1's target on the slice (<= 1.35)" if r <= 1.35 else
    "PARTIAL (1.35-1.45): E2 runs, the gap goes on the list" if r <= 1.45 else "the bench's picks do not carry (> 1.45): profile before E2"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
