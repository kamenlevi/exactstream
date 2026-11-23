"""C10 BOF2-O (CHECK55 track A, 32 rows; Leap, Sun 15 Nov 2026 18:2x; C9's harness and 12-layer slice): BOF2 (the exact
32-row fold, REF2-T's bits) on o_proj / out_proj ONLY. bof2_bench (578279c) killed BOF2 over its 5 shapes (-0.04 ms) but
o_proj / out_proj alone went 113.1 -> 73.6 us per call; this pick came after the data, so it is a new test, its line set
here before any run.
  C32o    REF2-T step 1 complete (v32one per-shape forms, seams to 32), one 32-row check per round;
  C32ob   the same with bofi.TABLE32 = {(2560, 6144): BOF2 bs 4} (12 calls per check on the slice).
BITS first (fresh caches, the same tokens): C32ob's logits == C32o's, every row.
PRE-REGISTERED (before any run; Leap 18:2x):
  expected on the slice: 12 calls x 39.5 us = -0.47 ms per 32-row check; T = -median(C32ob - C32o) / 0.47 (whole wall).
  lines: not bitwise -> a bug, no verdict; T >= 0.5 -> PASS (into E2's C32 arm Monday night); 0.25-0.5 BETWEEN;
  < 0.25 KILL. Prediction: T 0.6-0.9.
usage: [BigMoE venv] c10_bof2o.py [smoke]   (MEM ~21 GB at 12 layers; quiet ticket via qscreen.sh)"""
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
import bofi
bofi.install(ref2)
say(f"C10 BOF2-O {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB "
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


ARMS = {"C32o": False, "C32ob": True}
bofi.TABLE32[(2560, 6144)] = (4, 1)


def set_arm(a):
    v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0], ref2.CFG["seam_hi"] = True, True, 4, 32
    bofi.ON[0] = ARMS[a]


def prefill_off():
    on, bofi.ON[0] = bofi.ON[0], False
    try:
        return prefill()
    finally:
        bofi.ON[0] = on


def logits_check(cache, pos, n):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)[0]
    mx.eval(lg)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    mx.eval(states(cache))
    return lg


L = {}
for a in ARMS:
    set_arm(a)
    c = prefill_off()
    L[a] = logits_check(c, P, 32)
    del c
    mx.clear_cache()
nb = int(mx.sum(mx.all(L["C32o"] == L["C32ob"], axis=-1)).item())
say(f"bits: C32ob logits == C32o on {nb} / 32 rows (one 32-row check); BOF2 calls {bofi.CNT['bof2']}")
bofi.CNT["bof2"] = 0
C = {a: prefill_off() for a in ARMS}
POS = {a: P for a in ARMS}
WARM = os.environ.get("C9_WARM", "1") == "1"   # C9b / C10b (Sun 18:2x): an untimed check of the round's tokens on a
CW = prefill_off() if WARM else None             # throwaway cache before the timed arms, so no timed arm reads the
                                                 # round's PLE n-gram rows cold (Breakthrough's +8-12 ms first-arm effect)


def warm():
    if WARM:
        set_arm(names_[0])
        p = POS[names_[0]]                       # the timed arms' tokens this round (they advance in lockstep)
        check(CW, p, 32)


names_ = list(ARMS)


def run_arm(a):
    set_arm(a)
    c, p = C[a], POS[a]
    mx.synchronize()
    b, w = check(c, p, 32)
    POS[a] += 32
    return b, w


for i in range(2):
    for a in ARMS:
        run_arm(a)
bofi.CNT["bof2"] = 0
rounds = 3 if SMOKE else 24
T = {a: [] for a in ARMS}
G = {a: [] for a in ARMS}
D, DG = [], []
names = list(ARMS)
for r in range(rounds):
    order = names[r % len(names):] + names[:r % len(names)]
    warm()
    got, gotg = {}, {}
    for a in order:
        b, w = run_arm(a)
        T[a].append(w); G[a].append(w - b); got[a] = w; gotg[a] = w - b
    D.append(got["C32ob"] - got["C32o"]); DG.append(gotg["C32ob"] - gotg["C32o"])
bofi.ON[0] = False
m = lambda xs: 1e3 * float(np.median(xs))
say(f"{NL} layers, {rounds} rounds, warm-up {WARM}, SB {SBN} (contexts {P} -> {POS['C32o']}); ms per 32-row check:")
for a in names:
    say(f"  {a:5s} {m(T[a]):7.2f} ms (after the Python build {m(G[a]):6.2f} ms)")
q = lambda xs: f"{m(xs):+.3f} ms [{1e3 * np.percentile(xs, 25):+.3f}, {1e3 * np.percentile(xs, 75):+.3f}]"
say(f"  paired C32ob - C32o: whole {q(D)}; after the build {q(DG)}; BOF2 calls over the timed checks {bofi.CNT['bof2']}")
EXP = 0.47 * NL / 12
t = -m(D) / EXP
say(f"C10 BOF2-O verdict (pre-registered): bits {'BITWISE' if nb == 32 else 'NOT BITWISE (a bug: no verdict)'}; T {t:.2f} "
    f"(of -{EXP:.2f} ms) -> " + ("no verdict" if nb != 32 else "PASS: into E2's C32 arm" if t >= 0.5 else
                                 "BETWEEN" if t >= 0.25 else "KILL"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
