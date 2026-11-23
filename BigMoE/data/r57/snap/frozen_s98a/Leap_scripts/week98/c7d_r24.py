"""C7d R24 (bigger trees: NewIdeas' R(24) ask, Sun 22:3x; Leap, written Sun 15 Nov 2026 23:5x, before any run): what does a
24-row exact check cost against a 16- and a 32-row one, on C7c's slice, with C9b's warm-up?
Why: under REF2-T a 24-row check pads its second tile to 16 rows, so the desk (treek_costs_v1.json) charges it the 32-row
weight passes and saves only the expert union (93.3 vs 109.7 per layer, measured) and the cores' 8 rows: R(24) = R(32)
- 0.12 (-0.101 union, -0.023 cores; engine-scaled). NewIdeas: HE+ is flat between 24 and 32 rows within ~2 %, so R(24)
decides whether a 24-row tree is worth it.
Same slice and harness as C7c (c7c_one32.py: the 12-layer resident slice, SB every 4 layers, own caches, every cache
advancing 32 tokens per round, rounds rotated), plus C9b's untimed warm check of each round's tokens on a throwaway cache
(the cold-read lesson: no timed arm reads that round's PLE n-gram rows cold). Arms:
  A16   one 16-row check (+ an untimed 16-row check to keep pace);
  C24s  a 24-row check through v32's split (16 + 8, seams <= 16; today's engine path) + an untimed 8-row check;
  C24o  a 24-row check, step 1 complete (v32one's per-shape forms, seams to 32) + an untimed 8-row check;
  C32s  a 32-row check through v32's split;
  C32o  a 32-row check, step 1 complete (C7c's C32o).
Bits first (fresh prefills, one check of each arm's rows at the same position): C24o == C24s (24 / 24 rows); C24s[:16]
== A16 (16 / 16, SD == AR); C32o[:24] == C24o (24 / 24); C32o == C32s (32 / 32). Any mismatch: a bug, no timing verdict.
PRE-REGISTERED (before any run; Leap Sun 23:5x):
  primary: R24o = C24o / A16 (median of per-round ratios) and D = R32o - R24o (median of per-round paired differences).
  prediction: D ~0.10-0.12 (if the slice's expert share matches the engine's), R24o ~1.20-1.25, R32o ~1.33-1.35 (C7c
  1.345 without the warm-up), R24s ~R32s - D.
  lines: D >= 0.08 -> the desk R(24) stands (costs v1: 1.22 one pass; the engine's R(24) = E2's R(32) - D);
         0.03 <= D < 0.08 -> partial: R(24) = R(32) - D as measured;
         D < 0.03 -> a 24-row check costs ~ a 32-row one: R(24) := R(32) in the costs (a 24-row tree pays only through kept).
usage: [BigMoE venv] c7d_r24.py [smoke]   (MEM ~21 GB at 12 layers; quiet ticket via qscreen.sh; smoke: 4 layers, share)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("V32_MOE1", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
assert os.environ.get("REF2_DENSE_NATIVE", "0") != "1" and os.environ.get("V32_MR2", "0") != "1", "C7d runs without DENSE-N"
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
say(f"C7d {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB "
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


def logits_check(cache, pos, n):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)[0]
    mx.eval(lg)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    mx.eval(states(cache))
    return lg


# arm: (timed rows, (v32one on, W1 best forms, C7 vpt, seam_hi)); the untimed remainder keeps every cache at 32 per round
SPLIT, ONE = (False, False, 4, 16), (True, True, 4, 32)
ARMS = {"A16": (16, None), "C24s": (24, SPLIT), "C24o": (24, ONE), "C32s": (32, SPLIT), "C32o": (32, ONE)}


def set_arm(a):
    cfg = ARMS[a][1] or (True, True, 4, 16)
    v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0], ref2.CFG["seam_hi"] = cfg


# bits first: one check of each arm's rows at P on a fresh prefill (all arms see the same tokens and state)
LG = {}
for a in ARMS:
    set_arm(a)
    c = prefill()
    LG[a] = logits_check(c, P, ARMS[a][0])
    del c
    mx.clear_cache()
set_arm("A16")


def eq(x, y):
    e = mx.all(x == y, axis=-1)
    return int(mx.sum(e).item()), int(e.shape[0])


BITS = {"C24o == C24s": eq(LG["C24o"], LG["C24s"]), "C24s[:16] == A16": eq(LG["C24s"][:16], LG["A16"]),
        "C32o[:24] == C24o": eq(LG["C32o"][:24], LG["C24o"]), "C32o == C32s": eq(LG["C32o"], LG["C32s"])}
nbits_ok = all(n == m for n, m in BITS.values())
say(f"bits (rows equal / rows): {BITS} -> {'ALL BITWISE' if nbits_ok else 'NOT BITWISE (a bug: no timing verdict)'}")
del LG

C = {a: prefill() for a in ARMS}
POS = {a: P for a in ARMS}
CW = prefill()                                            # the throwaway cache for the per-round warm check (C9b)


def run_arm(a):
    n = ARMS[a][0]
    set_arm(a)
    c, p = C[a], POS[a]
    mx.synchronize()
    b, w = check(c, p, n)
    if n < 32:
        set_arm("A16")
        check(c, p + n, 32 - n)                           # untimed: keep pace (32 tokens per round)
    POS[a] += 32
    return b, w


def warm():
    set_arm("A16")
    p = POS["A16"]                                        # the timed arms' tokens this round (they advance in lockstep)
    check(CW, p, 16)
    check(CW, p + 16, 16)


for i in range(2):
    for a in ARMS:
        run_arm(a)
v32one.CNT.update({k: 0 for k in v32one.CNT})
v32one.MISS.clear()
v32.CNT.update({k: 0 for k in v32.CNT})
rounds = 3 if SMOKE else 20
T = {a: [] for a in ARMS}
Bt = {a: [] for a in ARMS}
RAT = {a: [] for a in ARMS}
DD, DS, G24 = [], [], []
names = list(ARMS)
for r in range(rounds):
    order = names[r % len(names):] + names[:r % len(names)]
    warm()
    got = {}
    for a in order:
        b, w = run_arm(a)
        T[a].append(w); Bt[a].append(b); got[a] = w
    for a in names:
        RAT[a].append(got[a] / got["A16"])
    DD.append((got["C32o"] - got["C24o"]) / got["A16"]); DS.append((got["C32s"] - got["C24s"]) / got["A16"])
    G24.append((got["C24s"] - got["C24o"]) / got["A16"])
set_arm("A16")
m = lambda xs: 1e3 * float(np.median(xs))
say(f"{NL} layers, {rounds} rounds + a warm check per round, SB {SBN} (contexts {P} -> {POS['A16']}); ms per check "
    f"(python build inside):")
for a in names:
    say(f"  {a:5s} {m(T[a]):7.2f} ms (build {m(Bt[a]):6.2f} ms, {np.median(Bt[a]) / np.median(T[a]):.2f} of it) = "
        f"{m(T[a]) / m(T['A16']):.3f} x A16")
q = lambda xs: f"{np.median(xs):.3f} [{np.percentile(xs, 25):.3f}-{np.percentile(xs, 75):.3f}]"
say("  per-round medians [IQR] x A16: " + "; ".join(f"{a} {q(RAT[a])}" for a in names if a != "A16"))
say(f"  paired / A16: D = C32o - C24o {q(DD)}; split C32s - C24s {q(DS)}; one pass at 24 (C24s - C24o) {q(G24)}")
say(f"  v32one counts: {dict(v32one.CNT)}; matmuls still split (shape: calls): {v32one.MISS}; v32 wrappers: {dict(v32.CNT)}")
d = float(np.median(DD))
r24 = float(np.median(RAT["C24o"]))
say(f"C7d verdict (pre-registered): R24o {r24:.3f}, R32o {np.median(RAT['C32o']):.3f}, D {d:.3f} -> " +
    ("no verdict (bits)" if not nbits_ok else "the desk R(24) stands (D >= 0.08)" if d >= 0.08 else
     "PARTIAL: R(24) = R(32) - D as measured (0.03 <= D < 0.08)" if d >= 0.03 else
     "24 rows cost ~32 rows (D < 0.03): R(24) := R(32)"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
