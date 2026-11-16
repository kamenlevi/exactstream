"""C5 TIME32 (WEEK98 pillar 2; Leap, Sun 15 Nov 2026 01:3x): what does the exact 17-32-row verify path cost against
16-row checks, on a resident 12-layer slice of the 125B (layers 0-11: 9 GDN incl. the PLE layer + 3 QSA attention)?
Arms (one process, interleaved rounds, paired; the same ref2 / DENSE-N config for all, so only the check form differs):
  A16   one 16-row verify + commit (full accept) + the acceptance read (the engine's per-check sync);
  B2x16 two 16-row verifies of rows 0-15 then 16-31, each with its commit and read (today's way to check 32 rows);
  C32   one 32-row verify through v32 (V32_MOE1: the experts as ONE call; v32mr2: GDN projections at full width with
        mr2; hc32: the HC read in one pass; DENSE-N copies) + commit + read.
Real code tokens after a 1,024-token prefill. v2 (01:4x): each arm on its OWN cache, every cache advancing 32 tokens per
round (A16: its timed 16-row check, then an untimed one), so all arms check at the same context lengths. v1 (01:39) ran
A16 on a cache advancing 16 per round and B / C on one advancing 64, so B / C checked at ~2.2K context against A's ~1.3K:
B2x16 / A16 came out 2.40 and R32 1.914, biased up; its fair read was 2 x C32 / B2x16 = 1.63.
Pre-registered (before any run): R32 = C32 / A16 per round, median.
  - R32 <= 1.4: the exact 32-row check pays for a ~98 % drafter at today's kernels (cheapcheck.md's 1.35-1.4 line).
  - 1.4 < R32 <= 1.7: it pays only with the kernel work (experts / dense toward the floor).
  - R32 > 1.7: look for a missed path before anything else.
  Prediction: R32 1.45-1.6 (the union read once: ~-25 % expert bytes; dense at R ~1.2-1.3; cores and router 2 x 16),
  B2x16 / A16 ~1.95-2.0.
Resident experts (ref2's experts_split), not the streamed tiers: no SSD misses, and the 45 GB gate is not in this path.
usage: [venv] c5_time32.py [smoke] (MEM ~22 GB at 12 layers; quiet ticket)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("REF2_DENSE_NATIVE", "1"), ("V32_MOE1", "1"), ("V32_MR2", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts/week98"))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
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
say(f"C5 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB active, "
    f"loaded in {time.time() - t00:.0f} s; ref2 on: {sorted(k for k, v in ref2.CFG.items() if v is True)}")
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts/*.py")))
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
    """one n-row check as the engine runs it: snapshot, verify, the acceptance read (sync), commit (full accept)"""
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    g = mx.argmax(lg[0], axis=-1).tolist()
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    return g


cA, cB, cC = prefill(), prefill(), prefill()
posA = posB = posC = P
v32.ST["on"] = False
check(cA, posA, 16); check(cA, posA + 16, 16); posA += 32   # DENSE-N copies and kernels built before timing
check(cB, posB, 16); check(cB, posB + 16, 16); posB += 32
v32.ST["on"] = True
check(cC, posC, 32); posC += 32


def A16():
    global posA
    v32.ST["on"] = False
    mx.synchronize(); t0 = time.perf_counter()
    check(cA, posA, 16)
    mx.eval(states(cA)); mx.synchronize()
    dt = time.perf_counter() - t0
    check(cA, posA + 16, 16)                              # untimed: cA keeps pace with cB / cC (32 per round)
    mx.eval(states(cA)); mx.synchronize()
    posA += 32
    return dt


def B2x16():
    global posB
    v32.ST["on"] = False
    mx.synchronize(); t0 = time.perf_counter()
    check(cB, posB, 16)
    check(cB, posB + 16, 16)
    mx.eval(states(cB)); mx.synchronize()
    posB += 32
    return time.perf_counter() - t0


def C32():
    global posC
    v32.ST["on"] = True
    mx.synchronize(); t0 = time.perf_counter()
    check(cC, posC, 32)
    mx.eval(states(cC)); mx.synchronize()
    posC += 32
    return time.perf_counter() - t0


for i in range(3):
    A16(); B2x16(); C32()
rounds = 4 if SMOKE else 16
T = {"A16": [], "B2x16": [], "C32": []}
R32, RB = [], []
for r in range(rounds):
    order = [("A16", A16), ("B2x16", B2x16), ("C32", C32)]
    order = order[r % 3:] + order[:r % 3]
    got = {}
    for name, fn in order:
        got[name] = fn()
        T[name].append(got[name])
    R32.append(got["C32"] / got["A16"]); RB.append(got["B2x16"] / got["A16"])
m = lambda xs: 1e3 * float(np.median(xs))
say(f"{NL} layers, {rounds} rounds (contexts {P} -> {posA} / {posB} / {posC}): A16 {m(T['A16']):.2f} ms | B2x16 {m(T['B2x16']):.2f} ms | C32 {m(T['C32']):.2f} ms")
say(f"  R32 = C32 / A16 median {np.median(R32):.3f} (rounds {', '.join(f'{x:.2f}' for x in R32)}); B2x16 / A16 {np.median(RB):.3f}; "
    f"C32 / B2x16 {np.median(T['C32']) / np.median(T['B2x16']):.3f}")
say(f"  v32 counts {dict(v32.CNT)}; v32mr2 {dict(v32mr2.CNT)}; hc32 {dict(hc32.CNT)}")
r = float(np.median(R32))
say(f"C5 verdict (pre-registered): R32 {r:.3f} -> " + ("PAYS at today's kernels (<= 1.4)" if r <= 1.4 else
    "pays only with the kernel work (1.4-1.7)" if r <= 1.7 else "look for a missed path (> 1.7)"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
