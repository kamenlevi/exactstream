"""C8 REFC (CHECK55 track B; Leap, Sun 15 Nov 2026 17:4x): does CB1's standalone contract gain (58c2439: -5.98 ms per
16-row engine check, -6.41 per 32-row, over 7 dense kinds + HC down) carry into a real check? The transfer rule (NewIdeas,
after ISO2's DENSE-N): a standalone gain counts only after a paired A/B in the real check. C7c's harness and slice
(12-layer resident, SB every 4 layers to 32 rows, REF2's exact-45 forms + gdn_body, own caches, rotated rounds), with
refc.py (the contract's forms: fixed K-chunks per shape, S4 at 2,048-key partitions) toggled per arm:
  A16    REF2-T, one 16-row check (+ an untimed 16-row check to keep pace);   A16c   the same under the contract;
  C32o   REF2-T step 1 complete (v32one per-shape forms, seams to 32);        C32oc  the same under the contract.
BITS first (fresh caches; the contract is a NEW reference, so its checks are compared with its own checks):
  (b1) one 32-row check == two 16-row checks (logits, every row), contract and REF2-T (control, C3);
  (b2) info: 16 one-row checks == one 16-row check's rows (the AR path pads to 16), contract and REF2-T.
PRE-REGISTERED (before any run; Leap 17:4x):
  expected on the slice from CB1's per-call savings x the slice's calls (9 GDN + 3 QSA layers, the head, 25 HC reads):
    -1.61 ms per 16-row check, -1.57 per 32-row (an upper bound: CB1's joint forms took the native layout on 3 shapes,
    refc runs MLX's layout); transfer T16 = -median(A16c - A16) / 1.61, T32 = -median(C32oc - C32o) / 1.57.
  lines: (b1) not bitwise for the contract -> a bug, no verdict;
         T16 >= 0.5 -> PASS: the contract arms join Monday night's E2 (the engine decides);
         0.25 <= T16 < 0.5 -> BETWEEN: E2 arm only if the slot has room; T16 < 0.25 -> KILL: no engine arm, profile.
  prediction: T16 0.6-0.8, T32 0.6-0.9; (b2) bitwise for both (rows pad to 16).
usage: [BigMoE venv] c8_refc.py [smoke]   (MEM ~23 GB at 12 layers; quiet ticket via qscreen.sh)"""
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98"))
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
import refc
refc.install(ref2)
say(f"C8 REFC {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; SB {SBN}; dense_native {ref2.CFG['dense_native']}")
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/*.py")))
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


ARMS = {"A16": (16, False, 16), "A16c": (16, True, 16), "C32o": (32, False, 32), "C32oc": (32, True, 32)}


def set_arm(a):
    n, con, seam = ARMS[a]
    v32one.ON[0], v32one.W1[0], v32one.QMV_VPT[0], ref2.CFG["seam_hi"] = True, True, 4, seam
    refc.ON[0] = con


def prefill_off():
    """prefills run with the contract off (its S4 and matmul forms are 1-32-row forms)"""
    on, refc.ON[0] = refc.ON[0], False
    try:
        return prefill()
    finally:
        refc.ON[0] = on


def logits_check(cache, pos, n):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)[0]
    mx.eval(lg)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n, verified_tokens=n)
    assert ok
    mx.eval(states(cache))
    return lg


def rows_equal(x, y):
    return int(mx.sum(mx.all(x == y, axis=-1)).item())


BITS = {}
for con in (True, False):
    tag = "contract" if con else "REF2-T"
    set_arm("C32oc" if con else "C32o")
    l32 = logits_check(prefill_off(), P, 32)
    set_arm("A16c" if con else "A16")
    c2 = prefill_off()
    l16 = mx.concatenate([logits_check(c2, P, 16), logits_check(c2, P + 16, 16)])
    BITS[(tag, "b1")] = rows_equal(l32, l16)
    c3 = prefill_off()
    l1 = mx.concatenate([logits_check(c3, P + j, 1) for j in range(16)])
    BITS[(tag, "b2")] = rows_equal(l1, l16[:16])
    del c2, c3
    mx.clear_cache()
say("bits: " + "; ".join(f"{t} {k} {v} / {32 if k == 'b1' else 16} rows" for (t, k), v in BITS.items()))
refc.CNT.update({k: 0 for k in refc.CNT})
C = {a: prefill_off() for a in ARMS}
POS = {a: P for a in ARMS}


def run_arm(a):
    set_arm(a)
    c, p = C[a], POS[a]
    mx.synchronize()
    if ARMS[a][0] == 16:
        b, w = check(c, p, 16)
        check(c, p + 16, 16)                              # untimed: keep pace (32 tokens per round)
    else:
        b, w = check(c, p, 32)
    POS[a] += 32
    return b, w


for i in range(2):
    for a in ARMS:
        run_arm(a)
refc.CNT.update({k: 0 for k in refc.CNT})
rounds = 3 if SMOKE else 14
T = {a: [] for a in ARMS}
Bt = {a: [] for a in ARMS}
D16, D32 = [], []
names = list(ARMS)
for r in range(rounds):
    order = names[r % len(names):] + names[:r % len(names)]
    got = {}
    for a in order:
        b, w = run_arm(a)
        T[a].append(w); Bt[a].append(b); got[a] = w
    D16.append(got["A16c"] - got["A16"]); D32.append(got["C32oc"] - got["C32o"])
refc.ON[0] = False
m = lambda xs: 1e3 * float(np.median(xs))
say(f"{NL} layers, {rounds} rounds, SB {SBN} (contexts {P} -> {POS['A16']}); ms per check (python build inside):")
for a in names:
    say(f"  {a:6s} {m(T[a]):7.2f} ms (build {m(Bt[a]):6.2f} ms) = {m(T[a]) / m(T['A16']):.3f} x A16")
q = lambda xs: f"{m(xs):+.3f} ms [{1e3 * np.percentile(xs, 25):+.3f}, {1e3 * np.percentile(xs, 75):+.3f}]"
say(f"  paired: A16c - A16 {q(D16)}; C32oc - C32o {q(D32)}; refc counts over the timed checks {dict(refc.CNT)}")
EXP16, EXP32 = 1.61 * NL / 12, 1.57 * NL / 12
t16, t32 = -m(D16) / EXP16, -m(D32) / EXP32
b1 = BITS[("contract", "b1")] == 32
say(f"C8 REFC verdict (pre-registered): bits (b1) contract {'BITWISE' if b1 else 'NOT BITWISE (a bug: no verdict)'}; "
    f"T16 {t16:.2f} (of -{EXP16:.2f} ms), T32 {t32:.2f} (of -{EXP32:.2f} ms) -> "
    + ("no verdict" if not b1 else "PASS: the contract arms join E2" if t16 >= 0.5 else
       "BETWEEN: E2 arm if the slot has room" if t16 >= 0.25 else "KILL: no engine arm, profile first"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
