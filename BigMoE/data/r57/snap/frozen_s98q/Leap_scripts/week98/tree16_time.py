"""TREE16-TIME (Leap, Sun 15 Nov 2026 15:0x): what does a 16-node tree check cost over a 16-row chain check, as built
(L1-L3, all bitwise)? A prior for BigMoE's M4 engine arm. 4-layer slice without the PLE layer (layers 4-7, G G G A), the
engine's SB (a submit every 4 layers), one cache per arm at the same context (each check rolled back after its read).
Arms (rotated rounds, paired):
  C16  a 16-row chain verify (ref2's exact forms) + the acceptance read;
  Tc   tree_verify on a chain-shaped 16-node tree: the tree code's own cost (route (b), the tree GDN body, no forks);
  Tb   tree_verify on a bushy 16-node tree (main chain of 10, 6 side nodes, 3 forks): + the forks;
  Tbb  Tb with route (b)'s heads batched (one scores matmul / gather / softmax / pv_tree per kv head; ALL BITWISE in
       tree_att_test, added Sun 15:0x after a 2-round smoke put Tb at ~+20 % engine-scaled: the 12 head chunks' ops).
Timed: the verify forward + the read (mx.synchronize around), not the rollback.
PRE-REGISTERED (before any run; Leap 15:0x; the verdict arm is Tbb since 15:0x, Tb reported): engine-scaled overhead =
(Tbb - C16) x 12 (48 / 4 layers; the head is in
both arms), against the engine's chain-16 ~62 ms:
  <= 2 % (~1.2 ms) -> as priced (TREE16 price +0.5-1.5 ms); M4 as booked;
  2-5 % -> route (b)'s dispatch count (12 head chunks x ~9 ops) is the first thing to trim before M4;
  > 5 % -> look for a slow piece (per-row tail steps, the gathers) before M4.
usage: [BigMoE venv] tree16_time.py   (MEM ~8 GB; quiet ticket)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/week98"))
sys.path.append(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Breakthrough_tree16"))   # B1, read-only
t00 = time.time()


def say(s):
    print(s, flush=True)


MD = os.path.expanduser("~/BigMoE/models/bare")
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
        inner.layers = inner.layers[4:8]
        inner._ple_stage_idx = None
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
Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (4 if 2 <= int(rows) <= 32 else 0)
model = rt.model
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):
    ref2.CFG[k_] = True
import tree16
from tree_meta import tree_meta
tree16.install(ref2)
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P = 1024
say(f"TREE16-TIME {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice layers 4-7, SB 4, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active")


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


rng = np.random.default_rng(11)
toks = [int(t) for t in rng.integers(1000, 50000, 16)]
chain_par = [-1] + list(range(15))
def dfs_relabel(parents):
    n = len(parents)
    kids = [[] for _ in range(n)]
    for c in range(1, n):
        kids[parents[c]].append(c)
    order, stack = [], [0]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(reversed(kids[u]))
    new = {u: i for i, u in enumerate(order)}
    return [-1] + [new[parents[u]] for u in order[1:]]


bushy_par = dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 1, 12, 3, 14])   # main chain 0-9, side branches at depths 1, 2, 4
meta_c = tree_meta(toks, chain_par, P, ids[P - 9:P])
meta_b = tree_meta(toks, bushy_par, P, ids[P - 9:P])
C = {a: prefill() for a in ("C16", "Tc", "Tb", "Tbb")}


def run(a):
    cache = C[a]
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    mx.synchronize()
    t0 = time.perf_counter()
    import tree_att as TA
    TA.BATCH_HEADS[0] = a == "Tbb"
    with verify_scope():
        if a == "C16":
            lg = model(mx.array([toks]), cache=cache)[0]
        else:
            lg, _ = tree16.tree_verify(model, cache, toks, meta_c if a == "Tc" else meta_b)
    mx.argmax(lg, axis=-1).tolist()
    mx.synchronize()
    dt = time.perf_counter() - t0
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=16)   # roll back to the root
    # (keep 0 is refused; keeping node 0, the root, is the same row in every arm: each check advances the context by 1)
    assert ok
    mx.eval(states(cache))
    return dt


for _ in range(3):
    for a in C:
        run(a)
R = int(os.environ.get("TREE_TIME_ROUNDS", "24"))
T = {a: [] for a in C}
DB, DC, DBB = [], [], []
names = list(C)
for r in range(R):
    order = names[r % len(names):] + names[:r % len(names)]
    got = {a: run(a) for a in order}
    for a in names:
        T[a].append(got[a])
    DB.append(got["Tb"] - got["C16"]); DC.append(got["Tc"] - got["C16"]); DBB.append(got["Tbb"] - got["C16"])
m = lambda xs: 1e3 * float(np.median(xs))
say(f"4 layers, {R} rotated rounds, context {P}: C16 {m(T['C16']):.3f} ms | Tc {m(T['Tc']):.3f} | Tb {m(T['Tb']):.3f} | "
    f"Tbb {m(T['Tbb']):.3f}")
q = lambda xs: f"{m(xs):+.3f} ms [{1e3 * np.percentile(xs, 25):+.3f}, {1e3 * np.percentile(xs, 75):+.3f}]"
say(f"  paired: Tc - C16 {q(DC)} (the tree code); Tb - C16 {q(DB)} (+ forks); Tbb - C16 {q(DBB)} (heads batched)")
eng = 12 * m(DBB)
say(f"TREE16-TIME verdict (pre-registered): engine-scaled overhead {eng:+.2f} ms = {100 * eng / 62:+.1f} % of a ~62 ms chain-16 -> "
    + ("as priced (<= 2 %)" if eng <= 1.24 else "trim route (b)'s dispatches before M4 (2-5 %)" if eng <= 3.1 else
       "look for a slow piece before M4 (> 5 %)"))
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
