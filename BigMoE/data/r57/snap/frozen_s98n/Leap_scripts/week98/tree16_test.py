"""tree16_test.py (Leap, Sun 15 Nov 2026 15:1x; TREE16 L3's own test, before BigMoE's M3 harness): tree_verify vs chain
verifies on a 4-layer slice WITHOUT the PLE layer (layers 4-7, G G G A; the model's PLE stage switched off), so branching
trees run end to end before B2 lands. For every tree: tree_verify from a fresh 1,024-token prefill; then, for every
root-to-leaf path, a chain verify of the path's tokens from its own fresh prefill. Node n at depth d: tree logits[n] and
hidden_w[n] == the path's chain row d, bitwise. B1's tree_meta builds the metadata (Breakthrough, imported read-only).
usage: [BigMoE venv] tree16_test.py   (MEM ~8 GB; share lock)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98"))
sys.path.append(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Breakthrough_tree16"))   # B1, read-only
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
        inner._ple_stage_idx = None                       # no PLE layer in this slice
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
assert all(getattr(l, "ple", None) is None for l in inner.layers), "the slice must not hold the PLE layer"
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body", "hc_b"):
    ref2.CFG[k_] = True
import tree16
from tree_meta import tree_meta


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


def gdn_trees(rng):
    out = {"chain": [-1] + list(range(15)), "bushy3": dfs_relabel([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
           "bushy-deep": dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 1, 11, 2, 13, 14]),
           "deep-narrow": dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13])}
    for i in range(3):
        out[f"random{i}"] = dfs_relabel([-1] + [int(rng.integers(max(0, c - 4), c)) for c in range(1, 16)])
    return out


tree16.install(ref2)
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P = 1024
say(f"TREE16-L3 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice layers 4-7 ({KINDS}), "
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


def chain(tokens):
    cache = prefill()
    with verify_scope():
        lg, hw = model(mx.array([tokens]), cache=cache, return_hidden=True)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg, hw)
    return lg, hw[0]


rng = np.random.default_rng(7)
allok = True
cases = dict(gdn_trees(rng))
cases["small7"] = dfs_relabel([-1, 0, 0, 1, 2, 3, 3])
for name, par in cases.items():
    T = len(par)
    toks = [int(t) for t in rng.integers(1000, 50000, T)]
    meta = tree_meta(toks, par, P, ids[P - 9:P])
    cache = prefill()
    with verify_scope():
        lt, ht = tree16.tree_verify(model, cache, toks, meta)
    lt = lt.astype(mx.float32)
    mx.eval(lt, ht)
    kids = [[] for _ in range(T)]
    for c in range(1, T):
        kids[par[c]].append(c)
    leaves = [u for u in range(T) if not kids[u]]
    bad = 0
    for leaf in leaves:
        path, u = [], leaf
        while u >= 0:
            path.append(u); u = par[u]
        path = path[::-1]
        lc, hc = chain([toks[n] for n in path])
        for d, n in enumerate(path):
            ok = bool(mx.all(lt[n] == lc[d]).item()) and bool(mx.all(ht[n] == hc[d]).item())
            bad += int(not ok)
    allok &= bad == 0
    say(f"  {name:12s} T {T:2d}, {len(leaves)} paths: {'ALL BITWISE' if bad == 0 else f'{bad} node-rows DIFFER'} "
        f"(tree16 counts {dict(tree16.CNT)})")
say(f"TREE16-L3 verdict: {'ALL BITWISE (logits + hidden_w of every node == its path chain row)' if allok else 'NOT BITWISE'}"
    f" ({time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB)")
