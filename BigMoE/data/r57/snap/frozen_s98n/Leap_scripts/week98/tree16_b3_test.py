"""tree16_b3_test.py (Leap, Sun 15 Nov 2026 17:4x; TREE16 B3 wired into tree16 / tree_att): Breakthrough's
tree16_qsa_e2e.py (281470d) with its two COPIES removed. Leap's live tree16._att_tree and tree_att.tree_sdpa run, with B3
plugged in through the new slot tree16.QSA_TREE[0] = qsa_tree (and B2 through PLE_TREE). Layers 0-3 (the PLE layer and
the attention layer), contexts P = 1024 (below ~2K: B3 returns None, the history-only indexer call must keep the bits)
and 3000 / 3002 (B3's selection live). Node n at depth d: tree logits[n] and hidden_w[n] == the path's chain row d,
bitwise, for every root-to-leaf path from its own fresh prefill.
usage: [BigMoE venv] python -B tree16_b3_test.py   (MEM ~9.4 GB; share lock; Breakthrough's modules read-only)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("MTPLX_QSA_PREFILL", "0")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/week98"))
sys.path.append(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Breakthrough_tree16"))      # B1 + B2 + B3 (read-only)
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
        inner.layers = inner.layers[0:4]                  # layers 0-3: the PLE layer (1) and the attention layer (3)
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
assert any(getattr(l, "ple", None) is not None for l in inner.layers), "the slice must hold the PLE layer"
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body", "hc_b"):
    ref2.CFG[k_] = True
import tree16
import tree_att as TA
from tree_meta import tree_meta
import qsa_tree as B3


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
tree16.QSA_TREE[0] = B3                                  # B3 plugged in through Leap's slot (no copies)
from ple_tree import ple_tree, DEBUG as B2DEBUG
tree16.PLE_TREE[0] = ple_tree                             # B2 plugged in (Leap's hook)
files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
say(f"TREE16-B3-WIRED {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice layers 0-3 ({KINDS}), "
    f"{mx.get_active_memory() / 1e9:.1f} GB active")


@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield


def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]


def prefill(P):
    cache = model.make_cache()
    lg = model(mx.array([ids[0:P]]), cache=cache)
    mx.eval(lg, states(cache))
    return cache


def chain(tokens, P):
    cache = prefill(P)
    with verify_scope():
        lg, hw = model(mx.array([tokens]), cache=cache, return_hidden=True)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg, hw)
    return lg, hw[0]


rng = np.random.default_rng(7)
allok = True
cases = dict(gdn_trees(rng))
cases["small7"] = dfs_relabel([-1, 0, 0, 1, 2, 3, 3])
cases["wide8"] = [-1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0]
EOS = int(next(l for l in inner.layers if getattr(l, "ple", None) is not None).ple.ple_embedding.eos_id)
EOSAT = {"bushy3-eos": (cases["bushy3"], (0, 2, 5))}
for k_, (par_, _at) in EOSAT.items():
    cases[k_] = par_
nodes_tot = 0
for P in (1024, 3000, 3002):
    for name, par in cases.items():
        T = len(par)
        toks = [int(t) for t in rng.integers(1000, 50000, T)]
        for i_ in EOSAT.get(name, (None, ()))[1]:
            toks[i_] = EOS
        meta = tree_meta(toks, par, P, ids[P - 9:P])
        cache = prefill(P)
        live0 = tree16.CNT["b3_live"]
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
            lc, hc = chain([toks[n] for n in path], P)
            for d, n in enumerate(path):
                ok = bool(mx.all(lt[n] == lc[d]).item()) and bool(mx.all(ht[n] == hc[d]).item())
                bad += int(not ok)
        nodes_tot += T
        allok &= bad == 0
        say(f"  P {P} {name:12s} T {T:2d}, {len(leaves)} paths: {'ALL BITWISE' if bad == 0 else f'{bad} node-rows DIFFER'}"
            f" (B3 live selections {tree16.CNT['b3_live'] - live0}; tree16 counts {dict(tree16.CNT)})")
say(f"TREE16-B3-WIRED verdict: {'ALL BITWISE (logits + hidden_w of every node == its path chain row)' if allok else 'NOT BITWISE'}"
    f" over {nodes_tot} nodes at 1K + 3K contexts; B3 selections live {tree16.CNT['b3_live']}, dense {tree16.CNT['b3_dense']}, calls {B3.DEBUG}; "
    f"B2 staged calls {B2DEBUG['staged']}, graph calls {B2DEBUG['graph']} ({time.time() - t00:.0f} s; peak "
    f"{mx.get_peak_memory() / 1e9:.1f} GB)")
