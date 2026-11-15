"""test_ple_tree.py (Breakthrough, TREE16 B2 bit test; spec section 3 / 5): on the 125B's layers 0-3 (GGGA; layer 1 is the PLE
layer), resident, with ref2 + v32 installed as for a 16-row check (exact-45's forms), after a 1,024-token prefill:
for every tree, each root-to-leaf path is verified as a chain (rolled back after), recording the PLE layer's input and
output rows; then ple_tree runs once over the tree's nodes (inputs = each node's chain input row, which Leap's L1-L3 make
bitwise) and every node's output must equal its chain row bit for bit. Also: the verify stash ple_tree leaves, no cache
slot written, no stale stage. Trees: a pure chain, bushy, deep narrow, two with EOS tokens, random preorder, T = 5 / 1.
Usage: [venv] python -I -B test_ple_tree.py   (MEM ~8 GB; share lock)"""
import glob
import os
import sys
import time

for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("V32_MOE1", "1"), ("HF_HUB_OFFLINE", "1"), ("MTPLX_NGRAM_PREWARM", "off")):
    os.environ.setdefault(k_, v_)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
t00 = time.time()
NL = 4
orig = mu.load_model


def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a:
        a[0] = True
    else:
        kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        model.language_model.model.layers = model.language_model.model.layers[:NL]
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(os.path.expanduser("~/BigMoE/models/bare"), mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind

model = rt.model
inner = model.language_model.model
import ref2

ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):
    ref2.CFG[k_] = True
import v32

v32.install(ref2)
from tree_meta import tree_meta
from ple_tree import ple_tree, DEBUG

PLE_L = next(i for i, l in enumerate(inner.layers) if "ple" in l)
ple_layer = inner.layers[PLE_L].ple
EOS = int(ple_layer.ple_embedding.eos_id)
print(f"slice {''.join('G' if l.is_linear else 'A' for l in inner.layers)}; PLE layer {PLE_L}; eos {EOS}; "
      f"sidecar {'yes' if ple_layer.ple_embedding.ngram_embedding._sidecar is not None else 'no'}", flush=True)

REC = {"on": False, "calls": []}
_ple0 = Q.PLELayer.__call__


def _ple_rec(self, hidden, input_ids, cache):
    out = _ple0(self, hidden, input_ids, cache)
    if REC["on"]:
        REC["calls"].append((hidden, input_ids, out))
    return out


Q.PLELayer.__call__ = _ple_rec


def verify_scope():
    import contextlib

    @contextlib.contextmanager
    def cm():
        with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
                model.verify_capture_scope():
            yield
    return cm()


def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]


files = sorted(glob.glob(os.path.expanduser("~/Leap/scripts/*.py")))
ids = [int(t) for t in rt.tokenizer.encode("".join(open(f).read() for f in files))]
P = 1024
cache = model.make_cache()
mx.eval(model(mx.array([ids[:P]]), cache=cache), states(cache))
rng = np.random.default_rng(20261115)
pool = np.array(ids[P:P + 4000])


def random_preorder(T):
    parents, stack = [-1], [0]
    for n in range(1, T):
        k = int(rng.integers(0, len(stack)))
        stack = stack[:len(stack) - k] or [0]
        parents.append(stack[-1])
        stack.append(n)
    return parents


def tokens_for(parents):
    """main-chain nodes take the text's own continuation; side nodes a token from later in the text"""
    T = len(parents)
    depth = [0] * T
    for n in range(1, T):
        depth[n] = depth[parents[n]] + 1
    main = [0]
    for n in range(1, T):
        if parents[n] == main[-1] and n == main[-1] + 1:
            main.append(n)
    return [ids[P + depth[n]] if n in main else int(rng.choice(pool)) for n in range(T)]


TREES = [("chain16", [-1] + list(range(15))),
         ("bushy", [-1, 0, 1, 2, 2, 1, 5, 0, 7, 7, 0, 10, 11, 0, 13, 13]),
         ("deep_narrow", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2])]
TREES += [(f"random{i}", random_preorder(16)) for i in range(4)]
TREES += [("small5", [-1, 0, 1, 0, 3]), ("single", [-1])]
toks = {nm: tokens_for(par) for nm, par in TREES}
TREES += [("bushy_eos", TREES[1][1]), ("deep_eos", TREES[2][1])]
toks["bushy_eos"] = list(toks["bushy"]); toks["bushy_eos"][2] = EOS; toks["bushy_eos"][9] = EOS; toks["bushy_eos"][0] = EOS
toks["deep_eos"] = list(toks["deep_narrow"]); toks["deep_eos"][4] = EOS; toks["deep_eos"][5] = EOS

tot_eq = tot = 0
fails = []
slot_ok = True
for nm, parents in TREES:
    tokens = toks[nm]
    T = len(tokens)
    meta = tree_meta(tokens, parents, P, ids[P - 9:P])
    H, Y = {}, {}
    for path in meta["paths"]:
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        REC["calls"].clear()
        REC["on"] = True
        with verify_scope():
            lg = model(mx.array([[tokens[n] for n in path]]), cache=cache)
        REC["on"] = False
        assert len(REC["calls"]) == 1, len(REC["calls"])
        h, _iid, out = REC["calls"][0]
        mx.eval(lg, h, out)
        CS.rollback_after_verify(cache, snap, len(path))
        mx.eval(states(cache))
        for j, n in enumerate(path):
            if n in Y:                                   # a shared prefix: the chain rows must agree across paths
                assert bool(mx.array_equal(Y[n], out[0, j])) and bool(mx.array_equal(H[n], h[0, j])), (nm, n)
            H[n], Y[n] = h[0, j], out[0, j]
    st0 = [mx.array(cache[PLE_L][k]) if cache[PLE_L][k] is not None else None for k in (2, 3)]
    hid = mx.stack([H[n] for n in range(T)])[None]
    iid = mx.array([tokens])
    with verify_scope():
        yt = ple_tree(ple_layer, hid, iid, meta, cache[PLE_L])
    mx.eval(yt)
    stash = getattr(cache[PLE_L], "_mtplx_verify_ple", None)
    slot_ok &= stash is not None and stash[0] is hid and stash[1] is iid
    for k, a in zip((2, 3), st0):                         # ple_tree writes no PLE slot
        b = cache[PLE_L][k]
        slot_ok &= (a is None and b is None) or (a is not None and b is not None and bool(mx.array_equal(a, b)))
    eq = [bool(mx.array_equal(yt[0, n], Y[n])) for n in range(T)]
    tot_eq += sum(eq)
    tot += T
    bad = [n for n in range(T) if not eq[n]]
    md = max((float(mx.max(mx.abs(yt[0, n].astype(mx.float32) - Y[n].astype(mx.float32)))) for n in bad), default=0.0)
    print(f"{nm:12s} T {T:2d}, {len(meta['paths'])} paths: nodes bitwise {sum(eq)} / {T}" +
          (f"; mismatches {bad} (max |diff| {md:.3g})" if bad else ""), flush=True)
    if bad:
        fails.append(nm)
print(f"B2 bit test: {tot_eq} / {tot} nodes bitwise over {len(TREES)} trees; stash + no slot writes {'OK' if slot_ok else 'FAIL'}; "
      f"n-gram rows via {'the staged path' if DEBUG['staged'] else 'the graph path'}; "
      f"{'PASS' if tot_eq == tot and slot_ok else 'FAIL: ' + ', '.join(fails)}; {time.time() - t00:.0f} s, peak "
      f"{mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
