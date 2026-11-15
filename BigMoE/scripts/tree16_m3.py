"""TREE16 M3 (BigMoE; spec ~/Leap/data/week98/tree16_spec.md, ce2047a): the bit harness on a PLE-free slice. Sun 15 Nov
2026 15:2x, pulled forward (the orchestrator 14:5x). Slice = layers 4-7 (G G G A) with the PLE stage off, loaded as Leap's
L3 test does (scripts/week98/tree16_test.py, 1569cdc); ref2 + the exact CFG forms; tree16.install(ref2); B1's tree_meta.
- T1: for every tree, tree_verify from a fresh 1,024-token prefill vs a chain verify of every root-to-leaf path from its own
  fresh prefill. Every node's logits and hidden_w == its path's chain row, bitwise.
- T2: for several accepted paths per tree (main-chain prefixes incl. the root alone, side branches, partial side paths):
  - tree route: prefill, snapshot (CS.snapshot_untrimmable_cache_lazy), tree_verify in the verify scopes, tree_commit
    (scripts/tree16_commit.py, with ref2), then a 4-row probe check;
  - chain route: prefill, snapshot, a chain verify of the path's k tokens, the loop's commit with a full accept
    (commit_verified_window keep = verified = k), then the same probe;
  - the probe's logits and hidden_w bitwise, plus the QSA offsets.
Trees: chain, bushy3, bushy-deep, deep-narrow, small7 (Leap's shapes) and 6 random trees (T 16 and smaller).
usage: [BigMoE venv] [TREE16_SLICE=0-3] python scripts/tree16_m3.py -> data/deepdraft/tree16/m3_slice[_0-3].json (~8 GB)"""
import contextlib, glob, json, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
sys.path.append(os.path.expanduser("~/Breakthrough/scripts/m1/tree16"))   # B1, read-only
t00 = time.time()
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/tree16")
SL = os.environ.get("TREE16_SLICE", "4-7")              # 4-7: G G G A, no PLE (Leap's L3 slice); 0-3: the PLE layer at index 1
SA, SB = (int(x) for x in SL.split("-")); SB += 1


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
        inner.layers = inner.layers[SA:SB]
        inner._ple_stage_idx = next((i for i, l in enumerate(inner.layers) if getattr(l, "ple", None) is not None), None)
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
from mtplx import cache_state as CS
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
HAS_PLE = inner._ple_stage_idx is not None
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body", "hc_b"):
    ref2.CFG[k_] = True
import tree16
from tree_meta import tree_meta
import tree16_commit as TC
tree16.install(ref2)
if HAS_PLE:                                              # Breakthrough's B2 (fbd3ede): the PLE layer per tree path
    from ple_tree import ple_tree
    tree16.PLE_TREE[0] = ple_tree
files = sorted(glob.glob(os.path.expanduser("~/Leap/scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P = 1024
say(f"TREE16-M3 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice layers {SL} ({KINDS}; PLE "
    f"{'at ' + str(inner._ple_stage_idx) + ' via B2 ple_tree' if HAS_PLE else 'none'}); "
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


def fwd(cache, tokens):
    with verify_scope():
        lg, hw = model(mx.array([tokens]), cache=cache, return_hidden=True)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg, hw)
    return lg, hw[0]


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


def qsa_offsets(cache):
    return [int(c.offset) for c in cache if callable(getattr(c, "is_trimmable", None)) and c.is_trimmable()]


rng = np.random.default_rng(11)
cases = {"chain": [-1] + list(range(15)),
         "bushy3": dfs_relabel([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
         "bushy-deep": dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 1, 11, 2, 13, 14]),
         "deep-narrow": dfs_relabel([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13]),
         "small7": dfs_relabel([-1, 0, 0, 1, 2, 3, 3])}
for i in range(6):
    T_ = 16 if i < 4 else int(rng.integers(5, 16))
    cases[f"random{i}"] = dfs_relabel([-1] + [int(rng.integers(max(0, c - 4), c)) for c in range(1, T_)])
R = {"slice": SL, "kinds": KINDS, "ple": HAS_PLE, "P": P, "trees": {}}
t1_all = t2_all = True
for name, par in cases.items():
    T = len(par)
    toks = [int(t) for t in rng.integers(1000, 50000, T)]
    meta = tree_meta(toks, par, P, ids[P - 9:P])
    kids = [[] for _ in range(T)]
    for c in range(1, T):
        kids[par[c]].append(c)
    leaves = [u for u in range(T) if not kids[u]]

    def root_path(u):
        p = []
        while u >= 0:
            p.append(u); u = par[u]
        return p[::-1]
    # T1
    cache = prefill()
    with verify_scope():
        lt, ht = tree16.tree_verify(model, cache, toks, meta)
    lt = lt.astype(mx.float32)
    mx.eval(lt, ht)
    bad1 = 0
    for leaf in leaves:
        path = root_path(leaf)
        lc, hc = fwd(prefill(), [toks[n] for n in path])
        for d, n in enumerate(path):
            bad1 += int(not (bool(mx.all(lt[n] == lc[d]).item()) and bool(mx.all(ht[n] == hc[d]).item())))
    # T2: the paths to accept
    D = 1
    while D < T and par[D] == D - 1:
        D += 1
    acc = {(0,), tuple(range(D))}
    if D > 2:
        acc.add(tuple(range(D // 2)))
    side = [u for u in leaves if u >= D] + [u for u in range(D, T) if kids[u]]
    for u in side[:4]:
        acc.add(tuple(root_path(u)))
    probe = [int(t) for t in rng.integers(1000, 50000, 4)]
    res2 = []
    for path in sorted(acc, key=lambda p: (len(p), p)):
        path = list(path)
        k = len(path)
        c1 = prefill(); s1 = CS.snapshot_untrimmable_cache_lazy(c1)          # the tree route
        with verify_scope():
            lt_, ht_ = tree16.tree_verify(model, c1, toks, meta)
        mx.eval(lt_, ht_)
        kind = TC.tree_commit(model, c1, s1.states, path, T, P, text=inner, ref2=ref2)
        mx.eval(states(c1))
        lp1, hp1 = fwd(c1, probe)
        c2 = prefill(); s2 = CS.snapshot_untrimmable_cache_lazy(c2)          # the chain route
        fwd(c2, [toks[n] for n in path])
        assert model.commit_verified_window(c2, s2.states, keep_tokens=k, verified_tokens=k)
        mx.eval(states(c2))
        lp2, hp2 = fwd(c2, probe)
        same = bool(mx.all(lp1 == lp2).item()) and bool(mx.all(hp1 == hp2).item()) and qsa_offsets(c1) == qsa_offsets(c2)
        res2.append(dict(path=path, k=k, kind=kind, bitwise=same,
                         max_abs=float(mx.max(mx.abs(lp1 - lp2)).item()) if not same else 0.0))
    ok2 = all(r_["bitwise"] for r_ in res2)
    t1_all &= bad1 == 0; t2_all &= ok2
    R["trees"][name] = dict(T=T, main_chain=D, paths=len(leaves), t1_bad_rows=bad1, t2=res2)
    say(f"  {name:12s} T {T:2d} (main {D}), {len(leaves)} paths: T1 {'ALL BITWISE' if bad1 == 0 else f'{bad1} rows DIFFER'}; "
        f"T2 {sum(r_['bitwise'] for r_ in res2)}/{len(res2)} commits bitwise "
        f"({', '.join(r_['kind'][0] + str(r_['k']) for r_ in res2)})")
R.update(t1=t1_all, t2=t2_all, wall_s=round(time.time() - t00, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 2))
os.makedirs(OUT, exist_ok=True)
json.dump(R, open(os.path.join(OUT, "m3_slice.json" if SL == "4-7" else f"m3_slice_{SL}.json"), "w"), indent=1)
say(f"TREE16-M3 verdict: T1 {'ALL BITWISE' if t1_all else 'NOT BITWISE'}; T2 {'ALL BITWISE' if t2_all else 'NOT BITWISE'} "
    f"(tree_commit + probe == chain commit + probe) ({R['wall_s']:.0f} s; peak {R['peak_gb']:.1f} GB)")
