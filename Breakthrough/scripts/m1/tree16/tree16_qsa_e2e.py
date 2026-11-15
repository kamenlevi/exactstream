"""tree16_qsa_e2e.py (Breakthrough, Sun 15 Nov 2026 17:4x; TREE16 B3 end to end, pre-registered in NOTES.md cef6107).
A copy of tree16_ple_e2e.py (itself a copy of Leap's tree16_test.py with layers 0-3, B2 and EOS trees). Changes:
- the contexts are P = 3000 and 3002 tokens, where the QSA indexer's selection is live ((P + 16) // 4 > 512) and every
  row is sparse;
- B3 plugs in through COPIES of Leap's tree16._att_tree (f44571b) and tree_att.tree_sdpa (8b1ff70, its BATCH_HEADS
  branch). Leap's files are untouched; the copies are installed over Q.Attention.__call__ after tree16.install.
  - _att_tree_b3 runs the stock indexer call under MTPLX's _QSA_HISTORY_ONLY: the cache's raw keys at L..L+T-1 in node
    order (tree_commit's contract), no selection. Then it calls B3's qsa_tree.tree_select on the nodes' qk rows.
  - tree_sdpa_b3 takes route (b)'s mask from B3's rows (qsa_tree.route_b_mask) instead of the all-cache causal mask.
For every tree: tree_verify from a fresh P-token prefill; then, for every root-to-leaf path, a chain verify of the path's
tokens from its own fresh prefill. Node n at depth d: tree logits[n] and hidden_w[n] == the path's chain row d, bitwise.
usage: [BigMoE venv] python -I -B tree16_qsa_e2e.py   (MEM ~9 GB; share lock)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("MTPLX_QSA_PREFILL", "0")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts/m1/tree16"))   # B1 + B2 + B3 (mine)
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


# ---- COPY of Leap's tree_att.tree_sdpa (8b1ff70), BATCH_HEADS branch, with route (b)'s mask from B3 when sel is given
def tree_sdpa_b3(q, k, v, scale, L, anc, depth, T, ref2, sel=None):
    assert TA.BATCH_HEADS[0]
    CFG = ref2.CFG
    HQ, HK, D = q.shape[1], k.shape[1], q.shape[3]
    gq = HQ // HK
    Nc = L + 16
    runN = lambda c, n: c if c >= CFG["kb_from"] else n if c == 0 else max(n, CFG["sdpa_bounds"][c - 1])
    cls = [ref2._klass(L + depth[n] + 16) for n in range(T)]
    out = None
    neg = mx.array(mx.finfo(q.dtype).min, q.dtype)
    for c in sorted(set(cls)):
        Kr = runN(c, Nc)
        Kp = -(-Kr // 32) * 32
        pad = Kr - (L + T)
        kk = k if pad == 0 else mx.concatenate([k, mx.zeros(k.shape[:2] + (pad, D), k.dtype)], axis=2)
        vv = v if pad == 0 else mx.concatenate([v, mx.zeros(v.shape[:2] + (pad, D), v.dtype)], axis=2)
        idx = np.zeros((16, Kr), np.int32)
        msk = np.zeros((16, Kr), bool)
        for n in range(T):
            idx[n, :] = np.arange(Kr)
            msk[n, :L] = True
            for j in range(16):
                cpos = L + j
                if cpos >= Kr:
                    break
                if j <= depth[n]:
                    idx[n, cpos] = L + anc[n][j]
                    msk[n, cpos] = True
                else:
                    idx[n, cpos] = cpos if cpos < L + T else L
        for n in range(T, 16):
            idx[n], msk[n] = idx[T - 1], msk[T - 1]
        idx_m = mx.array(idx)
        msk_m = mx.array(msk) if sel is None else B3.route_b_mask(sel, Kr)        # <- B3
        q16 = q if T == 16 else mx.concatenate([q, mx.repeat(q[:, :, T - 1:T], 16 - T, axis=2)], axis=2)
        qs = q16 * mx.array(scale, q.dtype)
        outs = []
        for j in range(HK):
            kj, vj = kk[:, j:j + 1], vv[:, j:j + 1]
            sc = mx.matmul(qs[:, j * gq:(j + 1) * gq], kj.swapaxes(-1, -2))
            sg = mx.take_along_axis(sc, mx.broadcast_to(idx_m[None, None], sc.shape), axis=-1)
            sg = mx.where(msk_m[None, None], sg, neg)
            pr = mx.softmax(sg, axis=-1, precise=True)
            prp = pr if Kp == Kr else mx.concatenate([pr, mx.zeros(pr.shape[:3] + (Kp - Kr,), pr.dtype)], axis=-1)
            vjp = vj[0, 0] if Kp == Kr else mx.concatenate([vj[0, 0], mx.zeros((Kp - Kr, D), vj.dtype)], axis=0)
            y = TA.pv_tree(prp[0].reshape(gq * 16, Kp), vjp, L, anc, depth, T, k_rule=Kr)
            outs.append(y.reshape(gq, 16, D)[:, :T])
        oc = mx.concatenate(outs, axis=0).reshape(1, HQ, T, D)
        sel_c = mx.array([cls[n] == c for n in range(T)])[None, None, :, None]
        out = oc if out is None else mx.where(sel_c, oc, out)
    return out


SELN = {"live": 0, "dense": 0}


# ---- COPY of Leap's tree16._att_tree (f44571b) with B3's selection (the stock indexer call kept for the cache only)
def _att_tree_b3(self, x, cache):
    t = tree16._TREE[0]
    if t is None or t["T"] == 1:
        return tree16._W["att"](self, x, cache)
    from mtplx.attention_math import attention_gate
    B, S, _ = x.shape
    assert B == 1 and S == t["T"], (x.shape, t["T"])
    assert not getattr(cache, "fixed_capacity", False), "tree16: the eager verify path only"
    L = int(cache.offset)
    if t["L"] is None:
        t["L"] = L
    fused = getattr(self, "qkv_fused", None)
    if fused is not None:
        outs = fused(x)
        if len(outs) == 4:
            q, k, v, idx_rows = outs
        else:
            (q, k, v), idx_rows = outs, None
    else:
        idx_rows = None
    sel = None
    if self.indexer is not None:
        tok = Q._QSA_HISTORY_ONLY.set(True)               # the stock call: raw keys / pooled blocks only, no selection
        try:
            st = self.indexer(x, L, cache, qk_rows=idx_rows) if fused is not None else self.indexer(x, L, cache)
        finally:
            Q._QSA_HISTORY_ONLY.reset(tok)
        assert st is None
        qk = idx_rows if idx_rows is not None else self.indexer.index_qk_proj(x)
        sel = B3.tree_select(self.indexer, qk, L, cache, t["meta"])
        SELN["dense" if sel is None else "live"] += 1
    if fused is None:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    q, gate = mx.split(q.reshape(B, S, self.n_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, S, -1)
    k = k.reshape(B, S, self.n_kv_heads, -1)
    v = v.reshape(B, S, self.n_kv_heads, -1)
    q = self.q_norm(q)
    k = self.k_norm(k)
    pos = mx.array([L + d for d in t["depth"]], mx.int32)
    cos, sin = Q._rope_cos_sin_half(pos, self._inv_freq, self._rope_attention_scaling)
    q = Q._apply_partial_rope_half(q, cos, sin)
    k = Q._apply_partial_rope_half(k, cos, sin)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    k = Q._after_index_block_writes(k, cache)
    k, v = cache.kv.update_and_fetch(k, v)
    out = tree_sdpa_b3(q, k, v, self.scale, L, t["anc"], t["depth"], t["T"], tree16._S["ref2"], sel=sel)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    tree16.CNT["att"] += 1
    return Q._linear(self.o_proj, attention_gate(out, gate))


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
Q.Attention.__call__ = _att_tree_b3                       # B3's copy over Leap's _att_tree (same fallbacks via _W)
from ple_tree import ple_tree, DEBUG as B2DEBUG
tree16.PLE_TREE[0] = ple_tree                             # B2 plugged in (Leap's hook)
files = sorted(glob.glob(os.path.expanduser("~/Leap/scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
say(f"TREE16-L3+B2+B3 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice layers 0-3 ({KINDS}), "
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
for P in (3000, 3002):
    for name, par in cases.items():
        T = len(par)
        toks = [int(t) for t in rng.integers(1000, 50000, T)]
        for i_ in EOSAT.get(name, (None, ()))[1]:
            toks[i_] = EOS
        meta = tree_meta(toks, par, P, ids[P - 9:P])
        cache = prefill(P)
        live0 = SELN["live"]
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
            f" (B3 live selections {SELN['live'] - live0}; tree16 counts {dict(tree16.CNT)})")
say(f"TREE16-L3+B2+B3 verdict: {'ALL BITWISE (logits + hidden_w of every node == its path chain row)' if allok else 'NOT BITWISE'}"
    f" over {nodes_tot} nodes at 3K context; B3 selections live {SELN['live']}, dense {SELN['dense']}, calls {B3.DEBUG}; "
    f"B2 staged calls {B2DEBUG['staged']}, graph calls {B2DEBUG['graph']} ({time.time() - t00:.0f} s; peak "
    f"{mx.get_peak_memory() / 1e9:.1f} GB)")
