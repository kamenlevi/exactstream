"""test_qsa_tree.py (Breakthrough, TREE16 B3 bit test; pre-registered in NOTES.md, cef6107).
Setup: the 125B's layers 0-3 (GGGA; layer 3 is attention with the QSA indexer), resident, ref2 + v32 installed with
exact-45's forms, at contexts L = 3000-3003 / 2036 / 2044.
- Every root-to-leaf path of every tree is checked as a chain (rolled back after), recording the indexer's mask and
  qk rows.
- B3's tree_select then runs over the tree's nodes; every node must get its chain row's mask, bit for bit.
- Again after the stock indexer call has written the nodes in node order (the integration order).
Kernel self-checks: B3's per-row-position kernels vs MTPLX's stock kernels at contiguous positions.
Sensitivity:
- the stock call's node-order mask vs the reference;
- nodes whose mask differs from the main-chain node at the same depth;
- blocks dropped per row.
Usage: [venv] python -I -B test_qsa_tree.py   (MEM ~9 GB; share lock)"""
import glob
import os
import sys
import time

for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("V32_MOE1", "1"), ("HF_HUB_OFFLINE", "1"), ("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_QSA_PREFILL", "0")):
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
import mtplx.kernels.qsa_indexer_prepare as QP
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind

model = rt.model
inner = model.language_model.model
import ref2

ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):
    ref2.CFG[k_] = True
if os.environ.get("B3T_IDX_PREP") == "0":       # extra (not pre-registered): the eager indexer path, as without K73_EXACT
    ref2.CFG["idx_prep"] = False
print(f"ref2 idx_prep {ref2.CFG['idx_prep']}", flush=True)
import v32

v32.install(ref2)
from tree_meta import tree_meta
from qsa_tree import tree_select, prep_queries_kernel, pool_keys_kernel, DEBUG

ATT = next(i for i, l in enumerate(inner.layers) if not l.is_linear)
att = inner.layers[ATT]
attn = getattr(att, "self_attn", None) or getattr(att, "attention", None) or getattr(att, "attn", None)
IDX = attn.indexer
print(f"slice {''.join('G' if l.is_linear else 'A' for l in inner.layers)}; attention layer {ATT}; indexer heads "
      f"{IDX.n_heads} x {IDX.head_dim}, ratio {IDX.ratio}, block_topk {IDX.block_topk}; qkv_fused "
      f"{'yes' if getattr(attn, 'qkv_fused', None) is not None else 'no'}", flush=True)

REC = {"on": False, "calls": []}
_idx0 = Q.QSAIndexer.__call__


def _idx_rec(self, hidden, pos_start, cache, qk_rows=None):
    out = _idx0(self, hidden, pos_start, cache, qk_rows)
    if REC["on"]:
        qk = qk_rows if qk_rows is not None else self.index_qk_proj(hidden)
        REC["calls"].append((int(pos_start), hidden, qk, out))
    return out


Q.QSAIndexer.__call__ = _idx_rec


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
rng = np.random.default_rng(20261115)


def random_preorder(T):
    parents, stack = [-1], [0]
    for n in range(1, T):
        k = int(rng.integers(0, len(stack)))
        stack = stack[:len(stack) - k] or [0]
        parents.append(stack[-1])
        stack.append(n)
    return parents


TREES = [("chain16", [-1] + list(range(15))),
         ("bushy", [-1, 0, 1, 2, 2, 1, 5, 0, 7, 7, 0, 10, 11, 0, 13, 13]),
         ("deep_narrow", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2])]
TREES += [(f"random{i}", random_preorder(16)) for i in range(4)]
TREES += [("small5", [-1, 0, 1, 0, 3]), ("single", [-1]),
          ("wide4", [-1, 0, 1, 2, 3, 0, 5, 6, 0, 8, 9, 0, 11, 12, 13, 14]),     # 4 branches: 4 closers at depth 3
          ("wide8", [-1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0])]       # 8 children: > 4 instances, 2 runs


def tokens_for(parents, P, pool):
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


def chain_check(cache, tokens, L):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    REC["calls"].clear()
    REC["on"] = True
    with verify_scope():
        lg = model(mx.array([tokens]), cache=cache)
    REC["on"] = False
    assert len(REC["calls"]) == 1, len(REC["calls"])
    pos, hid, qk, out = REC["calls"][0]
    assert pos == L, (pos, L)
    mx.eval(lg, hid, qk, *([] if out is None else [out]))
    CS.rollback_after_verify(cache, snap, len(tokens))
    mx.eval(states(cache))
    assert cache[ATT].offset == L and cache[ATT].pooled_len <= L // IDX.ratio, (cache[ATT].offset, cache[ATT].pooled_len)
    return hid, qk, out


CONTEXTS = [3000, 3001, 3002, 3003, 2036, 2044]
tot = dict(nodes=0, eq1=0, eq2=0, ctl_diff=0, ctl_nodes=0, path_dep=0, sparse_rows=0, dropped=0)
fails = []
kernel_ok = None
for L in CONTEXTS:
    cache = model.make_cache()
    mx.eval(model(mx.array([ids[:L]]), cache=cache), states(cache))
    ac = cache[ATT]
    pool = np.array(ids[L + 16:L + 4016])
    for nm, parents in TREES:
        tokens = tokens_for(parents, L, pool)
        T = len(tokens)
        meta = tree_meta(tokens, parents, L, ids[L - 9:L])
        M, QK, HID = {}, {}, {}
        for path in meta["paths"]:
            hid, qk, out = chain_check(cache, [tokens[n] for n in path], L)
            if kernel_ok is None and out is not None:          # one-time: B3's kernels vs MTPLX's stock kernels
                H, D = IDX.n_heads, IDX.head_dim
                q16 = mx.split(qk, [H * D], axis=-1)[0].reshape(1, 16, H, D)
                a = QP.qsa_indexer_prepare_queries_metal(q16, IDX.q_layernorm.weight, IDX._inv_freq, pos_start=L,
                                                         eps=IDX.rms_norm_eps,
                                                         attention_scaling=IDX._rope_attention_scaling)
                b = prep_queries_kernel(IDX, q16, mx.arange(L, L + 16, dtype=mx.int32))
                c = prep_queries_kernel(IDX, q16[:, ::-1], mx.arange(L + 15, L - 1, -1, dtype=mx.int32))[:, ::-1]
                raw = ac.raw_keys[:, (L // 4) * 4 - 64:(L // 4) * 4, :]
                pa = QP.qsa_indexer_pool_keys_metal(raw, IDX.k_layernorm.weight, IDX._inv_freq,
                                                    block_start=L // 4 - 16, compress_ratio=4, eps=IDX.rms_norm_eps,
                                                    attention_scaling=IDX._rope_attention_scaling)
                pb = pool_keys_kernel(IDX, raw, mx.arange(L // 4 - 16, L // 4, dtype=mx.int32))
                perm = np.random.default_rng(1).permutation(16)
                rawp = raw.reshape(1, 16, 4, -1)[:, mx.array(perm)].reshape(1, 64, -1)
                pc = pool_keys_kernel(IDX, rawp, mx.array((L // 4 - 16 + perm).astype(np.int32)))
                pc = pc[:, mx.array(np.argsort(perm))]
                kernel_ok = all(bool(mx.array_equal(x, y)) for x, y in ((a, b), (a, c), (pa, pb), (pa, pc)))
                print(f"kernel self-check (prepare-q in order / reversed, pool-k in order / permuted vs MTPLX's stock "
                      f"kernels): {'BITWISE' if kernel_ok else 'DIFFER'}", flush=True)
            for j, n in enumerate(path):
                if n in QK:                                    # a shared prefix: the chain rows must agree across paths
                    assert bool(mx.array_equal(QK[n], qk[0, j])) and bool(mx.array_equal(HID[n], hid[0, j])), (nm, n)
                    if out is not None:
                        assert bool(mx.array_equal(M[n], out[0, 0, j])), (nm, n, "mask")
                QK[n], HID[n] = qk[0, j], hid[0, j]
                M[n] = None if out is None else out[0, 0, j]
        qk_tree = mx.stack([QK[n] for n in range(T)])[None]
        sel1 = tree_select(IDX, qk_tree, L, ac, meta)
        dense_ref = all(M[n] is None for n in range(T))
        if dense_ref or sel1 is None:
            ok1 = dense_ref and sel1 is None
            eq1 = T if ok1 else 0
        else:
            mx.eval(sel1)
            eq1 = sum(bool(mx.array_equal(sel1[n], M[n])) for n in range(T))
        # the integration order: the stock indexer call writes the nodes in node order first (tree mode's call)
        hid_tree = mx.stack([HID[n] for n in range(T)])[None]
        stock = IDX(hid_tree, L, ac, qk_rows=qk_tree)
        sel2 = tree_select(IDX, qk_tree, L, ac, meta)
        if dense_ref or sel2 is None:
            eq2 = T if (dense_ref and sel2 is None) else 0
        else:
            mx.eval(sel2, *([] if stock is None else [stock]))
            eq2 = sum(bool(mx.array_equal(sel2[n], M[n])) for n in range(T))
        ctl = 0
        if stock is not None and not dense_ref:
            W_ = int(stock.shape[-1])
            ctl = sum(not bool(mx.array_equal(stock[0, 0, n], M[n][:W_])) for n in range(T))
            tot["ctl_nodes"] += T
        ac.trim(0)                                             # pooled_len back to L // 4 (the stock call's blocks)
        dep = int(np.asarray(meta["depth"]).max())
        pathdep = sparse = dropped = 0
        if not dense_ref:
            main = {int(meta["depth"][n]): n for n in range(T) if bool(meta["chain_rows"][n])}
            for n in range(T):
                d = int(meta["depth"][n])
                if not bool(meta["chain_rows"][n]) and d in main and not bool(mx.array_equal(M[n], M[main[d]])):
                    pathdep += 1
                vis = (L + d + 1) // 4
                if vis > IDX.block_topk:
                    sparse += 1
                    dropped += (L + d + 1) - int(mx.sum(M[n]).item())
        tot["nodes"] += T
        tot["eq1"] += eq1
        tot["eq2"] += eq2
        tot["ctl_diff"] += ctl
        tot["path_dep"] += pathdep
        tot["sparse_rows"] += sparse
        tot["dropped"] += dropped
        bad = eq1 != T or eq2 != T
        if bad:
            fails.append(f"{nm}@{L}")
        print(f"L {L} {nm:12s} T {T:2d}, {len(meta['paths'])} paths, depth {dep:2d}: B3 == chain {eq1:2d} / {T} "
              f"(after the stock call {eq2:2d} / {T}); stock node-order mask differs on {ctl:2d}; off-chain nodes "
              f"unlike the main-chain row {pathdep:2d}; sparse rows {sparse:2d}"
              + (f", tokens dropped / sparse row {dropped / sparse:.0f}" if sparse else "")
              + ("  <-- MISMATCH" if bad else ""), flush=True)
    del cache, ac
    mx.clear_cache()
ok = tot["eq1"] == tot["nodes"] and tot["eq2"] == tot["nodes"] and kernel_ok
print(f"B3 bit test: {tot['eq1']} / {tot['nodes']} nodes bitwise (and {tot['eq2']} after the stock call) over "
      f"{len(TREES)} trees x {len(CONTEXTS)} contexts; kernels {'BITWISE' if kernel_ok else 'DIFFER'}; "
      f"sensitivity: stock node-order mask wrong on {tot['ctl_diff']} / {tot['ctl_nodes']} nodes, "
      f"{tot['path_dep']} off-chain nodes unlike their main-chain row, {tot['sparse_rows']} sparse rows; "
      f"B3 calls {DEBUG}; {'PASS' if ok else 'FAIL: ' + ', '.join(fails)}; {time.time() - t00:.0f} s, peak "
      f"{mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
