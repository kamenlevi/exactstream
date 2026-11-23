"""TREE16 M2 (BigMoE; spec ~/Leap/data/week98/tree16_spec.md, ce2047a): the walk and tree_commit. Sun 15 Nov 2026 15:0x.

walk(tokens, parents, am): from node 0, move to the child whose token == argmax(logits[current]) (at most one child can
match: siblings are distinct tokens). Returns the accepted path [0, n1, ..., n(k-1)] and the bonus am[path[-1]].

tree_commit(model, cache, snapshot_states, path, T, L) leaves the cache exactly as a chain check of the path's k tokens
with a full accept would. It is MTPLX's commit_verified_window (qwen4_exp, the chain commit) with every "first keep rows"
replaced by "the path's rows in path order":
- QSA attention (trimmable): tree_verify left K / V (and the indexer's raw keys) for the T nodes at slots L..L+T-1 in node
  order. The path's rows are gathered into slots L..L+k-1, then the tail is trimmed to L + k. Pooled blocks from L // ratio
  on are dropped (tree_verify built them from node-order raw keys); the next forward rebuilds them from the path-ordered
  raw keys.
- GDN: restore the snapshot state, then gated_delta_update over the stash rows (q, k, v, a, b) gathered in path order;
  the conv state = the last (kernel - 1) rows of [snapshot conv rows; qkv rows in path order].
- PLE layer: restore its snapshot slots, stage the path's token ids (the n-gram history = the last 2 of [prev; path]),
  and run the layer on its captured inputs h_in gathered in path order (the dilated conv state follows, 9 rows back).
- Fast path: a main-chain prefix (path == [0, 1, ..., k-1]) with k < T is today's chain commit unchanged
  (commit_verified_window(keep_tokens=k, verified_tokens=T)). At k == T (a fully accepted chain-shaped tree) MTPLX's
  commit would skip the replay (a chain verify leaves the post-window state; tree_verify's does not), so it replays.
The t30b history: mtp_pairs(hidden_w, tokens, path, bonus) gives MTPHead.append's (hidden rows, next ids): the accepted
pairs (hidden_w[path[i]], tokens[path[i + 1]]), the last one with the bonus.
usage: python tree16_commit.py --selftest   (no model: the walk; the GDN and QSA parts against MTPLX's own
       commit_verified_window on the same rows re-ordered so the path comes first)"""
import numpy as np


def walk(tokens, parents, am):
    """tokens / parents [T] (spec order), am [T] = argmax(logits) per node -> (path, bonus)"""
    tokens, parents, am = (np.asarray(x) for x in (tokens, parents, am))
    kids = {}
    for n in range(1, len(tokens)):
        kids.setdefault(int(parents[n]), []).append(n)
    path, cur = [0], 0
    while True:
        nxt = [c for c in kids.get(cur, []) if int(tokens[c]) == int(am[cur])]
        assert len(nxt) <= 1, ("two children with the same token", cur, nxt)
        if not nxt:
            break
        cur = nxt[0]
        path.append(cur)
    return path, int(am[path[-1]])


def is_main_prefix(path):
    return all(int(n) == i for i, n in enumerate(path))


def _qsa_commit(entry, path, T, L):
    """the QSA layer's K / V and raw indexer keys into path order at L..L+k-1, then trim to L + k"""
    import mlx.core as mx
    k = len(path)
    off = int(entry.offset)
    assert off == L + T, (off, L, T)
    src = mx.array([L + int(n) for n in path], dtype=mx.int32)
    kv = entry.kv
    kv.keys[..., L:L + k, :] = mx.take(kv.keys, src, axis=2)
    kv.values[..., L:L + k, :] = mx.take(kv.values, src, axis=2)
    if entry.raw_keys is not None:
        entry.raw_keys[:, L:L + k, :] = mx.take(entry.raw_keys, src, axis=1)
    entry.trim(T - k)
    entry.pooled_len = min(entry.pooled_len, L // entry.ratio)    # rebuilt from the path-ordered raw keys


def _gdn_commit(entry, layer, pre, rows, path):
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_update
    qkv, q, k, v, a, b = rows
    idx = mx.array([int(n) for n in path], dtype=mx.int32)
    g = [mx.take(x, idx, axis=1) for x in (qkv, q, k, v, a, b)]
    gdn = layer.linear_attn
    conv_pre = pre[0]
    if conv_pre is None:
        conv_pre = mx.zeros((qkv.shape[0], gdn.conv_kernel_size - 1, qkv.shape[2]), dtype=qkv.dtype)
    _, new_state = gated_delta_update(g[1], g[2], g[3], g[4], g[5], gdn.A_log, gdn.dt_bias, pre[1], None,
                                      use_kernel=not gdn.training)
    conv_input = mx.concatenate([conv_pre, g[0]], axis=1)
    entry[0] = mx.contiguous(conv_input[:, -(gdn.conv_kernel_size - 1):, :])
    entry[1] = new_state
    entry._mtplx_verify_rows = None


def _ple_commit(entry, layer, pre, cap, path):
    import mlx.core as mx
    h_in, ids = cap
    idx = mx.array([int(n) for n in path], dtype=mx.int32)
    for j in range(len(pre)):
        entry[j] = pre[j]
    ple = layer.ple
    kept_ids = mx.take(ids, idx, axis=1)
    ple.ple_embedding.stage(kept_ids, entry, ple.NGRAM_IDX)
    layer(mx.take(h_in, idx, axis=1), input_ids=kept_ids, ssm_mask=None, cache=entry)
    entry._mtplx_verify_ple = None


def tree_commit(model, cache, snapshot_states, path, T, L, text=None, ref2=None):
    """leave `cache` as a chain check of the path's tokens with a full accept would (see the module doc); returns
    "chain" (the fast path) or "tree". `text` = the decoder (model.language_model.model by default). With `ref2` (as the
    loop installs it), the tree path runs as ref2's commit wrapper runs the chain commit (R-40 3t): in the verify phase
    (decode_verify + target_verify) when CFG commit_phase is on, with _IN_COMMIT set (the PLE seams step aside), and the
    commit-sub group cleared after."""
    path = [int(n) for n in path]
    k = len(path)
    assert 1 <= k <= T and path[0] == 0
    if is_main_prefix(path) and k < T:                   # today's chain commit, unchanged. NOT at k == T: MTPLX's commit
                                                         # skips the replay on a full accept (a chain verify leaves the
                                                         # post-window state), but tree_verify's recurrent state is not
                                                         # meaningful, so a fully accepted chain-shaped tree replays below
                                                         # (M3 T2, 14:5x: chain tree k 16 off by 5.8 through the fast path)
        txt = text if text is not None else model.language_model.model
        ok = txt.commit_verified_window(cache, snapshot_states, keep_tokens=k, verified_tokens=T)
        assert ok, "commit_verified_window refused the main-chain prefix"
        return "chain"
    txt = text if text is not None else model.language_model.model
    plan = []                                            # validate every entry before mutating any (as MTPLX does)
    for i, (layer, entry) in enumerate(zip(txt.layers, cache)):
        assert entry is not None, ("entry missing", i)
        if callable(getattr(entry, "is_trimmable", None)) and entry.is_trimmable():
            plan.append(("qsa", i, None)); continue
        pre = snapshot_states[i]
        assert pre is not None, ("snapshot missing", i)
        if getattr(layer, "ple", None) is not None:
            cap = getattr(entry, "_mtplx_verify_ple", None)
            assert cap is not None and cap[0].shape[1] == T, ("ple rows", i)
            plan.append(("ple", i, cap)); continue
        rows = getattr(entry, "_mtplx_verify_rows", None)
        assert rows is not None and rows[0].shape[1] == T, ("gdn rows", i)
        plan.append(("gdn", i, rows))
    import contextlib

    def run_plan():
        for kind, i, payload in plan:
            if kind == "qsa":
                _qsa_commit(cache[i], path, T, L)
            elif kind == "gdn":
                _gdn_commit(cache[i], txt.layers[i], snapshot_states[i], payload, path)
            else:
                _ple_commit(cache[i], txt.layers[i], snapshot_states[i], payload, path)
    if ref2 is None:
        run_plan()
        return "tree"
    assert not any(isinstance(p, (list, tuple)) and len(p) == 3 for p in (snapshot_states or ())), "ref2 LAZY COMMIT form"
    phase = contextlib.nullcontext()
    if ref2._ON[0] and ref2.CFG.get("commit_phase"):
        from mtplx.attention_context import attention_phase, model_forward_kind
        phase = contextlib.ExitStack()
        phase.enter_context(attention_phase("decode_verify")); phase.enter_context(model_forward_kind("target_verify"))
    ref2._IN_COMMIT[0] = True
    try:
        with phase:
            run_plan()
    finally:
        ref2._IN_COMMIT[0] = False
        if getattr(ref2, "_CSUB", None):
            ref2._CSUB.clear()
    return "tree"


def mtp_pairs(hidden_w, tokens, path, bonus):
    """MTPHead.append's arguments for the accepted path: hidden rows [1, k, Wd] (node order -> path order) and the
    next ids (tokens of path[1:], then the bonus)"""
    import mlx.core as mx
    idx = mx.array([int(n) for n in path], dtype=mx.int32)
    hw = hidden_w if hidden_w.ndim == 3 else hidden_w[None]
    return mx.take(hw, idx, axis=1), [int(tokens[n]) for n in path[1:]] + [int(bonus)]


def selftest():
    import types
    import mlx.core as mx
    from mtplx.models import qwen4_exp as Q
    rng = np.random.default_rng(5)
    # the walk
    tokens = np.array([9, 4, 6, 2, 7, 3], np.int32)      # 0 -> {1, 3}; 1 -> {2}; 3 -> {4, 5}
    parents = np.array([-1, 0, 1, 0, 3, 3], np.int32)
    am = np.array([2, 6, 1, 3, 8, 5], np.int32)          # 0 -> 2 (node 3), 3 -> 3 (node 5), 5 -> bonus 5
    path, bonus = walk(tokens, parents, am)
    assert path == [0, 3, 5] and bonus == 5, (path, bonus)
    assert walk(tokens, parents, np.array([4, 6, 0, 0, 0, 0]))[0] == [0, 1, 2]
    assert is_main_prefix([0, 1, 2]) and not is_main_prefix([0, 3, 5])

    # GDN + QSA against MTPLX's commit_verified_window on the same rows re-ordered so the path comes first
    T, L, ratio = 6, 10, 4
    Hk, Dk, Hv, Dv, C, Di = 2, 64, 2, 64, 12, 4         # the gated-delta kernel's shapes

    def gdn_layer():
        lin = types.SimpleNamespace(conv_kernel_size=4, training=False,
                                    A_log=mx.array(rng.normal(size=(Hv,)).astype(np.float32)),
                                    dt_bias=mx.array(rng.normal(size=(Hv,)).astype(np.float32)))
        return types.SimpleNamespace(linear_attn=lin, ple=None)
    layers = [gdn_layer(), types.SimpleNamespace(ple=None), gdn_layer()]
    rows = {i: (mx.array(rng.normal(size=(1, T, C)).astype(np.float32)),
                mx.array(rng.normal(size=(1, T, Hk, Dk)).astype(np.float32)) * 0.3,
                mx.array(rng.normal(size=(1, T, Hk, Dk)).astype(np.float32)) * 0.3,
                mx.array(rng.normal(size=(1, T, Hv, Dv)).astype(np.float32)),
                mx.array(rng.normal(size=(1, T, Hv)).astype(np.float32)),
                mx.array(rng.normal(size=(1, T, Hv)).astype(np.float32))) for i in (0, 2)}
    pre = {i: [mx.array(rng.normal(size=(1, 3, C)).astype(np.float32)),
               mx.array(rng.normal(size=(1, Hv, Dv, Dk)).astype(np.float32)) * 0.1] for i in (0, 2)}
    kvrows = [(mx.array(rng.normal(size=(1, 1, Di)).astype(np.float32)), mx.array(rng.normal(size=(1, 2, 1, 8)).astype(np.float32)),
               mx.array(rng.normal(size=(1, 2, 1, 8)).astype(np.float32))) for _ in range(L + T)]

    def build(order):
        """a cache whose tree rows sit in `order` (node ids per slot L..L+T-1)"""
        cache = []
        for i in range(3):
            if i == 1:
                qc = Q.QSACache(ratio)
                for j in list(range(L)) + [L + n for n in order]:
                    raw, k_, v_ = kvrows[j]
                    qc.write_raw(raw); qc.kv.update_and_fetch(k_, v_)
                qc.pooled_len = (L + T) // ratio         # as if tree_verify had pooled every block
                cache.append(qc)
            else:
                e = Q.ArraysCache(size=2)
                e[0], e[1] = mx.array(0.0), mx.array(0.0)    # the tree's post-verify state: not meaningful
                idx = mx.array(order, dtype=mx.int32)
                e._mtplx_verify_rows = tuple(mx.take(x, idx, axis=1) for x in rows[i])
                cache.append(e)
        return cache
    txt = types.SimpleNamespace(layers=layers, _refuse_commit=lambda i, why: (_ for _ in ()).throw(AssertionError((i, why))))
    snap = [pre[0], None, pre[2]]
    for path in ([0, 3, 5], [0, 1], [0, 3], [0, 1, 2], [0]):
        tree = build(list(range(T)))                     # node order
        kind = tree_commit(None, tree, snap, path, T, L, text=txt) if not is_main_prefix(path) else None
        rest = [n for n in range(T) if n not in path]
        chain = build(path + rest)                       # the same rows, the path first: a chain check
        ok = Q.Qwen4ExpTextModel.commit_verified_window(txt, chain, snap, keep_tokens=len(path), verified_tokens=T)
        assert ok
        if kind is None:                                 # a main-chain prefix: the fast path IS the chain commit
            continue
        k = len(path)
        for i in (0, 2):
            for j in (0, 1):
                a_, b_ = np.array(tree[i][j]), np.array(chain[i][j])
                assert (a_ == b_).all(), ("gdn", path, i, j, np.abs(a_ - b_).max())
        qt, qc = tree[1], chain[1]
        n_ = L + k
        assert qt.offset == qc.offset == n_
        for x, y in ((qt.kv.keys, qc.kv.keys), (qt.kv.values, qc.kv.values)):
            assert (np.array(x[..., :n_, :]) == np.array(y[..., :n_, :])).all(), ("qsa kv", path)
        assert (np.array(qt.raw_keys[:, :n_, :]) == np.array(qc.raw_keys[:, :n_, :])).all(), ("qsa raw", path)
        assert qt.pooled_len == min(qc.pooled_len, L // ratio) <= qc.pooled_len
    print("SELFTEST OK: walk; tree_commit's GDN states / conv rows and QSA K / V / raw keys == MTPLX's "
          "commit_verified_window on the path-first rows (bitwise), pooled blocks from L // ratio dropped")


if __name__ == "__main__":
    selftest()
