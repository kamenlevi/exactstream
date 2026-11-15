"""qsa_tree.py (Breakthrough, TREE16 piece B3; spec ~/Leap/data/week98/tree16_spec.md section 3, "Selection"; NOTES.md
here). The QSA indexer's block selection for a tree's T nodes above the dense line. Each node's mask is bitwise equal to
its row's mask when its root-to-node path is checked as a chain.

  sel = tree_select(indexer, qk_rows, L, cache, meta)

  indexer   the attention layer's QSAIndexer (MTPLX qwen4_exp).
  qk_rows   [1, T, (H + 1) * D], the nodes' indexer projection rows in node order: the fused QKV's 4th output, or
            indexer.index_qk_proj(x). The q part is pre-norm / pre-RoPE; the k part is the raw keys.
  L         the committed length (the tree's root sits at position L).
  cache     the layer's QSACache. Read only: raw keys before L and pooled blocks before L // ratio. The stock indexer
            call may already have written the nodes at L.. in node order; B3 never reads those.
  meta      B1's tree_meta (depth, anc).

Returns None when the chain's selection is the dense causal mask: (L + 16) // ratio <= block_topk, MTPLX's own shortcut,
with ref2 padding every check to 16 rows. Otherwise it returns a bool mx.array [T, L + 16]: node n's mask in its chain
column order (cache columns 0..L-1, then its path L..L+depth[n]; later columns False). Route (b) applies it where the
chain's SDPA applies its mask, as where(mask, s, finfo.min) on the node's chain-order score row. route_b_mask() pads it
to route (b)'s [16, Kr] layout.

Why each node's mask equals its chain row's (MTPLX's eager selector, QSAIndexer._select_eager, the exact-45 lane):
  - Queries: the chain prepares its 16 rows at L + row. A node's row is prepared at L + depth[n] by the same kernel,
    with the row's position read from a per-row array. The kernel is MTPLX's prepare kernel with that one expression
    changed, built at runtime from the installed module. When MTPLX's kernel is ineligible, the eager oracle runs with
    a positions array.
  - Path blocks: the chain pools blocks L // 4 .. (L + 16) // 4 - 1 from its own rows. On node n's path, block b is
    closed by the ancestor at depth 4b + 3 - L. Each (block, closer) instance pools the same 4 raw keys (cache rows
    before L, the path's node rows after) with the same per-block arithmetic (MTPLX's pool-keys kernel, its block start
    read per block).
  - Scores: the chain's matmul shape, [1, 16, H, D] @ [1, 1, D, nb]. Each node's row of dots is gathered into one
    [1, 16, H, nb] array, then the chain's relu-sum, divisor, validity, tie term, argpartition and mask ops run on it."""
import inspect

import numpy as np
import mlx.core as mx

R = 16                      # ref2 pads every check to 16 rows, so the chain's indexer always sees 16
DEBUG = {"calls": 0, "dense": 0, "runs": 0, "gathered": 0, "kernel_q": 0, "eager_q": 0, "kernel_k": 0, "eager_k": 0}
_KB = {}


def _patched_builder(fn_name, old, new, name_old, name_new):
    """MTPLX's kernel builder (mtplx.kernels.qsa_indexer_prepare), re-made from its installed source with ONE
    expression changed and the kernel renamed (MLX keys compiled kernels by name). Each edit must land exactly once."""
    if fn_name not in _KB:
        import mtplx.kernels.qsa_indexer_prepare as QP
        src = inspect.getsource(getattr(QP, fn_name))
        for a in (old, name_old):
            if src.count(a) != 1:
                raise RuntimeError(f"qsa_tree: {fn_name}'s source changed ({a!r} found {src.count(a)}x); re-check B3")
        ns = dict(vars(QP))
        exec(compile(src.replace(old, new).replace(name_old, name_new), f"<qsa_tree:{fn_name}>", "exec"), ns)
        _KB[fn_name] = ns[fn_name]
    return _KB[fn_name]


def prep_queries_kernel(indexer, raw_q, positions):
    """MTPLX's qsa_indexer_prepare_queries_metal with row s at positions[s] (int32 [S]) instead of pos_start + s"""
    import mtplx.kernels.qsa_indexer_prepare as QP
    build = _patched_builder("_prepare_queries_kernel", "pos_start[0] + int(row)", "pos_start[row]",
                             'f"mtplx_qsa_prepare_q_h', 'f"b3tree_qsa_prepare_q_h')
    w, inv = indexer.q_layernorm.weight, indexer._inv_freq
    head_dim, rotary_dim = QP._validate_common(raw_q, w, inv, expected_ndim=4)
    rows, heads = int(raw_q.shape[1]), int(raw_q.shape[2])
    assert positions.shape == (rows,) and positions.dtype == mx.int32
    kern = build(heads, head_dim, rotary_dim, float(indexer.rms_norm_eps),
                 QP._attention_scaling(indexer._rope_attention_scaling), raw_q.dtype)
    return kern(inputs=[raw_q, w, inv, positions], template=[("T", raw_q.dtype)],
                grid=(rows * heads * QP._SIMD, 1, 1), threadgroup=(QP._SIMD, 1, 1),
                output_shapes=[tuple(raw_q.shape)], output_dtypes=[raw_q.dtype])[0]


def pool_keys_kernel(indexer, fresh, block_ids):
    """MTPLX's qsa_indexer_pool_keys_metal with block i at block_ids[i] (int32 [N]) instead of block_start + i"""
    import mtplx.kernels.qsa_indexer_prepare as QP
    build = _patched_builder("_pool_keys_kernel", "block_start[0] + int(block)", "block_start[block]",
                             'f"mtplx_qsa_pool_k_d', 'f"b3tree_qsa_pool_k_d')
    w, inv = indexer.k_layernorm.weight, indexer._inv_freq
    head_dim, rotary_dim = QP._validate_common(fresh, w, inv, expected_ndim=3)
    ratio = int(indexer.ratio)
    blocks = int(fresh.shape[1]) // ratio
    assert blocks * ratio == int(fresh.shape[1]) and block_ids.shape == (blocks,) and block_ids.dtype == mx.int32
    kern = build(head_dim, rotary_dim, ratio, float(indexer.rms_norm_eps),
                 QP._attention_scaling(indexer._rope_attention_scaling), fresh.dtype)
    return kern(inputs=[fresh, w, inv, block_ids], template=[("T", fresh.dtype)],
                grid=(blocks * QP._SIMD, 1, 1), threadgroup=(QP._SIMD, 1, 1),
                output_shapes=[(1, blocks, head_dim)], output_dtypes=[fresh.dtype])[0]


def prep_queries_eager(indexer, q, positions):
    """QSAIndexer._prepare_queries_plain with a positions array (its tables are elementwise in the position)"""
    import mtplx.models.qwen4_exp as Q
    q = indexer.q_layernorm(q)
    if Q.qwen4_opdiet_enabled("rope"):
        cos, sin = Q._rope_cos_sin_half(positions, indexer._inv_freq, indexer._rope_attention_scaling)
        return Q._apply_partial_rope_half(q, cos, sin)
    cos, sin = Q._rope_cos_sin(positions, indexer._inv_freq, indexer._rope_attention_scaling)
    return Q._apply_partial_rope(q, cos, sin)


def pool_keys_eager(indexer, fresh, block_ids):
    """QSAIndexer._pool_keys_eager with per-block starts (an oracle; the engine runs the kernel: idx_prep on)"""
    import mtplx.models.qwen4_exp as Q
    ratio, D = int(indexer.ratio), int(indexer.head_dim)
    fr = fresh.reshape(1, int(fresh.shape[1]) // ratio, ratio, D)
    pooled = mx.mean(fr.astype(mx.float32), axis=2).astype(fr.dtype)
    pooled = indexer.k_layernorm(pooled)
    cos, sin = Q._rope_cos_sin(block_ids * ratio, indexer._inv_freq, indexer._rope_attention_scaling)
    return Q._apply_partial_rope(pooled[:, :, None, :], cos, sin)[:, :, 0, :]


def tree_select(indexer, qk_rows, L, cache, meta):
    DEBUG["calls"] += 1
    ratio, topk = int(indexer.ratio), int(indexer.block_topk)
    H, D = int(indexer.n_heads), int(indexer.head_dim)
    depth = [int(x) for x in meta["depth"]]
    T = len(depth)
    L = int(L)
    assert 1 <= T <= R and tuple(qk_rows.shape[:2]) == (1, T), (qk_rows.shape, T)
    assert not getattr(cache, "fixed_capacity", False), "qsa_tree: the eager verify path only"
    total = L + R
    nb = total // ratio
    if nb <= topk:                                     # the chain's own shortcut: dense == sparse
        DEBUG["dense"] += 1
        return None
    assert not indexer._uses_vision_positions(), "qsa_tree: text positions only"
    b0 = L // ratio                                    # blocks < b0 are committed; b0..nb-1 hold path positions
    assert int(cache.pooled_len) >= b0, ("qsa_tree: the committed pooled blocks are missing; run the stock indexer "
                                         "call first, as tree mode does", int(cache.pooled_len), b0)
    anc = [[int(x) for x in meta["anc"][n][:depth[n] + 1]] for n in range(T)]
    rows16 = list(range(T)) + [T - 1] * (R - T)        # padding rows: copies of the last node (discarded)
    positions = mx.array(np.array([L + depth[n] for n in rows16], np.int32))

    # queries, prepared at each node's own position
    q, k = mx.split(qk_rows, [H * D], axis=-1)
    q = q.reshape(1, T, H, D)
    k = k.reshape(1, T, D)
    if T < R:
        q = mx.concatenate([q, mx.repeat(q[:, T - 1:T], R - T, axis=1)], axis=1)
    if indexer._prepare_kernel_supported(q, indexer.q_layernorm.weight, expected_ndim=4):
        qp = prep_queries_kernel(indexer, q, positions)
        DEBUG["kernel_q"] += 1
    else:
        qp = prep_queries_eager(indexer, q, positions)
        DEBUG["eager_q"] += 1

    # path-block instances: node n's block b (complete once ratio*b + ratio - 1 <= L + depth[n]) is closed by its
    # ancestor at depth ratio*b + ratio - 1 - L
    W = nb - b0
    inst, need = {}, np.full((R, W), -1, np.int64)
    for r, n in enumerate(rows16):
        for i in range(W):
            dc = ratio * (b0 + i) + ratio - 1 - L
            if dc <= depth[n]:
                need[r, i] = inst.setdefault((b0 + i, anc[n][dc]), len(inst))
    NI = len(inst)
    insts = sorted(inst.items(), key=lambda t: t[1])
    c0 = ratio * b0                                    # the straddling block's committed rows c0..L-1
    src = k if c0 == L else mx.concatenate([cache.raw_keys[:, c0:L, :], k], axis=1)
    if NI:
        ridx = [p - c0 if p < L else (L - c0) + anc[c][p - L]
                for (b, c), _ in insts for p in range(ratio * b, ratio * b + ratio)]
        fresh = mx.take(src, mx.array(np.array(ridx, np.int32)), axis=1)      # [1, NI * ratio, D]
        bids = mx.array(np.array([b for (b, _c), _ in insts], np.int32))
        if indexer._prepare_kernel_supported(fresh, indexer.k_layernorm.weight, expected_ndim=3):
            pk = pool_keys_kernel(indexer, fresh, bids)
            DEBUG["kernel_k"] += 1
        else:
            pk = pool_keys_eager(indexer, fresh, bids)
            DEBUG["eager_k"] += 1
        inst_t = mx.swapaxes(pk.astype(mx.float32), 1, 2)[:, None]            # [1, 1, D, NI], as write_pooled
    # the scores: the chain's own matmul shape per run (committed columns + up to W instances in the path columns)
    com = cache.pooled_f32_view(b0)                    # [1, 1, D, b0]: the chain's fp32 mirror of the committed blocks
    qf = qp.astype(mx.float32)
    runs = max(1, -(-NI // W))
    dots = []
    for r in range(runs):
        cols = inst_t[..., r * W:(r + 1) * W] if NI else mx.zeros((1, 1, D, 0), mx.float32)
        if cols.shape[-1] < W:
            cols = mx.concatenate([cols, mx.zeros((1, 1, D, W - cols.shape[-1]), mx.float32)], axis=-1)
        dots.append(mx.matmul(qf, mx.concatenate([com, cols], axis=-1)))    # [1, 16, H, nb]
    DEBUG["runs"] += runs
    gidx = np.tile(np.arange(nb, dtype=np.int32), (R, 1))
    for r in range(R):
        for i in range(W):
            j = int(need[r, i])
            if j >= 0:
                gidx[r, b0 + i] = (j // W) * nb + b0 + (j % W)
    if runs == 1 and all(int(need[r, i]) in (-1, i) for r in range(R) for i in range(W)):
        raw = dots[0]
    else:
        allc = dots[0] if runs == 1 else mx.concatenate(dots, axis=-1)
        raw = mx.take_along_axis(allc, mx.broadcast_to(mx.array(gidx)[None, :, None, :], (1, R, H, nb)), axis=-1)
        DEBUG["gathered"] += 1
    # QSAIndexer._select_eager's selection and mask, op for op, on each node's chain-order row
    scores = mx.maximum(raw, 0.0).sum(axis=2) / indexer._score_divisor()
    scores = scores[0]                                 # [16, nb]
    nb_q = (positions + 1) // ratio                    # complete blocks visible per row
    blk = mx.arange(nb, dtype=mx.int32)
    valid = blk[None, :] < nb_q[:, None]
    neg = mx.array(-mx.inf, dtype=mx.float32)
    k_eff = min(topk, nb)
    masked = mx.where(valid, scores, neg)
    masked = masked - blk.astype(mx.float32)[None, :] * 1e-12
    top_idx = mx.argpartition(masked, kth=nb - k_eff, axis=-1)[:, nb - k_eff:]
    selected = mx.zeros((R, nb), dtype=mx.bool_)
    selected = mx.put_along_axis(selected, top_idx.astype(mx.int64), mx.array(True), axis=-1)
    selected = selected & valid
    tok = mx.repeat(selected, ratio, axis=1)
    if nb * ratio < total:
        tok = mx.concatenate([tok, mx.zeros((R, total - nb * ratio), dtype=mx.bool_)], axis=1)
    tpos = mx.arange(total, dtype=mx.int32)
    tail = tpos[None, :] >= (nb_q[:, None] * ratio)
    causal = tpos[None, :] <= positions[:, None]
    return ((tok | tail) & causal)[:T]


def route_b_mask(sel, Kr):
    """B3's [T, L + 16] rows in route (b)'s msk layout: 16 rows (padding = copies of the last node), Kr columns (False
    past L + 16, where the chain's verify SDPA extends its mask with False)"""
    T, W = int(sel.shape[0]), int(sel.shape[1])
    m = sel if T == R else mx.concatenate([sel, mx.repeat(sel[T - 1:T], R - T, axis=0)], axis=0)
    if Kr > W:
        m = mx.concatenate([m, mx.zeros((R, Kr - W), dtype=mx.bool_)], axis=1)
    return m[:, :Kr]
