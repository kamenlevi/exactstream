"""ple_tree.py (Breakthrough, TREE16 piece B2; spec ~/Leap/data/week98/tree16_spec.md; NOTES.md here). The 125B's PLE
layer (MTPLX qwen4_exp PLELayer, the one decoder layer with .ple) for a tree's T nodes in node order, each node's output
bitwise equal to its row when its root-to-node path is checked as a chain.

  out = ple_tree(ple_layer, hidden [1, T, X], input_ids [1, T], meta, cache)

returns the PLE term the decoder layer adds (`hidden + self.ple(hidden, input_ids, cache)`). Leap's L3 plugs it in with
tree16.PLE_TREE[0] = ple_tree and skips the model-level PLE stage in tree mode (B2 owns the staging).
  - n-gram rows: MTPLX's own staged path on each node's one-position history [ngram_prev, token] (B1's meta): _rows_np
    over T histories, then ONE sidecar.gather_np over the T positions (as a chain-16 gathers 16), so the n-gram ids
    (MTPLX's hashing and EOS rule) and the rows (a gather + per-row dequantize) are the chain's by construction. With no
    sidecar: MTPLX's _graph_path on the same histories (cache None: no slot write).
  - row-local math: PLELayer.__call__'s own lines (key / value projections, the norms, the gate, attention_gate).
  - the dilated conv (kernel 4, dilation 3, a 9-row state) on norm_conv(gated): one MLX conv1d over T per-node 10-row
    windows whose rows 0 / 3 / 6 / 9 hold the node's taps 9 / 6 / 3 back (B1's ple_taps: a node row or a conv-state
    slot) and the node itself; the dilated kernel reads only those rows, so each node's output takes the chain's taps
    through the same conv call.
  - the verify stash, under MTPLX's verify capture only: cache._mtplx_verify_ple = (hidden, input_ids), as MTPLX's
    _forward leaves it for a chain (BigMoE's M2 _ple_commit replays the accepted path from it).
  - the cache's PLE slots (conv state 2, n-gram history 3) are never written: M2's commit owns them."""
import math
import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn

DEBUG = {"staged": 0, "graph": 0}


def _ngram_rows(emb_mod, input_ids, meta, state_idx):
    """each node's n-gram embedding rows, [1, T, D], through MTPLX's own staged path (or its graph path)"""
    T = int(input_ids.shape[-1])
    ids_np = np.asarray(input_ids, dtype=np.int64).reshape(T, 1)
    prev_np = np.asarray(meta["ngram_prev"], dtype=np.int64).reshape(T, emb_mod.context_len)
    sidecar = emb_mod.ngram_embedding._sidecar
    if sidecar is not None and os.environ.get("MTPLX_NGRAM_STAGE", "1") != "0":
        rows, _hist = emb_mod._rows_np(ids_np, prev_np)
        emb = sidecar.gather_np(np.asarray(rows).reshape(-1), prepared=None)
        DEBUG["staged"] += 1
        return emb.reshape(1, T, -1)
    DEBUG["graph"] += 1
    emb = emb_mod._graph_path(mx.array(ids_np), None, state_idx, prev=mx.array(prev_np))
    return emb.reshape(1, T, -1)


def _tree_conv(ple_layer, x, cache, meta):
    """silu(dilated conv) per node: x [1, T, C] = norm_conv(gated) in node order; the taps from B1's ple_taps"""
    _, T, C = x.shape
    K, dil = int(ple_layer.conv_kernel_size), int(ple_layer.conv_dilation)
    sl = int(ple_layer.conv_state_len)
    assert (K, dil, sl) == (4, 3, 9), (K, dil, sl)      # B1's ple_taps encode 3 / 6 / 9 back
    if cache is not None and cache[ple_layer.CONV_IDX] is not None:
        state = cache[ple_layer.CONV_IDX]
    else:
        state = mx.zeros((1, sl, C), dtype=x.dtype)
    # src rows: the 9 state slots (slot s at row s), the T nodes (node m at row 9 + m), one zero row
    src = mx.concatenate([state.reshape(sl, C).astype(x.dtype), x.reshape(T, C), mx.zeros((1, C), x.dtype)], axis=0)
    taps = np.asarray(meta["ple_taps"], np.int64)        # [T, 3]: 3 / 6 / 9 back; node m, or -k = state slot 9 - k
    W = dil * (K - 1) + 1                                # 10-row windows; the dilated kernel reads rows 0 / 3 / 6 / 9
    idx = np.full((T, W), sl + T, np.int32)              # the other rows point at the zero row
    for n in range(T):
        for i, col in enumerate((2, 1, 0)):              # rows 0 / 3 / 6 = 9 / 6 / 3 back (ple_taps columns 2 / 1 / 0)
            idx[n, i * dil] = sl + int(taps[n, col])     # node m -> 9 + m; slot code -k -> 9 - k
        idx[n, (K - 1) * dil] = sl + n                   # row 9 = the node itself
    win = mx.take(src, mx.array(idx.reshape(-1)), axis=0).reshape(T, W, C)
    out = mx.conv1d(win, ple_layer.conv_weight, stride=1, padding=0, dilation=dil, groups=C)
    return nn.silu(out[:, -1:, :]).reshape(1, T, C)


def ple_tree(ple_layer, hidden, input_ids, meta, cache):
    import mtplx.models.qwen4_exp as Q
    T = int(input_ids.shape[-1])
    assert hidden.shape[:2] == (1, T), (hidden.shape, T)
    emb_mod = ple_layer.ple_embedding
    if getattr(emb_mod, "_staged", None) is not None:
        raise RuntimeError("ple_tree: a model-level PLE stage is pending; tree mode must skip it (B2 owns the staging)")
    if Q._VERIFY_CAPTURE.get() and cache is not None:
        cache._mtplx_verify_ple = (hidden, input_ids)
    emb = _ngram_rows(emb_mod, input_ids, meta, ple_layer.NGRAM_IDX)
    # PLELayer.__call__'s own lines (MTPLX qwen4_exp), with the n-gram rows above and the conv below
    emb = emb.astype(hidden.dtype)
    key = ple_layer.norm_key(ple_layer.key_proj(emb))
    key = key.reshape(*key.shape[:-1], ple_layer.hc_count, ple_layer.hidden_size)
    value = ple_layer.value_proj(emb)
    query = ple_layer.norm_query(hidden)
    query = query.reshape(*query.shape[:-1], ple_layer.hc_count, ple_layer.hidden_size)
    gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(ple_layer.hidden_size)
    gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
    gated = Q.attention_gate(value[..., None, :], gate)
    gated = gated.reshape(*hidden.shape)
    return gated + _tree_conv(ple_layer, ple_layer.norm_conv(gated), cache, meta)
