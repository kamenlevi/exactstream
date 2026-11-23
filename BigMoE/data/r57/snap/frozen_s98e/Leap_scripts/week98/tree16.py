"""tree16.py (Leap, Sun 15 Nov 2026 15:0x; TREE16 L3, data/week98/tree16_spec.md section 3): the exact 16-node tree check.
    logits, hidden_w = tree_verify(model, cache, tokens, meta)
inside the caller's verify scopes (attention phase decode_verify, target_verify, exact_verify, the capture scope), the
caller's snapshot taken before (M2), with ref2 applied and install(ref2) called once. Each node's logits are bitwise its
row in its root-to-node path checked as a chain; hidden_w is each node's pre-mixer widened stream (the MTP head's input).
Pieces: tree_gdn (L1, ALL BITWISE), tree_att (L2, ALL BITWISE), B2's ple_tree for the PLE layer (Breakthrough; until it
lands, only chain-shaped trees take the PLE layer's own forward), row-local ops as a T-row check (ref2's <= 16-row forms).
Cache afterwards (BigMoE's tree_commit contract, Sun 14:5x): QSA K / V and the indexer's raw keys at L..L+T-1 in node
order (offset L + T); GDN _mtplx_verify_rows = (qkv, q, k, v, a, b) per node in node order, the recurrent / conv state not
meaningful (the commit replays the path from the snapshot); the PLE stash is B2's.
Above ~2K tokens each QSA layer selects per node with B3 when the caller sets QSA_TREE[0] = Breakthrough's qsa_tree
(Sun 17:4x): the stock indexer call runs history-only (raw keys + pooled blocks), qsa_tree.tree_select gives each node's
mask in its chain column order, and tree_att.tree_sdpa takes route (b)'s mask from it. With the slot unset the code is
de7fcc0's: the stock indexer call, and a raise when it selects (past ~2K)."""
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98"))
import tree_att as TA
import tree_gdn as TG

_TREE = [None]
_W = {}
_S = {}
PLE_TREE = [None]          # B2's ple_tree(ple_layer, hidden, input_ids, meta, cache), set by its import
QSA_TREE = [None]          # B3's qsa_tree module (tree_select), set by the caller: per-node QSA selection above ~2K
CNT = dict(att=0, gdn=0, ple=0, ple_chain=0, b3_live=0, b3_dense=0)


def _ctx(tokens, meta):
    T = len(tokens)
    anc_raw = meta["anc"]
    depth = [int(d) for d in meta["depth"]]
    anc = [[int(x) for x in anc_raw[n][:depth[n] + 1]] for n in range(T)]
    win = [[int(x) for x in meta["gdn_win"][n]] for n in range(T)]
    restore, save = [-1] * T, [-1] * T
    for (n, r, s) in meta["fork_ops"]:
        restore[int(n)], save[int(n)] = int(r), int(s)
    chain = all(anc[n] == list(range(n + 1)) for n in range(T))
    return dict(T=T, anc=anc, depth=depth, win=win, restore=restore, save=save, n_slots=max(1, int(meta["n_slots"])),
                chain=chain, meta=meta, L=None)


def tree_verify(model, cache, tokens, meta):
    """tokens: T in 1..16 node tokens (node order, spec section 1); meta: B1's tree_meta(tokens, parents, L, prev)"""
    if not _S:
        raise RuntimeError("tree16: install(ref2) first")
    ctx = _ctx(tokens, meta)
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    skip_stage = PLE_TREE[0] is not None and getattr(inner, "_ple_stage_idx", None) is not None
    saved = getattr(inner, "_ple_stage_idx", None)
    if skip_stage:                       # tree mode with B2: no model-level PLE stage (it would hash the nodes as one
        inner._ple_stage_idx = None      # chain); B2's ple_tree stages each node's own history (MTPLX's _forward skips the
    _TREE[0] = ctx                       # stage when the index is None, so ref2's seam (d) never defers one either)
    try:
        logits, hidden = model(mx.array([list(tokens)]), cache=cache, return_hidden=True)
    finally:
        _TREE[0] = None
        if skip_stage:
            inner._ple_stage_idx = saved
    return logits[0], hidden[0]


def _att_tree(self, x, cache):
    t = _TREE[0]
    if t is None or t["T"] == 1:                                   # a 1-node tree is a 1-row check
        return _W["att"](self, x, cache)
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_math import attention_gate
    B, S, _ = x.shape
    assert B == 1 and S == t["T"], (x.shape, t["T"])
    assert not getattr(cache, "fixed_capacity", False), "tree16: the eager verify path only"
    L = int(cache.offset)
    if t["L"] is None:
        t["L"] = L
    fused = getattr(self, "qkv_fused", None)
    idx_rows = None
    if fused is not None:
        outs = fused(x)
        if len(outs) == 4:
            q, k, v, idx_rows = outs
        else:
            (q, k, v), idx_rows = outs, None
    sel = None
    if self.indexer is not None:
        b3 = QSA_TREE[0]
        if b3 is None:      # as before B3: the stock call (writes the raw keys, pre-RoPE); a selection (> ~2K) raises
            st = self.indexer(x, L, cache, qk_rows=idx_rows) if fused is not None else self.indexer(x, L, cache)
            if st is not None:
                raise NotImplementedError("tree16: the context is past the indexer budget; set tree16.QSA_TREE[0] "
                                          "= B3's qsa_tree for per-node selection")
        else:
            # The stock call in MTPLX's history-only mode: it writes the raw keys (pre-RoPE) and pooled blocks at
            # L..L+T-1 in node order (tree_commit's contract) and selects nothing (it would select for the nodes as one
            # chain). Then B3 selects per node: None below ~2K tokens (every key).
            tok = Q._QSA_HISTORY_ONLY.set(True)
            try:
                st = self.indexer(x, L, cache, qk_rows=idx_rows) if fused is not None else self.indexer(x, L, cache)
            finally:
                Q._QSA_HISTORY_ONLY.reset(tok)
            assert st is None, "tree16: the history-only indexer call returned a selection"
            qk = idx_rows if idx_rows is not None else self.indexer.index_qk_proj(x)
            sel = b3.tree_select(self.indexer, qk, L, cache, t["meta"])
            CNT["b3_dense" if sel is None else "b3_live"] += 1
    if fused is None:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    q, gate = mx.split(q.reshape(B, S, self.n_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, S, -1)
    k = k.reshape(B, S, self.n_kv_heads, -1)
    v = v.reshape(B, S, self.n_kv_heads, -1)
    q = self.q_norm(q)
    k = self.k_norm(k)
    pos = mx.array([L + d for d in t["depth"]], mx.int32)          # each node at its depth (the chain's row position)
    cos, sin = Q._rope_cos_sin_half(pos, self._inv_freq, self._rope_attention_scaling)
    q = Q._apply_partial_rope_half(q, cos, sin)
    k = Q._apply_partial_rope_half(k, cos, sin)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    k = Q._after_index_block_writes(k, cache)
    k, v = cache.kv.update_and_fetch(k, v)                         # the T nodes at L..L+T-1, node order
    out = TA.tree_sdpa(q, k, v, self.scale, L, t["anc"], t["depth"], t["T"], _S["ref2"], sel=sel)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    CNT["att"] += 1
    return Q._linear(self.o_proj, attention_gate(out, gate))


def _gdn_tree(self, inputs, mask=None, cache=None):
    t = _TREE[0]
    if t is None:
        return _W["gdn"](self, inputs, mask, cache)
    import mtplx.models.qwen4_exp as Q
    RF = _S["ref2"]
    B, S = inputs.shape[0], inputs.shape[1]
    assert B == 1 and S == t["T"] and mask is None and cache is not None
    fi = self.__dict__.get("in_proj_fused")
    if not (RF.CFG["gdn_exact"] and isinstance(fi, RF._InProjExact) and RF._lz_form(cache) != 3
            and cache[1] is not None and cache[1].dtype == mx.float32):
        raise RuntimeError("tree16: the GDN tree path needs gdn_exact, R-83's exact in-proj and no lazy commit")
    qkv, z, b, a = fi(inputs)                                      # <= 16 rows: ref2's certified 16-row forms
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    y, q, k, v = TG.gdn_body_tree(qkv, z, a, b, cache[0], self.conv1d.weight, self.A_log, self.dt_bias, cache[1],
                                  self.norm.weight, self.norm.eps, RF._gdnx()._scale(), t["win"], t["restore"],
                                  t["save"], t["n_slots"])
    if Q._VERIFY_CAPTURE.get():
        cache._mtplx_verify_rows = (qkv, q, k, v, a, b)            # per node, node order (tree_commit replays the path)
    cache.advance(S)
    CNT["gdn"] += 1
    return Q._linear(self.out_proj, y.reshape(B, S, -1))


def _ple_tree(self, hidden, input_ids, cache):
    t = _TREE[0]
    if t is None:
        return _W["ple"](self, hidden, input_ids, cache)
    if PLE_TREE[0] is not None:
        CNT["ple"] += 1
        return PLE_TREE[0](self, hidden, input_ids, t["meta"], cache)
    if t["chain"]:                                                 # a chain-shaped tree: node order is the chain
        CNT["ple_chain"] += 1
        return _W["ple"](self, hidden, input_ids, cache)
    raise NotImplementedError("tree16: a branching tree reaches the PLE layer; B2's ple_tree is not installed")


def install(ref2):
    """after ref2.apply() (and v32.install if any): the tree wrappers fall back to the installed hooks outside tree_verify"""
    import mtplx.models.qwen4_exp as Q
    if _S:
        return
    _S["ref2"] = ref2
    _W.update(att=Q.Attention.__call__, gdn=Q.GatedDeltaNet.__call__, ple=Q.PLELayer.__call__)
    Q.Attention.__call__ = _att_tree
    Q.GatedDeltaNet.__call__ = _gdn_tree
    Q.PLELayer.__call__ = _ple_tree
