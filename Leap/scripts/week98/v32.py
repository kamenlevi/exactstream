"""v32.py (Leap, Sun 15 Nov 2026 01:xx; WEEK98 pillar 2, data/week98/plan_1115.md 'the 17-32-row path', C2 / C3):
REF2's verify at 17-32 rows, exact by construction.

REF2's verify forms are certified per row for 1-16 rows: each row equals the 1-row AR. Above 16 rows every form falls
back to stock MLX, which is not exact. install() runs after ref2.apply() and wraps ref2's hooks, so a 17-32-row call in
the verify phase (MTPLX attention phase 'decode_verify': the verify forward and the commit replay, never the prefill)
runs through the certified forms:
  - row-local ops (quantized matmul incl. the 8-bit router and the LM head, bf16 Linear, the HC read, the MoE's expert
    call, and the whole MoE block, which at >= 32 rows would otherwise take MTPLX's prefill combine with a different
    sigmoid lowering) as rows [0, 16) and [16, n), concatenated;
  - the cache-dependent cores (a GDN layer, an attention layer) as chunk 1 then chunk 2 in sequence, so chunk 2 sees
    chunk 1's state and keys exactly as the 1-row AR would. The GDN capture-commit stash (qkv, q, k, v, a, b) of the
    two chunks is concatenated along the row axis, so the commit replays one window of n rows. (The PLE capture is
    taken at the model level, over all n rows, so it needs nothing.)
  - optional (V32_MR2=1): a 4-bit weight with a native copy (DENSE-N or DENSE-N-RAM) takes D32's 'mr2' kernel at 17-32
    rows: one weight read, bitwise == the 16-row native kernel per row at the same K split (D32, 10-26).
The HC write (elementwise), norms, RoPE and the MoE combine stay stock at n rows; C3's forward test checks them.
This is the correctness scaffold for the single-pass 32-row check: the experts' one 32-row call (the widened gate,
BigMoE's piece 1) and mr2 replace splits one at a time, each tested bitwise against it.
Not supported (install() refuses): ref2's lazy GDN commit, the HC pending carry (D5), NEVER-WAIT."""
import os
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn

R = 16
HI = 32
ST = {"on": False, "mr2": os.environ.get("V32_MR2") == "1", "moe1": os.environ.get("V32_MOE1") == "1"}
CNT = dict(qmm=0, lin=0, gr=0, att=0, gdn=0, moe=0, moeblk=0, moe1=0, tierblk=0, mr2=0)
_W = {}
_RF = [None]
_PH = [None]                                         # mtplx.attention_context.current_attention_phase
_CAP = [None]                                        # MTPLX's _VERIFY_CAPTURE (set only inside a verify forward)


def _rows(x):
    n = 1
    for d in x.shape[:-1]:
        n *= d
    return n


def _live(rows):
    """a 17-32-row call in the verify phase (the verify forward or the commit replay; never the prefill)"""
    return R < rows <= HI and ST["on"] and _RF[0]._ON[0] and _PH[0]() == "decode_verify"


def _halves(x2):
    return x2[:R], x2[R:]


# ---- row-local ops
def _qmm(x, w, *a, **kw):
    rows = _rows(x)
    if not _live(rows):
        return _W["qmm"](x, w, *a, **kw)
    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(rows, K)
    if ST["mr2"]:
        y = _mr2(x2, w, a, kw)
        if y is not None:
            CNT["mr2"] += 1
            return y.reshape(*lead, y.shape[-1])
    u, v = _halves(x2)
    y = mx.concatenate([_W["qmm"](u, w, *a, **kw), _W["qmm"](v, w, *a, **kw)], axis=0)
    CNT["qmm"] += 1
    return y.reshape(*lead, y.shape[-1])


def _lin(self, x):
    rows = _rows(x)
    if not _live(rows):
        return _W["lin"](self, x)
    lead = x.shape[:-1]
    u, v = _halves(x.reshape(rows, x.shape[-1]))
    y = mx.concatenate([_W["lin"](self, u), _W["lin"](self, v)], axis=0)
    CNT["lin"] += 1
    return y.reshape(*lead, y.shape[-1])


def _gr(self, hyper_input, pending=None):
    """the HC read: hyper_input [B, S, X] (rows = B x S); returns (mixed, written, inject) or one array"""
    rows = _rows(hyper_input)
    if not _live(rows):
        return _W["gr"](self, hyper_input, pending)
    lead = hyper_input.shape[:-1]
    h2 = hyper_input.reshape(rows, -1)
    p2 = None if pending is None else tuple(p.reshape(rows, -1) for p in pending)
    outs = []
    for s0, s1 in ((0, R), (R, rows)):
        hh = h2[s0:s1].reshape(1, s1 - s0, -1)
        pp = None if p2 is None else tuple(p[s0:s1].reshape(1, s1 - s0, -1) for p in p2)
        outs.append(_W["gr"](self, hh, pp))
    CNT["gr"] += 1
    if isinstance(outs[0], tuple):
        m_ = mx.concatenate([o[0].reshape(o[0].shape[1], -1) for o in outs], axis=0).reshape(*lead, -1)
        w_ = mx.concatenate([o[1].reshape(o[1].shape[1], -1) for o in outs], axis=0).reshape(hyper_input.shape)
        i_ = mx.concatenate([o[2].reshape(o[2].shape[1], -1) for o in outs], axis=0).reshape(*lead, -1)
        return m_, w_, i_
    return mx.concatenate([o.reshape(o.shape[1], -1) for o in outs], axis=0).reshape(*lead, -1)


def _moe(key):
    def f(self, x, indices, *a, **kw):
        rows = _rows(x)
        if not _live(rows):
            return _W[key](self, x, indices, *a, **kw)
        RF = _RF[0]
        if ST["moe1"] and RF.CFG["experts"] and not a and not kw:
            # V32_MOE1: the experts as ONE 17-32-row call (the union read once), through ref2's own resident expert
            # kernels (each (row, expert) pair in its own tile row, so per row the same arithmetic as a <= 16-row call):
            # experts_nax for the fused gate-up class, experts_split for the unfused SwitchGLU (ref2's _split_call)
            lead, k = x.shape[:-1], indices.shape[-1]
            x2, i2 = x.reshape(rows, x.shape[-1]), indices.reshape(rows, k).astype(mx.int32)
            y = None
            if key == "sw" and hasattr(self, "gu_weight"):
                from qmmg25 import experts_nax
                y = experts_nax(x2, (self.gu_weight, self.gu_scales, self.gu_biases),
                                (self.down_proj.weight, self.down_proj.scales, self.down_proj.biases), i2,
                                RF.CFG["experts_cfg_a"], RF.CFG["experts_cfg_b"])
            elif key == "swb" and hasattr(self, "gate_proj"):
                W = [RF._q4(self.gate_proj), RF._q4(self.up_proj), RF._q4(self.down_proj)]
                if all(w is not None for w in W):
                    y = RF.experts_split(x2, *W, i2)
            if y is not None:
                CNT["moe1"] += 1
                return y.reshape(*lead, k, -1)
        lead, K, k = x.shape[:-1], x.shape[-1], indices.shape[-1]
        x2, i2 = x.reshape(rows, K), indices.reshape(rows, k)
        y1 = _W[key](self, x2[:R].reshape(1, R, K), i2[:R].reshape(1, R, k), *a, **kw)
        y2 = _W[key](self, x2[R:].reshape(1, rows - R, K), i2[R:].reshape(1, rows - R, k), *a, **kw)
        CNT["moe"] += 1
        y = mx.concatenate([y1.reshape(R, k, -1), y2.reshape(rows - R, k, -1)], axis=0)
        return y.reshape(*lead, k, y.shape[-1])
    return f


def _tier_block(self, x, kind):
    """V32_MOE1 on the streamed exact-45 engine (the switch carries a tier _bm): the loop's own verify block at 17-32 rows,
    as r57_screen_rel1.py's K-73 _exact_block ('k73': router_fused, shared_cat, moe_tail) or R-69 _fused_block ('r69')
    does it at 1-16 rows: the top-k in gate-rank order (REF2-T decodes ranks 0-1 and 2-9 by position), the scores and the
    shared expert handed to the tier call (TIER_COMBINE / OOO combine inside it), ONE switch call for all rows, then the
    tail. Each piece is row-generic (router_fused: one threadgroup per row; moe_tail elementwise) or split 2 x 16 by the
    qmm wrapper (the router GEMM, shared_cat). Before Sun 02:4x v32 called MTPLX's _verify_call here: unsorted top-k and
    no scores, so the tiers decoded the wrong ranks at 4 / 3 bits and MLX combined after (BigMoE g32_test 02:34: rounding
    drift on every row, argmax intact)."""
    bm = self.switch_mlp._bm
    EM = sys.modules.get("evalsuite.model")
    if kind == "k73":
        from router_fused import router_fused
        from glue_fused import moe_tail, shared_cat
        inds, scores = router_fused(self.gate(x), self.top_k)
        g, s = shared_cat(self, x)
        shared = s
    else:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    if EM is not None and getattr(EM.STATE, "capture", None) is not None:
        EM.STATE.capture.setdefault(bm["L"], []).append(inds)
    bm["tier"]["scores"], bm["tier"]["shared"] = scores, shared
    try:
        y = self.switch_mlp(x, inds)
    finally:
        bm["tier"].pop("scores", None)
        bm["tier"].pop("shared", None)
    if y.ndim == inds.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2)
    CNT["tierblk"] += 1
    return moe_tail(y, g, s, exp="precise") if kind == "k73" else y + shared


def _tier_kind(self):
    """'k73' / 'r69' when the installed block is the exact-45 loop's verify block on a tier switch, else None"""
    bm = getattr(getattr(self, "switch_mlp", None), "_bm", None)
    if bm is None or "tier" not in bm:
        return None
    name = getattr(_W.get("blk"), "__name__", "")
    kind = "k73" if name == "_exact_block" else "r69" if name == "_fused_block" else None
    if kind is None:
        raise RuntimeError(f"v32: a tier switch under an unknown MoE block ({name!r}); its 17-32-row form is not known")
    EM = sys.modules.get("evalsuite.model")
    import bigmoe.tier as BT
    if EM is None or not EM.STATE.sorted or not (BT._COMBINE and BT._GLU == "3c") or (kind == "k73" and not self.norm_topk_prob):
        raise RuntimeError("v32: the tier block's conditions (EvalSuite sorted, TIER_COMBINE, 3c, norm_topk_prob) do not hold")
    return kind


def _moeblock(self, x):
    """the whole MoE block as two <= 16-row calls. MTPLX's SparseMoeBlock takes its prefill-combine path at >= 32 rows
    (_MOE_PREFILL_COMBINE_MIN_ROWS = 32) even in decode_verify, ahead of _verify_call, whose shared-expert gate uses the
    verify lowering of the sigmoid (attention_gate; MLX 0.32.2's fused sigmoid differs at some bf16 inputs). Split at
    the block, each half takes _verify_call, as a 16-row check does."""
    rows = _rows(x)
    if not _live(rows):
        return _W["blk"](self, x)
    kind = _tier_kind(self)
    if kind is not None:                               # the streamed exact-45 engine: its sync-free gate takes ONE MoE call
        if not ST["moe1"]:                             # per layer per forward (BigMoE g32_test B3: a 2 x 16 split = garbage)
            raise RuntimeError("v32: on the streamed tier engine the MoE block must be one call (V32_MOE1=1)")
        CNT["moeblk"] += 1
        return _tier_block(self, x, kind)
    if ST["moe1"] and hasattr(self, "_verify_call") and getattr(self, "sharding_group", None) is None:
        CNT["moeblk"] += 1                             # V32_MOE1: the block's verify path at n rows (not the prefill
        return self._verify_call(x)                    # combine); router / shared expert split by the qmm wrapper
    lead, Dm = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(rows, Dm)
    y1 = _W["blk"](self, x2[:R].reshape(1, R, Dm))
    y2 = _W["blk"](self, x2[R:].reshape(1, rows - R, Dm))
    CNT["moeblk"] += 1
    return mx.concatenate([y1.reshape(R, -1), y2.reshape(rows - R, -1)], axis=0).reshape(*lead, -1)


# ---- cache-dependent cores
def _att(self, x, cache):
    S = x.shape[1]
    if not (x.shape[0] == 1 and _live(S)):
        return _W["att"](self, x, cache)
    o1 = _W["att"](self, x[:, :R], cache)
    o2 = _W["att"](self, x[:, R:], cache)
    CNT["att"] += 1
    return mx.concatenate([o1, o2], axis=1)


def _gdn(self, inputs, mask=None, cache=None):
    S = inputs.shape[1]
    if not (inputs.shape[0] == 1 and mask is None and cache is not None and _live(S)):
        return _W["gdn"](self, inputs, mask, cache)
    cap = _CAP[0].get()                              # the stash exists only inside a verify forward (not the commit)
    if cap:
        cache._mtplx_verify_rows = None
    o1 = _W["gdn"](self, inputs[:, :R], mask, cache)
    s1 = getattr(cache, "_mtplx_verify_rows", None) if cap else None
    if cap:
        cache._mtplx_verify_rows = None
    o2 = _W["gdn"](self, inputs[:, R:], mask, cache)
    s2 = getattr(cache, "_mtplx_verify_rows", None) if cap else None
    if cap:
        if s1 is None or s2 is None:
            raise RuntimeError("v32: a GDN chunk did not stash its rows inside a verify forward")
        cache._mtplx_verify_rows = tuple(mx.concatenate([u, v], axis=1) for u, v in zip(s1, s2))
    CNT["gdn"] += 1
    return mx.concatenate([o1, o2], axis=1)


# ---- D32's mr2 on native copies (optional)
def _mr2(x2, w, a, kw):
    RF = _RF[0]
    scales = a[0] if len(a) > 0 else kw.get("scales")
    biases = a[1] if len(a) > 1 else kw.get("biases")
    transpose = a[2] if len(a) > 2 else kw.get("transpose", True)
    group_size = a[3] if len(a) > 3 else kw.get("group_size")
    bits = a[4] if len(a) > 4 else kw.get("bits")
    mode = a[5] if len(a) > 5 else kw.get("mode", "affine")
    if not (transpose and mode == "affine" and bits == 4 and group_size == 64 and x2.dtype == mx.bfloat16):
        return None
    e = RF._DNR.get(id(w)) if RF._DNR else None
    if e is not None and e[0] is w:                     # DENSE-N-RAM: w itself is the native copy
        _, N, K = e
        wn, sn, bn = w, scales, biases
    else:
        if not RF.CFG["dense_native"] or biases is None:
            return None
        e = RF._DN.get(id(w))
        if e is None or e[0] is not w:                  # no native copy yet: the 16-row path makes it
            return None
        _, wn, sn, bn = e
        N, K = w.shape[0], x2.shape[-1]
    if x2.shape[-1] != K or scales.dtype != mx.bfloat16:
        return None
    ks = RF._dn_ks(N, K)
    nt = RF.CFG["dense_native_nt"] if ks == 1 and N % (32 * RF.CFG["dense_native_nt"]) == 0 else 1
    import d32
    rows = x2.shape[0]
    xp = x2 if rows == HI else mx.concatenate([x2, mx.zeros((HI - rows, x2.shape[1]), x2.dtype)], axis=0)
    return d32.mm("q4", xp, (wn, sn, bn), N, ks, nt, 2, 1)[:rows]


def install(ref2):
    """after ref2.apply(): wrap its hooks (idempotent); ST['on'] switches the 17-32-row path"""
    import mtplx.models.qwen4_exp as Q
    from mlx_lm.models import switch_layers as SL
    if ref2.CFG["gdn_lazy_commit"] or ref2.CFG["hc_pend_mlp"] or ref2.CFG["neverwait"]:
        raise RuntimeError("v32: ref2's lazy GDN commit, HC pending carry and NEVER-WAIT are not supported")
    _RF[0] = ref2
    from mtplx.attention_context import current_attention_phase
    _PH[0] = current_attention_phase
    _CAP[0] = Q._VERIFY_CAPTURE
    if not _W:
        _W.update(qmm=mx.quantized_matmul, lin=nn.Linear.__call__, gr=Q.GatedResidual.__call__,
                  att=Q.Attention.__call__, gdn=Q.GatedDeltaNet.__call__,
                  sw=Q._FusedGateUpSwitchGLU.__call__, swb=SL.SwitchGLU.__call__, blk=Q.SparseMoeBlock.__call__)
        mx.quantized_matmul = _qmm
        nn.Linear.__call__ = _lin
        Q.GatedResidual.__call__ = _gr
        Q.Attention.__call__ = _att
        Q.GatedDeltaNet.__call__ = _gdn
        Q._FusedGateUpSwitchGLU.__call__ = _moe("sw")
        SL.SwitchGLU.__call__ = _moe("swb")
        Q.SparseMoeBlock.__call__ = _moeblock         # after a loop's own block patch (K-73), if any: it wraps that
    ST["on"] = True


def uninstall():
    import mtplx.models.qwen4_exp as Q
    from mlx_lm.models import switch_layers as SL
    if _W:
        mx.quantized_matmul = _W["qmm"]
        nn.Linear.__call__ = _W["lin"]
        Q.GatedResidual.__call__ = _W["gr"]
        Q.Attention.__call__ = _W["att"]
        Q.GatedDeltaNet.__call__ = _W["gdn"]
        Q._FusedGateUpSwitchGLU.__call__ = _W["sw"]
        SL.SwitchGLU.__call__ = _W["swb"]
        Q.SparseMoeBlock.__call__ = _W["blk"]
        _W.clear()
    ST["on"] = False
