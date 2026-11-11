"""R-76 DRAFT-FUSE (Leap, Sun 2026-11-01): DFlash2's forward with fewer kernels and the same bits. install() patches the
drafter module (/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/dflash_model_mlx.py, which BigMoE's r57 harness also imports: nothing changes until
install() is called):
- GroupedDynamicCausalConv.prepare / finish: _grouped_dynamic_convolve (kernel 2, groups of 16) as ONE kernel per call
  instead of ~11 (zeros, the shifted copy, then per tap kernel * v, add, dynamic * v, add), replaying the eager bf16 ops
  in their order: o = +0; per tap: v = row l - tap (+0 above the block's first row), o = o + bf16(base) * v,
  o = o + dynamic * v, each op rounded to bf16 as MLX's binary kernels do (bitwise, r76_draft_fuse.py).
- DFlashAttention: the sliding-window / causal mask depends only on (ctx_len, L, window, causal) and was rebuilt in each
  of the 5 layers: built once per key and reused (the same values).
- install(forms=True) adds R-78: the 16-row block projections in the MLX spelling that is bitwise equal and faster at that
  row count (L1-23's trick): x @ Wt with Wt = W.T made contiguous (the MLP's gate / up / down and q_proj; Wt REPLACES W,
  so no extra memory) or (W @ x.T).T (k / v on the block rows, o_proj, the conv kernel projections). Other row counts
  (the 8-row context through k / v) keep nn.Linear's x @ W.T.
  FORM_ROWS (Wed 2026-11-04; MTP Head's DRAFT17, the orchestrator's route): the row counts that take the forms, default
  16..64. R-78b (scripts/r78b_rows.py, data/r78b_rows.txt) found both forms bitwise == x @ W.T at EVERY row count
  16-64 (and 80-1000), on all nine projection shapes, and different at 2-15. So block passes of 17 / 32 positions
  keep the drafts bitwise and lose form C's per-call weight copy (+8.6 / +9.0 ms per pass, TrainLab REPORT
  'DRAFT17'). DRAFT_FORM_ROWS="..." (env, read at import) adds rows; never add rows below 16.
uninstall() restores the originals."""
import os
from functools import lru_cache
import mlx.core as mx

_CONV = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= n_el[0]) return;
    constexpr int H = H_C, GS = GS_C, KS = KS_C, NG = H_C / GS_C;
    const uint h = i % H;
    const uint l = (i / H) % L_C;
    const uint b = i / (H * L_C);
    const uint g = h / GS;
    T o = T(0);
    for (int t = 0; t < KS; ++t) {
        const T v = ((int)l >= t) ? x[(size_t)(b * L_C + (l - t)) * H + h] : T(0);
        const T k = static_cast<T>(base[(size_t)t * H + h]);
        const T p = k * v;
        o = o + p;
        const T d = dyn[((size_t)(b * L_C + l) * DROW) + DOFF + t * NG + g];
        const T q = d * v;
        o = o + q;
    }
    out[i] = o;
"""


@lru_cache(None)
def _conv_k():
    return mx.fast.metal_kernel(name="dflash_gdconv", input_names=["x", "dyn", "base", "n_el"], output_names=["out"],
                                source=_CONV)


def gd_conv(hidden, dyn_full, half, base, group_size):
    """== _grouped_dynamic_convolve(hidden, dyn_full[..., half, :, :], base, group_size), bitwise. hidden [B, L, H];
    dyn_full [B, L, 2 * KS * groups] (kernel_projection's output, contiguous); base [KS, H] (any float dtype)."""
    B, L, H = hidden.shape
    KS = base.shape[0]
    NG = H // group_size
    assert dyn_full.shape[-1] == 2 * KS * NG and hidden.dtype == dyn_full.dtype, (dyn_full.shape, hidden.dtype)
    n = hidden.size
    (o,) = _conv_k()(inputs=[hidden, dyn_full, base, mx.array([n], mx.uint32)],
                     template=[("T", hidden.dtype), ("H_C", H), ("GS_C", group_size), ("KS_C", KS), ("L_C", L),
                               ("DROW", 2 * KS * NG), ("DOFF", half * KS * NG)],
                     grid=(n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[hidden.shape], output_dtypes=[hidden.dtype])
    return o


_ORIG = {}
_MASKS = {}


_FORMS = []


class _LinT:
    """call-only stand-in for a bias-free nn.Linear whose weight is kept transposed (R-78 form C)"""
    pass


FORM_ROWS = set(range(16, 65)) | {int(r) for r in os.environ.get("DRAFT_FORM_ROWS", "").split(",") if r.strip()}


def _rows16(x):
    return x.ndim >= 2 and x.size // x.shape[-1] in FORM_ROWS


def _forms_on(draft):
    import mlx.nn as nn
    for layer in draft.layers:
        a = layer.self_attn
        for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj, a.q_proj):          # form C
            assert isinstance(lin, nn.Linear) and "bias" not in lin
            wt = mx.contiguous(lin.weight.T)
            mx.eval(wt)
            cls = type(lin)
            sub = type(cls.__name__ + "T", (cls,), {"__call__": lambda self, x: (x @ self._wt) if _rows16(x)
                                                     else x @ mx.contiguous(self._wt.T).T})
            object.__setattr__(lin, "_wt", wt)
            lin.pop("weight")
            lin.__class__ = sub
            _FORMS.append(("C", lin, cls))
        for lin in (a.k_proj, a.v_proj, a.o_proj, layer.attention_conv.kernel_projection,
                    layer.mlp_conv.kernel_projection):                                             # form B
            cls = type(lin)
            sub = type(cls.__name__ + "B", (cls,), {"__call__": lambda self, x, _cls=cls: (
                (self["weight"] @ x.reshape(-1, x.shape[-1]).T).T.reshape(*x.shape[:-1], -1) if _rows16(x)
                else _cls.__call__(self, x))})
            lin.__class__ = sub
            _FORMS.append(("B", lin, cls))


def _forms_off():
    for kind, lin, cls in _FORMS:
        if kind == "C":
            lin["weight"] = mx.contiguous(lin._wt.T)
            object.__delattr__(lin, "_wt")
        lin.__class__ = cls
    _FORMS.clear()


def install(forms=False, draft=None):
    import dflash_model_mlx as DF
    if forms and not _FORMS:
        assert draft is not None, "install(forms=True) needs the draft model"
        _forms_on(draft)
    if _ORIG:
        return
    C = DF.GroupedDynamicCausalConv
    _ORIG.update(prepare=C.prepare, finish=C.finish, att=DF.DFlashAttention.__call__)

    def prepare(self, hidden):
        dyn = self.kernel_projection(hidden)
        return gd_conv(hidden, dyn, 0, self.base_kernel[0], self.group_size), dyn

    def finish(self, hidden, dynamic):
        return gd_conv(hidden, dynamic, 1, self.base_kernel[1], self.group_size)

    C.prepare, C.finish = prepare, finish

    def att(self, x, x_ctx, rope, cache):                 # DFlashAttention.__call__ with the mask memoized
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        key = (ctx_len, L, self.sliding_window if self.is_sliding else None, bool(self.is_causal))
        mask = _MASKS.get(key)
        if mask is None:
            mask = DF.create_causal_mask(L, offset=ctx_len) if self.is_causal else None
            if self.is_sliding:
                query = ctx_len + mx.arange(L)[:, None]
                kk = mx.arange(ctx_len + L)[None]
                context = (kk < ctx_len) & (query - kk < self.sliding_window)
                block = kk >= ctx_len
                if self.is_causal:
                    block = block & (kk <= query)
                mask = context | block
            if mask is not None:
                mx.eval(mask)
            if len(_MASKS) > 64:
                _MASKS.clear()
            _MASKS[key] = mask
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

    DF.DFlashAttention.__call__ = att


def uninstall():
    import dflash_model_mlx as DF
    _forms_off()
    if not _ORIG:
        return
    DF.GroupedDynamicCausalConv.prepare, DF.GroupedDynamicCausalConv.finish = _ORIG["prepare"], _ORIG["finish"]
    DF.DFlashAttention.__call__ = _ORIG["att"]
    _ORIG.clear()
    _MASKS.clear()
