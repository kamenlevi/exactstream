"""DKV (Leap, Wed 2026-11-04 18:0x; TABLE_350 'DKV'): a copy-free sliding-window cache for the DFlash2 drafter, bitwise
== mlx_lm's RotatingKVCache + draft_fused's attention. Each propose of the reference copies every layer's whole window
of K and V twice: RotatingKVCache._update_concat (temporal order + trim + append = a concat) and the block's
`concatenate([keys, prop_keys])` (DSDPA-0: the second copy alone ~0.19 / 0.35 ms per propose at ctx 1,008 / 2,054).
LinRotCache keeps the window in temporal order in a preallocated linear buffer [1, 8, cap, 128]. A multi-token update
trims the front exactly as _update_concat does (keep 0: the oldest (len - max_size + 1) rows), then writes the new
context rows AND the block's rows in one in-place slice update (the old views are dropped first, so MLX donates the
buffer). SDPA reads the view [a, b + L): the same rows in the same order as the reference's concat. Every other case
runs the reference's own code on the same state: 1-token updates (_update_in_place's rotated buffer, whose mask quirk
stays as is), a state the base class left (adopted into a fresh buffer at the next multi-token update: one copy, like
the reference's), and the buffer's end (compaction: one copy every ~(cap - max_size) / S proposes).
install(addmask=False): DFlashDraftModel.make_cache -> LinRotCache for sliding layers; DFlashAttention.__call__ takes
the copy-free path for LinRotCache (other caches: the previous call); DF._trim_recent_cache trims a LinRotCache window
in place. addmask=True: the memoized mask as an additive bf16 (0 / -inf) array (DSDPA-0: bitwise == bool, 4-16 % less
SDPA time). Inert until install(); uninstall() restores all three."""
import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

COUNTS = {"fast": 0, "slow": 0, "adopt": 0, "compact": 0, "trim_fast": 0}


class LinRotCache(RotatingKVCache):
    def __init__(self, max_size, keep=0, cap=None):
        assert keep == 0, "LinRotCache: keep 0 only (the drafter's)"
        super().__init__(max_size, keep)
        self.cap = int(cap or 2 * max_size + 128)
        self._bk = self._bv = None
        self._a = self._b = 0
        self._wk = self._wv = None
        self._fastok = False                             # True only while the base class has not touched the state

    def _win(self):
        return self._fastok and self.keys is not None and self.keys is self._wk and self._idx == self._b - self._a

    def update_and_fetch(self, keys, values):            # the base class's paths (1-token updates; direct callers)
        self._fastok = False
        return super().update_and_fetch(keys, values)

    def _fresh(self, k, v, n_extra):
        """a fresh buffer holding k / v (temporal order) at [0, n) with room for n_extra more rows"""
        n = k.shape[2]
        cap = max(self.cap, n + n_extra + 64)
        z = lambda a: mx.zeros((a.shape[0], a.shape[1], cap - n, a.shape[3]), a.dtype)
        self._wk = self._wv = self.keys = self.values = None
        self._bk, self._bv = mx.concatenate([k, z(k)], axis=2), mx.concatenate([v, z(v)], axis=2)
        self._a, self._b = 0, n

    def _set_views(self):
        self._wk, self._wv = self._bk[..., self._a:self._b, :], self._bv[..., self._a:self._b, :]
        self.keys, self.values = self._wk, self._wv
        self._idx = self._b - self._a
        self._fastok = True

    def update_with_prop(self, ck, cv, pk, pv):
        """the reference's update_and_fetch(ck, cv) followed by concatenate([keys, pk]) / ([values, pv]):
        returns (K, V, ctx_len) with K / V = [the window, the block's rows]"""
        S, L = ck.shape[2], pk.shape[2]
        if S < 2:                                        # _update_in_place: the reference's own code and copies
            COUNTS["slow"] += 1
            k, v = self.update_and_fetch(ck, cv)
            return mx.concatenate([k, pk], axis=2), mx.concatenate([v, pv], axis=2), k.shape[2]
        if self.keys is None:                            # the first update: the reference stores ck as is (_idx = S)
            self._fresh(ck[..., :0, :], cv[..., :0, :], S + L)
        elif not self._win():                            # a state the base class left: its temporal order, adopted
            COUNTS["adopt"] += 1
            self._fresh(self._temporal_order(self.keys), self._temporal_order(self.values), S + L)
        a, b = self._a, self._b
        if self.keys is not None or b > a:               # _update_concat's trim (keep 0); not on the first update
            trim = (b - a) - self.max_size + 1
            if trim > 0:
                a += trim
        if b + S + L > self._bk.shape[2]:                # the buffer's end: compact (one copy of the window)
            COUNTS["compact"] += 1
            wk, wv = self._bk[..., a:b, :], self._bv[..., a:b, :]
            self._fresh(wk, wv, S + L)
            a, b = self._a, self._b
        self._wk = self._wv = self.keys = self.values = None   # no live views of the buffer: the write below is in place
        self._bk[..., b:b + S + L, :] = mx.concatenate([ck, pk], axis=2)
        self._bv[..., b:b + S + L, :] = mx.concatenate([cv, pv], axis=2)
        self._a, self._b = a, b + S
        self._set_views()
        self.offset += S
        COUNTS["fast"] += 1
        return self._bk[..., a:b + S + L, :], self._bv[..., a:b + S + L, :], b + S - a

    def trim_recent(self, n):
        """DF._trim_recent_cache's RotatingKVCache branch (temporal order, drop the last n rows, offset -= n, _idx =
        the new length) on the window, in place"""
        COUNTS["trim_fast"] += 1
        self._b = max(self._a, self._b - n)
        self._set_views()
        self.offset -= n


_ORIG = {}
_MASKS = {}


def install(addmask=False):
    import dflash_model_mlx as DF
    if _ORIG:
        return
    _ORIG.update(att=DF.DFlashAttention.__call__, make=DF.DFlashDraftModel.make_cache, trim=DF._trim_recent_cache)
    prev, trim0 = _ORIG["att"], _ORIG["trim"]

    def make_cache(self):
        caches = _ORIG["make"](self)
        return [LinRotCache(c.max_size, c.keep) if type(c) is RotatingKVCache else c for c in caches]

    def trim_recent_cache(cache, num_tokens):
        if num_tokens <= 0:
            return
        rest = []
        for c in cache:
            n = min(getattr(c, "offset", num_tokens), num_tokens)
            if isinstance(c, LinRotCache) and n > 0 and c._win():
                c.trim_recent(n)
            else:
                if isinstance(c, LinRotCache):
                    c._fastok = False                    # the original branch rewrites keys / _idx directly
                rest.append(c)
        if rest:
            trim0(rest, num_tokens)

    def att(self, x, x_ctx, rope, cache):
        if not isinstance(cache, LinRotCache):
            return prev(self, x, x_ctx, rope, cache)
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
        keys, values, ctx_len = cache.update_with_prop(ctx_keys, ctx_values, prop_keys, prop_values)
        key = (ctx_len, L, self.sliding_window if self.is_sliding else None, bool(self.is_causal), bool(addmask))
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
            if mask is not None and addmask and mask.dtype == mx.bool_:
                mask = mx.where(mask, mx.array(0.0, x.dtype), mx.array(-float("inf"), x.dtype))
            if mask is not None:
                mx.eval(mask)
            if len(_MASKS) > 64:
                _MASKS.clear()
            _MASKS[key] = mask
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

    DF.DFlashAttention.__call__ = att
    DF.DFlashDraftModel.make_cache = make_cache
    DF._trim_recent_cache = trim_recent_cache


def uninstall():
    import dflash_model_mlx as DF
    if _ORIG:
        DF.DFlashAttention.__call__ = _ORIG["att"]
        DF.DFlashDraftModel.make_cache = _ORIG["make"]
        DF._trim_recent_cache = _ORIG["trim"]
        _ORIG.clear()
