"""Queue L2-06 (Breakthrough, 2026-10-25; NewIdeas QUEUE): 8-bit KV for the 125B's QSA (full-attention) layers.
Each QSA layer (12 of 48, plus the MTP layer) keeps K and V in bf16: 2 x 2 heads x 256 x 2 B = 2 KB per token per layer,
~26 KB per token in all, the bulk of the ~40 KB/token state growth (B12 + MTP at 64K: +2.6 GB active at the first token).
QKV stores them as MLX affine 8-bit, group 64 (4 groups per 256-wide head row, bf16 scales and biases): 272 B per head row
instead of 512, i.e. 53 % of the bf16 bytes. Every read dequantizes the whole buffer, so attention still gets bf16 arrays
and every kernel path (dense, rows-gather, flash-skip on the backing arrays) runs unchanged: a transient of one layer's
bf16 KV per read, and the only change is the quantized values. The indexer's raw and pooled key streams stay bf16.
Opt-in: install() once after load_b12, then ON[0] = True before model.make_cache(); caches made while ON[0] is False
are the stock QSAKVCache. GS / BITS: B12_KV8_GS (64), B12_KV8_BITS (8)."""
import os

GS = int(os.environ.get("B12_KV8_GS", "64"))
BITS = int(os.environ.get("B12_KV8_BITS", "8"))
ON = [False]
CONF = {"part": os.environ.get("B12_KV8_PART", "kv"), "gs": GS, "bits": BITS}   # read by each new cache (screens vary it)
_CLS = []


def _q(x, gs, bits):
    """[B, H, R, D] bf16 -> (w, scales, biases), each [B, H, R, *]; bits 16 = kept as is"""
    import mlx.core as mx
    if bits >= 16:
        return x
    B, H, R, D = x.shape
    w, s, b = mx.quantize(x.reshape(B * H * R, D), group_size=gs, bits=bits)
    return tuple(a.reshape(B, H, R, a.shape[-1]) for a in (w, s, b))


def _dq(t, gs, bits, rows=None):
    import mlx.core as mx
    if not isinstance(t, tuple):
        return t if rows is None else t[..., :rows, :]
    w, s, b = (a if rows is None else a[..., :rows, :] for a in t)
    B, H, R, _ = w.shape
    out = mx.dequantize(w.reshape(B * H * R, -1), s.reshape(B * H * R, -1), b.reshape(B * H * R, -1),
                        group_size=gs, bits=bits)
    return out.reshape(B, H, R, -1)


def _make_cls(Q):
    import mlx.core as mx

    class QKV(Q.QSAKVCache):
        """QSAKVCache with 8-bit storage. keys / values (properties) are the dequantized full-capacity buffers."""

        def __init__(self):
            object.__setattr__(self, "_k", None)
            object.__setattr__(self, "_v", None)
            c = CONF
            object.__setattr__(self, "_qk", (c["gs"], c["bits"] if "k" in c["part"] else 16))
            object.__setattr__(self, "_qv", (c["gs"], c["bits"] if "v" in c["part"] else 16))
            super().__init__()                         # sets keys = values = None (the setters below), offset = 0

        # -- storage access -------------------------------------------------------------------------------------
        @property
        def keys(self):
            return None if self._k is None else _dq(self._k, *self._qk)

        @keys.setter
        def keys(self, x):
            object.__setattr__(self, "_k", None if x is None else _q(x, *self._qk))

        @property
        def values(self):
            return None if self._v is None else _dq(self._v, *self._qv)

        @values.setter
        def values(self, x):
            object.__setattr__(self, "_v", None if x is None else _q(x, *self._qv))

        def _cap(self):
            return 0 if self._k is None else int((self._k[0] if isinstance(self._k, tuple) else self._k).shape[2])

        def _grow(self, needed, keys, values):
            target = Q.qsa_rows_target_value()
            cap = target if target >= needed else ((needed + self.step - 1) // self.step) * self.step
            B, H, _, Dk = keys.shape
            Dv = values.shape[3]
            prev = int(self.offset)
            new = []
            for t, D, dt, qc in ((self._k, Dk, keys.dtype, self._qk), (self._v, Dv, values.dtype, self._qv)):
                z = _q(mx.zeros((B, H, cap - (prev if t is not None else 0), D), dt), *qc)
                if t is None or prev == 0:
                    new.append(z)
                elif isinstance(t, tuple):
                    new.append(tuple(mx.concatenate([a[..., :prev, :], b], axis=2) for a, b in zip(t, z)))
                else:
                    new.append(mx.concatenate([t[..., :prev, :], z], axis=2))
            object.__setattr__(self, "_k", new[0])
            object.__setattr__(self, "_v", new[1])

        def update_and_fetch(self, keys, values):
            prev = int(self.offset)
            needed = prev + int(keys.shape[2])
            if self._k is None or needed > self._cap():
                self._grow(needed, keys, values)
            for t, x, qc in ((self._k, keys, self._qk), (self._v, values, self._qv)):
                if isinstance(t, tuple):
                    for a, b in zip(t, _q(x, *qc)):
                        a[..., prev:needed, :] = b
                else:
                    t[..., prev:needed, :] = x
            self.offset = needed
            return _dq(self._k, *self._qk, rows=needed), _dq(self._v, *self._qv, rows=needed)

        @property
        def state(self):
            if self._k is None:
                return None, None
            return _dq(self._k, *self._qk, rows=int(self.offset)), _dq(self._v, *self._qv, rows=int(self.offset))

        @state.setter
        def state(self, v):
            self.keys, self.values = v
            self.offset = self._cap()

        @property
        def nbytes(self):
            return 0 if self._k is None else sum(a.nbytes for t in (self._k, self._v)
                                                 for a in (t if isinstance(t, tuple) else (t,)))

    return QKV


def install():
    import mtplx.models.qwen4_exp as Q
    if _CLS:
        return _CLS[0]
    QKV = _make_cls(Q)
    _CLS.append(QKV)
    init0 = Q.QSACache.__init__

    def init(self, *a, **kw):
        init0(self, *a, **kw)
        if ON[0]:
            self.kv = QKV()
    Q.QSACache.__init__ = init
    return QKV


def selftest():
    """CPU: a QKV fed like decode (a prefill chunk, then 4-row steps, a trim, a state round trip) returns the bf16 rows
    within the 8-bit rounding bound, with the stock cache's offsets and shapes."""
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    import mtplx.models.qwen4_exp as Q
    QKV = install()
    ref, q = Q.QSAKVCache(), QKV()
    rng = np.random.default_rng(0)
    worst = 0.0
    for S in (300, 4, 4, 4, 1, 4):
        k = mx.array(rng.standard_normal((1, 2, S, 256)).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array(rng.standard_normal((1, 2, S, 256)).astype(np.float32)).astype(mx.bfloat16)
        rk, rv = ref.update_and_fetch(k, v)
        qk, qv = q.update_and_fetch(k, v)
        assert rk.shape == qk.shape and rv.shape == qv.shape and ref.offset == q.offset, (rk.shape, qk.shape)
        worst = max(worst, float(mx.max(mx.abs(rk.astype(mx.float32) - qk.astype(mx.float32)))),
                    float(mx.max(mx.abs(rv.astype(mx.float32) - qv.astype(mx.float32)))))
        if S == 1:
            ref.trim(3); q.trim(3)
    assert q.keys.shape[2] == q._cap() and q.keys.shape[:2] == (1, 2)
    st = q.state
    q2 = QKV(); q2.state = st
    rt_err = float(mx.max(mx.abs(q2.state[0].astype(mx.float32) - st[0].astype(mx.float32))))   # requantized: not bitwise
    assert q2.offset == q.offset and rt_err <= 2 * worst, (q2.offset, q.offset, rt_err)
    print(f"b12_kv8 selftest: offsets {q.offset} = {ref.offset}, max |err| {worst:.4f} ({CONF}), "
          f"state round trip {rt_err:.4f}, bytes {q.nbytes} vs {ref.keys.nbytes + ref.values.nbytes} bf16 -> OK", flush=True)


if __name__ == "__main__":
    selftest()
