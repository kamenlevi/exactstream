"""Reference MoE for engine tests: MTPLX's SparseMoeBlock replaced by an exact, synchronous-residency version (all 10 ranks
ensured through BigMoE's ext, pool SwitchGLU, top-10 renormalized, gated shared expert), as NewIdeas'/Breakthrough's KLD
probes do with BT_SYNCGATE=1."""
import numpy as np
import mlx.core as mx

K = 10


def install(ext):
    from mtplx.models import qwen4_exp
    from mtplx.attention_math import attention_gate

    def ref_block(self, x):
        sw = self.switch_mlp
        L = sw._bm["L"]
        S = x.shape[1]
        lg = self.gate(x)[0].astype(mx.float32)
        p = mx.softmax(lg, axis=-1, precise=True)
        top = mx.argpartition(-p, K - 1, axis=-1)[:, :K]
        pk = mx.take_along_axis(p, top, axis=-1)
        o = mx.argsort(-pk, axis=-1)
        ids, pk = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(pk, o, axis=-1)
        w = (pk / pk.sum(-1, keepdims=True)).astype(x.dtype)
        flat = [int(v) for v in np.array(ids).reshape(-1)]
        ext.ensure(L, mx.array(np.array(sorted(set(flat)), np.int32)))
        slots = np.array(ext.lookup(L, flat)).reshape(S, K).astype(np.uint32)
        base = type(sw).__mro__[1]
        yk = base.__call__(sw, x[0][:, None, :], mx.array(slots)[:, None, :])[:, 0]
        y = (yk * w[..., None]).sum(axis=-2)
        return (y + attention_gate(self.shared_expert(x[0]), self.shared_expert_gate(x[0])))[None]

    qwen4_exp.SparseMoeBlock.__call__ = ref_block


def capture_layers():
    """Wrap DecoderLayer.__call__ to record (input, output) per call; returns the list (cleared by the caller)."""
    from mtplx.models import qwen4_exp
    rec = []
    orig = qwen4_exp.DecoderLayer.__call__

    def wrapped(self, hidden, **kw):
        out = orig(self, hidden, **kw)
        rec.append((hidden, out))
        return out

    qwen4_exp.DecoderLayer.__call__ = wrapped
    return rec
