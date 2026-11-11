"""T1 of ~/FableGod/LOSSY350_A.md (Fable A, Thu 5 Nov 2026): input sparsity for the routed experts on the decision map.

What an xs arm does: in every MoE block the routed experts (the switch: gate / up / down projections) read x with all but
the c largest-|x| members of each 64-wide input group zeroed (the kernel's quant groups: 2560 inputs = 40 groups). The
block's router, its shared expert and the shared-expert gate keep the full x. Nothing else changes.

Why here and not at dm_common.py line 310: the map prefills each window in 512-row chunks, so nearly every row takes the
>= 32-row path (SparseMoeBlock._prefill_call, or BigMoE's streamed sync call), which never reaches line 310. Both paths
hand x to the block's switch_mlp, through its __call__(x, indices) or its sorted_experts(x, indices). wrap() wraps those
two entry points on the switch's RUNTIME class (BigMoE's "...Streamed" subclass), once per class, so every path is covered
and the model's own code runs unchanged behind the wrapper.

Off by default: with XS["c"] None or >= 64 the wrapper hands the very same x object on, so the forward is the one of the
unpatched harness, bit for bit. dm_common imports this module only under DM_XS=1.

Not covered: rows served by dm_t3 (the *_t3 arms compute their record-served pairs outside the switch). BigMoE's mid-token
look-ahead routers read the switch's x, so in a streamed DECODE they would see the masked x (prefetch only, never an
output); the map's prefill path does not use them.
"""
import mlx.core as mx

G = 64                                                    # the quant group width of the expert records
XS = {"c": None, "calls": 0, "rows": 0}                   # c: members kept of each 64 inputs; None or >= 64 = off
_WRAPPED = {}                                             # runtime switch class -> (its __call__, its sorted_experts or None)

def on():
    return XS["c"] is not None and XS["c"] < G

def mask(x, c):
    """x with all but the c largest-|x| members of each 64-wide group of its last axis zeroed; same shape and dtype.
    Ties at the threshold are kept (a group may keep more than c); c None or >= 64 returns x itself."""
    if c is None or c >= G:
        return x
    D = x.shape[-1]
    assert D % G == 0 and c >= 1, (D, c)
    a = mx.abs(x.astype(mx.float32)).reshape(*x.shape[:-1], D // G, G)
    thr = mx.sort(a, axis=-1)[..., G - c:G - c + 1]       # the c-th largest |x| of each group
    return mx.where((a >= thr).reshape(x.shape), x, mx.zeros_like(x))

_DEPTH = [0]

def _run(fn, self, x, indices, a, kw):
    """fn(self, x', indices): x' = mask(x) at the outermost wrapped call while an xs arm is on; otherwise x itself (the
    same object), also for a wrapped call nested inside another one (BigMoE's streamed call hands its already masked x
    to the base class), so rows are masked and counted once."""
    if _DEPTH[0] or not on():
        return fn(self, x, indices, *a, **kw)
    XS["calls"] += 1; XS["rows"] += x.size // x.shape[-1]
    _DEPTH[0] += 1
    try:
        return fn(self, mask(x, XS["c"]), indices, *a, **kw)
    finally:
        _DEPTH[0] -= 1

def wrap(sw):
    """Wrap the runtime class of this switch module (idempotent; a pass-through while no xs arm is on)."""
    cls = type(sw)
    if cls in _WRAPPED:
        return
    had = ("__call__" in cls.__dict__, "sorted_experts" in cls.__dict__)
    oc = cls.__call__
    ose = getattr(cls, "sorted_experts", None)
    def call(self, x, indices, *a, **kw):
        return _run(oc, self, x, indices, a, kw)
    cls.__call__ = call
    if ose is not None:
        def sorted_experts(self, x, indices, *a, **kw):
            return _run(ose, self, x, indices, a, kw)
        cls.sorted_experts = sorted_experts
    _WRAPPED[cls] = (oc, ose, had)

def unwrap():
    """Put every wrapped class back (tests; the harness never needs it: the wrapper is inert while no arm is on)."""
    for cls, (oc, ose, had) in list(_WRAPPED.items()):
        for name, orig, own in (("__call__", oc, had[0]), ("sorted_experts", ose, had[1])):
            if orig is None:
                continue
            if own:
                setattr(cls, name, orig)
            else:
                delattr(cls, name)
    _WRAPPED.clear()
