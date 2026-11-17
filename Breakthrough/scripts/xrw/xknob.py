"""xknob.py (Breakthrough, Tue 17 Nov 2026 10:4x; EXPERT REWRITE): the ONE opt-in knob Leap asked for (09:34): install()
wraps ooo1cq.tier_glu_nax6n_1cq once; with ON[0] False every call goes to the original (bitwise today: xrw_check item a);
with ON[0] True a p11-store call goes to the candidate MODE[0] ("w4a8" = int8 x int4b, FAST; "tgl16" = bf16 threadgroup
left), any other store to the original. COH on routes through the same wrap: coh1cq swaps ooo1cq's _ka / _kb and calls
ooo1cq.tier_glu_nax6n_1cq, looked up at call time; the candidates' pool reads are always coherent(system).
    import xknob; xknob.install(); xknob.MODE[0] = "w4a8"; xknob.ON[0] = True"""
import os
import sys

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ON = [False]
MODE = ["w4a8"]
COUNTS = {"cand": 0, "orig": 0, "fallback_not_p11": 0}
_ORIG = [None]


def impl(mode=None):
    mode = mode or MODE[0]
    if mode == "w4a8":
        import w4a8
        return w4a8.tier_glu_nax6n_1cq_w4a8
    if mode == "tgl16":
        import tgl16
        return tgl16.tier_glu_nax6n_1cq_tgl16
    if mode in ("tgl16g", "a8b16", "tgl16e", "tgl16eh", "tgl16ep", "tgl16eu", "w4a8u"):   # bench variants (tgl16v.py)
        import tgl16v
        return getattr(tgl16v, f"tier_glu_nax6n_1cq_{mode}")
    if mode.startswith("tgl16abl"):                        # timing ablations of tgl16, e.g. tgl16abl4 = no MMA
        import functools
        import tgl16v
        return functools.partial(tgl16v.tier_glu_nax6n_1cq_tgl16abl, abl=int(mode[8:]))
    raise ValueError(mode)


def install():
    import ooo1cq as OQ
    if _ORIG[0] is not None:
        return OQ.tier_glu_nax6n_1cq
    _ORIG[0] = OQ.tier_glu_nax6n_1cq

    def tier_glu_nax6n_1cq(*a, **kw):
        if ON[0]:
            if a[1]["gate_proj"]["s3"].dtype == mx.uint32:
                COUNTS["cand"] += 1
                return impl()(*a, **kw)
            COUNTS["fallback_not_p11"] += 1
        COUNTS["orig"] += 1
        return _ORIG[0](*a, **kw)
    OQ.tier_glu_nax6n_1cq = tier_glu_nax6n_1cq
    return tier_glu_nax6n_1cq


def uninstall():
    import ooo1cq as OQ
    if _ORIG[0] is not None:
        OQ.tier_glu_nax6n_1cq = _ORIG[0]
        _ORIG[0] = None
