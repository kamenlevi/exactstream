"""bofi.py (Leap, Sun 15 Nov 2026 18:1x; CHECK55 track A, BOF PASS b380726): an opt-in installer of BOF (bof.qmm16bof, a
blocked ordered fold of qmm16c ks 1: REF2-T's OWN bits, bitwise 5 shapes x 4 configs) over ref2 (+ v32 / v32one / refc).
install(ref2), then ON[0] = True: every <= 16-row 4-bit g64 call on a shape in TABLE runs qmm16bof instead of ref2's
qmm16c ks 1 (the same bits, so no reference change); everything else falls through. 17-32-row calls keep v32one's
qmm16c2a / qmm16c2 (ks 1: also the same bits).
TABLE (bof_bench, cold): o_proj / out_proj 2,560 x 6,144 bs 4 (85.9 -> 48.6 us); QSA qkv_fused 13,312 x 2,560 bs 8
(107.0 -> 101.4); shared down 2,560 x 640 bs 4 (10.8 -> 7.4); lm_head 248,320 x 2,560 bs 4 (1,612.7 -> 1,454.0). The GDN
in-proj (115.6 -> 114.5) stays on qmm16c."""
import math
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts/week98"))
from bof import qmm16bof, qmm16bof2

ON = [False]
TABLE = {(2560, 6144): (4, 1), (13312, 2560): (8, 1), (2560, 640): (4, 1), (248320, 2560): (4, 1)}
TABLE32 = {}        # C10 (Sun 18:2x): {(2560, 6144): (4, 1)} turns on BOF2 for o_proj / out_proj at 17-32 rows (bof2_bench)
CNT = dict(bof=0, bof2=0)
_S = {}


def _qmmb(x, w, *a, **kw):
    if ON[0]:
        scales = a[0] if len(a) > 0 else kw.get("scales")
        biases = a[1] if len(a) > 1 else kw.get("biases")
        transpose = a[2] if len(a) > 2 else kw.get("transpose", True)
        group_size = a[3] if len(a) > 3 else kw.get("group_size")
        bits = a[4] if len(a) > 4 else kw.get("bits")
        mode = a[5] if len(a) > 5 else kw.get("mode", "affine")
        K, N = int(x.shape[-1]), int(w.shape[0])
        rows = math.prod(x.shape[:-1])
        RF = _S["ref2"]
        if (transpose and mode == "affine" and biases is not None and group_size == 64 and bits == 4
                and x.dtype == mx.bfloat16 and 1 <= rows <= 16 and (N, K) in TABLE and w.ndim == 2
                and w.dtype == mx.uint32 and scales.dtype == mx.bfloat16 and RF.CFG["dense_min_n"] <= N
                and not RF.CFG["dense_native"] and (not RF._DNR or RF._DNR.get(id(w)) is None)):
            bs, rg = TABLE[(N, K)]
            CNT["bof"] += 1
            return qmm16bof(x.reshape(rows, K), w, scales, biases, bs=bs, rg=rg, term=0).reshape(*x.shape[:-1], N)
        if (transpose and mode == "affine" and biases is not None and group_size == 64 and bits == 4
                and x.dtype == mx.bfloat16 and 16 < rows <= 32 and (N, K) in TABLE32 and w.ndim == 2
                and w.dtype == mx.uint32 and scales.dtype == mx.bfloat16 and RF.CFG["dense_min_n"] <= N
                and not RF.CFG["dense_native"] and (not RF._DNR or RF._DNR.get(id(w)) is None)):
            bs, rg = TABLE32[(N, K)]
            CNT["bof2"] += 1
            return qmm16bof2(x.reshape(rows, K), w, scales, biases, bs=bs, rg=rg, term=0).reshape(*x.shape[:-1], N)
    return _S["qmm"](x, w, *a, **kw)


def install(ref2):
    """after ref2.apply() (and v32 / v32one / refc if used): idempotent; ON[0] stays False"""
    if _S:
        return
    _S.update(ref2=ref2, qmm=mx.quantized_matmul)
    mx.quantized_matmul = _qmmb
