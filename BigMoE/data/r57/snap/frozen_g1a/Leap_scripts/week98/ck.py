"""ck.py (Leap, Sun 15 Nov 2026 17:4x; CHECK55 track B): the reference contract's kernels, importable (CB1 58c2439 measured
them; refc.py installs them; cb1.py imports them from here).
  qmm16c2a_ks(x, w, scales, biases, ks, nt): v32one's one-weight-pass 32-row qmm16c2a with qmm16c's K split. Each 16-row
    tile's bits == qmm16c(ks) on its rows (CB1: every shape and ks bitwise); the two tiles' partials are reduced one
    after the other in qmm16c's order (group 0 + 1 + ...), so the threadgroup memory is qmm16c's.
  s4_partials_ps(a, w, ps) / s4_partials2i_ps(a, w, ps): leapk's S4 (16 rows) and one32's interleaved S4 (32 rows) at
    partition size ps (MLX's split-K replica uses 4,096); the same (cs, NP) interface as leapk.s4_partials /
    one32.s4_partials2i, so leapk's partition sums (_ACCUM / _ACCUM_GATE / kernel B, NP as a template) take them."""
import os
import sys
from functools import lru_cache
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1a/Leap_scripts/week98"))
import qmm16 as Q16
import leapk as LK
import one32

BF = mx.bfloat16

_RED2 = """    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];
    if (KS > 1) {                                   // tile 0 then tile 1, each in qmm16c's order (group 0 + 1 + ...)
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg == 0)
            for (int gg = 0; gg < KS - 1; gg++)
                for (short i = 0; i < 16; i++) acc[i] = acc[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) acc2[i] = acc2[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
    }
"""


def _src2a_ks():
    s = one32._SRC2A
    subs = [("    const uint nt = simdgroup_index_in_threadgroup % NT;\n",
             "    const uint kg = simdgroup_index_in_threadgroup / NT;\n"
             "    const uint nt = simdgroup_index_in_threadgroup % NT;\n"),
            ("    for (int k0 = 0; k0 < K_C; k0 += G) {\n",
             "    const int kper = K_C / KS;\n"
             "    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {\n"),
            ("    for (short i = 0; i < 16; i++) {\n        const int m = fm + 8 * ((i & 7) >> 2);\n",
             _RED2 + "    for (short i = 0; i < 16; i++) {\n        const int m = fm + 8 * ((i & 7) >> 2);\n")]
    for old, new in subs:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


@lru_cache(None)
def _k2aks():
    return mx.fast.metal_kernel(name="leap_cb1_qmm16c2a_ks", input_names=["x", "w", "scales", "biases"],
                                output_names=["y"], source=_src2a_ks(), header=Q16._HDR)


def qmm16c2a_ks(x, w, scales, biases, ks=1, nt=1):
    """qmm16c (ks, nt) for 17-32 rows in one weight pass; each 16-row tile's bits == qmm16c(ks) on those rows"""
    M, K = x.shape
    N = w.shape[0]
    assert 16 < M <= 32 and N % (32 * nt) == 0 and K % (64 * ks) == 0
    xp = x if M == 32 else mx.concatenate([x, mx.zeros((32 - M, K), x.dtype)], 0)
    (y,) = _k2aks()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt)],
                    grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(32, N)],
                    output_dtypes=[x.dtype])
    return y[:M]


_S4P = {}


def _s4k(N, K, ps, pf, rows):
    key = (N, K, ps, pf, rows)
    if key not in _S4P:
        NP, NT = -(-K // ps), -(-N // 32)
        src = LK._S4_SRC if rows == 16 else one32._s4_src2i()
        _S4P[key] = mx.fast.metal_kernel(
            name=f"leap_cb1_s4_{N}_{K}_{ps}_{rows}" if pf == 4 else f"leap_ck_s4_{N}_{K}_{ps}_{rows}_{pf}",
            input_names=["x", "w"], output_names=["cs"], source=src,
            header=LK._nax_header() + "\n" + LK.consts(GN=N, GK=K, GPS=ps, GNP=NP, GNT=NT, GPF=pf))
    return _S4P[key]


def s4_partials_ps(a, w, ps, pf=4):
    """leapk.s4_partials at partition size ps: a [1, 16, K] bf16, w [N, K] bf16 -> (cs [NP, 16, N] fp32, NP)"""
    N, K = w.shape
    NP, NT = -(-K // ps), -(-N // 32)
    (cs,) = _s4k(N, K, ps, pf, 16)(inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
                                   output_shapes=[(NP, 16, N)], output_dtypes=[mx.float32])
    return cs, NP


def s4_partials2i_ps(a, w, ps, pf=4):
    """one32.s4_partials2i at partition size ps: a [1, 32, K] bf16 -> (cs [NP, 32, N] fp32, NP)"""
    N, K = w.shape
    NP, NT = -(-K // ps), -(-N // 32)
    (cs,) = _s4k(N, K, ps, pf, 32)(inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP * 2, 1, 1),
                                   threadgroup=(32, 1, 1), output_shapes=[(NP, 32, N)], output_dtypes=[mx.float32])
    return cs, NP
