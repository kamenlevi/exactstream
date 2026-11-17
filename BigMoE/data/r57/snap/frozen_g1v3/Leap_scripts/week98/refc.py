"""refc.py (Leap, Sun 15 Nov 2026 17:4x; CHECK55 track B; CB1 PASS 58c2439): the reference CONTRACT's dense forms as an
opt-in installer over ref2 (+ v32 + v32one), for timing A/Bs on the slice / engine. A NEW reference, not REF2-T's bits:
every shape below uses one fixed K-chunk (CB1's joint ks) at 16 and at 32 rows, so SD == AR holds by construction under
it, but it must never stand in for exact-45's reference unless Kamen adopts the switch (next week).
install(ref2) after ref2.apply() and, for 32 rows, v32.install + v32one.install; then ON[0] = True turns the contract on
for the calls that follow (timing arms toggle it per check). OFF, every hook falls through to what was installed before.
Contract forms (4-bit g64; MLX's weight layout, so no DENSE-N copies; the native layout has the same bits):
  GDN qkv | z 16,384 x 2,560: ks 4 | GDN b | a 96 x 2,560: ks 10 (not folded into the in-proj here) |
  o_proj / out_proj 2,560 x 6,144: ks 4 | QSA qkv_fused 13,312 x 2,560: ks 4 | shared cat 1,281 x 2,560: ks 5 on a
  copy padded to 1,312 rows (output sliced back) | shared down 2,560 x 640: ks 10 | lm_head 248,320 x 2,560: ks 4.
  <= 16 rows: qmm16c(ks) (pads to 16); 17-32 rows: ck.qmm16c2a_ks(ks) (one weight pass, each tile == qmm16c(ks)).
  HC read: S4 at PS[0] = 2,048 (leapk.s4_partials / s4_gemm at 16 rows; one32.s4_partials2i / 2 at 32 rows).
Untouched: the 8-bit router / indexer, HC up, PLE, experts, attention, GDN body (REF2-T's forms)."""
import math
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98"))
from qmm16 import qmm16c
import ck

ON = [False]
PS = [2048]
# (N, K): (ks, nt at 16 rows, nt at 32 rows, padded N or None)
CON = {(16384, 2560): (4, 1, 1, None), (96, 2560): (10, 1, 1, None), (2560, 6144): (4, 2, 2, None),
       (13312, 2560): (4, 2, 1, None), (1281, 2560): (5, 1, 1, 1312), (2560, 640): (10, 1, 1, None),
       (248320, 2560): (4, 1, 2, None)}
CNT = dict(c16=0, c32=0, s4=0, s4x2=0, s4g=0)
_S = {}
_PADW = {}


def _padded(w, s, b, Np):
    e = _PADW.get(id(w))
    if e is None or e[0] is not w:
        z = lambda a: mx.concatenate([a, mx.zeros((Np - a.shape[0],) + a.shape[1:], a.dtype)])
        e = (w, z(w), z(s), z(b))
        mx.eval(e[1:])
        _PADW[id(w)] = e
    return e[1], e[2], e[3]


def _form(x2, w, s, b, N, K):
    """x2 [rows <= 32, K] -> [rows, N] under the contract"""
    ks, nt16, nt32, Np = CON[(N, K)]
    if Np:
        w, s, b = _padded(w, s, b, Np)
    if x2.shape[0] <= 16:
        y = qmm16c(x2, w, s, b, ks=ks, nt=nt16)
        CNT["c16"] += 1
    else:
        y = ck.qmm16c2a_ks(x2, w, s, b, ks=ks, nt=nt32)
        CNT["c32"] += 1
    return y[:, :N] if Np else y


def _qmmc(x, w, *a, **kw):
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
                and x.dtype == mx.bfloat16 and 1 <= rows <= 32 and (N, K) in CON and w.ndim == 2
                and w.dtype == mx.uint32 and scales.dtype == mx.bfloat16 and (not RF._DNR or RF._DNR.get(id(w)) is None)):
            return _form(x.reshape(rows, K), w, scales, biases, N, K).reshape(*x.shape[:-1], N)
    return _S["qmm"](x, w, *a, **kw)


def _c2a(x, w, s, b, nt=1):                       # v32one's 32-row GDN in-proj call (one32.qmm16c2a)
    N, K = int(w.shape[0]), int(x.shape[-1])
    if ON[0] and (N, K) in CON:
        return _form(x, w, s, b, N, K)
    return _S["c2a"](x, w, s, b, nt=nt)


def _c2(x, w, s, b, ks=1, nt=1):                  # one32.qmm16c2 (two-tile grid form)
    N, K = int(w.shape[0]), int(x.shape[-1])
    if ON[0] and (N, K) in CON:
        return _form(x, w, s, b, N, K)
    return _S["c2"](x, w, s, b, ks=ks, nt=nt)


def _qv32(x, w, s, b, bits, k_lanes, vecs_per_tg=4, rows=32):   # one32.qmvw32 (v32one's 32-row small shapes, b | a)
    N, K = int(w.shape[0]), int(x.shape[-1])
    if ON[0] and bits == 4 and (N, K) in CON:
        return _form(x, w, s, b, N, K)
    return _S["qv32"](x, w, s, b, bits, k_lanes, vecs_per_tg, rows)


def _s4p(a, w, pf=4):                             # leapk.s4_partials (the 16-row HC read)
    if ON[0]:
        CNT["s4"] += 1
        return ck.s4_partials_ps(a, w, PS[0], pf)
    return _S["s4p"](a, w, pf)


def _s4g(a, w, pf=4):                             # leapk.s4_gemm (partials + leapk's ordered partition sum)
    if ON[0]:
        LK = _S["LK"]
        cs, NP = ck.s4_partials_ps(a, w, PS[0], pf)
        N = int(w.shape[0])
        (y,) = LK._ACCUM(inputs=[cs], template=[("T", mx.bfloat16), ("N", N), ("NP", NP)], grid=(16 * N, 1, 1),
                         threadgroup=(256, 1, 1), output_shapes=[(1, 16, N)], output_dtypes=[mx.bfloat16])
        CNT["s4g"] += 1
        return y
    return _S["s4g"](a, w, pf)


def _s42i(a, w, pf=4):                            # one32.s4_partials2i / s4_partials2 (the 32-row HC read)
    if ON[0]:
        CNT["s4x2"] += 1
        return ck.s4_partials2i_ps(a, w, PS[0], pf)
    return _S["s42i"](a, w, pf)


def _s42(a, w, pf=4):
    if ON[0]:
        CNT["s4x2"] += 1
        return ck.s4_partials2i_ps(a, w, PS[0], pf)     # the same per-tile arithmetic as the tile-on-y form
    return _S["s42"](a, w, pf)


def install(ref2):
    """after ref2.apply() (and v32 / v32one if 32-row checks run): idempotent; ON[0] stays False"""
    if _S:
        return
    import leapk as LK
    import one32
    _S.update(ref2=ref2, LK=LK, qmm=mx.quantized_matmul, c2a=one32.qmm16c2a, c2=one32.qmm16c2, qv32=one32.qmvw32,
              s4p=LK.s4_partials, s4g=LK.s4_gemm, s42i=one32.s4_partials2i, s42=one32.s4_partials2)
    mx.quantized_matmul = _qmmc
    one32.qmm16c2a, one32.qmm16c2, one32.qmvw32 = _c2a, _c2, _qv32
    LK.s4_partials, LK.s4_gemm = _s4p, _s4g
    one32.s4_partials2i, one32.s4_partials2 = _s42i, _s42
