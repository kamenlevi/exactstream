"""tree_att.py (Leap, Sun 15 Nov 2026 14:4x; TREE16 L2, data/week98/tree16_spec.md 'L2 design notes'): attention route
(b) for the exact tree check. Step 1 (this file so far): a replica of MLX's split-K NAX GEMM in the nn layout, the P·V
of the exact path's unfused SDPA (Breakthrough's profile: steel_gemm_splitk_nax_nn bm64 bn64 bk256 wm2 wn2 +
steel_gemm_splitk_accum_bfloat16_float32). Each output element accumulates its partition's k-steps of 32 in order with
NAX 16 x 32 x 32 tiles (gemm_nax.h's per-element arithmetic, independent of the tile shape), the partitions are written
as fp32 partials and summed in order. leapk's S4 is the same scheme in the nt layout, bitwise to MLX for the HC GEMM.
The partition size MLX picks for P·V is pinned by tree_att_probe.py (match vs mx.matmul).
Step 2 (next): per-row V tails in the tail k-steps for the tree's nodes.
NOT RUN YET (written while C7c waits for memory)."""
from functools import lru_cache
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16a/Leap_scripts"))

BF = mx.bfloat16

# one simdgroup per (16-row tile, 32-column tile, partition); A [M, K] row-major, V [K, N] row-major (nn)
_NN_SRC = """
    using namespace mlx::steel;
    const int tg = int(threadgroup_position_in_grid.x);
    const int ct = tg % GNT;                     // 32-column tile
    const int rt = (tg / GNT) % GMT;             // 16-row tile
    const int part = tg / (GNT * GMT);           // K partition
    const int c_col = ct * 32;
    const short sgp_sn = short(min(32, GN - c_col));
    const int k0 = part * GPS, k1 = min(k0 + GPS, GK);
    const int iters = (k1 - k0) / 32;
    const device T* Ap = x + size_t(rt) * 16 * GK + k0;
    const device T* Bp = v + size_t(k0) * GN + c_col;
    NAXTile<float, 1, 2> Dt;
    const bool full_n = sgp_sn == 32;
    Dt.clear();
    for (int s = 0; s < iters; s++) {
        NAXTile<T, 1, 2> At;
        NAXTile<T, 2, 2> Bt;
        At.load(Ap + s * 32, GK);
        if (full_n) {
            Bt.load(Bp + size_t(s) * 32 * GN, GN);
        } else {
            Bt.load_safe(Bp + size_t(s) * 32 * GN, GN, short2(sgp_sn, 32));
        }
        tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<false>{});
    }
    device float* Cp = cs + (size_t(part) * GM + size_t(rt) * 16) * GN + c_col;
    if (full_n) {
        Dt.store(Cp, GN);
    } else {
        Dt.store_safe(Cp, GN, short2(sgp_sn, 16));
    }
"""
_K = {}

_ACC = mx.fast.metal_kernel(
    name="leap_tree_att_accum", input_names=["cs"], output_names=["y"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= GMN) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * GMN + i];
    }
    y[i] = static_cast<T>(out);
""")


def nn_partials(a, v, ps):
    """a [M, K] bf16 (M % 16 == 0, K % 32 == 0), v [K, N] bf16 -> fp32 partials [NP, M, N] over K partitions of ps"""
    import leapk as LK
    M, K = a.shape
    N = v.shape[1]
    assert M % 16 == 0 and K % 32 == 0 and ps % 32 == 0
    NP, NT, MT = (K + ps - 1) // ps, (N + 31) // 32, M // 16
    key = (M, N, K, ps)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"leap_tree_att_nn_{M}_{N}_{K}_{ps}", input_names=["x", "v"], output_names=["cs"], source=_NN_SRC,
            header=LK._nax_header() + "\n" + LK.consts(GM=M, GN=N, GK=K, GPS=ps, GNP=NP, GNT=NT, GMT=MT))
    (cs,) = _K[key](inputs=[a, v], template=[("T", BF)], grid=(32 * NT * MT * NP, 1, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(NP, M, N)], output_dtypes=[mx.float32])
    return cs, NP


def pv_replica(a, v, ps):
    """MLX's split-K NAX nn GEMM a @ v, replicated: partitions of ps, fp32 partials summed in order, cast to bf16"""
    M, N = a.shape[0], v.shape[1]
    cs, NP = nn_partials(a, v, ps)
    (y,) = _ACC(inputs=[cs], template=[("T", BF), ("GMN", M * N), ("NP", NP)], grid=(M * N, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(M, N)], output_dtypes=[BF])
    return y
