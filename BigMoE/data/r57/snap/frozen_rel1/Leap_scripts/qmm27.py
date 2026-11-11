> (a quote from the project owner, kept private)
4-bit. The installed SDK's matmul2d takes bfloat x uint4b_format -> float (MPPTensorOpsMatMul2d.h); MLX's affine codes
are unsigned 0..15, packed two per byte low nibble first. So per 64-value group: one matmul2d of the x rows (bf16,
tensor_inline) with the expert's packed codes (uint4b_format, tensor_inline over the raw buffer) into a float cooperative
tensor (sum of x * q), then acc += scale[n] * part + bias[n] * rowsum[m] per element (coordinates from
get_multidimensional_index). No software unpacking at all (L1-24 / L1-26: the ALU-bound part).
Numerics: the tensor unit's own accumulation inside each group (new rounding, like qmm16c's; fixed tiles, so
batch-invariant by construction: verified in test_qmm27.py).
experts_int4(x, gu, dn, ids) as qmmg25.experts_nax; qmm16i(x, w, scales, biases) dense (16 padded rows)."""
import os, sys
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
from gmr24 import segments

_HDR = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

# one simdgroup = one 16-row x 32-column tile of one segment (expert); xg holds the segment rows contiguously (pairs in
# expert order, padded by 16 rows); xs [P + 16, K / 64] float = per-row group sums of xg.
_SRC = r"""
    using namespace metal;
    constexpr int G = 64;
    const int t = threadgroup_position_in_grid.x;
    const int n0 = t * 32;
#if GATHER
    const int s = threadgroup_position_in_grid.y;
    const int p0 = seg_start[s];
    if (p0 >= P) return;
    const int cnt = seg_start[s + 1] - p0;
    const int e = sexp[p0];
#else
    const int p0 = 0, cnt = MROWS, e = 0;
#endif
    constexpr int KG = K_C / G;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, G, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tX((device bfloat*)(xg + (size_t)p0 * K_C),
                                                                dextents<int32_t, 2>(K_C, 16));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tW(
        (device uint8_t*)(w + (size_t)e * N_C * (K_C / 8)), dextents<int32_t, 2>(K_C, N_C));
    auto a0 = tX.slice<G, 16>(0, 0);
    auto b0 = tW.slice<G, 32>(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto part = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
    const device T* se = scales + ((size_t)e * N_C + n0) * KG;
    const device T* be = biases + ((size_t)e * N_C + n0) * KG;
    const device float* xsp = xs + (size_t)p0 * KG;
    for (int g = 0; g < KG; g++) {
        #pragma unroll
        for (uint16_t i = 0; i < part.get_capacity(); ++i) if (part.is_valid_element(i)) part[i] = 0.0f;
        auto a = tX.slice<G, 16>(g * G, 0);
        auto b = tW.slice<G, 32>(g * G, n0);
        op.run(a, b, part);
        #pragma unroll
        for (uint16_t i = 0; i < part.get_capacity(); ++i) {
            if (part.is_valid_element(i)) {
                auto idx = part.get_multidimensional_index(i);         // (column n, row m) within the 16 x 32 tile
                const int n = idx[0], m = idx[1];
                const float s_ = static_cast<float>(se[(size_t)n * KG + g]);
                const float b_ = static_cast<float>(be[(size_t)n * KG + g]);
                acc[i] = acc[i] + (s_ * part[i] + b_ * xsp[(size_t)m * KG + g]);
            }
        }
    }
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto idx = acc.get_multidimensional_index(i);
            const int n = idx[0], m = idx[1];
            if (m < cnt) y[(size_t)(p0 + m) * N_C + n0 + n] = static_cast<T>(acc[i]);
        }
    }
"""


@lru_cache(None)
def _k(gather):
    names = ["xg", "xs", "w", "scales", "biases"] + (["sexp", "seg_start"] if gather else [])
    return mx.fast.metal_kernel(name=f"leap_qmm27_{'g' if gather else 'd'}", input_names=names, output_names=["y"],
                                source=_SRC.replace("#if GATHER", f"#if {int(gather)}"), header=_HDR)


def _rowsums(xg):
    P, K = xg.shape
    return xg.astype(mx.float32).reshape(P, K // 64, 64).sum(-1)


def qmm16i(x, w, scales, biases):
    """dense y [M, N] for M <= 16 (padded to 16 rows)"""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % 32 == 0 and K % 64 == 0
    xg = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k(False)(inputs=[xg, _rowsums(xg), w, scales, biases],
                     template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("MROWS", M), ("P", 16)],
                     grid=(N // 32 * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, N)], output_dtypes=[x.dtype])
    return y[:M]


def qmmgi(xg, w, scales, biases, sexp, seg, P):
    """xg [P + 16, K]: rows in pair (expert-sorted) order, 16 padding rows; y [P, N]"""
    K = xg.shape[-1]
    N = w.shape[1]
    (y,) = _k(True)(inputs=[xg, _rowsums(xg), w, scales, biases, sexp, seg],
                    template=[("T", xg.dtype), ("K_C", K), ("N_C", N), ("MROWS", 16), ("P", P)],
                    grid=(N // 32 * 32, P, 1), threadgroup=(32, 1, 1), output_shapes=[(P, N)], output_dtypes=[xg.dtype],
                    init_value=0)
    return y


def experts_int4(x, gu, dn, ids):
    R, K = x.shape
    order, sexp, prow, seg, P = segments(ids)
    xg = mx.concatenate([x[prow], mx.zeros((16, K), x.dtype)], 0)
    g = qmmgi(xg, *gu, sexp, seg, P)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    hg = mx.concatenate([h, mx.zeros((16, h.shape[-1]), h.dtype)], 0)
    y = qmmgi(hg, *dn, sexp, seg, P)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)
