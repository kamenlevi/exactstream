"""QMM32 (Leap, Fri 2026-11-06 09:1x; TABLE_350 'D17-NAX'): qmm16c (qmm16.py v3, code MMA) for 17-32 rows. Each simdgroup decodes
its weight fragment once per K group and runs the 16 x 32 x 16 MMA twice, for rows 0-15 and 16-31, so the weights are read
once (D17-NAX's two halves read them twice and concatenate). Per-row arithmetic and order are qmm16c's, so rows 0-15 ==
qmm16c(x[:16]) and rows 16-31 == qmm16c(x[16:32]) bitwise. KS splits K as qmm16c (partials 2x: (KS - 1) x 4 KB per NT).
qmm32c(x, w, scales, biases, ks=1, nt=1): x [M <= 32, K] bf16 -> y [M, N]. New module: nothing live imports it until a
caller is switched."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from qmm16 import _HDR

_SRC32 = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int n0 = (threadgroup_position_in_grid.x * NT + nt) * 32;
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_a2 = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_c2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc[16], acc2[16];
    for (short i = 0; i < 16; i++) { acc[i] = 0.0f; acc2[i] = 0.0f; }

    constexpr int KW = K_C / 8, KG = K_C / G;
    const int kper = K_C / KS;
    const device uint32_t* wr[4];                             // B fragment rows (as v1)
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * KW + 2 * fq;
    // the 8 output columns of this lane: block bk (0, 1), column j: n0 + 16 bk + 4 fq + j
    const device T* sc_ = scales + (size_t)(n0 + 4 * fq) * KG;
    const device T* bi_ = biases + (size_t)(n0 + 4 * fq) * KG;
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;
    const device T* xr2 = x + (size_t)(fm + 16) * K_C + 16 * fq;
    const device T* xr3 = x + (size_t)(fm + 24) * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        uint32_t wv[4][2];
        for (short r = 0; r < 4; r++) { wv[r][0] = wr[r][k0 / 8]; wv[r][1] = wr[r][k0 / 8 + 1]; }
        for (short i = 0; i < 16; i++) { ct_c[i] = 0.0f; ct_c2[i] = 0.0f; }
        float xs0 = 0.0f, xs1 = 0.0f, xs2 = 0.0f, xs3 = 0.0f;
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        const device T* x2 = xr2 + k0;
        const device T* x3 = xr3 + k0;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) {
                const T a0 = x0[4 * s + j], a1 = x1[4 * s + j], a2 = x2[4 * s + j], a3 = x3[4 * s + j];
                ct_a[j] = a0; ct_a[4 + j] = a1; ct_a2[j] = a2; ct_a2[4 + j] = a3;
                xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
                xs2 += static_cast<float>(a2); xs3 += static_cast<float>(a3);
            }
            for (short r = 0; r < 4; r++) {
                const uint32_t word = wv[r][s >> 1];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++)
                    ct_b[base + j] = as_type<T>((ushort)(0x4300u | ((word >> (16 * (s & 1) + 4 * j)) & 0xFu)));
            }
            op.run(ct_a, ct_b, ct_c);
            op.run(ct_a2, ct_b, ct_c2);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        xs2 += simd_shuffle_xor(xs2, 1); xs2 += simd_shuffle_xor(xs2, 8);
        xs3 += simd_shuffle_xor(xs3, 1); xs3 += simd_shuffle_xor(xs3, 8);
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const size_t o = (size_t)(16 * bk + j) * KG + g;
                const float s_ = static_cast<float>(sc_[o]);
                const float b_ = static_cast<float>(bi_[o]) - 128.0f * s_;
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;   // rows fm and fm + 8
                acc[i0] = acc[i0] + (s_ * ct_c[i0] + b_ * xs0);
                acc[i1] = acc[i1] + (s_ * ct_c[i1] + b_ * xs1);
                acc2[i0] = acc2[i0] + (s_ * ct_c2[i0] + b_ * xs2);
                acc2[i1] = acc2[i1] + (s_ * ct_c2[i1] + b_ * xs3);
            }
    }

    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 32 * 32];
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) {
            part[(((kg - 1) * NT + nt) * 32 + i) * 32 + lane] = acc[i];
            part[(((kg - 1) * NT + nt) * 32 + 16 + i) * 32 + lane] = acc2[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) {
                acc[i] = acc[i] + part[((gg * NT + nt) * 32 + i) * 32 + lane];
                acc2[i] = acc2[i] + part[((gg * NT + nt) * 32 + 16 + i) * 32 + lane];
            }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
        y[(size_t)(m + 16) * N_C + n] = static_cast<T>(acc2[i]);
    }
"""


@lru_cache(None)
def _k32():
    return mx.fast.metal_kernel(name="leap_qmm32_codemma", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRC32, header=_HDR)


def qmm32c(x, w, scales, biases, ks=1, nt=1):
    M, K = x.shape
    N = w.shape[0]
    assert M <= 32 and N % (32 * nt) == 0 and K % (64 * ks) == 0 and (ks - 1) * nt * 4 <= 20
    xp = x if M == 32 else mx.concatenate([x, mx.zeros((32 - M, K), x.dtype)], 0)
    (y,) = _k32()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt)],
                  grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(32, N)],
                  output_dtypes=[x.dtype])
    return y[:M]
