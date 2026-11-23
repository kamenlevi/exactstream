"""bof.py (Leap, Sun 15 Nov 2026 17:5x; CHECK55 16-row lever (b), track A = REF2-T's own bits): qmm16c at ks 1 with its
K loop spread over BS simdgroups, BLOCKED ORDERED FOLD.
qmm16c (ks 1) folds every 64-weight group's term into one fp32 accumulator in group order:
    acc = acc + (s_g * P'_g + b'_g * X_g),   g = 0, 1, ..., K / 64 - 1
which leaves a narrow call K-serial (CB1: o_proj / out_proj 87 us at ks 1 vs 44 at ks 4). A plain K split changes that
fold (it is the contract's ks). Here a threadgroup per 32-column tile has BS simdgroups; each block of BS x RG groups is
computed in parallel (the same MMA, row sums and term expression per group as qmm16c), the terms land in threadgroup
memory, and simdgroup 0 folds them in group order before the next block. The fold is qmm16c's, so the bits should be
REF2-T's, provided the term compiles to the same arithmetic: TERM picks how (s * P + b * X) is written (0: as qmm16c
writes it; 1: fma(s, P, b * X); 2: fma(b, X, s * P); 3: no contraction). bof_bench.py tests which one matches.
qmm16bof(x, w, scales, biases, bs=8, rg=1, term=0): x [M <= 16, K] (padded to 16), MLX's 4-bit g64 layout."""
import os
import sys
from functools import lru_cache
import mlx.core as mx

_PS = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts")
if _PS not in sys.path:                              # appended: no shadowing of later imports (BigMoE 01:1x)
    sys.path.append(_PS)
import qmm16 as Q16

_TERMS = {0: "#define TERM(s, c, b, x) ((s) * (c) + (b) * (x))\n",
          1: "#define TERM(s, c, b, x) fma((s), (c), (b) * (x))\n",
          2: "#define TERM(s, c, b, x) fma((b), (x), (s) * (c))\n",
          3: "#pragma clang fp contract(off)\n#define TERM(s, c, b, x) ((s) * (c) + (b) * (x))\n"}

_SRCB = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int n0 = threadgroup_position_in_grid.x * 32;
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
    float acc[16];
    for (short i = 0; i < 16; i++) acc[i] = 0.0f;

    constexpr int KW = K_C / 8, KG = K_C / G;
    const device uint32_t* wr[4];
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * KW + 2 * fq;
    const device T* sc_ = scales + (size_t)(n0 + 4 * fq) * KG;
    const device T* bi_ = biases + (size_t)(n0 + 4 * fq) * KG;
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;

    threadgroup float part[BS * RG * 16 * 32];
    for (int gb = 0; gb < KG; gb += BS * RG) {
        for (short rg = 0; rg < RG; rg++) {
            const int g = gb + (int)sg * RG + rg;
            if (g < KG) {
                const int k0 = g * G;
                uint32_t wv[4][2];
                for (short r = 0; r < 4; r++) { wv[r][0] = wr[r][k0 / 8]; wv[r][1] = wr[r][k0 / 8 + 1]; }
                for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
                float xs0 = 0.0f, xs1 = 0.0f;
                const device T* x0 = xr0 + k0;
                const device T* x1 = xr1 + k0;
                for (short s = 0; s < 4; s++) {
                    for (short j = 0; j < 4; j++) {
                        const T a0 = x0[4 * s + j], a1 = x1[4 * s + j];
                        ct_a[j] = a0; ct_a[4 + j] = a1;
                        xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
                    }
                    for (short r = 0; r < 4; r++) {
                        const uint32_t word = wv[r][s >> 1];
                        const short base = (r >> 1) * 8 + (r & 1) * 4;
                        for (short j = 0; j < 4; j++)
                            ct_b[base + j] = as_type<T>((ushort)(0x4300u | ((word >> (16 * (s & 1) + 4 * j)) & 0xFu)));
                    }
                    op.run(ct_a, ct_b, ct_c);
                }
                xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
                xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
                threadgroup float* pp = part + (size_t)(((int)sg * RG + rg) * 16) * 32 + lane;
                for (short bk = 0; bk < 2; bk++)
                    for (short j = 0; j < 4; j++) {
                        const size_t o = (size_t)(16 * bk + j) * KG + g;
                        const float s_ = static_cast<float>(sc_[o]);
                        const float b_ = static_cast<float>(bi_[o]) - 128.0f * s_;
                        const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;   // rows fm and fm + 8
                        pp[i0 * 32] = TERM(s_, ct_c[i0], b_, xs0);
                        pp[i1 * 32] = TERM(s_, ct_c[i1], b_, xs1);
                    }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const int nb = min(BS * RG, KG - gb);
            for (int j = 0; j < nb; j++)                    // qmm16c's fold, in group order
                for (short i = 0; i < 16; i++) acc[i] = acc[i] + part[(j * 16 + i) * 32 + lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg != 0) return;
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
    }
"""


@lru_cache(None)
def _kb(term):
    return mx.fast.metal_kernel(name=f"leap_bof_t{term}", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRCB, header=Q16._HDR + "\n" + _TERMS[term])


def qmm16bof(x, w, scales, biases, bs=8, rg=1, term=0):
    """qmm16c (ks 1) bits with the K loop over bs simdgroups (blocks of bs x rg groups, folded in group order)"""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % 32 == 0 and K % 64 == 0 and bs * rg <= 16
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _kb(term)(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("BS", bs), ("RG", rg)],
                     grid=(N // 32 * 32 * bs, 1, 1), threadgroup=(32 * bs, 1, 1), output_shapes=[(16, N)],
                     output_dtypes=[x.dtype])
    return y[:M]


# ---- 32 rows, one weight pass (Sun 18:1x): each simdgroup's group computes BOTH 16-row tiles' terms from one decoded
# weight fragment (as one32.qmm16c2a), and simdgroup 0 folds each tile in group order: per tile the bits of qmm16c ks 1,
# so of REF2-T's 32-row forms (qmm16c2a / qmm16c2 at ks 1). Threadgroup memory BS x RG x 2 tiles x 2 KB (<= 32 KB).
_SRCB2 = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int n0 = threadgroup_position_in_grid.x * 32;
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_a2 = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc[16], acc2[16];
    for (short i = 0; i < 16; i++) { acc[i] = 0.0f; acc2[i] = 0.0f; }

    constexpr int KW = K_C / 8, KG = K_C / G;
    const device uint32_t* wr[4];
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * KW + 2 * fq;
    const device T* sc_ = scales + (size_t)(n0 + 4 * fq) * KG;
    const device T* bi_ = biases + (size_t)(n0 + 4 * fq) * KG;
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;
    const device T* xr2 = x + (size_t)(16 + fm) * K_C + 16 * fq;
    const device T* xr3 = x + (size_t)(24 + fm) * K_C + 16 * fq;

    threadgroup float part[BS * RG * 2 * 16 * 32];
    for (int gb = 0; gb < KG; gb += BS * RG) {
        for (short rg = 0; rg < RG; rg++) {
            const int g = gb + (int)sg * RG + rg;
            if (g < KG) {
                const int k0 = g * G;
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
                        const T a0 = x0[4 * s + j], a1 = x1[4 * s + j];
                        const T a2 = x2[4 * s + j], a3 = x3[4 * s + j];
                        ct_a[j] = a0; ct_a[4 + j] = a1;
                        ct_a2[j] = a2; ct_a2[4 + j] = a3;
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
                threadgroup float* pp = part + (size_t)((((int)sg * RG + rg) * 2) * 16) * 32 + lane;
                threadgroup float* pq = pp + 16 * 32;
                for (short bk = 0; bk < 2; bk++)
                    for (short j = 0; j < 4; j++) {
                        const size_t o = (size_t)(16 * bk + j) * KG + g;
                        const float s_ = static_cast<float>(sc_[o]);
                        const float b_ = static_cast<float>(bi_[o]) - 128.0f * s_;
                        const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                        pp[i0 * 32] = TERM(s_, ct_c[i0], b_, xs0);
                        pp[i1 * 32] = TERM(s_, ct_c[i1], b_, xs1);
                        pq[i0 * 32] = TERM(s_, ct_c2[i0], b_, xs2);
                        pq[i1 * 32] = TERM(s_, ct_c2[i1], b_, xs3);
                    }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const int nb = min(BS * RG, KG - gb);
            for (int j = 0; j < nb; j++)
                for (short i = 0; i < 16; i++) {
                    acc[i] = acc[i] + part[((j * 2) * 16 + i) * 32 + lane];
                    acc2[i] = acc2[i] + part[((j * 2 + 1) * 16 + i) * 32 + lane];
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg != 0) return;
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
        y[(size_t)(16 + m) * N_C + n] = static_cast<T>(acc2[i]);
    }
"""


@lru_cache(None)
def _kb2(term):
    return mx.fast.metal_kernel(name=f"leap_bof2_t{term}", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRCB2, header=Q16._HDR + "\n" + _TERMS[term])


def qmm16bof2(x, w, scales, biases, bs=4, rg=1, term=0):
    """17-32 rows, one weight pass: each 16-row tile's bits == qmm16c ks 1 (so == one32.qmm16c2a / qmm16c2 at ks 1)"""
    M, K = x.shape
    N = w.shape[0]
    assert 16 < M <= 32 and N % 32 == 0 and K % 64 == 0 and bs * rg <= 8
    xp = x if M == 32 else mx.concatenate([x, mx.zeros((32 - M, K), x.dtype)], 0)
    (y,) = _kb2(term)(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("BS", bs), ("RG", rg)],
                      grid=(N // 32 * 32 * bs, 1, 1), threadgroup=(32 * bs, 1, 1), output_shapes=[(32, N)],
                      output_dtypes=[x.dtype])
    return y[:M]
