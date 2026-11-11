"""L1-15 (Leap, Sat 2026-10-24): a 16-row 4-bit affine (g64) matmul on the M5's neural accelerators (Metal 4 tensor ops,
MPP matmul2d 16 x 32 x 16), y [16, N] = x [16, K] @ dequant(w [N, K])^T.
One fixed kernel for every row count: decode pads its 1 row to 16, so each row's bits never depend on how many rows run
(batch-invariant by construction, "exact under its own numerics"). The 16 x 16 fragment layout and the op descriptor are
MLX's own (steel/gemm/nax.h BaseNAXFrag + mma, mlx 0.32.4): lane (fm, fn) holds rows fm, fm + 8 and columns fn .. fn + 3.
Per 64-value quantization group the K order is permuted (the same permutation for x and w, so the product is unchanged):
k-step s, fragment column c -> offset 16 * (c / 4) + 4 * s + c % 4, so each lane reads 16 consecutive codes (2 words)
of each of its weight rows. KS simdgroups split K for one 16 x 32 tile and are summed in a fixed order (group 0 + 1 + ...).
Dequantization: bf16(scale * code + bias) in fp32 arithmetic; accumulation fp32; output one bf16 rounding."""
from functools import lru_cache
import mlx.core as mx

_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_SRC = r"""
    constexpr int G = 64;                                     // quantization group = one K chunk
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;      // K group of this simdgroup
    const uint nt = simdgroup_index_in_threadgroup % NT;      // n tile within the threadgroup
    const int n0 = (threadgroup_position_in_grid.x * NT + nt) * 32;   // this tile's 32 output columns
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));         // fragment row 0..7 (and fm + 8)
    const short fq = ((qid & 2) | (lane & 1));                // fragment column block: columns 4 fq .. 4 fq + 3

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;

    constexpr int KW = K_C / 8, KG = K_C / G;                 // uint32 words and groups per weight row
    const int kper = K_C / KS;
    // this lane's 4 weight rows: fragment 0 rows fm, fm + 8; fragment 1 rows 16 + fm, 24 + fm
    const device uint32_t* wr[4];
    const device T* sr[4];
    const device T* br[4];
    for (short r = 0; r < 4; r++) {
        const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
        wr[r] = w + (size_t)n * KW;
        sr[r] = scales + (size_t)n * KG;
        br[r] = biases + (size_t)n * KG;
    }
    const device T* xr0 = x + (size_t)fm * K_C;               // x rows fm and fm + 8
    const device T* xr1 = x + (size_t)(fm + 8) * K_C;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        uint32_t wv[4][2];
        float sc[4], bi[4];
        for (short r = 0; r < 4; r++) {
            const device uint32_t* p = wr[r] + (k0 + 16 * fq) / 8;
            wv[r][0] = p[0]; wv[r][1] = p[1];
            sc[r] = (float)sr[r][g]; bi[r] = (float)br[r][g];
        }
        const device T* x0 = xr0 + k0 + 16 * fq;
        const device T* x1 = xr1 + k0 + 16 * fq;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) {
                ct_a[j] = x0[4 * s + j];                      // A rows fm (elements 0..3) and fm + 8 (4..7)
                ct_a[4 + j] = x1[4 * s + j];
            }
            for (short r = 0; r < 4; r++) {
                const uint32_t word = wv[r][s >> 1];
                const short base = (r >> 1) * 8 + (r & 1) * 4; // B fragment (r >> 1), its row half (r & 1)
                for (short j = 0; j < 4; j++) {
                    const float code = (float)((word >> (16 * (s & 1) + 4 * j)) & 0xFu);
                    ct_b[base + j] = static_cast<T>(sc[r] * code + bi[r]);
                }
            }
            op.run(ct_a, ct_b, ct_c);
        }
    }

    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = ct_c[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) ct_c[i] = ct_c[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
    }
    // destination element i: fragment i / 8 (columns 0-15 / 16-31), row fm + 8 * ((i % 8) / 4), column 4 fq + i % 4
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(ct_c[i]);
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="leap_qmm16_nax", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRC, header=_HDR)


def qmm16(x, w, scales, biases, ks=2, nt=1):
    """x [M <= 16, K] bf16 (padded to 16 rows here), w [N, K / 8] uint32, scales / biases [N, K / 64] -> y [M, N].
    ks: K split per tile (summed in a fixed order: part of the numerics); nt: n tiles per threadgroup (speed only)."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % (32 * nt) == 0 and K % (64 * ks) == 0
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt)],
                grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(16, N)],
                output_dtypes=[x.dtype])
    return y[:M]


# ---- v2: NB blocks of 32 output columns per simdgroup (the x fragment is loaded once for all of them: half the x
# reads per weight byte at NB = 2), and the next group's weight words are loaded before the current group's ops run.
# Same per-output arithmetic order as v1 (same permutation, same fp32 accumulation), so v1 and v2 agree bitwise at KS 1.
_SRC2 = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup;
    const int n0 = threadgroup_position_in_grid.x * (32 * NB);
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    using ct_a_t = metal::remove_addrspace_t<decltype(ct_a)>;
    using ct_b_t = metal::remove_addrspace_t<decltype(ct_b)>;
    auto ct_c0 = op.template get_destination_cooperative_tensor<ct_a_t, ct_b_t, float>();
    auto ct_c1 = op.template get_destination_cooperative_tensor<ct_a_t, ct_b_t, float>();
    for (short i = 0; i < 16; i++) { ct_c0[i] = 0.0f; ct_c1[i] = 0.0f; }

    constexpr int KW = K_C / 8, KG = K_C / G, NR = 4 * NB;
    const int kper = K_C / KS;
    const device uint32_t* wr[NR];
    const device T* sr[NR];
    const device T* br[NR];
    for (short r = 0; r < NR; r++) {                          // row r: block r / 4, fragment (r / 2) % 2, half r % 2
        const int n = n0 + (r >> 2) * 32 + ((r >> 1) & 1) * 16 + fm + (r & 1) * 8;
        wr[r] = w + (size_t)n * KW + 2 * fq;
        sr[r] = scales + (size_t)n * KG;
        br[r] = biases + (size_t)n * KG;
    }
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;

    const int kb = kg * kper, ke = (kg + 1) * kper;
    uint32_t wn[NR][2];
    T sn[NR], bn[NR];
    for (short r = 0; r < NR; r++) {
        wn[r][0] = wr[r][kb / 8]; wn[r][1] = wr[r][kb / 8 + 1];
        sn[r] = sr[r][kb / G]; bn[r] = br[r][kb / G];
    }
    for (int k0 = kb; k0 < ke; k0 += G) {
        uint32_t wv[NR][2];
        float sc[NR], bi[NR];
        for (short r = 0; r < NR; r++) {
            wv[r][0] = wn[r][0]; wv[r][1] = wn[r][1]; sc[r] = (float)sn[r]; bi[r] = (float)bn[r];
        }
        if (k0 + G < ke) {                                    // prefetch the next group
            for (short r = 0; r < NR; r++) {
                wn[r][0] = wr[r][(k0 + G) / 8]; wn[r][1] = wr[r][(k0 + G) / 8 + 1];
                sn[r] = sr[r][(k0 + G) / G]; bn[r] = br[r][(k0 + G) / G];
            }
        }
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = x0[4 * s + j]; ct_a[4 + j] = x1[4 * s + j]; }
            for (short blk = 0; blk < NB; blk++) {
                for (short r4 = 0; r4 < 4; r4++) {
                    const short r = blk * 4 + r4;
                    const uint32_t word = wv[r][s >> 1];
                    const short base = (r4 >> 1) * 8 + (r4 & 1) * 4;
                    for (short j = 0; j < 4; j++) {
                        const float code = (float)((word >> (16 * (s & 1) + 4 * j)) & 0xFu);
                        ct_b[base + j] = static_cast<T>(sc[r] * code + bi[r]);
                    }
                }
                if (blk == 0) op.run(ct_a, ct_b, ct_c0); else op.run(ct_a, ct_b, ct_c1);
            }
        }
    }

    threadgroup float part[(KS > 1 ? KS - 1 : 1) * 32 * 32];
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) {
            part[((kg - 1) * 32 + i) * 32 + lane] = ct_c0[i];
            if (NB > 1) part[((kg - 1) * 32 + 16 + i) * 32 + lane] = ct_c1[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) {
                ct_c0[i] = ct_c0[i] + part[(gg * 32 + i) * 32 + lane];
                if (NB > 1) ct_c1[i] = ct_c1[i] + part[(gg * 32 + 16 + i) * 32 + lane];
            }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(ct_c0[i]);
        if (NB > 1) y[(size_t)m * N_C + n + 32] = static_cast<T>(ct_c1[i]);
    }
"""


@lru_cache(None)
def _k2():
    return mx.fast.metal_kernel(name="leap_qmm16_nax_v2", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRC2, header=_HDR)


def qmm16b(x, w, scales, biases, ks=2, nb=2):
    """v2 of qmm16: nb blocks of 32 columns per simdgroup, prefetched weight words"""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % (32 * nb) == 0 and K % (64 * ks) == 0 and nb in (1, 2)
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k2()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NB", nb)],
                 grid=(N // (32 * nb) * 32 * ks, 1, 1), threadgroup=(32 * ks, 1, 1), output_shapes=[(16, N)],
                 output_dtypes=[x.dtype])
    return y[:M]


# ---- v3 "code MMA": the neural accelerator multiplies x by the raw codes (as bf16 128 + c, built with bit operations, exact),
# and the affine dequantization is applied once per output and 64-value group instead of once per weight:
#   y[m, n] = sum_g ( s[n, g] * P'[m, n, g] + (b[n, g] - 128 s[n, g]) * X[m, g] ),
#   P'[m, n, g] = sum_{k in g} x[m, k] (128 + c[n, k]) (fp32 accumulate),  X[m, g] = sum_{k in g} x[m, k] (fp32).
# ~3x less ALU per weight than v1, and no bf16 rounding of dequantized weights. X[m, g] is summed in-kernel in a fixed
# order (each lane its 16 values in order, then xor-shuffles over the 4 lanes of the row: 1, then 8).
_SRC3 = r"""
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
    float acc[16];
    for (short i = 0; i < 16; i++) acc[i] = 0.0f;

    constexpr int KW = K_C / 8, KG = K_C / G;
    const int kper = K_C / KS;
    const device uint32_t* wr[4];                             // B fragment rows (as v1)
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * KW + 2 * fq;
    // the 8 output columns of this lane: block bk (0, 1), column j: n0 + 16 bk + 4 fq + j
    const device T* sc_ = scales + (size_t)(n0 + 4 * fq) * KG;
    const device T* bi_ = biases + (size_t)(n0 + 4 * fq) * KG;
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
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
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const size_t o = (size_t)(16 * bk + j) * KG + g;
                const float s_ = static_cast<float>(sc_[o]);
                const float b_ = static_cast<float>(bi_[o]) - 128.0f * s_;
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;   // rows fm and fm + 8
                acc[i0] = acc[i0] + (s_ * ct_c[i0] + b_ * xs0);
                acc[i1] = acc[i1] + (s_ * ct_c[i1] + b_ * xs1);
            }
    }

    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) acc[i] = acc[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
    }
"""


@lru_cache(None)
def _k3():
    return mx.fast.metal_kernel(name="leap_qmm16_codemma", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRC3, header=_HDR)


def qmm16c(x, w, scales, biases, ks=1, nt=1):
    """v3 (code MMA) of qmm16: same interface; its own numerics (more accurate: no bf16 rounding of weights)"""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % (32 * nt) == 0 and K % (64 * ks) == 0
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k3()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt)],
                 grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(16, N)],
                 output_dtypes=[x.dtype])
    return y[:M]
