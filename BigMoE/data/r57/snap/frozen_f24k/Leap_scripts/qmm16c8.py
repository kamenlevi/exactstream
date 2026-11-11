"""BP-5 (Leap, Wed 2026-11-04 16:2x; TABLE_350 'BP-5'): qmm16c8, an 8-bit affine (g64) 16-row matmul on the M5's
matrix units: qmm16.qmm16c's v3 "code MMA" (MPP matmul2d 16 x 32 x 16, fragment maps and K permutation as qmm16.py's
docstring) with byte codes. The B operand is bf16(code) (exact for 0..255); per output and 64-value group
y += s * P + b * X with P = sum x * code and X = sum x (fp32, X summed in qmm16c's fixed order). One fixed kernel for
every row count (rows padded to 16), so a row's bits never depend on the batch. KS simdgroups split K, summed in order.
For the drafter (exact default): bf16 projections at 16 rows run ~3.8-5.9 ms per propose (BP-2a / BP-3); MLX's own
8-bit quantized_matmul at 16 rows moves only ~95 GB/s, so an 8-bit drafter needs this kernel.
qmm16c8(x [M <= 16, K] bf16, w [N, K/4] uint32 (MLX 8-bit packing), scales / biases [N, K/64]) -> y [M, N]"""
from functools import lru_cache
import mlx.core as mx

_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_SRC8 = r"""
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

    constexpr int KW = K_C / 4, KG = K_C / G;
    const int kper = K_C / KS;
    const device uint32_t* wr[4];                             // B fragment rows; this lane's 16 codes of a group = 4 words
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * KW + 4 * fq;
    const device T* sc_ = scales + (size_t)(n0 + 4 * fq) * KG;
    const device T* bi_ = biases + (size_t)(n0 + 4 * fq) * KG;
    const device T* xr0 = x + (size_t)(fm < M_C ? fm : 0) * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8 < M_C ? fm + 8 : 0) * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        uint32_t wv[4][4];
        for (short r = 0; r < 4; r++) for (short s = 0; s < 4; s++) wv[r][s] = wr[r][k0 / 4 + s];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        float xs0 = 0.0f, xs1 = 0.0f;
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) {
                const T a0 = fm < M_C ? x0[4 * s + j] : static_cast<T>(0.0f);   // rows >= M_C read as zero (no padding pass)
                const T a1 = fm + 8 < M_C ? x1[4 * s + j] : static_cast<T>(0.0f);
                ct_a[j] = a0; ct_a[4 + j] = a1;
                xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
            }
            for (short r = 0; r < 4; r++) {
                const uint32_t word = wv[r][s];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++)
                    ct_b[base + j] = static_cast<T>(static_cast<float>((word >> (8 * j)) & 0xFFu));
            }
            op.run(ct_a, ct_b, ct_c);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const size_t o = (size_t)(16 * bk + j) * KG + g;
                const float s_ = static_cast<float>(sc_[o]);
                const float b_ = static_cast<float>(bi_[o]);
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
        if (m < M_C) y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
    }
"""


@lru_cache(None)
def _k8():
    return mx.fast.metal_kernel(name="leap_qmm16c8", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRC8, header=_HDR)


def qmm16c8(x, w, scales, biases, ks=1, nt=1, row_bound=False):
    """8-bit g64 code MMA at <= 16 rows: y [M, N]. Default: rows padded to 16 (one kernel variant per shape, as the
    engine's D8-NAX ran it on Wed). row_bound=True (BP-11b, opt-in): rows >= M read as zero inside the kernel (no padding
    pass, one variant per M); a row's bits are the same either way (the op's rows are independent)"""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % (32 * nt) == 0 and K % (64 * ks) == 0 and w.shape[1] * 4 == K, (x.shape, w.shape)
    if not row_bound:
        xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
        (y,) = _k8()(inputs=[xp, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks),
                                                               ("NT", nt), ("M_C", 16)],
                     grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(16, N)],
                     output_dtypes=[x.dtype])
        return y[:M]
    (y,) = _k8()(inputs=[x, w, scales, biases], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt),
                                                         ("M_C", M)],
                 grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(M, N)],
                 output_dtypes=[x.dtype])
    return y
