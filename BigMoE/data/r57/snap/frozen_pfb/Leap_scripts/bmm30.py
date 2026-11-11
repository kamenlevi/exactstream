"""L1-29 (Leap, Sun 2026-10-25): a bf16 16-row tensor-unit matmul for the 125B's hyper-connection mixers in a 16-row check.
L1-23: MLX's best batch-invariant 16-row forms take 5.32 ms per check for HC mix down (320 x 10240, x97) vs a 2.27 ms
floor, and 2.66 for HC mix up (10240 x 320, x97; floor 2.27). This is qmm16c's MMA skeleton with bf16 weights read
directly (no dequant): lane (fm, fq) holds tile rows fm / fm + 8 and, per 64-value K group, the 16 consecutive values
16 fq .. 16 fq + 15 of its 4 weight rows (the same K permutation for x and w, so the product is unchanged); KS
simdgroups split K and are summed in a fixed order; one fixed tile for every row count (decode pads), so batch-invariant
by construction. New rounding vs MLX (fp32 accumulation in the tensor unit's order).
bmm16(x [M <= 16, K], w [N, K] bf16, ks, nt) -> [M, N]"""
from functools import lru_cache
import mlx.core as mx

_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_SRC = r"""
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
    for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
    const int kper = K_C / KS;
    const device T* wr[4];
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * K_C + 16 * fq;
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;
    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        for (short st = 0; st < 4; st++) {
            for (short j = 0; j < 4; j++) {
                ct_a[j] = x0[4 * st + j];
                ct_a[4 + j] = x1[4 * st + j];
            }
            for (short r = 0; r < 4; r++) {
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                const device T* wp = wr[r] + k0 + 4 * st;
                for (short j = 0; j < 4; j++) ct_b[base + j] = wp[j];
            }
            op.run(ct_a, ct_b, ct_c);
        }
    }
    float acc[16];
    for (short i = 0; i < 16; i++) acc[i] = ct_c[i];
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
def _k():
    return mx.fast.metal_kernel(name="leap_bmm16", input_names=["x", "w"], output_names=["y"], source=_SRC, header=_HDR)


def bmm16(x, w, ks=1, nt=1):
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and N % (32 * nt) == 0 and K % (64 * ks) == 0 and w.shape[1] == K
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k()(inputs=[xp, w], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt)],
                grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(16, N)],
                output_dtypes=[x.dtype])
    return y[:M]
