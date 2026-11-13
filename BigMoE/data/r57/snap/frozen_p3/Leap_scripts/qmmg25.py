"""L1-25 (Leap, Sat 2026-10-24): routed experts of a verify check on the M5's tensor units. L1-24 showed the expert path is
ALU-bound (MLX's single-row arithmetic, ~7 us per routed pair), not bandwidth-bound. Here each distinct expert's rows
(<= 16 per check) run as one 16-row NAX tile: qmm16c's code-MMA kernel (qmm16.py, L1-15 / L1-23) with a segment
prologue (expert weights, gathered token rows) and masked stores. Every output row depends only on its own input row and
the expert's weights through a fixed tile (padding rows are copies of a real row and never stored), so the result is
batch-invariant by construction; its rounding differs from MLX's qmv (more accurate: fp32-exact dequant), so adopting it
is a one-time reference change for decode and verify alike.
experts_nax(x [R, K], gu (w, s, b), dn (w, s, b), ids [R, k], cfg_a=(ks, nt), cfg_b=(ks, nt)) -> [R, k, D]"""
import os, sys
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
from qmm16 import _HDR
from gmr24 import segments

_SRC = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int n0 = (threadgroup_position_in_grid.x * NT + nt) * 32;
    const int s = threadgroup_position_in_grid.y;
    const int p0 = seg_start[s];
    if (p0 >= P) return;                                      // whole threadgroup: past the last segment
    const int cnt = seg_start[s + 1] - p0;
    const int e = sexp[p0];
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
    const device uint32_t* we = w + (size_t)e * N_C * KW;
    const device T* se = scales + (size_t)e * N_C * KG;
    const device T* be = biases + (size_t)e * N_C * KG;
    const device uint32_t* wr[4];
    for (short r = 0; r < 4; r++) wr[r] = we + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * KW + 2 * fq;
    const device T* sc_ = se + (size_t)(n0 + 4 * fq) * KG;
    const device T* bi_ = be + (size_t)(n0 + 4 * fq) * KG;
    // tile rows fm and fm + 8 = the segment's rows; rows past cnt copy row 0 (computed, never stored)
    const int r0 = prow[p0 + (fm < cnt ? fm : 0)];
    const int r1 = prow[p0 + (fm + 8 < cnt ? fm + 8 : 0)];
    const device T* xr0 = x + (size_t)r0 * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)r1 * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        uint32_t wv[4][2];
        for (short r = 0; r < 4; r++) { wv[r][0] = wr[r][k0 / 8]; wv[r][1] = wr[r][k0 / 8 + 1]; }
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        float xs0 = 0.0f, xs1 = 0.0f;
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        for (short st = 0; st < 4; st++) {
            for (short j = 0; j < 4; j++) {
                const T a0 = x0[4 * st + j], a1 = x1[4 * st + j];
                ct_a[j] = a0; ct_a[4 + j] = a1;
                xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
            }
            for (short r = 0; r < 4; r++) {
                const uint32_t word = wv[r][st >> 1];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++)
                    ct_b[base + j] = as_type<T>((ushort)(0x4300u | ((word >> (16 * (st & 1) + 4 * j)) & 0xFu)));
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
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
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
        if (m < cnt) y[(size_t)(p0 + m) * N_C + n] = static_cast<T>(acc[i]);
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="leap_qmmg25", input_names=["x", "w", "scales", "biases", "prow", "sexp", "seg_start"],
                                output_names=["y"], source=_SRC, header=_HDR)


def qmmg(x, w, scales, biases, prow, sexp, seg, P, ks=1, nt=1):
    """y [P, N]: row p = x[prow[p]] @ dequant(w[sexp[p]])^T, one 16-row tile per expert segment"""
    K = x.shape[-1]
    N = w.shape[1]
    assert N % (32 * nt) == 0 and K % (64 * ks) == 0 and w.shape[2] * 8 == K
    (y,) = _k()(inputs=[x, w, scales, biases, prow, sexp, seg],
                template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", P)],
                grid=(N // 32 * 32 * ks, P, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(P, N)],
                output_dtypes=[x.dtype], init_value=0)
    return y


def experts_nax(x, gu, dn, ids, cfg_a=(1, 1), cfg_b=(1, 1)):
    R = x.shape[0]
    order, sexp, prow, seg, P = segments(ids)
    g = qmmg(x, *gu, prow, sexp, seg, P, *cfg_a)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    y = qmmg(h, *dn, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)
