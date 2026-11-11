"""L1-28 (Leap, Sun 2026-10-25): the L1-26 native-layout kernel with TWO 32-column tiles per simdgroup. The x fragments, the
per-group row sums and the K loop are shared by both tiles (half the x loads and row-sum ALU per weight byte); each tile
keeps its own B fragments, cooperative accumulator and epilogue, with the same arithmetic per output as qmm16n / qmm16c,
so the output is meant to be bit-identical to them. Same native store (qmm26.repack).
qmm16n2(x, wn, sn, bn, N, ks, nt); experts_native2(x, gu_n, dn_n, ids, cfg_a, cfg_b) as qmm26."""
import os, sys
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
from qmm16 import _HDR
from gmr24 import segments

_SRC = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int t = (threadgroup_position_in_grid.x * NT + nt) * 2;    // first of this simdgroup's two 32-column tiles
    const int n0 = t * 32;
#if GATHER
    const int s = threadgroup_position_in_grid.y;
    const int p0 = seg_start[s];
    if (p0 >= P) return;
    const int cnt = seg_start[s + 1] - p0;
    const int e = sexp[p0];
#else
    const int p0 = 0, cnt = 16, e = 0;
#endif
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
    auto ct_d = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc[16], acd[16];
    for (short i = 0; i < 16; i++) { acc[i] = 0.0f; acd[i] = 0.0f; }

    constexpr int KG = K_C / G, NTILES = N_C / 32;
    const int kper = K_C / KS;
    // native weights: [e][t][g][lane][8 words]; scales / biases: [e][t][g][32]
    const device uint4* wq = (const device uint4*)(wn + ((((size_t)e * NTILES + t) * KG) * 32 + lane) * 8);
    const device uint4* wq2 = wq + (size_t)KG * 64;                  // the next tile's block
    const device T* sc_ = sn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const device T* bi_ = bn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const device T* sc2 = sc_ + (size_t)KG * 32;
    const device T* bi2 = bi_ + (size_t)KG * 32;
#if GATHER
    const int r0 = prow[p0 + (fm < cnt ? fm : 0)];
    const int r1 = prow[p0 + (fm + 8 < cnt ? fm + 8 : 0)];
#else
    const int r0 = fm, r1 = fm + 8;
#endif
    const device T* xr0 = x + (size_t)r0 * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)r1 * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        const uint4 q0 = wq[(size_t)g * 64];                          // words 0..3: (st 0, p 0) (st 0, p 1) (st 1, p 0) (st 1, p 1)
        const uint4 q1 = wq[(size_t)g * 64 + 1];                      // words 4..7: steps 2, 3
        const uint wv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint4 r0 = wq2[(size_t)g * 64];
        const uint4 r1 = wq2[(size_t)g * 64 + 1];
        const uint wv2[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        for (short i = 0; i < 16; i++) { ct_c[i] = 0.0f; ct_d[i] = 0.0f; }
        float xs0 = 0.0f, xs1 = 0.0f;
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        for (short st = 0; st < 4; st++) {
            for (short j = 0; j < 4; j++) {
                const T a0 = x0[4 * st + j], a1 = x1[4 * st + j];
                ct_a[j] = a0; ct_a[4 + j] = a1;
                xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
            }
            for (short p = 0; p < 2; p++) {
                const uint word = wv[st * 2 + p];
                for (short j = 0; j < 4; j++) {
                    const uint two = ((word >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                    ct_b[p * 8 + j] = as_type<T>((ushort)(two & 0xFFFFu));
                    ct_b[p * 8 + 4 + j] = as_type<T>((ushort)(two >> 16));
                }
            }
            op.run(ct_a, ct_b, ct_c);
            for (short p = 0; p < 2; p++) {
                const uint word = wv2[st * 2 + p];
                for (short j = 0; j < 4; j++) {
                    const uint two = ((word >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                    ct_b[p * 8 + j] = as_type<T>((ushort)(two & 0xFFFFu));
                    ct_b[p * 8 + 4 + j] = as_type<T>((ushort)(two >> 16));
                }
            }
            op.run(ct_a, ct_b, ct_d);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        const device T* scg = sc_ + (size_t)g * 32;
        const device T* big = bi_ + (size_t)g * 32;
        const device T* scg2 = sc2 + (size_t)g * 32;
        const device T* big2 = bi2 + (size_t)g * 32;
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                const float s_ = static_cast<float>(scg[16 * bk + j]);
                const float b_ = static_cast<float>(big[16 * bk + j]) - 128.0f * s_;
                acc[i0] = acc[i0] + (s_ * ct_c[i0] + b_ * xs0);
                acc[i1] = acc[i1] + (s_ * ct_c[i1] + b_ * xs1);
                const float s2 = static_cast<float>(scg2[16 * bk + j]);
                const float b2 = static_cast<float>(big2[16 * bk + j]) - 128.0f * s2;
                acd[i0] = acd[i0] + (s2 * ct_d[i0] + b2 * xs0);
                acd[i1] = acd[i1] + (s2 * ct_d[i1] + b2 * xs1);
            }
    }

    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 32 * 32];
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) {
            part[(((kg - 1) * NT + nt) * 32 + i) * 32 + lane] = acc[i];
            part[(((kg - 1) * NT + nt) * 32 + 16 + i) * 32 + lane] = acd[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) {
                acc[i] = acc[i] + part[((gg * NT + nt) * 32 + i) * 32 + lane];
                acd[i] = acd[i] + part[((gg * NT + nt) * 32 + 16 + i) * 32 + lane];
            }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        if (m < cnt) {
            y[(size_t)(p0 + m) * N_C + n] = static_cast<T>(acc[i]);
            y[(size_t)(p0 + m) * N_C + n + 32] = static_cast<T>(acd[i]);
        }
    }
"""


@lru_cache(None)
def _k(gather):
    names = ["x", "wn", "sn", "bn"] + (["prow", "sexp", "seg_start"] if gather else [])
    return mx.fast.metal_kernel(name=f"leap_qmm28_{'g' if gather else 'd'}", input_names=names, output_names=["y"],
                                source=_SRC.replace("#if GATHER", f"#if {int(gather)}"), header=_HDR)


def qmm16n2(x, wn, sn, bn, N, ks=1, nt=1):
    M, K = x.shape
    assert M <= 16 and N % (64 * nt) == 0 and K % (64 * ks) == 0
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k(False)(inputs=[xp, wn, sn, bn], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", 16)],
                     grid=(N // 64 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(16, N)],
                     output_dtypes=[x.dtype])
    return y[:M]


def qmmgn2(x, wn, sn, bn, N, prow, sexp, seg, P, ks=1, nt=1):
    K = x.shape[-1]
    assert N % (64 * nt) == 0 and K % (64 * ks) == 0
    (y,) = _k(True)(inputs=[x, wn, sn, bn, prow, sexp, seg],
                    template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", P)],
                    grid=(N // 64 * 32 * ks, P, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(P, N)],
                    output_dtypes=[x.dtype], init_value=0)
    return y


def experts_native2(x, gu_n, dn_n, ids, cfg_a=(4, 1), cfg_b=(1, 1), Ngu=1280, Ndn=2560):
    R = x.shape[0]
    order, sexp, prow, seg, P = segments(ids)
    g = qmmgn2(x, *gu_n, Ngu, prow, sexp, seg, P, *cfg_a)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    y = qmmgn2(h, *dn_n, Ndn, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)
