"""L1-26 (Leap, Sat 2026-10-24): a fragment-native 4-bit weight layout for qmm16c (dense) and experts_nax (L1-25).
L1-24 / L1-25: the 16-row paths are ALU-bound on unpacking. qmm16c's lane reads 2 words from each of 4 weight rows per
64-value group (8-byte pieces of 4 rows) and turns each nibble into a bf16 operand with 3 ops (shift, mask, or 0x4300).
Native layout, per (32-column tile t, group g, lane): the lane's 8 words stored contiguously (32 bytes; the simdgroup's
1 KB per tile-group is one contiguous block), each word pairing the same 4 codes of the lane's two rows fm and fm + 8
(low / high 16 bits), so one (shift, mask 0x000F000F, or 0x43004300) yields two bf16 operands. Scales / biases per
(tile, group) as 32 contiguous columns. The MMA gets the same values in the same order as qmm16c, and the epilogue is
unchanged, so the output is meant to be bit-identical to qmm16c (no new reference beyond L1-23 / L1-25's).
repack(w, s, b) -> (wn, sn, bn) for [N, K/8] or [E, N, K/8]; qmm16n(x, wn, sn, bn, N, ks, nt);
experts_native(x, gu_n, dn_n, ids, cfg_a, cfg_b) as qmmg25.experts_nax."""
import os, sys
from functools import lru_cache
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
from qmm16 import _HDR
from gmr24 import segments


def _lane_maps():
    lane = np.arange(32)
    qid = lane >> 2
    fm = (qid & 4) | ((lane >> 1) & 3)
    fq = (qid & 2) | (lane & 1)
    return fm, fq


def repack(w, s, b):
    """MLX 4-bit (g64) [.., N, K/8] uint32 + [.., N, K/64] scales / biases -> native layout:
    wn [.., N/32, K/64, 32 lanes, 8 words] (word index = st * 2 + p: step st 0..3, row pair p: rows r 2p / 2p+1),
    sn / bn [.., N/32, K/64, 32 columns]"""
    W = np.array(w).view(np.uint32)
    lead = W.shape[:-2]
    N, KW = W.shape[-2:]
    KG = KW // 8
    W = W.reshape(lead + (N // 32, 32, KG, 8))                        # [.., t, row in tile, g, word in group]
    fm, fq = _lane_maps()
    out = np.zeros(lead + (N // 32, KG, 32, 8), np.uint32)
    for lane in range(32):
        for r in range(4):
            row = (r >> 1) * 16 + int(fm[lane]) + (r & 1) * 8          # row within the tile
            for st in range(4):
                wsel = 2 * int(fq[lane]) + (st >> 1)                   # word within the group
                half = (W[..., row, :, wsel] >> np.uint32(16 * (st & 1))) & np.uint32(0xFFFF)   # [.., t, g]
                out[..., lane, st * 2 + (r >> 1)] |= half << np.uint32(16 * (r & 1))
    S = np.array(s.astype(mx.float32)).reshape(lead + (N // 32, 32, KG))
    B = np.array(b.astype(mx.float32)).reshape(lead + (N // 32, 32, KG))
    sn = np.ascontiguousarray(np.moveaxis(S, -1, -2))                   # [.., t, g, 32 columns], contiguous (a strided
    bn = np.ascontiguousarray(np.moveaxis(B, -1, -2))                   # input is copied by metal_kernel on every call)
    return (mx.array(out), mx.array(sn).astype(s.dtype), mx.array(bn).astype(b.dtype))


_SRC = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int t = threadgroup_position_in_grid.x * NT + nt;          // 32-column tile
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
    float acc[16];
    for (short i = 0; i < 16; i++) acc[i] = 0.0f;

    constexpr int KG = K_C / G, NTILES = N_C / 32;
    const int kper = K_C / KS;
    // native weights: [e][t][g][lane][8 words]; scales / biases: [e][t][g][32]
    const device uint4* wq = (const device uint4*)(wn + ((((size_t)e * NTILES + t) * KG) * 32 + lane) * 8);
    const device T* sc_ = sn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const device T* bi_ = bn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
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
            for (short p = 0; p < 2; p++) {
                const uint word = wv[st * 2 + p];
                for (short j = 0; j < 4; j++) {
                    const uint two = ((word >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                    ct_b[p * 8 + j] = as_type<T>((ushort)(two & 0xFFFFu));          // row r = 2p   (fm, or 16 + fm)
                    ct_b[p * 8 + 4 + j] = as_type<T>((ushort)(two >> 16));          // row r = 2p+1 (fm + 8, or 24 + fm)
                }
            }
            op.run(ct_a, ct_b, ct_c);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        const device T* scg = sc_ + (size_t)g * 32;
        const device T* big = bi_ + (size_t)g * 32;
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const float s_ = static_cast<float>(scg[16 * bk + j]);
                const float b_ = static_cast<float>(big[16 * bk + j]) - 128.0f * s_;
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
def _k(gather):
    names = ["x", "wn", "sn", "bn"] + (["prow", "sexp", "seg_start"] if gather else [])
    return mx.fast.metal_kernel(name=f"leap_qmm26_{'g' if gather else 'd'}", input_names=names, output_names=["y"],
                                source=_SRC.replace("#if GATHER", f"#if {int(gather)}"), header=_HDR)


def qmm16n(x, wn, sn, bn, N, ks=1, nt=1):
    """dense: y [16, N] (x padded to 16 rows), bit-identical to qmm16c on the same weights"""
    M, K = x.shape
    assert M <= 16 and N % (32 * nt) == 0 and K % (64 * ks) == 0
    xp = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], 0)
    (y,) = _k(False)(inputs=[xp, wn, sn, bn], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", 16)],
                     grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(16, N)],
                     output_dtypes=[x.dtype])
    return y[:M]


def qmmgn(x, wn, sn, bn, N, prow, sexp, seg, P, ks=1, nt=1):
    K = x.shape[-1]
    assert N % (32 * nt) == 0 and K % (64 * ks) == 0
    (y,) = _k(True)(inputs=[x, wn, sn, bn, prow, sexp, seg],
                    template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", P)],
                    grid=(N // 32 * 32 * ks, P, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(P, N)],
                    output_dtypes=[x.dtype], init_value=0)
    return y


def experts_native(x, gu_n, dn_n, ids, cfg_a=(4, 1), cfg_b=(1, 2), Ngu=1280, Ndn=2560):
    R = x.shape[0]
    order, sexp, prow, seg, P = segments(ids)
    g = qmmgn(x, *gu_n, Ngu, prow, sexp, seg, P, *cfg_a)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    y = qmmgn(h, *dn_n, Ndn, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)
