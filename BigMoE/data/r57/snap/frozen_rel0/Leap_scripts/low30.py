"""L1-30 (Leap, Sun 2026-10-25): native-layout tensor-unit expert kernel at 4 / 3 / 2 bits (qmm26's skeleton, BITS a
template constant). Per (32-column tile t, group g, lane) the lane's codes for its 4 rows x 16 K values sit contiguously,
two rows paired per word (low / high 16 bits), so one (shift, mask, or 0x43004300) yields two bf16 operands 128 + code:
  4-bit: 8 words, word st * 2 + p, field j at bit 4 j (= qmm26's repack; must be bitwise equal to experts_native)
  2-bit: 4 words, word (st >> 1) * 2 + p, field (st & 1) * 4 + j at bit 2 x field
  3-bit: the low 2 bits as the 2-bit layout + a 1-bit plane wh [.., lane, 2 words]: bit st * 4 + j of word p (6 ops)
Epilogue as qmm26 (scale, bias - 128 x scale, row sums of x).
repack_codes(Q, s, b, bits) -> (wn, wh, sn, bn) for Q [E, N, K] uint8 codes; experts_low(x, gu, dn, ids, bits, cfg_a, cfg_b)."""
import os, sys
from functools import lru_cache
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
from qmm16 import _HDR
from gmr24 import segments
from qmm26 import _lane_maps


def repack_codes(Q, s, b, bits):
    """Q [E, N, K] uint8 codes (< 2^bits), s / b [E, N, K/64] (mx) -> wn [E, N/32, K/64, 32, words], wh (3-bit: 1-bit plane
    [E, N/32, K/64, 32, 2]; else a dummy), sn / bn [E, N/32, K/64, 32] contiguous"""
    E, N, K = Q.shape
    NT, KG = N // 32, K // 64
    Qr = Q.reshape(E, NT, 32, KG, 64)
    fm, fq = _lane_maps()
    words = {4: 8, 3: 4, 2: 4}[bits]
    wn = np.zeros((E, NT, KG, 32, words), np.uint32)
    wh = np.zeros((E, NT, KG, 32, 2), np.uint32) if bits == 3 else np.zeros((256,), np.uint32)
    for lane in range(32):
        for p in range(2):
            for h in range(2):
                row = p * 16 + int(fm[lane]) + h * 8
                for st in range(4):
                    for j in range(4):
                        c = Qr[:, :, row, :, 16 * int(fq[lane]) + 4 * st + j].astype(np.uint32)     # [E, NT, KG]
                        if bits == 4:
                            wn[..., lane, st * 2 + p] |= c << np.uint32(16 * h + 4 * j)
                        else:
                            f = (st & 1) * 4 + j
                            wn[..., lane, (st >> 1) * 2 + p] |= (c & np.uint32(3)) << np.uint32(16 * h + 2 * f)
                            if bits == 3:
                                wh[..., lane, p] |= (c >> np.uint32(2)) << np.uint32(16 * h + st * 4 + j)
    S = np.array(s.astype(mx.float32)).reshape(E, NT, 32, KG)
    B = np.array(b.astype(mx.float32)).reshape(E, NT, 32, KG)
    sn = np.ascontiguousarray(np.moveaxis(S, -1, -2))
    bn = np.ascontiguousarray(np.moveaxis(B, -1, -2))
    return mx.array(wn), mx.array(wh), mx.array(sn).astype(s.dtype), mx.array(bn).astype(b.dtype)


_SRC = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int t = threadgroup_position_in_grid.x * NT + nt;
    const int n0 = t * 32;
    const int s = threadgroup_position_in_grid.y;
    const int p0 = seg_start[s];
    if (p0 >= P) return;
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

    constexpr int KG = K_C / G, NTILES = N_C / 32;
    constexpr int WPL = BITS == 4 ? 8 : 4;
    const int kper = K_C / KS;
    const size_t lb = (((size_t)e * NTILES + t) * KG) * 32 + lane;     // (e, t, g = 0, lane)
    const device uint* wq = wn + lb * WPL;
    const device uint* hq = wh + lb * 2;
    const device T* sc_ = sn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const device T* bi_ = bn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const int r0 = prow[p0 + (fm < cnt ? fm : 0)];
    const int r1 = prow[p0 + (fm + 8 < cnt ? fm + 8 : 0)];
    const device T* xr0 = x + (size_t)r0 * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)r1 * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        uint wv[8];
        uint hv[2] = {0u, 0u};
        const uint4 q0 = *((const device uint4*)(wq + (size_t)g * 32 * WPL));
        wv[0] = q0.x; wv[1] = q0.y; wv[2] = q0.z; wv[3] = q0.w;
        if (BITS == 4) {
            const uint4 q1 = *((const device uint4*)(wq + (size_t)g * 32 * WPL + 4));
            wv[4] = q1.x; wv[5] = q1.y; wv[6] = q1.z; wv[7] = q1.w;
        }
        if (BITS == 3) {
            const uint2 h = *((const device uint2*)(hq + (size_t)g * 64));
            hv[0] = h.x; hv[1] = h.y;
        }
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
                for (short j = 0; j < 4; j++) {
                    uint two;
                    if (BITS == 4) {
                        two = ((wv[st * 2 + p] >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                    } else {
                        const short f = (st & 1) * 4 + j;
                        two = ((wv[(st >> 1) * 2 + p] >> (2 * f)) & 0x00030003u) | 0x43004300u;
                        if (BITS == 3) {
                            const short i = st * 4 + j;
                            two |= (i >= 2 ? (hv[p] >> (i - 2)) : (hv[p] << (2 - i))) & 0x00040004u;
                        }
                    }
                    ct_b[p * 8 + j] = as_type<T>((ushort)(two & 0xFFFFu));
                    ct_b[p * 8 + 4 + j] = as_type<T>((ushort)(two >> 16));
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
def _k():
    return mx.fast.metal_kernel(name="leap_low30", input_names=["x", "wn", "wh", "sn", "bn", "prow", "sexp", "seg_start"],
                                output_names=["y"], source=_SRC, header=_HDR)


def qmmg_low(x, wn, wh, sn, bn, N, bits, prow, sexp, seg, P, ks=1, nt=1):
    K = x.shape[-1]
    assert N % (32 * nt) == 0 and K % (64 * ks) == 0
    (y,) = _k()(inputs=[x, wn, wh, sn, bn, prow, sexp, seg],
                template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", P), ("BITS", bits)],
                grid=(N // 32 * 32 * ks, P, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(P, N)],
                output_dtypes=[x.dtype], init_value=0)
    return y


def experts_low(x, gu, dn, ids, bits, cfg_a=(4, 1), cfg_b=(1, 2), Ngu=1280, Ndn=2560):
    R = x.shape[0]
    order, sexp, prow, seg, P = segments(ids)
    g = qmmg_low(x, *gu, Ngu, bits, prow, sexp, seg, P, *cfg_a)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    y = qmmg_low(h, *dn, Ndn, bits, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)
