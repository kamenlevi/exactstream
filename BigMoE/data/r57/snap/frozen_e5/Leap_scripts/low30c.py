"""L1-30c (Leap, Sun 2026-10-25): low30b's kernel (MODE, PAIRK) plus PREF: load group g + 1's words while group g is
unpacked and multiplied (register double buffer), and an optional grid height ny (rows of segments launched; default
P = routed pairs, most of which exit at once). Outputs are bitwise equal to low30b MODE 0 for every PREF / ny.
low30b's docstring follows.

MODE (profile; outputs are garbage except MODE 0): 0 full, 1 no tensor-unit multiply (unpacked words folded into a
sink), 2 no unpacking (operands fixed once, weight words folded into a sink), 3 no activation work (x operands fixed
once, no row sums), 4 weight / scale reads only.
PAIRK (layout): 0 = low30's (one word's two halves feed operand slots i and i + 4: rows fm / fm + 8 of one k);
1 = codes paired along K (halves feed slots i and i + 1: k and k + 1 of one row), so two operands can move as one
register. Same values in the same slots -> bitwise equal to PAIRK 0.
  PAIRK 1 words per lane per group: 4-bit word p * 4 + h * 2 + (st >> 1), field (st & 1) * 2 + (j >> 1) at 4 f;
  2-bit (and 3-bit low plane) word p * 2 + h, field st * 2 + (j >> 1) at 2 f; 3-bit high plane word p, bit h * 8 +
  st * 2 + (j >> 1); half = j & 1 everywhere.
repack_k(Q, s, b, bits, pairk) -> (wn, wh, sn, bn); experts_prof(x, gu, dn, ids, bits, cfg_a, cfg_b, mode, pairk)."""
import os, sys
from functools import lru_cache
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
from qmm16 import _HDR
from gmr24 import segments
from qmm26 import _lane_maps
from low30 import repack_codes


def repack_k(Q, s, b, bits, pairk):
    if not pairk:
        return repack_codes(Q, s, b, bits)
    E, N, K = Q.shape
    NT, KG = N // 32, K // 64
    Qr = Q.reshape(E, NT, 32, KG, 64)
    fm, fq = _lane_maps()
    wn = np.zeros((E, NT, KG, 32, {4: 8, 3: 4, 2: 4}[bits]), np.uint32)
    wh = np.zeros((E, NT, KG, 32, 2), np.uint32) if bits == 3 else np.zeros((256,), np.uint32)
    for lane in range(32):
        for p in range(2):
            for h in range(2):
                row = p * 16 + int(fm[lane]) + h * 8
                for st in range(4):
                    for j in range(4):
                        c = Qr[:, :, row, :, 16 * int(fq[lane]) + 4 * st + j].astype(np.uint32)
                        half = 16 * (j & 1)
                        if bits == 4:
                            wn[..., lane, p * 4 + h * 2 + (st >> 1)] |= c << np.uint32(half + 4 * ((st & 1) * 2 + (j >> 1)))
                        else:
                            wn[..., lane, p * 2 + h] |= (c & np.uint32(3)) << np.uint32(half + 2 * (st * 2 + (j >> 1)))
                            if bits == 3:
                                wh[..., lane, p] |= (c >> np.uint32(2)) << np.uint32(half + h * 8 + st * 2 + (j >> 1))
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
    uint sink = 0u;
    if (MODE == 2) for (short i = 0; i < 16; i++) ct_b[i] = as_type<T>((ushort)0x4300u);
    if (MODE == 3) for (short i = 0; i < 8; i++) ct_a[i] = as_type<T>((ushort)0x3F80u);

    constexpr int KG = K_C / G, NTILES = N_C / 32;
    constexpr int WPL = BITS == 4 ? 8 : 4;
    const int kper = K_C / KS;
    const size_t lb = (((size_t)e * NTILES + t) * KG) * 32 + lane;
    const device uint* wq = wn + lb * WPL;
    const device uint* hq = wh + lb * 2;
    const device T* sc_ = sn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const device T* bi_ = bn + (((size_t)e * NTILES + t) * KG) * 32 + 4 * fq;
    const int r0 = prow[p0 + (fm < cnt ? fm : 0)];
    const int r1 = prow[p0 + (fm + 8 < cnt ? fm + 8 : 0)];
    const device T* xr0 = x + (size_t)r0 * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)r1 * K_C + 16 * fq;

#define LOADA(gg) (*((const device uint4*)(wq + (size_t)(gg) * 32 * WPL)))
#define LOADB(gg) (*((const device uint4*)(wq + (size_t)(gg) * 32 * WPL + 4)))
#define LOADH(gg) (*((const device uint2*)(hq + (size_t)(gg) * 64)))
    const int kb = kg * kper, ke = (kg + 1) * kper;
    uint4 pa = uint4(0u), pb = uint4(0u);
    uint2 ph = uint2(0u);
    if (PREF) {
        pa = LOADA(kb / G);
        if (BITS == 4) pb = LOADB(kb / G);
        if (BITS == 3) ph = LOADH(kb / G);
    }
    for (int k0 = kb; k0 < ke; k0 += G) {
        const int g = k0 / G;
        uint4 q0, q1 = uint4(0u);
        uint2 hh = uint2(0u);
        if (PREF) {
            q0 = pa; q1 = pb; hh = ph;
            if (k0 + G < ke) {
                pa = LOADA(g + 1);
                if (BITS == 4) pb = LOADB(g + 1);
                if (BITS == 3) ph = LOADH(g + 1);
            }
        } else {
            q0 = LOADA(g);
            if (BITS == 4) q1 = LOADB(g);
            if (BITS == 3) hh = LOADH(g);
        }
        uint wv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        uint hv[2] = {hh.x, hh.y};
        if (MODE == 2 || MODE == 4) {
            sink ^= wv[0] ^ wv[1] ^ wv[2] ^ wv[3] ^ hv[0] ^ hv[1];
            if (BITS == 4) sink ^= wv[4] ^ wv[5] ^ wv[6] ^ wv[7];
        }
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        float xs0 = 0.0f, xs1 = 0.0f;
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        for (short st = 0; st < 4; st++) {
            if (MODE <= 2) {
                for (short j = 0; j < 4; j++) {
                    const T a0 = x0[4 * st + j], a1 = x1[4 * st + j];
                    ct_a[j] = a0; ct_a[4 + j] = a1;
                    xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
                }
            }
            if (MODE == 0 || MODE == 1 || MODE == 3) {
                for (short p = 0; p < 2; p++) {
                    for (short u = 0; u < 4; u++) {
                        // PAIRK 0: u = j, halves -> slots (h 0, j) / (h 1, j); PAIRK 1: u = h * 2 + i, halves -> (h, 2i) / (h, 2i + 1)
                        uint two;
                        if (PAIRK == 0) {
                            const short j = u;
                            if (BITS == 4) {
                                two = ((wv[st * 2 + p] >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                            } else {
                                const short f = (st & 1) * 4 + j;
                                two = ((wv[(st >> 1) * 2 + p] >> (2 * f)) & 0x00030003u) | 0x43004300u;
                                if (BITS == 3) {
                                    const short b = st * 4 + j;
                                    two |= (b >= 2 ? (hv[p] >> (b - 2)) : (hv[p] << (2 - b))) & 0x00040004u;
                                }
                            }
                        } else {
                            const short h = u >> 1, i = u & 1;
                            if (BITS == 4) {
                                const short f = (st & 1) * 2 + i;
                                two = ((wv[p * 4 + h * 2 + (st >> 1)] >> (4 * f)) & 0x000F000Fu) | 0x43004300u;
                            } else {
                                const short f = st * 2 + i;
                                two = ((wv[p * 2 + h] >> (2 * f)) & 0x00030003u) | 0x43004300u;
                                if (BITS == 3) {
                                    const short b = h * 8 + st * 2 + i;
                                    two |= (b >= 2 ? (hv[p] >> (b - 2)) : (hv[p] << (2 - b))) & 0x00040004u;
                                }
                            }
                        }
                        if (MODE == 1) {
                            sink ^= two;
                        } else if (PAIRK == 0) {
                            ct_b[p * 8 + u] = as_type<T>((ushort)(two & 0xFFFFu));
                            ct_b[p * 8 + 4 + u] = as_type<T>((ushort)(two >> 16));
                        } else {
                            const short h = u >> 1, i = u & 1;
                            ct_b[p * 8 + h * 4 + 2 * i] = as_type<T>((ushort)(two & 0xFFFFu));
                            ct_b[p * 8 + h * 4 + 2 * i + 1] = as_type<T>((ushort)(two >> 16));
                        }
                    }
                }
            }
            if (MODE == 0 || MODE == 2 || MODE == 3) op.run(ct_a, ct_b, ct_c);
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
    if (MODE != 0) acc[0] += static_cast<float>(sink & 1u);

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
    return mx.fast.metal_kernel(name="leap_low30c", input_names=["x", "wn", "wh", "sn", "bn", "prow", "sexp", "seg_start"],
                                output_names=["y"], source=_SRC, header=_HDR)


def qmmg_prof(x, wn, wh, sn, bn, N, bits, prow, sexp, seg, P, ks, nt, mode, pairk, pref=0, ny=None):
    K = x.shape[-1]
    (y,) = _k()(inputs=[x, wn, wh, sn, bn, prow, sexp, seg],
                template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("P", P), ("BITS", bits),
                          ("MODE", mode), ("PAIRK", pairk), ("PREF", pref)],
                grid=(N // 32 * 32 * ks, P if ny is None else ny, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(P, N)],
                output_dtypes=[x.dtype], init_value=0)
    return y


def experts_prof(x, gu, dn, ids, bits, cfg_a, cfg_b, mode=0, pairk=0, pref=0, ny=None, Ngu=1280, Ndn=2560):
    R = x.shape[0]
    order, sexp, prow, seg, P = segments(ids)
    g = qmmg_prof(x, *gu, Ngu, bits, prow, sexp, seg, P, *cfg_a, mode, pairk, pref, ny)
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    y = qmmg_prof(h, *dn, Ndn, bits, mx.arange(P, dtype=mx.int32), sexp, seg, P, *cfg_b, mode, pairk, pref, ny)
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], -1)
