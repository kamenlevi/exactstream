"""NAX7 (Leap, Mon 2026-11-02 21:4x; Fable mission 27 design A; orchestrator 21:2x, Leap's #1): NAX6 (tierglu_nax6.py) with
the same per-output arithmetic and a different schedule. N7-1's knockouts: NAX6's kernel A without its matrix ops is
memory-bound at ~its bytes, and the ops add ~80 % of their own time on top (each simdgroup waits for its codes, decodes,
runs two ops and consumes the result at once).
Kernel A here, per simdgroup (projection pj, K chunk kc, as NAX6): the codes / LSB / scale / bias of quant group g + 2
are loaded into registers right after group g + 1 is decoded, and that decode runs between op.run(g) and g's epilogue,
so the loads have a full iteration to arrive and the decode overlaps the matrix unit. Every output still gets NAX6's
sequence: per quant group cg = 0, two 16 x 32 x 32 ops (k halves in order), acc += s * cg + b * xsum; the K chunks'
accumulators summed in chunk order; the same swiglu. So tier_glu_nax7 == tier_glu_nax6 bitwise (checked by
n7_screen.py). Kernel B and the small dispatches are NAX6's for now."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_grouped import make_grp
from tierglu_nax import rowsum64

# The pipelined core for S_K / S_O, with the caller's setup as NAX6's _CORE4 (e / el / tier / n0 / lane / sg / kc, ar0 /
# ar1, as0 / as1); GS_C must be 1.
_CORE7 = r"""
    constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8;
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const device uint32_t* wrow = (const device uint32_t*)((const device uint8_t*)w3p + ((size_t)e * S_O + n0 + lane) * KB3);
    const device T* srow = s3p + ((size_t)e * S_O + n0 + lane) * KG;
    const device T* brow;
    const device uint32_t* lrow = (const device uint32_t*)l1p;
    if (tier == 4) {
        lrow = (const device uint32_t*)((const device uint8_t*)l1p + ((size_t)el * S_O + n0 + lane) * KL);
        brow = blp + ((size_t)el * S_O + n0 + lane) * KG;
    } else {
        brow = (tier == 2 ? b2p : b3p) + ((size_t)e * S_O + n0 + lane) * KG;
    }
    const float smul = tier == 4 ? 0.5f : 1.0f;
    threadgroup uint32_t* tbw = tb_all + sg * (2 * 32 * 8);         // 2 buffers x 32 rows x 8 words (64 nibbles)
    threadgroup float* sbw = sb_all + sg * (2 * 64);                 // 2 buffers x (32 scales, 32 biases)
    constexpr int GPC = KG / KS_C;                                   // quant groups per K chunk
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    uint pw0, pw1, pw2, pw3, pw4, pw5, pl0 = 0, pl1 = 0;             // the next group's raw codes / LSB words
    float ps, pb;                                                    // and its scale (x smul) / bias
    auto fetch = [&](int g) {
        const device uint32_t* c = wrow + 6 * g;
        pw0 = c[0]; pw1 = c[1]; pw2 = c[2]; pw3 = c[3]; pw4 = c[4]; pw5 = c[5];
        if (tier == 4) { pl0 = lrow[2 * g]; pl1 = lrow[2 * g + 1]; }
        ps = (float)srow[g] * smul;
        pb = (float)brow[g];
    };
    auto decode = [&](int slot) {                                    // NAX6's decode of the fetched group into buffer slot
        threadgroup uint32_t* tb = tbw + slot * (32 * 8);
        threadgroup float* sb = sbw + slot * 64;
        uint ch[8] = {pw0 & 0xFFFFFFu, (pw0 >> 24) | ((pw1 & 0xFFFFu) << 8), (pw1 >> 16) | ((pw2 & 0xFFu) << 16), pw2 >> 8,
                      pw3 & 0xFFFFFFu, (pw3 >> 24) | ((pw4 & 0xFFFFu) << 8), (pw4 >> 16) | ((pw5 & 0xFFu) << 16), pw5 >> 8};
        uint nb[8];
        if (tier == 4) {
            for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? pl0 : pl1) >> (8 * (q & 3))) & 0xFFu);
        } else if (tier == 2) {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
        } else {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
        }
        threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8);
        dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
        dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
        sb[lane] = ps;
        sb[32 + lane] = pb;
    };
    const int g0 = kc * GPC, g1 = (kc + 1) * GPC;
    fetch(g0);
    decode(g0 & 1);
    if (g0 + 1 < g1) fetch(g0 + 1);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = g0; g < g1; ++g) {
        const int cur = g & 1;
        threadgroup uint32_t* tb = tbw + cur * (32 * 8);
        threadgroup float* sb = sbw + cur * 64;
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        for (int hh = 0; hh < 2; ++hh) {
            const int k0 = 64 * g + 32 * hh;
            const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
            const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
            const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
            const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
            for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
            TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
            op.run(ca, tbt, cg);
        }
        if (g + 1 < g1) {                                            // overlaps the ops above
            decode((g + 1) & 1);
            if (g + 2 < g1) fetch(g + 2);
        }
        const float x0 = as0[g], x1 = as1[g];
        for (int i = 0; i < 16; ++i) {
            const int col = 16 * (i >> 3) + fn0 + (i & 3);
            acc[i] += sb[col] * cg[i] + sb[32 + col] * (((i >> 2) & 1) ? x1 : x0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_A7 = N6._A4.replace(N6._CORE4, _CORE7)               # GS_C = 1: NAX6's buffer sizes are the ones _CORE7 indexes
assert _A7 != N6._A4 and _CORE7 in _A7


@lru_cache(None)
def _ka7():
    return mx.fast.metal_kernel(
        name="tierglu_nax7_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A7, header=N6._HDR4)


def kernel_a7(x, xs, msb, lsb, ms, ls, grp, plan, b2=None, ks=4):
    """NAX7's kernel A (gate / up + swiglu -> h [M * R, N]); == NAX6's kernel A bitwise"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg = grp
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    g, u = msb["gate_proj"], msb["up_proj"]
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1),
            ("K_C", K), ("N_C", N), ("KS_C", ks)]
    (h,) = _ka7()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl, grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    return h


def tier_glu_nax7(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, grp=None, nsg_b=2, ms_np=None, ls_np=None,
                  ks=4, ks_b=2, xs=None):
    """tier_glu_nax6's signature and bits: NAX7's kernel A, NAX6's kernel B"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    D = msb["down_proj"]["s3"].shape[1]
    N = msb["gate_proj"]["s3"].shape[1]
    if grp is None:
        grp = make_grp(msb_slot, lsb_slot, plan, M, ms_np, ls_np, 16)
    pidx, seg, nsg_, nseg = grp
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    if r4:
        ls = lsb_slot.astype(mx.int32).reshape(-1)
        L = lsb
    else:
        ls = mx.zeros((1,), mx.int32)
        L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    xs = rowsum64(x) if xs is None else xs
    h = kernel_a7(x, xs, msb, lsb, ms, ls, grp, plan, b2=b2, ks=ks)
    hs = rowsum64(h)
    d = msb["down_proj"]
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    (y,) = N6._kb4()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             ms, ls, pidx, seg, nsg_],
                     template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                     grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                     output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---------------------------------------------------------------- NAX7-WS (N7-5): loader / op simdgroups
# 16 simdgroups per threadgroup: sg 0-7 are NAX6's (projection pj = sg & 1, K chunk kc = sg >> 1) and run only its op
# sequence on decoded tiles; sg 8-15 are their loaders (loader 8 + c serves op simdgroup c): each fetches quant group
# g + 2's codes / LSB / scale / bias into registers and decodes group g + 1 into a two-slot ring while the ops of group g
# run. One threadgroup barrier per stage. Bitwise == NAX6: the same nibbles, the same per-output ops and epilogue, the
# chunk sums in order and the swiglu by simdgroup 1, as NAX6.
_A7W = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C;
    constexpr int KB3 = K * 3 / 8, KG = K / 64, KL = K / 8;
    constexpr int GPC = KG / KS_C;                                   // quant groups per K chunk
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const bool loader = sg >= 2 * KS_C;
    const uint c8 = sg % (2 * KS_C);                                 // the op simdgroup this one is or serves
    const uint pj = c8 & 1u, kc = c8 >> 1;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier0 = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pf];
    int tier = tier0;
    uint el = 0;
    if (tier == 4) {
        const int el_ = lsb_slot[(pf / NR) * R4_C + popcount((uint)MASK4_C & ((1u << r0) - 1u))];
        if (el_ < 0) tier = 3; else el = (uint)el_;                // LP1: no LSB read -> tier 3
    }
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    threadgroup uint32_t ring[2 * 2 * KS_C * 32 * 8];               // [slot][op simdgroup][32 rows x 8 words]; later the chunk sums
    threadgroup float rsb[2 * 2 * KS_C * 64];                        // [slot][op simdgroup][32 scales, 32 biases]
    const int g0 = kc * GPC;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    // loader state (loaders only)
    const device uint32_t* w3p = pj == 0 ? w3g : w3u;
    const device T* s3p = pj == 0 ? s3g : s3u;
    const device T* b3p = pj == 0 ? b3g : b3u;
    const device T* b2p = pj == 0 ? b2g : b2u;
    const device uint32_t* l1p = pj == 0 ? l1g : l1u;
    const device T* blp = pj == 0 ? bg : bu;
    const device uint32_t* wrow = (const device uint32_t*)((const device uint8_t*)w3p + ((size_t)e * N + n0 + lane) * KB3);
    const device T* srow = s3p + ((size_t)e * N + n0 + lane) * KG;
    const device T* brow;
    const device uint32_t* lrow = (const device uint32_t*)l1p;
    if (tier == 4) {
        lrow = (const device uint32_t*)((const device uint8_t*)l1p + ((size_t)el * N + n0 + lane) * KL);
        brow = blp + ((size_t)el * N + n0 + lane) * KG;
    } else {
        brow = (tier == 2 ? b2p : b3p) + ((size_t)e * N + n0 + lane) * KG;
    }
    const float smul = tier == 4 ? 0.5f : 1.0f;
    uint pw0 = 0, pw1 = 0, pw2 = 0, pw3 = 0, pw4 = 0, pw5 = 0, pl0 = 0, pl1 = 0;
    float ps = 0.0f, pb = 0.0f;
    auto fetch = [&](int g) {
        const device uint32_t* c = wrow + 6 * g;
        pw0 = c[0]; pw1 = c[1]; pw2 = c[2]; pw3 = c[3]; pw4 = c[4]; pw5 = c[5];
        if (tier == 4) { pl0 = lrow[2 * g]; pl1 = lrow[2 * g + 1]; }
        ps = (float)srow[g] * smul;
        pb = (float)brow[g];
    };
    auto decode = [&](int slot) {
        threadgroup uint32_t* tb = ring + (slot * 2 * KS_C + c8) * (32 * 8);
        threadgroup float* sb = rsb + (slot * 2 * KS_C + c8) * 64;
        uint ch[8] = {pw0 & 0xFFFFFFu, (pw0 >> 24) | ((pw1 & 0xFFFFu) << 8), (pw1 >> 16) | ((pw2 & 0xFFu) << 16), pw2 >> 8,
                      pw3 & 0xFFFFFFu, (pw3 >> 24) | ((pw4 & 0xFFFFu) << 8), (pw4 >> 16) | ((pw5 & 0xFFu) << 16), pw5 >> 8};
        uint nb[8];
        if (tier == 4) {
            for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? pl0 : pl1) >> (8 * (q & 3))) & 0xFFu);
        } else if (tier == 2) {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
        } else {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
        }
        threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8);
        dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
        dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
        sb[lane] = ps;
        sb[32 + lane] = pb;
    };
    // op state (op simdgroups only)
    const uint ma0 = (uint)pidx[q0 + (fm0 < nq ? fm0 : 0)] / NR, ma1 = (uint)pidx[q0 + (fm0 + 8 < nq ? fm0 + 8 : 0)] / NR;
    const device T* ar0 = x + (size_t)ma0 * K;
    const device T* ar1 = x + (size_t)ma1 * K;
    const device float* as0 = xs + (size_t)ma0 * (K / 64);
    const device float* as1 = xs + (size_t)ma1 * (K / 64);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    // stage t: loaders decode group g0 + t (fetched at t - 1) into slot t & 1 and fetch g0 + t + 1; op simdgroups run group
    // g0 + t - 1 from slot (t - 1) & 1. Every barrier is in uniform control flow.
    for (int t = 0; t <= GPC; ++t) {
        if (loader) {
            if (t == 0) {
                fetch(g0);
                decode(0);
                if (1 < GPC) fetch(g0 + 1);
            } else if (t < GPC) {
                decode(t & 1);
                if (t + 1 < GPC) fetch(g0 + t + 1);
            }
        } else if (t > 0) {
            const int g = g0 + t - 1, slot = (t - 1) & 1;
            threadgroup uint32_t* tb = ring + (slot * 2 * KS_C + c8) * (32 * 8);
            threadgroup float* sb = rsb + (slot * 2 * KS_C + c8) * 64;
            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
            for (int hh = 0; hh < 2; ++hh) {
                const int k0 = 64 * g + 32 * hh;
                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
                for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
                TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
                op.run(ca, tbt, cg);
            }
            const float x0 = as0[g], x1 = as1[g];
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + fn0 + (i & 3);
                acc[i] += sb[col] * cg[i] + sb[32 + col] * (((i >> 2) & 1) ? x1 : x0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* part = (threadgroup float*)ring;              // [projection][chunk][16 x 32]
    if (!loader) {
        for (int i = 0; i < 16; ++i) part[(pj * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1) {                                                  // gate = chunk sums in order, up likewise, then swiglu
        for (int i = 0; i < 16; ++i) {
            const int row = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
            float gs = part[(0 * KS_C + 0) * 512 + row * 32 + col], us = part[(1 * KS_C + 0) * 512 + row * 32 + col];
            for (int c = 1; c < KS_C; ++c) { gs += part[(0 * KS_C + c) * 512 + row * 32 + col]; us += part[(1 * KS_C + c) * 512 + row * 32 + col]; }
            if (row < nq) {
                const T gv = static_cast<T>(gs);
                const T ub = static_cast<T>(us);
                h[(size_t)pidx[q0 + row] * N + n0 + col] = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);
            }
        }
    }
"""


@lru_cache(None)
def _ka7w():
    return mx.fast.metal_kernel(
        name="tierglu_nax7w_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A7W, header=N6._HDR4)


def kernel_a7w(x, xs, msb, lsb, ms, ls, grp, plan, b2=None, ks=4):
    """NAX7-WS kernel A (loader / op simdgroups); == NAX6's kernel A bitwise"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg = grp
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    g, u = msb["gate_proj"], msb["up_proj"]
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("K_C", K), ("N_C", N),
            ("KS_C", ks)]
    (h,) = _ka7w()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                           pidx, seg, nsg_],
                   template=tmpl, grid=(N // 32 * 128 * ks, nseg, 1), threadgroup=(128 * ks, 1, 1),
                   output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    return h


# ---------------------------------------------------------------- NAX7-F (N7-10): NAX6 + combine_phase in three dispatches
# K0 = gpu_grp's grouping (threadgroup 0, NAX6's _GRP verbatim) + rowsum64(x) + the counters zeroed; K1 = NAX6's kernel A,
# then the group's last threadgroup (a device counter) computes its pairs' h row sums in rowsum64's order; K2 = NAX6's
# kernel B, then the 10th pair to finish a (row, 64-column block) does that block's combine in combine_phase's LY order.
# Release fences before the counters, acquire fences after them; no spin-waits. Bitwise == tier_glu_nax6 + combine_phase.
_GRP_BODY = N6._GRP[N6._GRP.index("    if ((int)i < P) {"):]
_K0 = r"""
    const uint tg0 = threadgroup_position_in_grid.x;
    const uint lt = thread_index_in_threadgroup;
    threadgroup int key_s[P_C];
    threadgroup int skey[P_C];
    threadgroup int start_s[P_C];
    if (tg0 == 0) {
        const int P = P_C;
        const uint i = lt;
""" + _GRP_BODY + r"""
    } else {
        const uint j = (tg0 - 1) * TGS_C + lt;
        if (j < (uint)(M_C * KG_C)) {                                  // rowsum64(x), verbatim order
            const uint r = j / KG_C, gg = j % KG_C;
            const device T* p = x + (size_t)r * (KG_C * 64) + gg * 64;
            float s = 0.0f;
            for (int q = 0; q < 64; ++q) s += (float)p[q];
            xs[(size_t)r * KG_C + gg] = s;
        }
        for (uint z = j; z < (uint)(P_C + M_C * NB_C); z += NXT_C * TGS_C) {
            if (z < (uint)P_C) cntA[z] = 0; else cntC[z - P_C] = 0;
        }
    }
"""
_A7F = N6._A4 + r"""
    // NAX7-F: the group's last threadgroup computes its pairs' h row sums (rowsum64's sequential order)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup int last_s[1];
    if (sg == 0 && lane == 0) {
        const int old = atomic_fetch_add_explicit((device atomic_int*)cntA + gq, 1, memory_order_relaxed);
        last_s[0] = (old == (int)(N / 32) - 1) ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_s[0]) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        for (uint j = sg * 32 + lane; j < (uint)nq * (N / 64); j += 64 * KS_C) {
            const uint row = j / (N / 64), gg = j % (N / 64);
            const device T* p = h + (size_t)pidx[q0 + row] * N + gg * 64;
            float s = 0.0f;
            for (int q = 0; q < 64; ++q) s += (float)p[q];
            hs[(size_t)pidx[q0 + row] * (N / 64) + gg] = s;
        }
    }
"""
_B7F = N6._B4 + r"""
    // NAX7-F: the 10th pair to finish a (row, column block) does that block's combine (combine_phase's LY order)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup int dn[16];
    threadgroup int ndn[1];
    const uint tid = sg * 32 + lane;
    const uint blk = threadgroup_position_in_grid.x;
    if (tid == 0) ndn[0] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)nq) {
        const uint mrow = (uint)pidx[q0 + tid] / NR;
        const int old = atomic_fetch_add_explicit((device atomic_int*)cntC + mrow * (D / (32 * NSG_C)) + blk, 1, memory_order_relaxed);
        if (old == NR - 1) {
            const int k = atomic_fetch_add_explicit((threadgroup atomic_int*)ndn, 1, memory_order_relaxed);
            dn[k] = (int)mrow;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ndn[0] > 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        for (int qd = 0; qd < ndn[0]; ++qd) {
            const uint m = (uint)dn[qd];
            for (uint c = tid; c < (uint)(32 * NSG_C); c += 32 * NSG_C * KS_C) {
                const uint d = blk * (32 * NSG_C) + c;
                T part[LY_C];
                for (int j = 0; j < LY_C; ++j) part[j] = static_cast<T>(0);
                for (int rr = 0; rr < NR; ++rr) {
                    const T yv = y[(size_t)(m * NR + rr) * D + d];
                    const T prod = TgMultiply()(yv, scores[m * NR + rr]);
                    part[rr % LY_C] = prod + part[rr % LY_C];
                }
                T tot = part[0];
                for (int j = 1; j < LY_C; ++j) tot = part[j] + tot;
                out[(size_t)m * D + d] = tot;
            }
        }
    }
"""


@lru_cache(None)
def _k0f():
    return mx.fast.metal_kernel(name="tierglu_nax7f_k0", input_names=["msb_slot", "lsb_slot", "x"],
                                output_names=["pidx", "seg", "nseg", "xs", "cntA", "cntC"], source=_K0, header=N6._HDR4)


@lru_cache(None)
def _ka7f():
    return mx.fast.metal_kernel(
        name="tierglu_nax7f_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "cntA"],
        output_names=["h", "hs"], source=_A7F, header=N6._HDR4)


@lru_cache(None)
def _kb7f():
    return mx.fast.metal_kernel(
        name="tierglu_nax7f_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg",
                     "scores", "cntC"],
        output_names=["y", "out"], source=_B7F, header=N6._HDR4)


def tier_glu_nax7f(x, msb, lsb, msb_slot, lsb_slot, scores, plan="4433333333", b2=None, nsg_b=2, ks=4, ks_b=2, ly=8):
    """the block's combined experts [M, D] in three dispatches; == combine_phase(tier_glu_nax6(...), all phase 1) bitwise"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 64 and K % 64 == 0 and N % 64 == 0 and D % (32 * nsg_b) == 0
    P = M * R
    assert P <= 1024
    KG, NB = K // 64, D // (32 * nsg_b)
    TGS = max(32, (P + 31) // 32 * 32)
    NXT = max(1, (M * KG + TGS - 1) // TGS)
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    if r4:
        ls = lsb_slot.astype(mx.int32).reshape(-1)
        L = lsb
    else:
        ls = mx.zeros((1,), mx.int32)
        L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    pidx, seg, nsg_, xs, cntA, cntC = _k0f()(
        inputs=[ms, ls, x], template=[("T", x.dtype), ("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2),
                                      ("R4_C", max(r4, 1)), ("M_C", M), ("KG_C", KG), ("NB_C", NB), ("TGS_C", TGS),
                                      ("NXT_C", NXT)],
        grid=(TGS * (1 + NXT), 1, 1), threadgroup=(TGS, 1, 1),
        output_shapes=[(P,), (P + 1,), (1,), (M, KG), (P,), (M * NB,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.float32, mx.int32, mx.int32])
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    h, hs = _ka7f()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                            L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                            pidx, seg, nsg_, cntA],
                    template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                    grid=(N // 32 * 64 * ks, P, 1), threadgroup=(64 * ks, 1, 1),
                    output_shapes=[(P, N), (P, N // 64)], output_dtypes=[x.dtype, mx.float32])
    y, out = _kb7f()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             ms, ls, pidx, seg, nsg_, scores.astype(x.dtype).reshape(-1), cntC],
                     template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b), ("LY_C", ly)],
                     grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, P, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                     output_shapes=[(P, D), (M, D)], output_dtypes=[x.dtype, x.dtype])
    return out
