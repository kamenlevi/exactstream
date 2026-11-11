"""K-62 NAX3 (Leap, Sun 2026-11-01): MoonShots' 3c tier GLU (tierglu.tier_glu: MSB-3 base pool for every rank, LSB-1 sidecar
for the plan's 4-bit ranks) on the M5's matrix units, dequantizing each expert ONCE per call instead of once per (row, rank)
pair. K-61 and MoonShots' 10-22 microbench showed 3c is ALU-bound on per-pair dequant + dot work (~6.5 us per pair).
- Pairs are grouped by (base slot, tier, sidecar slot) as in tierglu_grouped (<= 16 pairs per group, padded to 16 rows).
- One simdgroup = one 16-row x 32-column output tile; Metal 4's mpp::tensor_ops::matmul2d (16 x 32 x 16, bf16 in, fp32
  accumulate, MLX's BaseNAXFrag element map; scripts/nax_probe.py) on A = the group's x (or h) rows and B = the raw codes q
  as bf16 (exact small integers: tier 3 q, tier 2 q & 6, tier 4 2q + lsb).
- Per 64-value quant group g: C_g = X_g Q_g (4 MMAs), then acc += s * C_g + b * xsum_g in fp32 (tier 4: s = s3 / 2).
  The codes enter as 128 + q (bf16 0x4300 | q, built two per 32-bit op; no convert) and the bias absorbs it (b - 128 s).
  Inside each 64-value group the k order is permuted (A and B alike) so a lane decodes 16 consecutive codes per row.
- Kernel A: simdgroup 0 = gate, 1 = up, same columns; swiglu as 3c's (bf16 g, u; TgSigmoid, TgMultiply). Kernel B: down.
- xsum / hsum per 64 values: one thread per (row, group), a fixed sequential fp32 sum (batch-invariant by construction).
NOT bit-identical to 3c (summation order). A pair's bits must not depend on its group's other pairs, the group size or its
row in the tile (INV; the NAX probe showed row-invariance of one MMA). tier_glu_nax(...) -> y [M, R, D] like tier_glu."""
import os, sys
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts"))
from tierglu import _HDR, PROJ, plan_masks
from tierglu_grouped import make_grp

_HDRN = _HDR + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

# shared body: S_K = reduction length, S_O = output columns of the projection, rows of A from `arow(i)`; the B operand rows
# are this simdgroup's 32 output columns (n0 = column tile start); codes w3p / scales s3p / biases bqp / lsb l1p
_CORE = r"""
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);           // B rows n0 + fm0 + 8 t; A rows fm0, fm0 + 8
    const short f4 = (qid & 2) | (lane & 1);                   // frag columns 4 f4 .. 4 f4 + 3
    const short fn0 = 4 * f4;                                  // C columns fn0 + c, 16 + fn0 + c
    constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8;
    // k order inside each 64-value quant group: lane quarter f4 at K-step s, element j holds value 16 f4 + 4 s + (0 2 1 3)[j]
    // (A and B alike), so a lane's 16 values per row and group are 16 consecutive codes: 6 bytes, 3 aligned ushorts
    const device uint16_t* wr = (const device uint16_t*)((const device uint8_t*)w3p + ((size_t)e * S_O + n0 + fm0) * KB3 + 6 * f4);
    const device T* sp = s3p + ((size_t)e * S_O + n0 + 4 * f4) * KG;   // C columns 4 f4 + c and 16 + 4 f4 + c
    const device T* bp;
    const device uint16_t* lr = (const device uint16_t*)l1p;
    if (tier == 4) {
        lr = (const device uint16_t*)((const device uint8_t*)l1p + ((size_t)el * S_O + n0 + fm0) * KL + 2 * f4);
        bp = blp + ((size_t)el * S_O + n0 + 4 * f4) * KG;
    } else {
        bp = (tier == 2 ? b2p : b3p) + ((size_t)e * S_O + n0 + 4 * f4) * KG;
    }
    const device T* xa0 = ar0 + 16 * f4;
    const device T* xa1 = ar1 + 16 * f4;
    const float smul = tier == 4 ? 0.5f : 1.0f;
    const uint qm2 = tier == 2 ? 0x00060006u : 0x00070007u;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto cg = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                             metal::remove_addrspace_t<decltype(cb)>, float>();
    float acc[16];
    float sink = 0.0f;
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    uint nlo[4], nhi[4], nlw[4];
    for (int t = 0; t < 4; ++t) {                              // group 0's codes; later groups are fetched one ahead
        const device uint16_t* c = wr + (t * 8 * KB3) / 2;
        nlo[t] = (uint)c[0] | ((uint)c[1] << 16);
        nhi[t] = c[2];
        nlw[t] = tier == 4 ? (uint)lr[(t * 8 * KL) / 2] : 0u;
    }
    for (int g = 0; g < KG; ++g) {
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        uint lo[4], hi[4], lw[4];
        for (int t = 0; t < 4; ++t) { lo[t] = nlo[t]; hi[t] = nhi[t]; lw[t] = nlw[t]; }
        if (PF_C && g + 1 < KG) {
            for (int t = 0; t < 4; ++t) {
                const device uint16_t* c = wr + (t * 8 * KB3 + 24 * (g + 1)) / 2;
                nlo[t] = (uint)c[0] | ((uint)c[1] << 16);
                nhi[t] = c[2];
                nlw[t] = tier == 4 ? (uint)lr[(t * 8 * KL + 8 * (g + 1)) / 2] : 0u;
            }
        }
        for (int s = 0; s < 4; ++s) {
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xa0 + 64 * g + 4 * s);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xa1 + 64 * g + 4 * s);
            ca[0] = a0[0]; ca[1] = a0[2]; ca[2] = a0[1]; ca[3] = a0[3];
            ca[4] = a1[0]; ca[5] = a1[2]; ca[6] = a1[1]; ca[7] = a1[3];
            for (int t = 0; t < 4; ++t) {
                const uint f = s == 0 ? (lo[t] & 0xFFFu) : s == 1 ? ((lo[t] >> 12) & 0xFFFu)
                             : s == 2 ? ((lo[t] >> 24) | ((hi[t] & 0xFu) << 8)) : (hi[t] >> 4);
                const uint m = f | (f << 10);                  // a@0 b@3 c@6 d@9 | a@10 b@13 c@16 d@19
                uint r0, r1;
                if (tier == 4) {
                    const uint nb = (lw[t] >> (4 * s)) & 0xFu;
                    const uint nn = nb | (nb << 14);
                    r0 = ((m & 0x00070007u) << 1) | (nn & 0x00010001u) | 0x43004300u;          // 128 + 2a + la, 128 + 2c + lc
                    r1 = (((m >> 3) & 0x00070007u) << 1) | ((nn >> 1) & 0x00010001u) | 0x43004300u;
                } else {
                    r0 = (m & qm2) | 0x43004300u;              // bf16 pair (128 + a, 128 + c)
                    r1 = ((m >> 3) & qm2) | 0x43004300u;       // (128 + b, 128 + d)
                }
                if (DIAG_C == 1) { r0 = 0x43014302u; r1 = 0x43034304u; }
                const vec<T, 2> p0 = as_type<vec<T, 2>>(r0), p1 = as_type<vec<T, 2>>(r1);
                cb[4 * t] = p0[0]; cb[4 * t + 1] = p0[1]; cb[4 * t + 2] = p1[0]; cb[4 * t + 3] = p1[1];
                if (DIAG_C == 2) sink += (float)p0[0] + (float)p0[1] + (float)p1[0] + (float)p1[1];
            }
            if (DIAG_C != 2) op.run(ca, cb, cg);
        }
        float sv[8], bv[8];
        for (int c = 0; c < 4; ++c) {
            sv[c] = (float)sp[c * KG + g] * smul;
            sv[4 + c] = (float)sp[(16 + c) * KG + g] * smul;
            bv[c] = (float)bp[c * KG + g] - 128.0f * sv[c];             // B = 128 + q: the bias absorbs the offset
            bv[4 + c] = (float)bp[(16 + c) * KG + g] - 128.0f * sv[4 + c];
        }
        const float x0 = as0[g], x1 = as1[g];
        for (int i = 0; i < 16; ++i) {
            const int c = (i & 3) + ((i >> 3) << 2);
            acc[i] += sv[c] * cg[i] + bv[c] * (((i >> 2) & 1) ? x1 : x0);
        }
        if (!PF_C && g + 1 < KG) {
            for (int t = 0; t < 4; ++t) {
                const device uint16_t* c = wr + (t * 8 * KB3 + 24 * (g + 1)) / 2;
                nlo[t] = (uint)c[0] | ((uint)c[1] << 16);
                nhi[t] = c[2];
                nlw[t] = tier == 4 ? (uint)lr[(t * 8 * KL + 8 * (g + 1)) / 2] : 0u;
            }
        }
    }
    if (DIAG_C == 2) acc[0] += sink * 1e-30f;
"""

_AN = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0: gate, 1: up
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pf];
    uint el = 0;
    if (tier == 4) el = (uint)lsb_slot[(pf / NR) * R4_C + popcount((uint)MASK4_C & ((1u << r0) - 1u))];
    const short fmA = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint ma0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)] / NR, ma1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)] / NR;
    const device T* ar0 = x + (size_t)ma0 * K;
    const device T* ar1 = x + (size_t)ma1 * K;
    const device float* as0 = xs + (size_t)ma0 * (K / 64);
    const device float* as1 = xs + (size_t)ma1 * (K / 64);
    const device uint32_t* w3p = sg == 0 ? w3g : w3u;
    const device T* s3p = sg == 0 ? s3g : s3u;
    const device T* b3p = sg == 0 ? b3g : b3u;
    const device T* b2p = sg == 0 ? b2g : b2u;
    const device uint32_t* l1p = sg == 0 ? l1g : l1u;
    const device T* blp = sg == 0 ? bg : bu;
    #define S_K K
    #define S_O N
""" + _CORE + r"""
    threadgroup T gate_t[16][32];
    if (sg == 0) {
        for (int i = 0; i < 16; ++i) gate_t[fm0 + ((i >> 2) & 1) * 8][fn0 + (i & 3) + ((i >> 3) << 4)] = static_cast<T>(acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1) {
        for (int i = 0; i < 16; ++i) {
            const int row = fm0 + ((i >> 2) & 1) * 8, col = fn0 + (i & 3) + ((i >> 3) << 4);
            if (row < nq) {
                const T gv = gate_t[row][col];
                const T ub = static_cast<T>(acc[i]);
                h[(size_t)pidx[q0 + row] * N + n0 + col] = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);
            }
        }
    }
"""

_BN = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = (threadgroup_position_in_grid.x * nsg + sg) * 32;
    if (n0 >= (uint)D) return;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pf];
    uint el = 0;
    if (tier == 4) el = (uint)lsb_slot[(pf / NR) * R4_C + popcount((uint)MASK4_C & ((1u << r0) - 1u))];
    const short fmA = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint pa0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)], pa1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)];
    const device T* ar0 = h + (size_t)pa0 * N;
    const device T* ar1 = h + (size_t)pa1 * N;
    const device float* as0 = hs + (size_t)pa0 * (N / 64);
    const device float* as1 = hs + (size_t)pa1 * (N / 64);
    const device uint32_t* w3p = w3d;
    const device T* s3p = s3d;
    const device T* b3p = b3d;
    const device T* b2p = b2d;
    const device uint32_t* l1p = l1d;
    const device T* blp = bd;
    #define S_K N
    #define S_O D
""" + _CORE + r"""
    for (int i = 0; i < 16; ++i) {
        const int row = fm0 + ((i >> 2) & 1) * 8, col = fn0 + (i & 3) + ((i >> 3) << 4);
        if (row < nq) y[(size_t)pidx[q0 + row] * D + n0 + col] = static_cast<T>(acc[i]);
    }
"""

_RS = r"""
    const uint g = thread_position_in_grid.x, r = thread_position_in_grid.y;
    if (g >= (uint)(L_C / 64)) return;
    const device T* p = a + (size_t)r * L_C + g * 64;
    float s = 0.0f;
    for (int i = 0; i < 64; ++i) s += (float)p[i];
    out[(size_t)r * (L_C / 64) + g] = s;
"""


@lru_cache(None)
def _kan():
    return mx.fast.metal_kernel(
        name="tierglu_nax_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_AN, header=_HDRN)


@lru_cache(None)
def _kbn():
    return mx.fast.metal_kernel(
        name="tierglu_nax_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_BN, header=_HDRN)


@lru_cache(None)
def _krs():
    return mx.fast.metal_kernel(name="rowsum64", input_names=["a"], output_names=["out"], source=_RS)


def rowsum64(a):
    """[rows, L] -> [rows, L / 64] fp32: a fixed sequential sum per (row, 64-value group)"""
    R_, L = a.shape
    (o,) = _krs()(inputs=[a], template=[("T", a.dtype), ("L_C", L)], grid=(L // 64, R_, 1), threadgroup=(L // 64, 1, 1),
                  output_shapes=[(R_, L // 64)], output_dtypes=[mx.float32])
    return o


def tier_glu_nax(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, grp=None, nsg_b=2, exq=0,
                 ms_np=None, ls_np=None, diag=0, pf=1):
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 16 and K % 64 == 0 and N % 64 == 0 and D % (32 * nsg_b) == 0
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
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("EXQ_C", exq), ("DIAG_C", diag), ("PF_C", pf)]
    xs = rowsum64(x)
    (h,) = _kan()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N)],
                  grid=(N // 32 * 64, nseg, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = _kbn()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b, nseg, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
