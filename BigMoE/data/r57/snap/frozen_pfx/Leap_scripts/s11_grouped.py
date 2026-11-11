"""S11 (Leap, Sun 2026-10-25; NewIdeas' question 10:36): 3c with the code decode shared by every pair of one distinct
expert. Per-pair 3c (MoonShots' tierglu, snapshot scripts/tierglu_v3d.py) decodes each expert's codes once per (row,
rank) pair. A 10.3-row check has 1.94 pairs per distinct expert (16 rows: ~2.6).
Grouped kernels here (copies of 3c's A and B):
  - one threadgroup = one chunk of up to GM pairs that share (expert slot, tier); pairs are sorted by that key on the
    GPU (one small kernel: stable rank by counting, no host sync), and threadgroups whose sorted position is not a
    chunk leader exit at once;
  - per 512 (A) / 256 (B) value block the lane loads every member's x / h chunk (3c's load_vector), then per weight row
    decodes its codes ONCE (3-bit bytes, or spread8 to 4-bit nibbles) and runs 3c's qdot for each member;
  - per pair, every float operation is 3c's, in 3c's order, so h, the per-rank y and the combined y must be BITWISE
    equal to 3c's (the combine is MoonShots' own _KC kernel, which matches 3c's fused combine bitwise).
Usage: [BigMoE venv] s11_grouped.py test | bench   -> ~/Leap/data/s11_{mode}.txt"""
import os, sys, time
from functools import lru_cache
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tierglu_v3d as TG
import b10_trellis as BT                                     # routing, passes, timing harness, 3c pools (read only)

MODE = sys.argv[1] if len(sys.argv) > 1 else "test"
OUT = os.path.expanduser(f"~/Leap/data/s11_{MODE}.txt")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


# sort pairs by key = slot * 4 + tier code (4-bit 2, 2-bit 1, 3-bit 0); order[i] = pair at sorted position i
_GRP = r"""
    constexpr int P = NP_C, NR = NR_C, GM = GM_C;
    threadgroup int key[P];
    const uint t = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
    for (uint i = t; i < (uint)P; i += nt) {
        const uint r = i % NR;
        key[i] = msb_slot[i] * 4 + (((MASK4_C >> r) & 1) ? 2 : (((MASK2_C >> r) & 1) ? 1 : 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < (uint)P; i += nt) {
        const int ki = key[i];
        int rank = 0, pos = 0;
        for (int j = 0; j < P; ++j) {
            const int kj = key[j];
            rank += (kj < ki || (kj == ki && j < (int)i)) ? 1 : 0;
            pos += (kj == ki && j < (int)i) ? 1 : 0;
        }
        order[rank] = (int)i;
        ks[rank] = ki;
        lead[rank] = (pos % GM == 0) ? 1 : 0;
    }
"""

# members of the chunk led by sorted position c: c, c + 1, ... while the key matches (at most GM)
_MEMBERS = r"""
    const uint c = threadgroup_position_in_grid.y;
    if (lead[c] == 0) return;
    const int kc = ks[c];
    uint pm[GM];
    int np = 0;
    for (int g = 0; g < GM; ++g) {
        const bool mem = (c + g < (uint)P) && (ks[c + g] == kc);
        pm[g] = mem ? (uint)order[c + g] : 0u;
        np += mem ? 1 : 0;
    }
    const uint r = pm[0] % NR, m = pm[0] / NR;              // tier, slot and LSB slot are shared by the chunk
    const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pm[0]];
"""

_AG = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C, RA = RA_C, G = 64, GM = GM_C, P = NP_C;
    constexpr int KB3 = K * 3 / 8, KG = K / G, KL = K / 8;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0: gate rows, 1: up rows
    const uint j0 = threadgroup_position_in_grid.x * RA;
""" + _MEMBERS + r"""
    const device uint8_t* w3 = (const device uint8_t*)(sg == 0 ? w3g : w3u) + ((size_t)e * N + j0) * KB3;
    const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * N + j0) * KG;
    float result[GM][RA];
    for (int g = 0; g < GM; ++g) for (int row = 0; row < RA; ++row) result[g][row] = 0.0f;
    if (tier == 4) {
        const uint i4 = popcount((uint)MASK4_C & ((1u << r) - 1u));
        const uint el = (uint)lsb_slot[m * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)(sg == 0 ? l1g : l1u) + ((size_t)el * N + j0) * KL;
        const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            float xt[GM][16];
            float sum[GM];
            for (int g = 0; g < GM; ++g)
                if (g < np) sum[g] = tg_load_vector<T, float, 16, 4>(x + (size_t)(pm[g] / NR) * K + lane * 16 + b * 512, xt[g]);
            for (int row = 0; row < RA; ++row) {
                uint q0, q1;
                if (LD_C == 1) {
                    const device uint16_t* cc = (const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6);
                    const uint u0 = cc[0], u1 = cc[1], u2 = cc[2];
                    const uint lw = *(const device uint16_t*)(l1 + row * KL + b * 64 + lane * 2);
                    q0 = spread8(u0 | ((u1 & 0xFFu) << 16), lw & 0xFFu);
                    q1 = spread8((u1 >> 8) | (u2 << 8), lw >> 8);
                } else {
                    const device uint8_t* cc = w3 + row * KB3 + b * 192 + lane * 6;
                    const device uint8_t* lb = l1 + row * KL + b * 64 + lane * 2;
                    q0 = spread8((uint)cc[0] | ((uint)cc[1] << 8) | ((uint)cc[2] << 16), lb[0]);
                    q1 = spread8((uint)cc[3] | ((uint)cc[4] << 8) | ((uint)cc[5] << 16), lb[1]);
                }
                const uint16_t ws[4] = {(uint16_t)(q0 & 0xFFFFu), (uint16_t)(q0 >> 16), (uint16_t)(q1 & 0xFFFFu), (uint16_t)(q1 >> 16)};
                const int gi = row * KG + b * 8 + lane / 4;
                const float s = (float)s3[gi] * 0.5f;
                const float bb = bl[gi];
                for (int g = 0; g < GM; ++g)
                    if (g < np) result[g][row] += tg_qdot4<float, 16>(ws, xt[g], s, bb, sum[g]);
            }
        }
    } else {
        const device T* b3 = tier == 2 ? (sg == 0 ? b2g : b2u) + ((size_t)e * N + j0) * KG
                                       : (sg == 0 ? b3g : b3u) + ((size_t)e * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            float xt[GM][16];
            float sum[GM];
            for (int g = 0; g < GM; ++g)
                if (g < np) sum[g] = tg_load_vector<T, float, 16, 3>(x + (size_t)(pm[g] / NR) * K + lane * 16 + b * 512, xt[g]);
            for (int row = 0; row < RA; ++row) {
                uint8_t wb[6];
                if (LD_C == 1) {
                    const device uint16_t* cc = (const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6);
                    const uint16_t u0 = cc[0] & (tier == 2 ? 0x6DB6 : 0xFFFF), u1 = cc[1] & (tier == 2 ? 0xB6DB : 0xFFFF),
                                   u2 = cc[2] & (tier == 2 ? 0xDB6D : 0xFFFF);
                    wb[0] = (uint8_t)u0; wb[1] = (uint8_t)(u0 >> 8); wb[2] = (uint8_t)u1;
                    wb[3] = (uint8_t)(u1 >> 8); wb[4] = (uint8_t)u2; wb[5] = (uint8_t)(u2 >> 8);
                } else {
                    tg_codes3(w3 + row * KB3, b * 512 + lane * 16, 6, wb, tier == 2);
                }
                const int gi = row * KG + b * 8 + lane / 4;
                const float s = s3[gi];
                const float bb = b3[gi];
                for (int g = 0; g < GM; ++g)
                    if (g < np) result[g][row] += tg_qdot3<float, 16>(wb, xt[g], s, bb, sum[g]);
            }
        }
    }
    threadgroup T gate_v[GM][RA];
    for (int g = 0; g < GM; ++g) {
        if (g < np) {
            for (int row = 0; row < RA; ++row) result[g][row] = simd_sum(result[g][row]);
            if (sg == 0 && lane == 0) {
                for (int row = 0; row < RA; ++row) gate_v[g][row] = static_cast<T>(result[g][row]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1 && lane < (uint)RA) {
        for (int g = 0; g < GM; ++g) {
            if (g < np) {
                float u = 0.0f;
                for (int row = 0; row < RA; ++row) u = (row == (int)lane) ? result[g][row] : u;
                const T gg = gate_v[g][lane];
                const T ub = static_cast<T>(u);
                h[(size_t)pm[g] * N + j0 + lane] = TgMultiply()(TgMultiply()(gg, TgSigmoid()(gg)), ub);
            }
        }
    }
"""

_BG = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C, RB = RB_C, G = 64, GM = GM_C, P = NP_C;
    constexpr int NB3 = N * 3 / 8, NG = N / G, NL = N / 8;
    constexpr int BS = 256;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
""" + _MEMBERS + r"""
    const device uint8_t* w3 = (const device uint8_t*)w3d + ((size_t)e * D + d0) * NB3;
    const device T* s3 = s3d + ((size_t)e * D + d0) * NG;
    float result[GM][RB];
    for (int g = 0; g < GM; ++g) for (int row = 0; row < RB; ++row) result[g][row] = 0.0f;
    if (tier == 4) {
        const uint i4 = popcount((uint)MASK4_C & ((1u << r) - 1u));
        const uint el = (uint)lsb_slot[m * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)l1d + ((size_t)el * D + d0) * NL;
        const device T* bl = bd + ((size_t)el * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float xt[GM][8];
            float sum[GM];
            for (int g = 0; g < GM; ++g)
                if (g < np) sum[g] = tg_load_vector<T, float, 8, 4>(h + (size_t)pm[g] * N + lane * 8 + k, xt[g]);
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* cc = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint q = spread8((uint)cc[0] | ((uint)cc[1] << 8) | ((uint)cc[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                const uint16_t ws[2] = {(uint16_t)(q & 0xFFFFu), (uint16_t)(q >> 16)};
                const int gi = row * NG + (k + lane * 8) / G;
                const float s = (float)s3[gi] * 0.5f;
                const float bb = (float)bl[gi];
                for (int g = 0; g < GM; ++g)
                    if (g < np) result[g][row] += tg_qdot4<float, 8>(ws, xt[g], s, bb, sum[g]);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            float xt[GM][8];
            float sum[GM];
            for (int g = 0; g < GM; ++g)
                if (g < np) sum[g] = tg_load_vector_safe<T, float, 8, 4>(h + (size_t)pm[g] * N + lane * 8 + k, xt[g], remaining);
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* cc = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint q = spread8((uint)cc[0] | ((uint)cc[1] << 8) | ((uint)cc[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                const uint16_t ws[2] = {(uint16_t)(q & 0xFFFFu), (uint16_t)(q >> 16)};
                const int gi = row * NG + (k + lane * 8) / G;
                const float s = (float)s3[gi] * 0.5f;
                const float bb = (float)bl[gi];
                for (int g = 0; g < GM; ++g)
                    if (g < np) result[g][row] += tg_qdot4<float, 8>(ws, xt[g], s, bb, sum[g]);
            }
        }
    } else {
        const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float xt[GM][8];
            float sum[GM];
            for (int g = 0; g < GM; ++g)
                if (g < np) sum[g] = tg_load_vector<T, float, 8, 3>(h + (size_t)pm[g] * N + lane * 8 + k, xt[g]);
            for (int row = 0; row < RB; ++row) {
                uint8_t wb[3];
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, tier == 2);
                const int gi = row * NG + (k + lane * 8) / G;
                const float s = (float)s3[gi];
                const float bb = (float)b3[gi];
                for (int g = 0; g < GM; ++g)
                    if (g < np) result[g][row] += tg_qdot3<float, 8>(wb, xt[g], s, bb, sum[g]);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            float xt[GM][8];
            float sum[GM];
            for (int g = 0; g < GM; ++g)
                if (g < np) sum[g] = tg_load_vector_safe<T, float, 8, 3>(h + (size_t)pm[g] * N + lane * 8 + k, xt[g], remaining);
            for (int row = 0; row < RB; ++row) {
                uint8_t wb[3];
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, tier == 2);
                const int gi = row * NG + (k + lane * 8) / G;
                const float s = (float)s3[gi];
                const float bb = (float)b3[gi];
                for (int g = 0; g < GM; ++g)
                    if (g < np) result[g][row] += tg_qdot3<float, 8>(wb, xt[g], s, bb, sum[g]);
            }
        }
    }
    for (int g = 0; g < GM; ++g) {
        if (g < np) {
            for (int row = 0; row < RB; ++row) {
                const float v = simd_sum(result[g][row]);
                if (lane == 0) y[(size_t)pm[g] * D + d0 + row] = static_cast<T>(v);
            }
        }
    }
"""


@lru_cache(None)
def _kg():
    return mx.fast.metal_kernel(name="s11_group", input_names=["msb_slot"], output_names=["order", "ks", "lead"],
                                source=_GRP)


@lru_cache(None)
def _kag():
    return mx.fast.metal_kernel(
        name="s11_a", input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                                   "msb_slot", "lsb_slot", "order", "ks", "lead"],
        output_names=["h"], source=_AG, header=TG._HDR)


@lru_cache(None)
def _kbg():
    return mx.fast.metal_kernel(
        name="s11_b", input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "order", "ks", "lead"],
        output_names=["y"], source=_BG, header=TG._HDR)


def grouped_glu(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", scores=None, b2=None, ra=4, rb=4, nsg_b=2, gm=4,
                ly=8, ld=0, parts=False):
    """== TG.tier_glu(x, msb, lsb, msb_slot, lsb_slot, plan, scores=scores) bitwise, with the decode shared per chunk
    of up to gm pairs of one (slot, tier). parts=True also returns (h, per-rank y [M, R, D])."""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = TG.plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    P = M * R
    assert K % 512 == 0 and N % ra == 0 and D % (rb * nsg_b) == 0 and P <= 4096
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in TG.PROJ}
    if r4:
        ls, L = lsb_slot.astype(mx.int32).reshape(-1), lsb
    else:
        ls, L = mx.zeros((1,), mx.int32), {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in TG.PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    order, ks, lead = _kg()(inputs=[ms], template=[("NP_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("GM_C", gm)],
                            grid=(256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(P,), (P,), (P,)],
                            output_dtypes=[mx.int32] * 3)
    g, u, d = (msb[p] for p in TG.PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("GM_C", gm), ("NP_C", P)]
    (h,) = _kag()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, order, ks, lead],
                  template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                  grid=(N // ra * 64, P, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    (yp,) = _kbg()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls,
                           order, ks, lead],
                   template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                   grid=(D // (rb * nsg_b) * 32 * nsg_b, P, 1), threadgroup=(32 * nsg_b, 1, 1),
                   output_shapes=[(P, D)], output_dtypes=[x.dtype])
    if scores is None:
        return yp.reshape(M, R, D)
    (y,) = TG._kc()(inputs=[yp, yp, mx.arange(P, dtype=mx.int32), scores.astype(x.dtype).reshape(-1)],
                    template=[("T", x.dtype), ("NR_C", R), ("D_C", D), ("LY_C", ly)],
                    grid=(D, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return (y, h, yp.reshape(M, R, D)) if parts else y


def tg_h(x, msb, lsb, msb_slot, lsb_slot, plan, ra=4, ld=0):
    """3c's kernel A alone (h [M * R, N]) for the bitwise check of h."""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = TG.plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    bb = {p: msb[p]["b3"] for p in TG.PROJ}
    ls, L = (lsb_slot.astype(mx.int32).reshape(-1), lsb) if r4 else \
        (mx.zeros((1,), mx.int32), {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in TG.PROJ})
    g, u = msb["gate_proj"], msb["up_proj"]
    (h,) = TG._ka()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                            L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"],
                            msb_slot.astype(mx.int32).reshape(-1), ls],
                    template=[("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)),
                              ("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                    grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    return h


def lsb_for(plan, i):
    r4 = plan.count("4")
    return i[:, :r4] if r4 else None


def same(a, b):
    a, b = np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16))
    return a.shape == b.shape and bool((a == b).all()), int((a != b).sum())


if __name__ == "__main__":
    t0 = time.time()
    say(f"S11 {MODE} ({time.strftime('%a %H:%M')}): grouped-decode 3c vs per-pair 3c, layer {BT.LAYER}, MLX {mx.__version__}"
        + (" [QUIET MODE]" if BT.QUIET else ""))
    X, ids, wts = BT.routing()
    TGm, msb, lsb = BT.load_3c()
    say(f"  routing: layer {BT.LAYER} top-{BT.R} on {X.shape[0]} captured tokens; 3c pools loaded ({time.time() - t0:.0f} s)")
    PLANS = ("3333333333", "4444444444", "4433333333")
    if MODE == "test":
        ok = True
        for S in (1, 4, 8, 16):
            for a in (0, 300, 700):
                x = X[a:a + S]
                ii, ww = mx.array(ids[a:a + S]), mx.array(wts[a:a + S]).astype(mx.bfloat16)
                for plan in PLANS:
                    ls = lsb_for(plan, ii)
                    ref = TG.tier_glu(x, msb, lsb, ii, ls, plan=plan, scores=ww)
                    ref_p = TG.tier_glu(x, msb, lsb, ii, ls, plan=plan)
                    ref_h = tg_h(x, msb, lsb, ii, ls, plan)
                    for gm, ra, rb, nsg_b, ld in ((4, 4, 4, 2, 0), (2, 4, 4, 2, 0), (8, 2, 2, 2, 0), (4, 4, 4, 2, 1)):
                        y, h, yp = grouped_glu(x, msb, lsb, ii, ls, plan=plan, scores=ww, gm=gm, ra=ra, rb=rb, nsg_b=nsg_b,
                                               ld=ld, parts=True)
                        # ld changes only how 3c loads codes (same values): compare h with 3c at the same ld
                        rh = ref_h if ld == 0 else tg_h(x, msb, lsb, ii, ls, plan, ld=1)
                        # ra / rb change only the grid split of rows (per-row arithmetic unchanged)
                        r1, r2, r3 = same(h, rh), same(yp, ref_p), same(y, ref)
                        good = r1[0] and r2[0] and r3[0]
                        ok &= good
                        if not good or (a == 0 and gm == 4 and ld == 0):
                            say(f"  S {S:2d} tok {a:3d} plan {plan} gm {gm} ra {ra} rb {rb} ld {ld}: h {r1}, per-rank y {r2}, "
                                f"combined y {r3}  {'OK' if good else 'MISMATCH'}")
        say(f"  BITWISE vs per-pair 3c on every case: {ok}")
        assert ok
    elif MODE == "diag":                                    # where the grouped cost goes: machinery vs sharing
        plan = "3333333333"
        P0 = None

        def unfused(y, i, w):                               # 3c per-rank + MoonShots' separate combine (grouped's tail)
            yp = TG.tier_glu(y, msb, lsb, i, None, plan=plan).reshape(-1, y.shape[1])
            (out,) = TG._kc()(inputs=[yp, yp, mx.arange(yp.shape[0], dtype=mx.int32), w.astype(y.dtype).reshape(-1)],
                              template=[("T", y.dtype), ("NR_C", 10), ("D_C", y.shape[1]), ("LY_C", 8)],
                              grid=(y.shape[1], y.shape[0], 1), threadgroup=(256, 1, 1),
                              output_shapes=[(y.shape[0], y.shape[1])], output_dtypes=[y.dtype])
            return out

        def group_only(y, i, w):                            # the sort kernel alone (+ a tiny MLX tail to chain it)
            o, k_, l_ = _kg()(inputs=[i.astype(mx.int32).reshape(-1)],
                              template=[("NP_C", i.size), ("NR_C", 10), ("MASK4_C", 0), ("MASK2_C", 0), ("GM_C", 2)],
                              grid=(256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(i.size,)] * 3,
                              output_dtypes=[mx.int32] * 3)
            return y * 0 + (o[:1] + k_[:1] + l_[:1]).astype(y.dtype)

        for S in (10, 16):
            ps = BT.passes(ids, wts, S)
            x = X[:S]
            arms = [("3c fused", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, None, plan=plan, scores=w)),
                    ("3c per-rank + separate combine", unfused),
                    ("sort kernel alone", group_only)]
            for gm in (1, 2):
                arms.append((f"grouped gm {gm}", lambda y, i, w, gm=gm: grouped_glu(y, msb, lsb, i, None, plan=plan, scores=w, gm=gm)))
            for nm, ms_ in BT.timed_arms(arms, x, ps, rounds=7).items():
                say(f"  S {S:2d} {nm:32s}: {ms_:.3f} ms / layer")
    else:
        rounds = int(os.environ.get("S11_ROUNDS", "5"))
        res = {}
        for S in (4, 8, 10, 11, 16):
            ps = BT.passes(ids, wts, S)
            x = X[:S]
            dist = np.mean([p[2] for p in ps])
            arms = []
            for plan in ("3333333333", "4433333333"):
                arms.append((f"3c {plan}", lambda y, i, w, plan=plan: TG.tier_glu(y, msb, lsb, i, lsb_for(plan, i), plan=plan, scores=w)))
                for gm, ra, rb, nsg_b in ((2, 4, 4, 2), (4, 4, 4, 2), (4, 2, 2, 2), (8, 2, 2, 2)):
                    arms.append((f"grouped {plan} gm {gm} ra {ra} rb {rb}",
                                 lambda y, i, w, plan=plan, c=dict(gm=gm, ra=ra, rb=rb, nsg_b=nsg_b):
                                 grouped_glu(y, msb, lsb, i, lsb_for(plan, i), plan=plan, scores=w, **c)))
            for nm, ms_ in BT.timed_arms(arms, x, ps, rounds=rounds).items():
                res[(S, nm)] = ms_
                say(f"  S {S:2d} (distinct {dist:5.1f}, pairs {S * BT.R:3d}) {nm:38s}: {ms_:.3f} ms / layer, "
                    f"{ms_ / (S * BT.R) * 1e3:.2f} us / pair, distinct bytes at {dist * 2.162688 / ms_:.0f} GB/s (3-bit)")
        for plan in ("3333333333", "4433333333"):
            base = {S: res[(S, f"3c {plan}")] for S in (4, 8, 10, 11, 16)}
            for nm in sorted({k[1] for k in res if k[1].startswith(f"grouped {plan}")}):
                d = {S: base[S] - res[(S, nm)] for S in base}
                say(f"  {nm}: saved ms / layer S4 {d[4]:+.3f} S8 {d[8]:+.3f} S10 {d[10]:+.3f} S11 {d[11]:+.3f} S16 {d[16]:+.3f}"
                    f" -> per 48-layer check at 10.3 rows {(d[10] + d[11]) / 2 * 48:+.1f} ms, at 16 rows {d[16] * 48:+.1f} ms")
    say(f"  done in {time.time() - t0:.0f} s")
    open(OUT, "w").write("\n".join(lines) + "\n")
