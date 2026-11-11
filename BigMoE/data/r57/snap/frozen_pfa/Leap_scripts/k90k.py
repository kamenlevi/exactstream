"""K90 kernels (Leap, Wed 2026-11-04 18:1x; MISSION 33, ~/FableGod/KERNEL90.md section 1).
Arm 0 STREAM: NAX6-N's kernels A and B (tierglu_nax6n: the same prologue, grid, threadgroups and load instructions for
the native code words, LSB words, scales and biases of every (group, tile)) with the decode, the matrix op, the epilogue,
the chunk sums and the stores removed: the loaded words are xor-folded into one uint per lane and stored only if it hits
a constant (never), so no load is dead code. Its time = the streaming time of NAX6-N's own access pattern: the
denominator R = B / t(STREAM).
stream(x, msb_n, lsb_n, msb_slot, lsb_slot, grp, plan) -> (h_dummy, y_dummy); bf16 native pools only."""
from functools import lru_cache
import mlx.core as mx
import os, sys
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6

_CORE_S = r"""
    constexpr int KG = S_K / 64;
    const device uint32_t* wrow = (const device uint32_t*)w3p + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32 * 6) + lane * 6;
    const device T* srow = s3p + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32) + lane;
    const device T* brow;
    const device uint32_t* lrow = (const device uint32_t*)l1p;
    if (tier == 4) {
        lrow = (const device uint32_t*)l1p + ((size_t)el * (S_O / 32) + n0 / 32) * (KG * 32 * 2) + lane * 2;
        brow = blp + ((size_t)el * (S_O / 32) + n0 / 32) * (KG * 32) + lane;
    } else {
        brow = (tier == 2 ? b2p : b3p) + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32) + lane;
    }
    constexpr int GPC = KG / GS_C / KS_C;
    uint xr = 0u;
    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {
        for (int j = 0; j < GS_C; ++j) {
            const int g = gp * GS_C + j;
            const device uint32_t* c = wrow + 192 * g;
            xr ^= c[0] ^ c[1] ^ c[2] ^ c[3] ^ c[4] ^ c[5];
            if (tier == 4) xr ^= lrow[64 * g] ^ lrow[64 * g + 1];
            xr ^= (uint)as_type<ushort>(srow[32 * g]) ^ ((uint)as_type<ushort>(brow[32 * g]) << 16);
        }
    }
"""
_PRE_A, _ = N6._A4.split(N6._CORE4)
_PRE_B, _ = N6._B4.split(N6._CORE4)
_A_S = _PRE_A + _CORE_S + "    if (xr == 0x9E3779B9u) h[(size_t)gq] = static_cast<T>(1.0f);\n"
_B_S = _PRE_B + _CORE_S + "    if (xr == 0x9E3779B9u) y[(size_t)gq] = static_cast<T>(1.0f);\n"


@lru_cache(None)
def _ksa():
    return mx.fast.metal_kernel(
        name="k90_stream_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A_S, header=N6._HDR4)


@lru_cache(None)
def _ksb():
    return mx.fast.metal_kernel(
        name="k90_stream_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B_S, header=N6._HDR4)


def stream(x, msb_n, lsb_n, msb_slot, lsb_slot, grp, plan="4433333333", nsg_b=2, ks=4, ks_b=2):
    """arm 0: kernels A and B of NAX6-N reading their bytes only; B depends on A's (dummy) output"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = mx.zeros((M, K // 64), mx.float32)
    (h,) = _ksa()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                  grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = mx.zeros((M * R, N // 64), mx.float32)
    (y,) = _ksb()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y


# ---- arm 3, NAX-CN: NAX-C (tierglu_naxc.py: int8 activations per (row, 64-group), int4 codes, int32 sums, the affine
# epilogue) on NAX6-N's native addressing. NAX-C's swaps (operand types, code - 8, int8 A loads, epilogue) and NAX6-N's
# (record addresses) touch disjoint lines, so NAX-CN == NAX-C on the row-major pools, bitwise (checked). ----
import tierglu_naxc as NC
import tierglu_nax6n as NN
_NATIVE_PAIRS = [                                       # tierglu_nax6n._CORE_N's address swaps, verbatim
    ("const device uint32_t* wrow = (const device uint32_t*)((const device uint8_t*)w3p + ((size_t)e * S_O + n0 + (DIAG_C == 3 ? 0 : lane)) * KB3);",
     "const device uint32_t* wrow = (const device uint32_t*)w3p + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32 * 6) + lane * 6;"),
    ("const device T* srow = s3p + ((size_t)e * S_O + n0 + lane) * KG;",
     "const device T* srow = s3p + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32) + lane;"),
    ("lrow = (const device uint32_t*)((const device uint8_t*)l1p + ((size_t)el * S_O + n0 + lane) * KL);",
     "lrow = (const device uint32_t*)l1p + ((size_t)el * (S_O / 32) + n0 / 32) * (KG * 32 * 2) + lane * 2;"),
    ("brow = blp + ((size_t)el * S_O + n0 + lane) * KG;",
     "brow = blp + ((size_t)el * (S_O / 32) + n0 / 32) * (KG * 32) + lane;"),
    ("brow = (tier == 2 ? b2p : b3p) + ((size_t)e * S_O + n0 + lane) * KG;",
     "brow = (tier == 2 ? b2p : b3p) + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32) + lane;"),
    ("const device uint32_t* c = wrow + 6 * g;", "const device uint32_t* c = wrow + 192 * g;"),
    ("const uint l0 = lrow[2 * g], l1 = lrow[2 * g + 1];", "const uint l0 = lrow[64 * g], l1 = lrow[64 * g + 1];"),
    ("sb[64 * j + lane] = (float)srow[g] * smul;", "sb[64 * j + lane] = (float)srow[32 * g] * smul;"),
    ("sb[64 * j + 32 + lane] = (float)brow[g];", "sb[64 * j + 32 + lane] = (float)brow[32 * g];"),
]
assert NN._swap(N6._CORE4, _NATIVE_PAIRS) == NN._CORE_N, "the copied address swaps no longer match tierglu_nax6n"
_CORE_CN = NN._swap(NC._CORE_C, _NATIVE_PAIRS)
_A_CN = NC._A_C.replace(NC._CORE_C, _CORE_CN)
_B_CN = NC._B_C.replace(NC._CORE_C, _CORE_CN)
assert _CORE_CN in _A_CN and _CORE_CN in _B_CN


@lru_cache(None)
def _kacn():
    return mx.fast.metal_kernel(
        name="k90_naxcn_a",
        input_names=["xq", "xs", "xqs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A_CN, header=N6._HDR4)


@lru_cache(None)
def _kbcn():
    return mx.fast.metal_kernel(
        name="k90_naxcn_b",
        input_names=["hq", "hs", "hqs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B_CN, header=N6._HDR4)


def tier_glu_naxcn(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2):
    """NAX-C's arithmetic on the native pools (bf16 scales) with NAX6's GPU grouping: [M, R, D]"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4 and M <= 64
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xq, xs, xqs = NC.quant64(x)
    (h,) = _kacn()(inputs=[xq, xs, xqs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                           pidx, seg, nsg_],
                   template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                   grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                   output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hq, hs, hqs = NC.quant64(h)
    (y,) = _kbcn()(inputs=[hq, hs, hqs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                           ms, ls, pidx, seg, nsg_],
                   template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                   grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                   output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---- arm 4, NAX8-P1: persistent kernels A and B (NAX6-N's inner loop and tails verbatim). Each threadgroup loops over
# work items from a device counter (thread 0: atomic fetch-add, relaxed; one threadgroup barrier broadcasts the item):
# A items (group g, gate / up tile t), t fastest (today's dispatch order), NA = nseg x N/32; B items (g, tile pair j),
# NB = nseg x D/64. Only real groups (no empty threadgroups). The counters are zeroed by gpu_grp_ctr (gpu_grp + one
# store each). Grid = 20 cores x RES threadgroups. The per-item arithmetic is NAX6-N's, so bitwise == NAX6. ----
_GRP_CTR = N6._GRP + "    if ((int)i == 0) { ctr[0] = 0; ctr[1] = 0; }\n"


@lru_cache(None)
def _kgrp_ctr():
    return mx.fast.metal_kernel(name="k90_grp_ctr", input_names=["msb_slot", "lsb_slot"],
                                output_names=["pidx", "seg", "nseg", "ctr"], source=_GRP_CTR)


def gpu_grp_ctr(msb_slot, lsb_slot, plan):
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = msb_slot.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024, P
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pidx, seg, nseg, ctr = _kgrp_ctr()(inputs=[ms, ls], template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2),
                                                                   ("R4_C", max(r4, 1))],
                                       grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                                       output_shapes=[(P,), (P + 1,), (1,), (32,)], output_dtypes=[mx.int32] * 4)   # 32: MLX binds
    # inputs of < 8 elements in the constant address space (no atomics there); ctr[0] / ctr[1] are the two counters
    return pidx, seg, nseg, P, ctr


def _persist(src, which, ntiles, n0_old, n0_new, tb_line, sb_line):
    head = (f"    threadgroup int item_s[1];\n{tb_line}\n{sb_line}\n"
            f"    device atomic_int* actr = (device atomic_int*)((device int*)ctr) + {which};\n"
            f"    const int NT_ = {ntiles}, NI_ = nseg[0] * NT_;\n"
            "    for (;;) {\n"
            "    if (thread_index_in_threadgroup == 0) item_s[0] = atomic_fetch_add_explicit(actr, 1, memory_order_relaxed);\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    const int item_ = item_s[0];\n"
            "    if (item_ >= NI_) break;\n"
            "    const uint gq = (uint)(item_ / NT_);")
    s = NN._swap(src, [(tb_line, ""), (sb_line, ""),
                       ("const uint gq = threadgroup_position_in_grid.y;\n    if ((int)gq >= nseg[0]) return;", head),
                       (n0_old, n0_new)])
    return s + "    }\n"


_A_P1 = _persist(NN._A_N, 0, "N / 32", "const uint n0 = threadgroup_position_in_grid.x * 32;",
                 "const uint n0 = (uint)(item_ % NT_) * 32;",
                 "    threadgroup uint32_t tb_all[2 * KS_C * 2 * 32 * 8 * GS_C];     // later reused for the chunk accumulators",
                 "    threadgroup float sb_all[2 * KS_C * 2 * 64 * GS_C];")
_B_P1 = _persist(NN._B_N, 1, "D / (32 * NSG_C)", "const uint n0 = (threadgroup_position_in_grid.x * NSG_C + ti) * 32;",
                 "const uint n0 = ((uint)(item_ % NT_) * NSG_C + ti) * 32;",
                 "    threadgroup uint32_t tb_all[NSG_C * KS_C * 2 * 32 * 8 * GS_C];  // later reused for the chunk accumulators",
                 "    threadgroup float sb_all[NSG_C * KS_C * 2 * 64 * GS_C];")


@lru_cache(None)
def _kap1():
    return mx.fast.metal_kernel(
        name="k90_p1_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "ctr"],
        output_names=["h"], source=_A_P1, header=N6._HDR4)


@lru_cache(None)
def _kbp1():
    return mx.fast.metal_kernel(
        name="k90_p1_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "ctr"],
        output_names=["y"], source=_B_P1, header=N6._HDR4)


CORES = 20


def tier_glu_nax8p1(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", res_a=3, res_b=4, nsg_b=2, ks=4, ks_b=2):
    """NAX6-N as persistent kernels A and B (bf16 native pools); == tier_glu_nax6n bitwise"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    pidx, seg, nsg_, nseg, ctr = gpu_grp_ctr(msb_slot, lsb_slot, plan)
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = NN.rowsum64(x)
    (h,) = _kap1()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                           pidx, seg, nsg_, ctr],
                   template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                   grid=(CORES * res_a * 64 * ks, 1, 1), threadgroup=(64 * ks, 1, 1),
                   output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = NN.rowsum64(h)
    (y,) = _kbp1()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                           ms, ls, pidx, seg, nsg_, ctr],
                   template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                   grid=(CORES * res_b * 32 * nsg_b * ks_b, 1, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                   output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---- probes 0b / 0c (Fable A 18:1x): STREAM with U quant groups of loads issued per simdgroup before the first use
# (same addresses, same grid); and STREAM as persistent kernels (P1's loop around the loads-only body) ----
_LOOP_S = '''    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {
        for (int j = 0; j < GS_C; ++j) {
            const int g = gp * GS_C + j;
            const device uint32_t* c = wrow + 192 * g;
            xr ^= c[0] ^ c[1] ^ c[2] ^ c[3] ^ c[4] ^ c[5];
            if (tier == 4) xr ^= lrow[64 * g] ^ lrow[64 * g + 1];
            xr ^= (uint)as_type<ushort>(srow[32 * g]) ^ ((uint)as_type<ushort>(brow[32 * g]) << 16);
        }
    }
'''
_LOOP_U = '''    const int gend = (kc + 1) * GPC * GS_C;
    for (int g0 = kc * GPC * GS_C; g0 < gend; g0 += U_C) {
        uint wv[U_C][6], lv[U_C][2], sv[U_C];
        for (int u = 0; u < U_C; ++u) {
            const int g = min(g0 + u, gend - 1);
            const device uint32_t* c = wrow + 192 * g;
            for (int w = 0; w < 6; ++w) wv[u][w] = c[w];
            lv[u][0] = tier == 4 ? lrow[64 * g] : 0u;
            lv[u][1] = tier == 4 ? lrow[64 * g + 1] : 0u;
            sv[u] = (uint)as_type<ushort>(srow[32 * g]) ^ ((uint)as_type<ushort>(brow[32 * g]) << 16);
        }
        for (int u = 0; u < U_C; ++u)
            xr ^= wv[u][0] ^ wv[u][1] ^ wv[u][2] ^ wv[u][3] ^ wv[u][4] ^ wv[u][5] ^ lv[u][0] ^ lv[u][1] ^ sv[u];
    }
'''
assert _CORE_S.count(_LOOP_S) == 1
_A_SU, _B_SU = _A_S.replace(_LOOP_S, _LOOP_U), _B_S.replace(_LOOP_S, _LOOP_U)
_A_SP = _persist(_A_S, 0, "N / 32", "const uint n0 = threadgroup_position_in_grid.x * 32;",
                 "const uint n0 = (uint)(item_ % NT_) * 32;",
                 "    threadgroup uint32_t tb_all[2 * KS_C * 2 * 32 * 8 * GS_C];     // later reused for the chunk accumulators",
                 "    threadgroup float sb_all[2 * KS_C * 2 * 64 * GS_C];")
_B_SP = _persist(_B_S, 1, "D / (32 * NSG_C)", "const uint n0 = (threadgroup_position_in_grid.x * NSG_C + ti) * 32;",
                 "const uint n0 = ((uint)(item_ % NT_) * NSG_C + ti) * 32;",
                 "    threadgroup uint32_t tb_all[NSG_C * KS_C * 2 * 32 * 8 * GS_C];  // later reused for the chunk accumulators",
                 "    threadgroup float sb_all[NSG_C * KS_C * 2 * 64 * GS_C];")
_IN_A = ["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
         "msb_slot", "lsb_slot", "pidx", "seg", "nseg"]
_IN_B = ["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"]


@lru_cache(None)
def _kprobe(which):
    src = {"au": _A_SU, "bu": _B_SU, "ap": _A_SP, "bp": _B_SP}[which]
    ins = (_IN_A if which[0] == "a" else _IN_B) + (["ctr"] if which[1] == "p" else [])
    return mx.fast.metal_kernel(name=f"k90_probe_{which}", input_names=ins, output_names=["h" if which[0] == "a" else "y"],
                                source=src, header=N6._HDR4)


def stream_probe(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", u=1, res=None, nsg_b=2, ks=4, ks_b=2):
    """STREAM with its grouping inside the call: u = quant groups of loads in flight per simdgroup (u 1 = arm 0's
    kernels); res = (RES_A, RES_B) -> the persistent loads-only kernels"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if res is None:
        pidx, seg, nsg_, nseg = N6.gpu_grp(msb_slot, lsb_slot, plan)
        ctr = None
    else:
        pidx, seg, nsg_, nseg, ctr = gpu_grp_ctr(msb_slot, lsb_slot, plan)
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, uu, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = mx.zeros((M, K // 64), mx.float32)
    ins_a = [x, xs, g["w3"], uu["w3"], g["s3"], uu["s3"], g["b3"], uu["b3"], bb["gate_proj"], bb["up_proj"],
             L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg, nsg_]
    if res is None and u == 1:
        ka, kb, ta, tb_ = _ksa(), _ksb(), [], []
    elif res is None:
        ka, kb, ta, tb_ = _kprobe("au"), _kprobe("bu"), [("U_C", u)], [("U_C", u)]
    else:
        ka, kb, ta, tb_ = _kprobe("ap"), _kprobe("bp"), [], []
        ins_a = ins_a + [ctr]
    ga = (N // 32 * 64 * ks, nseg, 1) if res is None else (CORES * res[0] * 64 * ks, 1, 1)
    (h,) = ka(inputs=ins_a, template=tmpl + ta + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=ga,
              threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = mx.zeros((M * R, N // 64), mx.float32)
    ins_b = [h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls, pidx, seg, nsg_]
    if res is not None:
        ins_b = ins_b + [ctr]
    gb = (D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1) if res is None else (CORES * res[1] * 32 * nsg_b * ks_b, 1, 1)
    (y,) = kb(inputs=ins_b, template=tmpl + tb_ + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)], grid=gb,
              threadgroup=(32 * nsg_b * ks_b, 1, 1), output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y


# ---- arm 7, NAX8-GU (kernel A only; B = NAX6-N's): gate and up in one simdgroup. Threadgroup = KS_C simdgroups (the K
# chunks); each decodes BOTH projections' codes for its chunk (its own double buffers: the same 20 KB in all), marshals the
# A operand once per (group, half) and runs two ops (gate, up) into two accumulators; every accumulator sees NAX6's op and
# epilogue order, so bitwise == NAX6. ----
def _dec(sfx):
    return f'''            {{
                const device uint32_t* c = wrow_{sfx} + 192 * g;
                const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
                uint ch[8] = {{w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                              w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8}};
                uint nb[8];
                if (tier == 4) {{
                    const uint l0 = lrow_{sfx}[64 * g], l1 = lrow_{sfx}[64 * g + 1];
                    for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
                }} else if (tier == 2) {{
                    for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
                }} else {{
                    for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
                }}
                threadgroup uint4* dst = (threadgroup uint4*)(tb_{sfx} + lane * 8 * GS_C + 8 * j);
                dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
                dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
                sb_{sfx}[64 * j + lane] = (float)srow_{sfx}[32 * g] * smul;
                sb_{sfx}[64 * j + 32 + lane] = (float)brow_{sfx}[32 * g];
            }}
'''


_CORE_GU = r"""
    constexpr int KG = S_K / 64;
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const size_t tbase = (size_t)e * (S_O / 32) + n0 / 32;
    const device uint32_t* wrow_g = (const device uint32_t*)w3g + tbase * (KG * 32 * 6) + lane * 6;
    const device uint32_t* wrow_u = (const device uint32_t*)w3u + tbase * (KG * 32 * 6) + lane * 6;
    const device T* srow_g = s3g + tbase * (KG * 32) + lane;
    const device T* srow_u = s3u + tbase * (KG * 32) + lane;
    const device T* brow_g;
    const device T* brow_u;
    const device uint32_t* lrow_g = (const device uint32_t*)l1g;
    const device uint32_t* lrow_u = (const device uint32_t*)l1u;
    if (tier == 4) {
        const size_t lbase = (size_t)el * (S_O / 32) + n0 / 32;
        lrow_g = (const device uint32_t*)l1g + lbase * (KG * 32 * 2) + lane * 2;
        lrow_u = (const device uint32_t*)l1u + lbase * (KG * 32 * 2) + lane * 2;
        brow_g = bg + lbase * (KG * 32) + lane;
        brow_u = bu + lbase * (KG * 32) + lane;
    } else {
        brow_g = (tier == 2 ? b2g : b3g) + tbase * (KG * 32) + lane;
        brow_u = (tier == 2 ? b2u : b3u) + tbase * (KG * 32) + lane;
    }
    const float smul = tier == 4 ? 0.5f : 1.0f;
    threadgroup uint32_t* tbw_g = tb_all + (kc * 2 + 0) * (2 * 32 * 8 * GS_C);
    threadgroup uint32_t* tbw_u = tb_all + (kc * 2 + 1) * (2 * 32 * 8 * GS_C);
    threadgroup float* sbw_g = sb_all + (kc * 2 + 0) * (2 * 64 * GS_C);
    threadgroup float* sbw_u = sb_all + (kc * 2 + 1) * (2 * 64 * GS_C);
    constexpr int GPC = KG / GS_C / KS_C;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg_g = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    auto cg_u = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc_g[16], acc_u[16];
    for (int i = 0; i < 16; ++i) { acc_g[i] = 0.0f; acc_u[i] = 0.0f; }
    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {
        threadgroup uint32_t* tb_g = tbw_g + (gp & 1) * (32 * 8 * GS_C);
        threadgroup uint32_t* tb_u = tbw_u + (gp & 1) * (32 * 8 * GS_C);
        threadgroup float* sb_g = sbw_g + (gp & 1) * (64 * GS_C);
        threadgroup float* sb_u = sbw_u + (gp & 1) * (64 * GS_C);
        for (int j = 0; j < GS_C; ++j) {
            const int g = gp * GS_C + j;
""" + _dec("g") + _dec("u") + r"""
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < GS_C; ++j) {
            const int g = gp * GS_C + j;
            for (int i = 0; i < 16; ++i) { cg_g[i] = 0.0f; cg_u[i] = 0.0f; }
            for (int hh = 0; hh < 2; ++hh) {
                const int k0 = 64 * g + 32 * hh;
                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
                for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
                TB tbt_g((threadgroup uchar*)tb_g + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                op.run(ca, tbt_g, cg_g);
                TB tbt_u((threadgroup uchar*)tb_u + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                op.run(ca, tbt_u, cg_u);
            }
            const float x0 = as0[g], x1 = as1[g];
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + fn0 + (i & 3);
                acc_g[i] += sb_g[64 * j + col] * cg_g[i] + sb_g[64 * j + 32 + col] * (((i >> 2) & 1) ? x1 : x0);
                acc_u[i] += sb_u[64 * j + col] * cg_u[i] + sb_u[64 * j + 32 + col] * (((i >> 2) & 1) ? x1 : x0);
            }
        }
    }
"""
_A_GU = NN._swap(N6._A4, [
    ("const uint pj = sg & 1u, kc = sg >> 1;", "const uint kc = sg;"),
    ("    const device uint32_t* w3p = pj == 0 ? w3g : w3u;\n", ""),
    ("    const device T* s3p = pj == 0 ? s3g : s3u;\n", ""),
    ("    const device T* b3p = pj == 0 ? b3g : b3u;\n", ""),
    ("    const device T* b2p = pj == 0 ? b2g : b2u;\n", ""),
    ("    const device uint32_t* l1p = pj == 0 ? l1g : l1u;\n", ""),
    ("    const device T* blp = pj == 0 ? bg : bu;\n", ""),
    (N6._CORE4, _CORE_GU),
    ("    for (int i = 0; i < 16; ++i) part[(pj * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];",
     "    for (int i = 0; i < 16; ++i) {\n"
     "        part[(0 * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc_g[i];\n"
     "        part[(1 * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc_u[i];\n"
     "    }"),
])


@lru_cache(None)
def _kagu():
    return mx.fast.metal_kernel(name="k90_gu_a", input_names=_IN_A, output_names=["h"], source=_A_GU, header=N6._HDR4)


def tier_glu_nax8gu(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2):
    """NAX6-N with kernel A as NAX8-GU (gate + up per simdgroup); B = NAX6-N's; bf16 native pools"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = NN.rowsum64(x)
    (h,) = _kagu()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                           pidx, seg, nsg_],
                   template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                   grid=(N // 32 * 32 * ks, nseg, 1), threadgroup=(32 * ks, 1, 1),
                   output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = NN.rowsum64(h)
    (y,) = NN._kbn()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             ms, ls, pidx, seg, nsg_],
                     template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                     grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                     output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---- arm 8b (S <= 16; kernel A only): the left operand as a tensor<device T> view of the check's 16 x rows (row stride
# K; x padded to 16 rows), no pidx gather and no ca[] writes; the op computes all 16 x rows against the expert, and a
# pair's result is read out of C at its x row (INV: a row's bits do not depend on its position or companions); the
# epilogue's xsum term uses that x row's sums; stores go through a 16-entry x row -> pair map built in the prologue. ----
_A_8B = NN._swap(NN._A_N, [
    ("const int q0 = seg[gq], nq = seg[gq + 1] - q0;",
     "const int q0 = seg[gq], nq = seg[gq + 1] - q0;\n"
     "    threadgroup int xmap[16];\n"
     "    if (thread_index_in_threadgroup < 16) xmap[thread_index_in_threadgroup] = -1;\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    if ((int)thread_index_in_threadgroup < nq) xmap[pidx[q0 + thread_index_in_threadgroup] / NR] = pidx[q0 + thread_index_in_threadgroup];\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);"),
    ("const uint ma0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)] / NR, ma1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)] / NR;",
     "const uint ma0 = (uint)fmA, ma1 = (uint)fmA + 8;"),
    ("    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();\n"
     "    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;\n"
     "    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();",
     "    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;\n"
     "    using TA = tensor<device T, dextents<int32_t, 2>, tensor_inline>;\n"
     "    auto cg = op.template get_destination_cooperative_tensor<TA, TB, float>();"),
    ("                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);\n"
     "                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);\n"
     "                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);\n"
     "                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);\n"
     "                for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }\n",
     "                TA ta((device T*)(x + k0), dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, K});\n"),
    ("op.run(ca, tbt, cg);", "op.run(ta, tbt, cg);"),
    ("            if (row < nq) {\n", "            const int pr_ = xmap[row];\n            if (pr_ >= 0) {\n"),
    ("h[(size_t)pidx[q0 + row] * N + n0 + col] =", "h[(size_t)pr_ * N + n0 + col] ="),
])


@lru_cache(None)
def _ka8b():
    return mx.fast.metal_kernel(name="k90_8b_a", input_names=_IN_A, output_names=["h"], source=_A_8B, header=N6._HDR4)


def tier_glu_nax8b(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2):
    """NAX6-N with kernel A's left operand as a device tensor view of the 16 x rows (S <= 16); B = NAX6-N's"""
    M, K = x.shape
    assert M <= 16, "arm 8b: S <= 16"
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    x16 = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)], axis=0)
    xs = NN.rowsum64(x16)
    (h,) = _ka8b()(inputs=[x16, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                           pidx, seg, nsg_],
                   template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                   grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                   output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = NN.rowsum64(h)
    (y,) = NN._kbn()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             ms, ls, pidx, seg, nsg_],
                     template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                     grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                     output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---- probe MUL-first (Fable A 18:2x; both kernels): the first op of each quant group in a non-accumulating multiply mode
# (cg = A B), so the 16 cg zero-writes go; the second op accumulates as today. Bitwise only if the unit's accumulate onto
# 0 equals its plain product (checked 96 / 96); else dropped. ----
_MF_PAIRS = [
    ("    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;\n",
     "    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;\n"
     "    constexpr auto desc_m = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,\n"
     "        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);\n"
     "    mpp::tensor_ops::matmul2d<desc_m, metal::execution_simdgroup> op_m;\n"),
    ("            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;\n", ""),
    ("op.run(ca, tbt, cg);", "if (hh == 0) op_m.run(ca, tbt, cg); else op.run(ca, tbt, cg);"),
]
_A_MF, _B_MF = NN._swap(NN._A_N, _MF_PAIRS), NN._swap(NN._B_N, _MF_PAIRS)


@lru_cache(None)
def _kamf():
    return mx.fast.metal_kernel(name="k90_mf_a", input_names=_IN_A, output_names=["h"], source=_A_MF, header=N6._HDR4)


@lru_cache(None)
def _kbmf():
    return mx.fast.metal_kernel(name="k90_mf_b", input_names=_IN_B, output_names=["y"], source=_B_MF, header=N6._HDR4)


def tier_glu_nax8mf(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2):
    """NAX6-N with MUL-first ops in kernels A and B (bf16 native pools)"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = NN.rowsum64(x)
    (h,) = _kamf()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                           pidx, seg, nsg_],
                   template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                   grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                   output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = NN.rowsum64(h)
    (y,) = _kbmf()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                           ms, ls, pidx, seg, nsg_],
                   template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                   grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                   output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---- probe decode LUT (Fable A 18:2x; both kernels): spread3's two 12-bit halves are independent, so spread3(x) =
# T[x & 0xFFF] | T[x >> 12] << 16 with T[t] = spread3(t) (4,096 uint16); spread8 = (that << 1) | its LSB spread
# (unchanged ops). The same nibbles, so bitwise by construction (checked 96 / 96). Two forms: "lut" = the table in
# threadgroup memory (8 KB), filled by each threadgroup (A 28 KB, B 18 KB); "lutd" = the table as a device input
# (4,096 uint16, built once on the host), read through L1: no fill, no barrier. ----
import numpy as _np
_LSB8 = r"""
inline uint lsb8(uint l) {                                 // spread8's low-bit spread, verbatim
    uint y = l & 0xFFu;
    y = (y | (y << 12)) & 0x000F000Fu;
    y = (y | (y << 6)) & 0x03030303u;
    y = (y | (y << 3)) & 0x11111111u;
    return y;
}
"""


def _lut_pairs(tab):
    lk = lambda v: f"((uint){tab}[{v} & 0xFFFu] | ((uint){tab}[{v} >> 12] << 16))"
    return [
        ("for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);",
         f"for (int q = 0; q < 8; ++q) nb[q] = ({lk('ch[q]')} << 1) | lsb8(((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);"),
        ("for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;",
         f"for (int q = 0; q < 8; ++q) nb[q] = {lk('ch[q]')} & 0x66666666u;"),
        ("for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);", f"for (int q = 0; q < 8; ++q) nb[q] = {lk('ch[q]')};"),
    ]


_FILL = ("if ((int)gq >= nseg[0]) return;",
         "if ((int)gq >= nseg[0]) return;\n"
         "    threadgroup ushort lut12[4096];\n"
         "    for (uint i_ = thread_index_in_threadgroup; i_ < 4096u; i_ += threads_per_threadgroup.x) lut12[i_] = (ushort)spread3(i_);\n"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);")
_A_LUT, _B_LUT = NN._swap(NN._A_N, [_FILL] + _lut_pairs("lut12")), NN._swap(NN._B_N, [_FILL] + _lut_pairs("lut12"))
_A_LUTD, _B_LUTD = NN._swap(NN._A_N, _lut_pairs("lutd")), NN._swap(NN._B_N, _lut_pairs("lutd"))


def _spread3_np(x):
    x = x.astype(_np.uint32)
    a = (x & 0xFFF) | ((x & 0xFFF000) << 4)
    a = (a & 0x003F003F) | ((a & 0x0FC00FC0) << 2)
    return (a & 0x07070707) | ((a & 0x38383838) << 1)


_LUTD = [None]
def _lutd():
    if _LUTD[0] is None:
        t = _spread3_np(_np.arange(4096))
        assert t.max() < 65536
        _LUTD[0] = mx.array(t.astype(_np.uint16))
    return _LUTD[0]


@lru_cache(None)
def _klut(which):
    src = {"a": _A_LUT, "b": _B_LUT, "ad": _A_LUTD, "bd": _B_LUTD}[which]
    ins = (_IN_A if which[0] == "a" else _IN_B) + (["lutd"] if which.endswith("d") else [])
    return mx.fast.metal_kernel(name=f"k90_lut_{which}", input_names=ins, output_names=["h" if which[0] == "a" else "y"],
                                source=src, header=N6._HDR4 + _LSB8)


def _tier_glu_lut(x, msb_n, lsb_n, msb_slot, lsb_slot, plan, dev, grp=None, nsg_b=2, ks=4, ks_b=2):
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    extra = [_lutd()] if dev else []
    xs = NN.rowsum64(x)
    (h,) = _klut("ad" if dev else "a")(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg, nsg_] + extra,
        template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
        grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = NN.rowsum64(h)
    (y,) = _klut("bd" if dev else "b")(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                ms, ls, pidx, seg, nsg_] + extra,
        template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


def tier_glu_nax8lut(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", **kw):
    return _tier_glu_lut(x, msb_n, lsb_n, msb_slot, lsb_slot, plan, False, **kw)


def tier_glu_nax8lutd(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", **kw):
    return _tier_glu_lut(x, msb_n, lsb_n, msb_slot, lsb_slot, plan, True, **kw)


# ---- probe PF-6 (Fable A 18:2x; both kernels): the next quant group's six code words loaded into registers before the
# current decode, nothing else moved (+6 registers, distance 1); the same values reach the same ops: bitwise. ----
_PF_PAIRS = [
    ("    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {        // this simdgroup's K chunk; GS_C quant groups per stage\n",
     "    uint pf0, pf1, pf2, pf3, pf4, pf5;\n"
     "    { const device uint32_t* c0 = wrow + 192 * (kc * GPC * GS_C);\n"
     "      pf0 = c0[0]; pf1 = c0[1]; pf2 = c0[2]; pf3 = c0[3]; pf4 = c0[4]; pf5 = c0[5]; }\n"
     "    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {        // this simdgroup's K chunk; GS_C quant groups per stage\n"),
    ("            const device uint32_t* c = wrow + 192 * g;\n"
     "            const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];\n",
     "            const uint w0 = pf0, w1 = pf1, w2 = pf2, w3 = pf3, w4 = pf4, w5 = pf5;\n"
     "            if (g + 1 < (int)((kc + 1) * GPC * GS_C)) {\n"
     "                const device uint32_t* cn = wrow + 192 * (g + 1);\n"
     "                pf0 = cn[0]; pf1 = cn[1]; pf2 = cn[2]; pf3 = cn[3]; pf4 = cn[4]; pf5 = cn[5];\n"
     "            }\n"),
]
_A_PF, _B_PF = NN._swap(NN._A_N, _PF_PAIRS), NN._swap(NN._B_N, _PF_PAIRS)


@lru_cache(None)
def _kpf(which):
    return mx.fast.metal_kernel(name=f"k90_pf6_{which}", input_names=_IN_A if which == "a" else _IN_B,
                                output_names=["h" if which == "a" else "y"], source=_A_PF if which == "a" else _B_PF,
                                header=N6._HDR4)


def tier_glu_nax8pf6(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2):
    """NAX6-N with the next quant group's code words prefetched into registers (both kernels)"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0 and r4
    N, D = msb_n["gate_proj"]["s3"].shape[1], msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xs = NN.rowsum64(x)
    (h,) = _kpf("a")(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                             L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                             pidx, seg, nsg_],
                     template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                     grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                     output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = NN.rowsum64(h)
    (y,) = _kpf("b")(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                             ms, ls, pidx, seg, nsg_],
                     template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                     grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                     output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
