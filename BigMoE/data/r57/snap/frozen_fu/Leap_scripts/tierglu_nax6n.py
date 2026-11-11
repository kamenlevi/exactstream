"""NAX6-N, a fragment-native record layout (Leap, Mon 2026-11-02 23:2x; pre-registered in ~/Leap/TABLE_350.md 'N7-13'):
NAX6's kernels (tierglu_nax6.py) with only their address arithmetic changed, on records repacked so that for each
(expert, 32-row tile, quant group) the 32 rows' codes (6 words each), LSB words (2 each), scales and biases (1 each) are
contiguous: one load instruction of a simdgroup touches ~6 lines of codes instead of 32. The same values reach the same
ops, so tier_glu_nax6n == tier_glu_nax6 bitwise (n7_native.py checks it).
repack(pools) builds the native copy of NAX6's msb / lsb dicts (numpy, on the CPU).
p11 (Tue 17:2x, TABLE_350 'P11'): pools whose s3 / b3 / b arrays are uint32 [C, 8804] (scripts/p11.py's 11-bit pieces)
run kernels that read them through P11P (the same floats as the bf16 pieces); bf16 pools are untouched. Tue 22:5x
(native+p11+pad): the MSB / LSB piece strides come from the pools' shapes (MSB pieces padded to 9216 words; LSB b 8804)."""
import os, sys
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_grouped import make_grp
from tierglu_nax import rowsum64
import p11 as P11

def _swap(src, pairs):
    for a, b in pairs:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    return src

_CORE_N = _swap(N6._CORE4, [
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
])
_A_N = N6._A4.replace(N6._CORE4, _CORE_N)
_B_N = N6._B4.replace(N6._CORE4, _CORE_N)
assert _CORE_N in _A_N and _CORE_N in _B_N
# native + p11: the scale / bias pointers become P11R views of the expert's 11-bit piece (a lane = one row of the
# 32-row tile, so its bit offset is fixed; srow[32 * g] / brow[32 * g] read group g)
_CORE_NP = _swap(_CORE_N, [
    ("constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8;",
     "constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8; constexpr uint NB11 = S_O * KG / 32, PW11 = PWM_C, PWL11 = PWL_C;"),
    ("const device T* srow = s3p + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32) + lane;",
     "const P11R srow = p11r(s3p, e, PW11, NB11, (n0 / 32) * (KG * 32) + lane);"),
    ("const device T* brow;", "P11R brow = {};"),
    ("brow = blp + ((size_t)el * (S_O / 32) + n0 / 32) * (KG * 32) + lane;",
     "brow = p11r(blp, el, PWL11, NB11, (n0 / 32) * (KG * 32) + lane);"),
    ("brow = (tier == 2 ? b2p : b3p) + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32) + lane;",
     "brow = p11r(tier == 2 ? b2p : b3p, e, PW11, NB11, (n0 / 32) * (KG * 32) + lane);"),
])
_A_NP = _swap(N6._A4, [
    ("const device T* s3p = pj == 0 ? s3g : s3u;", "const device uint* s3p = (const device uint*)(pj == 0 ? s3g : s3u);"),
    ("const device T* b3p = pj == 0 ? b3g : b3u;", "const device uint* b3p = (const device uint*)(pj == 0 ? b3g : b3u);"),
    ("const device T* b2p = pj == 0 ? b2g : b2u;", "const device uint* b2p = (const device uint*)(pj == 0 ? b2g : b2u);"),
    ("const device T* blp = pj == 0 ? bg : bu;", "const device uint* blp = (const device uint*)(pj == 0 ? bg : bu);"),
]).replace(N6._CORE4, _CORE_NP)
_B_NP = _swap(N6._B4, [
    ("const device T* s3p = s3d;", "const device uint* s3p = (const device uint*)s3d;"),
    ("const device T* b3p = b3d;", "const device uint* b3p = (const device uint*)b3d;"),
    ("const device T* b2p = b2d;", "const device uint* b2p = (const device uint*)b2d;"),
    ("const device T* blp = bd;", "const device uint* blp = (const device uint*)bd;"),
]).replace(N6._CORE4, _CORE_NP)
assert _CORE_NP in _A_NP and _CORE_NP in _B_NP


def repack(msb, lsb):
    """NAX6's pools -> the native layout: w3 [E, O, KG*6] u32 -> [E, O/32, KG, 32, 6]; s3 / b3 / b [E, O, KG] ->
    [E, O/32, KG, 32]; l1 [E, O, KG*2] u32 -> [E, O/32, KG, 32, 2] (flattened back to the original shapes' sizes)"""
    def tile(a, per):                                    # [E, O, KG * per] -> [E, O/32, KG, 32, per]
        a = np.array(a)
        E, O, W = a.shape
        KG = W // per
        return mx.array(np.ascontiguousarray(a.reshape(E, O // 32, 32, KG, per).transpose(0, 1, 3, 2, 4)).reshape(E, O, W))
    def tile_bf(a):                                      # bf16 [E, O, KG] via its uint16 view
        u = np.array(a.view(mx.uint16))
        E, O, KG = u.shape
        return mx.array(np.ascontiguousarray(u.reshape(E, O // 32, 32, KG).transpose(0, 1, 3, 2)).reshape(E, O, KG)).view(mx.bfloat16)
    m2 = {p: {"w3": tile(msb[p]["w3"], 6), "s3": tile_bf(msb[p]["s3"]), "b3": tile_bf(msb[p]["b3"])} for p in PROJ}
    l2 = {p: {"l1": tile(lsb[p]["l1"], 2), "b": tile_bf(lsb[p]["b"])} for p in PROJ}
    return m2, l2


def native_map(O, KG, per):
    """repack's per-expert permutation as a gather: native = rowmajor.reshape(-1)[native_map(O, KG, per)] within one
    expert's [O, KG * per] array (per 6 for w3, 2 for l1, 1 for s3 / b3 / b); the same map for every expert"""
    return np.arange(O * KG * per).reshape(O // 32, 32, KG, per).transpose(0, 2, 1, 3).reshape(-1)


# N7-15 (Mon 2026-11-02 night, BigMoE's load path): the GPU repack of freshly loaded slots, in place. The permutation never
# leaves a 32-row tile (tile t keeps its byte range), so one threadgroup moves a tile through threadgroup memory in one
# pass. An item = one gate / up tile or 4 down tiles (the same bytes: 30 KB of codes for this model).
_RPK_HDR = r"""
inline void rp_load(threadgroup uint* tb, device uint* src, int n4, uint tid) {
    for (int i = (int)tid; i < n4; i += 256) ((threadgroup uint4*)tb)[i] = ((device uint4*)src)[i];
}
inline void rp_store_w(device uint* dst, threadgroup uint* tb, int KG, int PW, uint tid) {
    // one 32-row tile of 4-byte words, row-major in tb -> dst[((g * 32 + r) * PW + w)] = tb[(r * KG + g) * PW + w]
    const int n4 = 8 * KG * PW;
    for (int i = (int)tid; i < n4; i += 256) {
        uint4 v;
        for (int c = 0; c < 4; ++c) {
            const int n = 4 * i + c, w = n % PW, q = n / PW, r = q % 32, g = q / 32;
            v[c] = tb[(r * KG + g) * PW + w];
        }
        ((device uint4*)dst)[i] = v;
    }
}
inline void rp_store_h(device ushort* dst, threadgroup ushort* tb, int KG, uint tid) {
    // one 32-row tile of bf16 scales / biases: dst[g * 32 + r] = tb[r * KG + g]
    const int n8 = 4 * KG;
    for (int i = (int)tid; i < n8; i += 256) {
        uint e[8];
        for (int c = 0; c < 8; ++c) { const int n = 8 * i + c, r = n % 32, g = n / 32; e[c] = tb[r * KG + g]; }
        ((device uint4*)dst)[i] = uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16), e[4] | (e[5] << 16), e[6] | (e[7] << 16));
    }
}
"""
_RPK_CLAIM = r"""
    // one threadgroup: the first pair naming each dirty slot claims it (clears its flag) and lists it; negative slots skip
    const int i = (int)thread_position_in_threadgroup.x;
    threadgroup atomic_int nc[2];
    if (i < 2) atomic_store_explicit(&nc[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup int ps[NM_C + NL_C];                                 // the call's slots, base then sidecar
    for (int p = i; p < NM_C + NL_C; p += NT_C) ps[p] = p >= NM_C ? sl[p - NM_C] : sm[p];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = i; p < NM_C + NL_C; p += NT_C) {
        const bool side = p >= NM_C;
        const int q0 = side ? NM_C : 0;
        const int s = ps[p];
        if (s < 0) continue;
        device int32_t* df = side ? (device int32_t*)dl : (device int32_t*)dm;
        if (df[s] == 0) continue;                                    // clean: almost every pair stops here
        bool first = true;
        for (int q = q0; q < p && first; ++q) first = ps[q] != s;
        if (first) {                                                 // a later pair with the same slot finds this one
            df[s] = 0;
            lst[q0 + atomic_fetch_add_explicit(&nc[side ? 1 : 0], 1, memory_order_relaxed)] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < 2) cnt[i] = atomic_load_explicit(&nc[i], memory_order_relaxed);
    if (i < 2) tok[i] = tok_in[i] + 1;                               // the claim chain's token (see repack_slots)
"""
_RPK_TILES = r"""
    // NG_C threadgroups walk the items of the claimed slots: base slots (w3 + s3 + b3) first, then sidecar slots (l1 + b)
    const uint tid = thread_position_in_threadgroup.x;
    const int tg = (int)threadgroup_position_in_grid.x;
    threadgroup uint tb[TBW_C];
    if (tg == 0) {
        for (int i = (int)tid; i < NM_C; i += 256) sm2[i] = sm[i];
        for (int i = (int)tid; i < NL_C; i += 256) sl2[i] = sl[i];
    }
    constexpr int TG_ = N_C / 32, IT = 2 * TG_ + D_C / 128, KGG = K_C / 64, KGD = N_C / 64;
    const int cm = cnt[0], cl = cnt[1];
    for (int it = tg; it < IT * (cm + cl); it += NG_C) {
        const bool side = it >= IT * cm;
        const int k = side ? it - IT * cm : it;
        const int slot = lst[(side ? NM_C : 0) + k / IT];
        const int j = k % IT;
        const int p = j < TG_ ? 0 : (j < 2 * TG_ ? 1 : 2);
        const int u = p == 0 ? j : (p == 1 ? j - TG_ : j - 2 * TG_);
        const int O = p < 2 ? N_C : D_C, KG = p < 2 ? KGG : KGD, nt = p < 2 ? 1 : 4;
        const size_t row0 = (size_t)slot * O + (size_t)(32 * nt * u);
        const int PW = side ? 2 : 6;                                  // words per (row, quant group): LSB 2, codes 6
        device uint* w = side ? (device uint*)(p == 0 ? l1g : (p == 1 ? l1u : l1d))
                              : (device uint*)(p == 0 ? w3g : (p == 1 ? w3u : w3d));
        w += row0 * (size_t)(KG * PW);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rp_load(tb, w, 8 * nt * KG * PW, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = 0; q < nt; ++q) rp_store_w(w + q * 32 * KG * PW, tb + q * 32 * KG * PW, KG, PW, tid);
        for (int f = 0; f < (side ? 1 : 2); ++f) {
            device ushort* h = side ? (device ushort*)(p == 0 ? bg : (p == 1 ? bu : bd))
                             : (f == 0 ? (device ushort*)(p == 0 ? s3g : (p == 1 ? s3u : s3d))
                                       : (device ushort*)(p == 0 ? b3g : (p == 1 ? b3u : b3d)));
            h += row0 * (size_t)KG;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            rp_load(tb, (device uint*)h, 4 * nt * KG, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int q = 0; q < nt; ++q) rp_store_h(h + q * 32 * KG, (threadgroup ushort*)tb + q * 32 * KG, KG, tid);
        }
    }
"""


@lru_cache(None)
def _krc():
    return mx.fast.metal_kernel(name="nax6n_repack_claim", input_names=["sm", "sl", "dm", "dl", "tok_in"],
                                output_names=["lst", "cnt", "tok"], source=_RPK_CLAIM, ensure_row_contiguous=False)


# claims on the same flags must see each other's clears. mx.depends orders MLX's graph but sets no GPU barrier between
# the two claim kernels (N7-15 self-test D, Tue 01:5x: every slot named by both calls was rewritten twice), and a flag
# array routed through mx.depends is a copy. So each claim takes the previous claim's token on the same flags as an
# input: a real buffer dependency, so MLX orders the second claim after the first.
_RPK_TOK = {}
_TOK0 = []


@lru_cache(None)
def _krt():
    return mx.fast.metal_kernel(
        name="nax6n_repack_tiles",
        input_names=["lst", "cnt", "sm", "sl", "w3g", "w3u", "w3d", "s3g", "s3u", "s3d", "b3g", "b3u", "b3d",
                     "l1g", "l1u", "l1d", "bg", "bu", "bd"],
        output_names=["sm2", "sl2"], source=_RPK_TILES, header=_RPK_HDR, ensure_row_contiguous=False)


def repack_slots(msb, lsb, msb_slot, lsb_slot, dirty_m, dirty_l=None, ng=64):
    """N7-15: rewrite IN PLACE, row-major -> native (== repack), every base-pool slot named in msb_slot whose dirty_m flag
    is set (w3 / s3 / b3 of gate, up, down) and every sidecar slot named in lsb_slot whose dirty_l flag is set (l1 / b),
    once per slot however many pairs name it; clears those flags; negative slots are skipped. The pools and the flags
    (int32 [slots], >= 2 entries) are written through their buffers, so they must be the engine's own contiguous arrays
    (never routed through mx.depends, which hands the kernel a copy). Calls on the same flags chain by themselves.
    Returns (msb_slot, lsb_slot) as new arrays: the calls that read the pools must take these, so MLX orders them after
    the rewrite."""
    g, u, d = (msb[p] for p in PROJ)
    N, D = g["w3"].shape[1], d["w3"].shape[1]
    K = g["w3"].shape[2] // 6 * 64
    assert d["w3"].shape[2] == N // 64 * 6 and N % 32 == 0 and D % 128 == 0, (N, D, K)
    tbw = max(32 * (K // 64) * 6, 128 * (N // 64) * 6)
    assert tbw * 4 <= 32768, tbw
    assert dirty_m.dtype == mx.int32 and dirty_m.size >= 2
    sm = msb_slot.astype(mx.int32).reshape(-1)
    NM = sm.size
    side = lsb is not None and lsb_slot is not None and dirty_l is not None
    if side:
        assert dirty_l.dtype == mx.int32 and dirty_l.size >= 2
        sl, dl, L = lsb_slot.astype(mx.int32).reshape(-1), dirty_l, lsb
    else:
        sl, dl = mx.full((2,), -1, mx.int32), mx.zeros((2,), mx.int32)
        L = {p: {"l1": msb[p]["w3"], "b": msb[p]["s3"]} for p in PROJ}   # never touched: no sidecar items
    NL = sl.size if side else 0
    nt = min(1024, max(32, (NM + NL + 31) // 32 * 32))
    if not _TOK0:
        _TOK0.append(mx.zeros((2,), mx.int32))
        mx.eval(_TOK0[0])
    prev = _RPK_TOK.get(id(dirty_m))
    tok_in = prev[1] if prev is not None and prev[0] is dirty_m else _TOK0[0]
    lst, cnt, tok = _krc()(inputs=[sm, sl, dirty_m, dl, tok_in], template=[("NM_C", NM), ("NL_C", NL), ("NT_C", nt)],
                           grid=(nt, 1, 1), threadgroup=(nt, 1, 1),
                           output_shapes=[(max(NM + NL, 2),), (2,), (2,)], output_dtypes=[mx.int32] * 3)
    _RPK_TOK[id(dirty_m)] = (dirty_m, tok)
    sm2, sl2 = _krt()(inputs=[lst, cnt, sm, sl, g["w3"], u["w3"], d["w3"], g["s3"], u["s3"], d["s3"],
                              g["b3"], u["b3"], d["b3"], L["gate_proj"]["l1"], L["up_proj"]["l1"], L["down_proj"]["l1"],
                              L["gate_proj"]["b"], L["up_proj"]["b"], L["down_proj"]["b"]],
                      template=[("NM_C", NM), ("NL_C", NL), ("NG_C", ng), ("K_C", K), ("N_C", N), ("D_C", D),
                                ("TBW_C", tbw)],
                      grid=(256 * ng, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(NM,), (max(NL, 1),)], output_dtypes=[mx.int32, mx.int32])
    return sm2.reshape(msb_slot.shape), (sl2.reshape(lsb_slot.shape) if side else lsb_slot)


@lru_cache(None)
def _kan():
    return mx.fast.metal_kernel(
        name="tierglu_nax6n_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A_N, header=N6._HDR4)


@lru_cache(None)
def _kbn():
    return mx.fast.metal_kernel(
        name="tierglu_nax6n_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B_N, header=N6._HDR4)


@lru_cache(None)
def _kan_p11():
    return mx.fast.metal_kernel(
        name="tierglu_nax6np_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A_NP, header=N6._HDR4 + P11.P11_METAL)


@lru_cache(None)
def _kbn_p11():
    return mx.fast.metal_kernel(
        name="tierglu_nax6np_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B_NP, header=N6._HDR4 + P11.P11_METAL)


def tier_glu_nax6n(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2, xs=None, b2=None):
    """tier_glu_nax6 on repack()'s native pools (plan without 2-bit ranks); == tier_glu_nax6 bitwise"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0, "the native repack covers the 3- / 4-bit tiers"
    p11 = msb_n["gate_proj"]["s3"].dtype == mx.uint32             # p11 pools: uint32 [C, 8804] pieces
    if p11:
        N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
        pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]   # strides from the pools (native+p11+pad)
        assert pwm >= P11.piece_words(N * 40) and pwm % 4 == 0 and pwl >= P11.piece_words(N * 40) and pwl % 4 == 0 and b2 is None
        assert all(msb_n[p][k].shape[-1] == pwm for p in PROJ for k in ("s3", "b3")) and all(lsb_n[p]["b"].shape[-1] == pwl for p in PROJ)
    else:
        N = msb_n["gate_proj"]["s3"].shape[1]
        D = msb_n["down_proj"]["s3"].shape[1]
    if grp is None:
        grp = N6.gpu_grp(msb_slot, lsb_slot, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    L = lsb_n
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    if p11:
        tmpl = tmpl + [("PWM_C", pwm), ("PWL_C", pwl)]
    xs = rowsum64(x) if xs is None else xs
    (h,) = (_kan_p11 if p11 else _kan)()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                  grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = (_kbn_p11 if p11 else _kbn)()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
