"""3c-N (Leap, Tue 2026-11-03 15:4x; TABLE_350 'NATIVE-3C'): MoonShots' fused 3c tier GLU (third_party/ms_gdn/tierglu.py =
BigMoE's planes_engine copy) reading pools in NAX6-N's fragment-native layout (tierglu_nax6n.repack / native_map: per
expert, per 32-row tile and quant group, the 32 rows' codes / LSB words / scales / biases contiguous). Only the address
arithmetic of kernels A and B changes: the 4 rows a threadgroup (A) or simdgroup (B) owns lie in one 32-row tile, so a
(row, group) record sits at tile base + row-in-tile x (24 / 8 / 1) + group x (768 / 256 / 32) for codes / LSB words /
scales and biases. The same values reach the same ops in the same order, so 3c-N on native pools == 3c on row-major
pools bitwise (native3c_screen.py checks it). Covers the paths REF2-T uses: tier_glu (A + B, or A + BC with scores) and
tier_glu_phase (AP + BP, then TG.combine_phase), and the pair kernels (AS / BS: tier_glu_pairs_n, tier_glu_split_n).
The native sources are derived from tierglu's CURRENT sources on each call (_sync): if BigMoE's LP1 patch rewrites them,
3c-N follows; kernel names carry a hash of their source, so a changed source never reuses a stale build.
p11 (Tue 17:2x, TABLE_350 'P11'): pools whose s3 / b3 / b arrays are uint32 [C, 8804] (scripts/p11.py's 11-bit pieces)
switch every kernel to P11P reads, which return the same floats as the bf16 pieces; bf16 pools are untouched."""
import hashlib, os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party/ms_gdn"))
import tierglu as TG
from tierglu import PROJ, plan_masks
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
import p11 as P11

def _swap(src, pairs):
    for a, b, n in pairs:
        assert src.count(a) == n, (a[:70], src.count(a), n)
        src = src.replace(a, b)
    return src

_A_PAIRS = [
    ("const device uint8_t* w3 = (const device uint8_t*)(sg == 0 ? w3g : w3u) + ((size_t)e * N + j0) * KB3;",
     "const device uint8_t* w3 = (const device uint8_t*)(sg == 0 ? w3g : w3u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 768) + (j0 % 32) * 24;", 1),
    ("const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * N + j0) * KG;",
     "const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32);", 1),
    ("const device uint8_t* l1 = (const device uint8_t*)(sg == 0 ? l1g : l1u) + ((size_t)el * N + j0) * KL;",
     "const device uint8_t* l1 = (const device uint8_t*)(sg == 0 ? l1g : l1u) + ((size_t)el * (N / 32) + j0 / 32) * (KG * 256) + (j0 % 32) * 8;", 1),
    ("const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;",
     "const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32);", 1),
    ("(sg == 0 ? b2g : b2u) + ((size_t)e * N + j0) * KG", "(sg == 0 ? b2g : b2u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32)", 1),
    ("(sg == 0 ? b3g : b3u) + ((size_t)e * N + j0) * KG", "(sg == 0 ? b3g : b3u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32)", 1),
    ("(const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6)",
     "(const device uint16_t*)(w3 + row * 24 + (b * 8 + lane / 4) * 768 + (lane % 4) * 6)", 2),
    ("*(const device uint16_t*)(l1 + row * KL + b * 64 + lane * 2)",
     "*(const device uint16_t*)(l1 + row * 8 + (b * 8 + lane / 4) * 256 + (lane % 4) * 2)", 1),
    ("const device uint8_t* c = w3 + row * KB3 + b * 192 + lane * 6;",
     "const device uint8_t* c = w3 + row * 24 + (b * 8 + lane / 4) * 768 + (lane % 4) * 6;", 1),
    ("const device uint8_t* lb = l1 + row * KL + b * 64 + lane * 2;",
     "const device uint8_t* lb = l1 + row * 8 + (b * 8 + lane / 4) * 256 + (lane % 4) * 2;", 1),
    ("tg_codes3(w3 + row * KB3, b * 512 + lane * 16, 6, wb, tier == 2);",
     "tg_codes3(w3 + row * 24 + (b * 8 + lane / 4) * 768 + (lane % 4) * 6, 0, 6, wb, tier == 2);", 1),
    ("const int gi = row * KG + b * 8 + lane / 4;", "const int gi = row + (b * 8 + lane / 4) * 32;", 2),
]
_B_PAIRS = [
    ("const device uint8_t* w3 = (const device uint8_t*)w3d + ((size_t)e * D + d0) * NB3;",
     "const device uint8_t* w3 = (const device uint8_t*)w3d + ((size_t)e * (D / 32) + d0 / 32) * (NG * 768) + (d0 % 32) * 24;", 1),
    ("const device T* s3 = s3d + ((size_t)e * D + d0) * NG;",
     "const device T* s3 = s3d + ((size_t)e * (D / 32) + d0 / 32) * (NG * 32) + (d0 % 32);", 1),
    ("const device uint8_t* l1 = (const device uint8_t*)l1d + ((size_t)el * D + d0) * NL;",
     "const device uint8_t* l1 = (const device uint8_t*)l1d + ((size_t)el * (D / 32) + d0 / 32) * (NG * 256) + (d0 % 32) * 8;", 1),
    ("const device T* bl = bd + ((size_t)el * D + d0) * NG;",
     "const device T* bl = bd + ((size_t)el * (D / 32) + d0 / 32) * (NG * 32) + (d0 % 32);", 1),
    ("const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * D + d0) * NG;",
     "const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * (D / 32) + d0 / 32) * (NG * 32) + (d0 % 32);", 1),
    ("const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;",
     "const device uint8_t* c = w3 + row * 24 + ((k + lane * 8) / 64) * 768 + ((k + lane * 8) % 64) * 3 / 8;", 2),
    ("l1[row * NL + (k + lane * 8) / 8]", "l1[row * 8 + ((k + lane * 8) / 64) * 256 + ((k + lane * 8) % 64) / 8]", 2),
    ("const int gi = row * NG + (k + lane * 8) / G;", "const int gi = row + ((k + lane * 8) / G) * 32;", 4),
    ("tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, tier == 2);",
     "tg_codes3(w3 + row * 24 + ((k + lane * 8) / 64) * 768, (k + lane * 8) % 64, 3, wb, tier == 2);", 2),
]
# native -> native + p11: each simdgroup decodes its rows' scales / biases from the expert's 11-bit piece once into
# threadgroup memory (p11_fill); P11T views serve the kernels' own s3[gi] / bl[gi] / b3[gi] reads from there
_OFF_A = "(j0 / 32) * (KG * 32) + (j0 % 32)"
_OFF_B = "(d0 / 32) * (NG * 32) + (d0 % 32)"
_A_P11 = [
    ("constexpr int KB3 = K * 3 / 8, KG = K / G, KL = K / 8;",
     "constexpr int KB3 = K * 3 / 8, KG = K / G, KL = K / 8; constexpr uint NB11 = N * KG / 32, PW11 = PWM_C, PWL11 = PWL_C;\n"
     "    threadgroup float p11_tg[2 * 2 * RA * KG];                // per simdgroup: RA x KG scales, then RA x KG biases", 1),
    ("const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32);",
     f"const P11P s3_p11 = p11p((const device uint*)(sg == 0 ? s3g : s3u), e, PW11, NB11, {_OFF_A});\n"
     "    threadgroup float* p11_s = p11_tg + sg * (2 * RA * KG);\n"
     "    p11_fill(p11_s, s3_p11, RA, KG, lane);\n"
     "    const P11T s3 = {p11_s, RA};", 1),
    ("const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32);",
     f"const P11P bl_p11 = p11p((const device uint*)(sg == 0 ? bg : bu), el, PWL11, NB11, {_OFF_A});\n"
     "        p11_fill(p11_s + RA * KG, bl_p11, RA, KG, lane);\n"
     "        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
     "        const P11T bl = {p11_s + RA * KG, RA};", 1),
    ("const device T* b3 = tier == 2 ? (sg == 0 ? b2g : b2u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32)\n"
     "                                       : (sg == 0 ? b3g : b3u) + ((size_t)e * (N / 32) + j0 / 32) * (KG * 32) + (j0 % 32);",
     f"const P11P b3_p11 = p11p((const device uint*)(tier == 2 ? (sg == 0 ? b2g : b2u) : (sg == 0 ? b3g : b3u)), e, PW11, NB11, {_OFF_A});\n"
     "        p11_fill(p11_s + RA * KG, b3_p11, RA, KG, lane);\n"
     "        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
     "        const P11T b3 = {p11_s + RA * KG, RA};", 1),
]
_B_P11 = [
    ("constexpr int NB3 = N * 3 / 8, NG = N / G, NL = N / 8;",
     "constexpr int NB3 = N * 3 / 8, NG = N / G, NL = N / 8; constexpr uint NB11 = D * NG / 32, PW11 = PWM_C, PWL11 = PWL_C;\n"
     "    constexpr int P11_NS = NR > 2 ? NR : 2;                  // simdgroups per threadgroup: nsg (2) or the ranks (BC)\n"
     "    threadgroup float p11_tg[P11_NS * 2 * RB * NG];", 1),
    ("const device T* s3 = s3d + ((size_t)e * (D / 32) + d0 / 32) * (NG * 32) + (d0 % 32);",
     f"const P11P s3_p11 = p11p((const device uint*)s3d, e, PW11, NB11, {_OFF_B});\n"
     "    threadgroup float* p11_s = p11_tg + simdgroup_index_in_threadgroup * (2 * RB * NG);\n"
     "    p11_fill(p11_s, s3_p11, RB, NG, lane);\n"
     "    const P11T s3 = {p11_s, RB};", 1),
    ("const device T* bl = bd + ((size_t)el * (D / 32) + d0 / 32) * (NG * 32) + (d0 % 32);",
     f"const P11P bl_p11 = p11p((const device uint*)bd, el, PWL11, NB11, {_OFF_B});\n"
     "        p11_fill(p11_s + RB * NG, bl_p11, RB, NG, lane);\n"
     "        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
     "        const P11T bl = {p11_s + RB * NG, RB};", 1),
    ("const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * (D / 32) + d0 / 32) * (NG * 32) + (d0 % 32);",
     f"const P11P b3_p11 = p11p((const device uint*)(tier == 2 ? b2d : b3d), e, PW11, NB11, {_OFF_B});\n"
     "        p11_fill(p11_s + RB * NG, b3_p11, RB, NG, lane);\n"
     "        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
     "        const P11T b3 = {p11_s + RB * NG, RB};", 1),
]
_IN_A = ["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu", "msb_slot", "lsb_slot"]
_IN_B = ["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot"]
_KINDS = {"a": ("_A", _A_PAIRS, _IN_A, "h"), "b": ("_B", _B_PAIRS, _IN_B, "y"), "bc": ("_BC", _B_PAIRS, _IN_B + ["scores"], "y"),
          "ap": ("_AP", _A_PAIRS, _IN_A + ["phase"], "h"), "bp": ("_BP", _B_PAIRS, _IN_B + ["phase"], "y"),
          "as": ("_AS", _A_PAIRS, _IN_A + ["pidx"], "h"), "bs": ("_BS", _B_PAIRS, _IN_B + ["pidx"], "y")}
_SRC = {}                                                    # kind -> (tierglu source, native source, native + p11 source)


def _sync():
    """re-derive any native source whose tierglu source changed (e.g. BigMoE's LP1 patch)"""
    for kind, (attr, pairs, _, _) in _KINDS.items():
        cur = getattr(TG, attr)
        if kind not in _SRC or _SRC[kind][0] is not cur:
            nat = _swap(cur, pairs)
            _SRC[kind] = (cur, nat, _swap(nat, _A_P11 if pairs is _A_PAIRS else _B_P11))


@lru_cache(None)
def _build(kind, src, p11=False):
    _, _, ins, out = _KINDS[kind]
    hdr = TG._HDR + (P11.P11_METAL if p11 else "")
    tag = hashlib.sha1((hdr + src).encode()).hexdigest()[:10]
    return mx.fast.metal_kernel(name=f"tierglu_n{'p' if p11 else ''}_{kind}_{tag}", input_names=ins, output_names=[out],
                                source=src, header=hdr)


def _kern(kind, p11=False):
    _sync()
    return _build(kind, _SRC[kind][2 if p11 else 1], p11)


def _dims(msb_n, b2=None):
    """(N, D, p11): p11 pools hold uint32 [C, >= 8804] pieces (native+p11+pad pads the MSB ones to 9216 words; the stride
    is read from the pools' shapes, the base word stays at 11 * NB11), so the row counts come from w3 [C, rows, ...]"""
    s3 = msb_n["gate_proj"]["s3"]
    if s3.dtype == mx.uint32:
        N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
        assert s3.shape[-1] >= P11.piece_words(N * 40) and s3.shape[-1] % 4 == 0, s3.shape
        assert all(msb_n[p][k].shape[-1] == s3.shape[-1] for p in PROJ for k in ("s3", "b3")), "one MSB piece stride"
        assert P11.piece_words(D * 10) == P11.piece_words(N * 40)
        assert b2 is None or all(v.dtype == mx.uint32 for v in b2.values()), "p11 pools need p11 b2 pools"
        return N, D, True
    return s3.shape[1], msb_n["down_proj"]["s3"].shape[1], False


_sync()


def _common(x, msb, lsb, msb_slot, lsb_slot, plan, b2):
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    if r4:
        ls, L = lsb_slot.astype(mx.int32).reshape(-1), lsb
    else:
        ls, L = mx.zeros((1,), mx.int32), {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in PROJ)
    ina = [x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls]
    inb = [d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls]
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    if msb["gate_proj"]["s3"].dtype == mx.uint32:             # p11: the pieces' strides from the pools (MSB s3 / b3 / b2; LSB b)
        pwl = L["gate_proj"]["b"].shape[-1]
        assert all(L[p]["b"].shape[-1] == pwl for p in PROJ) and all(v.shape[-1] == msb["gate_proj"]["s3"].shape[-1] for v in bb.values())
        tmpl = tmpl + [("PWM_C", msb["gate_proj"]["s3"].shape[-1]), ("PWL_C", pwl)]
    return R, ina, inb, tmpl


def tier_glu_n(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, scores=None, ly=8,
               rbc=4, ld=0):
    """TG.tier_glu on native pools (same arguments, same returns); == TG.tier_glu on row-major pools bitwise"""
    M, K = x.shape
    N, D, p11 = _dims(msb_n, b2)
    assert K % 512 == 0 and N % 32 == 0 and D % 32 == 0 and ra == 4 and rb == 4 and rbc == 4
    R, ina, inb, tmpl = _common(x, msb_n, lsb_n, msb_slot, lsb_slot, plan, b2)
    (h,) = _kern("a", p11)(inputs=ina, template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                      grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, N)],
                      output_dtypes=[x.dtype])
    if scores is not None:
        (y,) = _kern("bc", p11)(inputs=[h] + inb + [scores.astype(x.dtype).reshape(-1)],
                           template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                           grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1), output_shapes=[(M, D)],
                           output_dtypes=[x.dtype])
        return y
    (y,) = _kern("b", p11)(inputs=[h] + inb, template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                      grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                      output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


def tier_glu_phase_n(x, msb_n, lsb_n, msb_slot, lsb_slot, phase, which, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2,
                     ld=0):
    """TG.tier_glu_phase on native pools; == TG.tier_glu_phase on row-major pools bitwise (combine with TG.combine_phase)"""
    M, K = x.shape
    N, D, p11 = _dims(msb_n, b2)
    assert ra == 4 and rb == 4
    R, ina, inb, tmpl = _common(x, msb_n, lsb_n, msb_slot, lsb_slot, plan, b2)
    ph = phase.astype(mx.int32).reshape(-1)
    tmpl = tmpl + [("PHASE_C", which)]
    (h,) = _kern("ap", p11)(inputs=ina + [ph], template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                       grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, N)],
                       output_dtypes=[x.dtype])
    (y,) = _kern("bp", p11)(inputs=[h] + inb + [ph], template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                       grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                       output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y


def tier_glu_pairs_n(x, msb_n, lsb_n, msb_slot, lsb_slot, pidx, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, ld=0):
    """TG.tier_glu_pairs on native pools: per-rank outputs y [len(pidx), D] for the pairs pidx only"""
    M, K = x.shape
    N, D, p11 = _dims(msb_n, b2)
    assert ra == 4 and rb == 4
    R, ina, inb, tmpl = _common(x, msb_n, lsb_n, msb_slot, lsb_slot, plan, b2)
    P = pidx.size
    (h,) = _kern("as", p11)(inputs=ina + [pidx], template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                       grid=(N // ra * 64, P, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    (y,) = _kern("bs", p11)(inputs=[h] + inb + [pidx], template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                       grid=(D // (rb * nsg_b) * 32 * nsg_b, P, 1), threadgroup=(32 * nsg_b, 1, 1),
                       output_shapes=[(P, D)], output_dtypes=[x.dtype])
    return y


def tier_glu_split_n(x, msb_n, lsb_n, msb_slot, lsb_slot, hit, scores, plan="4433333333", b2=None, ly=8):
    """TG.tier_glu_split on native pools (hit pairs first, missed pairs second, then TG's combine)"""
    import numpy as np
    M, R = len(hit), len(plan)
    flat = np.asarray(hit, bool).reshape(-1)
    hi, mi = np.nonzero(flat)[0], np.nonzero(~flat)[0]
    loc = np.empty(M * R, np.int32)
    loc[hi] = np.arange(hi.size)
    loc[mi] = -(np.arange(mi.size) + 1)
    D = _dims(msb_n, b2)[1]
    run = lambda ix: tier_glu_pairs_n(x, msb_n, lsb_n, msb_slot, lsb_slot, mx.array(ix.astype(np.int32)), plan, b2) \
        if ix.size else mx.zeros((1, D), x.dtype)
    yh, ym = run(hi), run(mi)
    (out,) = TG._kc()(inputs=[yh, ym, mx.array(loc), scores.astype(x.dtype).reshape(-1)],
                      template=[("T", x.dtype), ("D_C", D), ("NR_C", R), ("LY_C", ly)],
                      grid=(D, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return out
