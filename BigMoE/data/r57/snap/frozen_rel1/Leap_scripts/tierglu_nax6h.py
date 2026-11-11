"""NAX6-H, the in-kernel hybrid (Leap, Mon 2026-11-02 23:1x; FAST track; BigMoE's fastBEh arm via ref2 CFG nax_hybrid; the
engine form of H-1c, TABLE_350 'H-1'): NAX6's kernels A and B (tierglu_nax6.py, its GPU grouping, tiles, chunks and chunk
sums) with one change: a group that holds ONE pair (nq == 1, known from NAX6's own grouping) skips the 16-row matrix op set
and runs a SIMD-ALU path instead: each lane (one output column, as in NAX6's decode) decodes its 64 codes per quant group
as NAX6 does and takes the fp32 dot with the pair's 64 activations, then NAX6's affine epilogue (s x dot + b x xsum),
the chunk partials and sums as NAX6. Groups of 2+ pairs (duplicates included) keep NAX6's matrix path unchanged.
No pair list, no host sync, no extra dispatch, no new combine; works inside OOO's phases (gpu_grp_phase) as NAX6 does.
FAST: a pair's bits depend on whether it is alone in its group, so rows are not invariant to their companions.
tier_glu_nax6h(..., hybrid=False) is NAX6 bitwise (HYB_C 0 compiles the original path)."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_grouped import make_grp
from tierglu_nax import rowsum64

_LOOP = "    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {"
_DEC0 = "            const device uint32_t* c = wrow + 6 * g;"
_DEC1 = "            threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8 * GS_C + 8 * j);"
assert N6._CORE4.count(_LOOP) == 1 and N6._CORE4.count(_DEC0) == 1 and N6._CORE4.count(_DEC1) == 1
_DECODE = N6._CORE4[N6._CORE4.index(_DEC0):N6._CORE4.index(_DEC1)]          # NAX6's decode of group g into nb[8], verbatim
_ALU = r"""
    float accv = 0.0f;
    const bool alu = HYB_C != 0 && nq == 1;
    if (alu) {
        for (int g = kc * GPC; g < (kc + 1) * GPC; ++g) {
""" + _DECODE + r"""
            const device T* xr = ar0 + 64 * g;
            float dot = 0.0f;
            for (int q = 0; q < 8; ++q) {
                const vec<T, 4> xa = *(const device vec<T, 4>*)(xr + 8 * q);
                const vec<T, 4> xb = *(const device vec<T, 4>*)(xr + 8 * q + 4);
                const uint w = nb[q];
                dot = fma(float(xa[0]), float(w & 0xFu), dot);
                dot = fma(float(xa[1]), float((w >> 4) & 0xFu), dot);
                dot = fma(float(xa[2]), float((w >> 8) & 0xFu), dot);
                dot = fma(float(xa[3]), float((w >> 12) & 0xFu), dot);
                dot = fma(float(xb[0]), float((w >> 16) & 0xFu), dot);
                dot = fma(float(xb[1]), float((w >> 20) & 0xFu), dot);
                dot = fma(float(xb[2]), float((w >> 24) & 0xFu), dot);
                dot = fma(float(xb[3]), float(w >> 28), dot);
            }
            accv += (float)srow[g] * smul * dot + (float)brow[g] * as0[g];
        }
    } else
"""
_CORE_H = N6._CORE4.replace(_LOOP, _ALU + _LOOP)
_PART_A = "    for (int i = 0; i < 16; ++i) part[(pj * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];"
_PART_B = "    for (int i = 0; i < 16; ++i) part[(ti * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];"
assert N6._A4.count(_PART_A) == 1 and N6._B4.count(_PART_B) == 1
_A_H = N6._A4.replace(N6._CORE4, _CORE_H).replace(
    _PART_A, "    if (alu) { part[(pj * KS_C + kc) * 512 + lane] = accv; } else {\n" + _PART_A + "\n    }")
_B_H = N6._B4.replace(N6._CORE4, _CORE_H).replace(
    _PART_B, "    if (alu) { part[(ti * KS_C + kc) * 512 + lane] = accv; } else {\n" + _PART_B + "\n    }")
assert _CORE_H in _A_H and _CORE_H in _B_H


@lru_cache(None)
def _kah():
    return mx.fast.metal_kernel(
        name="tierglu_nax6h_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A_H, header=N6._HDR4)


@lru_cache(None)
def _kbh():
    return mx.fast.metal_kernel(
        name="tierglu_nax6h_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B_H, header=N6._HDR4)


def tier_glu_nax6h(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, grp=None, nsg_b=2, ms_np=None, ls_np=None,
                   ks=4, ks_b=2, xs=None, hybrid=True):
    """tier_glu_nax6's signature, shapes and grouping; one-pair groups on the SIMD ALUs when hybrid (FAST)"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 64 and K % 64 == 0 and N % 64 == 0 and D % (32 * nsg_b) == 0
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
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1),
            ("HYB_C", int(bool(hybrid)))]
    assert (K // 64) % ks == 0 and (N // 64) % ks_b == 0, (K, N, ks, ks_b)
    xs = rowsum64(x) if xs is None else xs
    (h,) = _kah()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                  grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = _kbh()(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---------------------------------------------------------------- NAX6-NH: the hybrid on the native record layout (N7-13)
# The same ALU path with tierglu_nax6n's addresses (codes at 192 words per quant group, LSB at 64, scales / biases at 32).
def _native_hybrid(p11=False):
    """NAX6-NH's sources: the hybrid's ALU path spliced into NAX6-N's core; p11=True: into its p11 core (Wed 2026-11-04,
    for Friday's native+p11+pad store: the scale / bias reads become P11R views, which return the same floats)"""
    import tierglu_nax6n as NN
    core_n = NN._CORE_NP if p11 else NN._CORE_N
    a_n, b_n = (NN._A_NP, NN._B_NP) if p11 else (NN._A_N, NN._B_N)
    loop_n = "    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {"
    d0 = "            const device uint32_t* c = wrow + 192 * g;"
    assert core_n.count(loop_n) == 1 and core_n.count(d0) == 1 and core_n.count(_DEC1) == 1
    dec = core_n[core_n.index(d0):core_n.index(_DEC1)]
    alu = _ALU.replace(_DECODE, dec).replace("accv += (float)srow[g] * smul * dot + (float)brow[g] * as0[g];",
                                             "accv += (float)srow[32 * g] * smul * dot + (float)brow[32 * g] * as0[g];")
    assert alu != _ALU
    core = core_n.replace(loop_n, alu + loop_n)
    a = a_n.replace(core_n, core).replace(
        _PART_A, "    if (alu) { part[(pj * KS_C + kc) * 512 + lane] = accv; } else {\n" + _PART_A + "\n    }")
    b = b_n.replace(core_n, core).replace(
        _PART_B, "    if (alu) { part[(ti * KS_C + kc) * 512 + lane] = accv; } else {\n" + _PART_B + "\n    }")
    assert core in a and core in b
    return a, b


@lru_cache(None)
def _knh_p11():
    import p11 as P11
    a, b = _native_hybrid(p11=True)
    ka = mx.fast.metal_kernel(
        name="tierglu_nax6nhp_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=a, header=N6._HDR4 + P11.P11_METAL)
    kb = mx.fast.metal_kernel(
        name="tierglu_nax6nhp_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=b, header=N6._HDR4 + P11.P11_METAL)
    return ka, kb


@lru_cache(None)
def _knh():
    a, b = _native_hybrid()
    ka = mx.fast.metal_kernel(
        name="tierglu_nax6nh_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=a, header=N6._HDR4)
    kb = mx.fast.metal_kernel(
        name="tierglu_nax6nh_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=b, header=N6._HDR4)
    return ka, kb


def tier_glu_nax6nh(x, msb_n, lsb_n, msb_slot, lsb_slot, plan="4433333333", grp=None, nsg_b=2, ks=4, ks_b=2, xs=None,
                    hybrid=True, b2=None):
    """NAX6-H on tierglu_nax6n.repack()'s native pools (FAST); hybrid=False == NAX6-N == NAX6 bitwise. p11 pools (uint32
    s3 / b3 / b, native+p11 or native+p11+pad) take the p11 sources, as tier_glu_nax6n does"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    p11 = msb_n["gate_proj"]["s3"].dtype == mx.uint32
    if p11:
        import p11 as P11
        N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
        pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]
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
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1),
            ("HYB_C", int(bool(hybrid)))]
    if p11:
        tmpl = tmpl + [("PWM_C", pwm), ("PWL_C", pwl)]
    ka, kb = _knh_p11() if p11 else _knh()
    xs = rowsum64(x) if xs is None else xs
    (h,) = ka(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                      L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg, nsg_],
              template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
              threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = kb(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                      ms, ls, pidx, seg, nsg_],
              template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
              grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
              output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
