"""NAX-C, design C on the FAST track (Leap, Mon 2026-11-02 22:4x; EXPERT_KERNEL.md design C; pre-registered in
~/Leap/TABLE_350.md 'NAX-C'): NAX6 (tierglu_nax6.py: the GPU grouping, kernel A gate / up + swiglu, kernel B down, the
same tiles, chunks and chunk sums) with its matrix ops in int8 x int4 -> int32 (M27-C: 1.55x NAX6's bf16 x uint4 form):
- activations quantized per (row, 64-group): scale = max|x| / 127 (1 when the group is zero), x_q = round-to-nearest
  (x / scale) clamped to [-127, 127], qs = sum of x_q; a pass that replaces rowsum64 (x for kernel A, h for kernel B);
- the 3- / 4-bit codes decoded as in NAX6, stored as int4 (code - 8: one XOR per word);
- per quant group: cg = sum x_q (q - 8) in two 16 x 32 x 32 int ops (exact int32), then
  acc += scale (s (cg + 8 qs) + b qs), i.e. NAX6's affine dequant applied to the dequantized activations.
FAST track: not bitwise to anything; integer sums make a row's result independent of its companions (row-invariant)."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
from tierglu import PROJ, plan_masks
import tierglu_nax6 as N6
from tierglu_grouped import make_grp

def _swap(src, pairs):
    for a, b in pairs:
        assert src.count(a) >= 1, a
        src = src.replace(a, b)
    return src

_CORE_C = _swap(N6._CORE4, [
    ("auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();",
     "auto ca = op.template get_left_input_cooperative_tensor<int8_t, int4b_format, int32_t>();"),
    ("using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;",
     "using TB = tensor<threadgroup int4b_format, dextents<int32_t, 2>, tensor_inline>;"),
    ("auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();",
     "auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, int32_t>();"),
    ("nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);",
     "nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu) ^ 0x88888888u;"),
    ("nb[q] = spread3(ch[q]) & 0x66666666u;", "nb[q] = (spread3(ch[q]) & 0x66666666u) ^ 0x88888888u;"),
    ("for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);", "for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) ^ 0x88888888u;"),
    ("for (int i = 0; i < 16; ++i) cg[i] = 0.0f;", "for (int i = 0; i < 16; ++i) cg[i] = 0;"),
    ("const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);",
     "const vec<int8_t, 4> a00 = *(const device vec<int8_t, 4>*)(ar0 + k0 + fn0);"),
    ("const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);",
     "const vec<int8_t, 4> a10 = *(const device vec<int8_t, 4>*)(ar1 + k0 + fn0);"),
    ("const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);",
     "const vec<int8_t, 4> a01 = *(const device vec<int8_t, 4>*)(ar0 + k0 + 16 + fn0);"),
    ("const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);",
     "const vec<int8_t, 4> a11 = *(const device vec<int8_t, 4>*)(ar1 + k0 + 16 + fn0);"),
    ("const float x0 = as0[g], x1 = as1[g];", "const float sx0 = as0[g], sx1 = as1[g];\n            const int qs0 = aq0[g], qs1 = aq1[g];"),
    ("acc[i] += sb[64 * j + col] * cg[i] + sb[64 * j + 32 + col] * (((i >> 2) & 1) ? x1 : x0);",
     "const bool r1 = ((i >> 2) & 1) != 0;\n"
     "                const float sx = r1 ? sx1 : sx0;\n"
     "                const int qs = r1 ? qs1 : qs0;\n"
     "                acc[i] += sx * (sb[64 * j + col] * float(cg[i] + 8 * qs) + sb[64 * j + 32 + col] * float(qs));"),
])
_A_C = _swap(N6._A4, [
    (N6._CORE4, _CORE_C),
    ("const device T* ar0 = x + (size_t)ma0 * K;", "const device int8_t* ar0 = xq + (size_t)ma0 * K;"),
    ("const device T* ar1 = x + (size_t)ma1 * K;", "const device int8_t* ar1 = xq + (size_t)ma1 * K;"),
    ("const device float* as1 = xs + (size_t)ma1 * (K / 64);",
     "const device float* as1 = xs + (size_t)ma1 * (K / 64);\n"
     "    const device int* aq0 = xqs + (size_t)ma0 * (K / 64);\n"
     "    const device int* aq1 = xqs + (size_t)ma1 * (K / 64);"),
])
_B_C = _swap(N6._B4, [
    (N6._CORE4, _CORE_C),
    ("const device T* ar0 = h + (size_t)pa0 * N;", "const device int8_t* ar0 = hq + (size_t)pa0 * N;"),
    ("const device T* ar1 = h + (size_t)pa1 * N;", "const device int8_t* ar1 = hq + (size_t)pa1 * N;"),
    ("const device float* as1 = hs + (size_t)pa1 * (N / 64);",
     "const device float* as1 = hs + (size_t)pa1 * (N / 64);\n"
     "    const device int* aq0 = hqs + (size_t)pa0 * (N / 64);\n"
     "    const device int* aq1 = hqs + (size_t)pa1 * (N / 64);"),
])
_Q8 = r"""
    const uint g = thread_position_in_grid.x, r = thread_position_in_grid.y;
    if (g >= (uint)(L_C / 64)) return;
    const device T* p = a + (size_t)r * L_C + g * 64;
    float amax = 0.0f;
    for (int i = 0; i < 64; ++i) amax = max(amax, fabs((float)p[i]));
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    int qs = 0;
    device int8_t* qo = q + (size_t)r * L_C + g * 64;
    for (int i = 0; i < 64; ++i) {
        const int v = clamp((int)rint((float)p[i] * inv), -127, 127);
        qo[i] = (int8_t)v;
        qs += v;
    }
    scale[(size_t)r * (L_C / 64) + g] = sc;
    qsum[(size_t)r * (L_C / 64) + g] = qs;
"""


@lru_cache(None)
def _kq8():
    return mx.fast.metal_kernel(name="naxc_q8", input_names=["a"], output_names=["q", "scale", "qsum"], source=_Q8)


def quant64(a):
    """[rows, L] -> (int8 [rows, L], scale fp32 [rows, L / 64], sum of the int8 values int32 [rows, L / 64])"""
    R_, L = a.shape
    return _kq8()(inputs=[a], template=[("T", a.dtype), ("L_C", L)], grid=(L // 64, R_, 1), threadgroup=(L // 64, 1, 1),
                  output_shapes=[(R_, L), (R_, L // 64), (R_, L // 64)], output_dtypes=[mx.int8, mx.float32, mx.int32])


@lru_cache(None)
def _kac():
    return mx.fast.metal_kernel(
        name="tierglu_naxc_a",
        input_names=["xq", "xs", "xqs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_A_C, header=N6._HDR4)


@lru_cache(None)
def _kbc():
    return mx.fast.metal_kernel(
        name="tierglu_naxc_b",
        input_names=["hq", "hs", "hqs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B_C, header=N6._HDR4)


def tier_glu_naxc(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, grp=None, nsg_b=2, ms_np=None, ls_np=None,
                  ks=4, ks_b=2):
    """tier_glu_nax6's signature and shapes ([M, R, D] per-rank outputs), int8 x int4 matrix ops (FAST)"""
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
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", 0), ("GS_C", 1)]
    xq, xs, xqs = quant64(x)
    (h,) = _kac()(inputs=[xq, xs, xqs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                  grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hq, hs, hqs = quant64(h)
    (y,) = _kbc()(inputs=[hq, hs, hqs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
