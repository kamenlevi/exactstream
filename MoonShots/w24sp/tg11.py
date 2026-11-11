"""W24-sp step (2): today's 3c (BigMoE's live copy, ~/MoonShots/patches/planes_engine/tierglu.py, imported read-only)
with every scale / bias read from the 11-bit planar format instead of bf16:
  plane A = sign + 7 mantissa bits (1 byte per value), plane B = 3-bit exponent offset from the piece's base exponent
  (MLX 3-bit packing, 8 values in 3 bytes), base = 1 byte per piece. bf16 bits = (A & 0x80) << 8 | (B + base) << 7 |
  (A & 0x7F); the kernel builds the float bits of that bf16 directly, so every scale / bias is the same float as
  (float)bf16 and the rest of 3c's arithmetic is untouched (meant to be bit-identical).
Pools per projection: msb[p] = {"w3": as today, "sb": [C_m, SBW] u8 = sA | sB | bA | bB | s base | b base | pad},
lsb[p] = {"l1": as today, "sb": [C_l, SBL] u8 = bA | bB | base | pad}; G = 25,600 values per piece (640 x 40 / 2560 x 10).
Tiers 4 and 3 only (no 2-bit bias pool). tier_glu11(...) has tier_glu's signature minus b2."""
import os, sys
from functools import lru_cache
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
import tierglu as TG                                        # today's 3c, read only

G = 25600                                                   # values per scale / bias piece
PB = G + G * 3 // 8                                         # 35,200: one packed piece (A + B)
SBB = 2 * PB                                                # MSB block: offset of the s / b base bytes
SBW = (SBB + 2 + 15) // 16 * 16                             # 70,416: MSB block per projection (16-byte aligned)
SBL = (PB + 1 + 15) // 16 * 16                              # 35,216: LSB block per projection

_HDR = TG._HDR + r"""
// [W24-sp] 11-bit planar scale / bias: A[0..G) = sign + 7 mantissa bits, then B = 3-bit exponent offsets from the
// piece's base (MLX 3-bit packing: value i at bits 3(i % 8) of the 3-byte group i / 8). Returns the exact bf16 as float.
inline float tg_sb11(const device uint8_t* A, uint g, uint i, uint base) {
    const device uint8_t* B = A + g + (i >> 3) * 3;
    const uint w = (uint)B[0] | ((uint)B[1] << 8) | ((uint)B[2] << 16);
    const uint a = A[i];
    return as_type<float>(((a & 0x80u) << 24) | ((((w >> ((i & 7u) * 3u)) & 7u) + base) << 23) | ((a & 0x7Fu) << 16));
}
"""

_RA = [  # kernel A: (today's text, 11-bit text)
    ("    const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * N + j0) * KG;\n",
     "    const device uint8_t* sbm = (sg == 0 ? sbg : sbu) + (size_t)e * SBW_C;       // packed s3 | b3 of expert e\n"
     "    const uint i0 = j0 * KG, sbase = sbm[SBB_C], bbase = sbm[SBB_C + 1];\n"),
    ("        const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;\n",
     "        const device uint8_t* sbl = (sg == 0 ? sblg : sblu) + (size_t)el * SBL_C;\n"
     "        const uint lbase = sbl[PB_C];\n"),
    ("                const float s = (float)s3[gi] * 0.5f;\n                const float bb = bl[gi];\n",
     "                const float s = tg_sb11(sbm, G_C, i0 + gi, sbase) * 0.5f;\n"
     "                const float bb = tg_sb11(sbl, G_C, i0 + gi, lbase);\n"),
    ("        const device T* b3 = tier == 2 ? (sg == 0 ? b2g : b2u) + ((size_t)e * N + j0) * KG\n"
     "                                       : (sg == 0 ? b3g : b3u) + ((size_t)e * N + j0) * KG;\n",
     "        const device uint8_t* b3p = sbm + PB_C;\n"),
    ("                const float s = s3[gi];\n                const float bb = b3[gi];\n",
     "                const float s = tg_sb11(sbm, G_C, i0 + gi, sbase);\n"
     "                const float bb = tg_sb11(b3p, G_C, i0 + gi, bbase);\n"),
]
_RB = [  # kernels B and BC: (today's text, 11-bit text, count)
    ("    const device T* s3 = s3d + ((size_t)e * D + d0) * NG;\n",
     "    const device uint8_t* sbm = sbd + (size_t)e * SBW_C;\n"
     "    const uint i0 = d0 * NG, sbase = sbm[SBB_C], bbase = sbm[SBB_C + 1];\n", 1),
    ("        const device T* bl = bd + ((size_t)el * D + d0) * NG;\n",
     "        const device uint8_t* sbl = sbld + (size_t)el * SBL_C;\n        const uint lbase = sbl[PB_C];\n", 1),
    ("(float)s3[gi] * 0.5f, (float)bl[gi], sum)",
     "tg_sb11(sbm, G_C, i0 + gi, sbase) * 0.5f, tg_sb11(sbl, G_C, i0 + gi, lbase), sum)", 2),
    ("        const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * D + d0) * NG;\n",
     "        const device uint8_t* b3p = sbm + PB_C;\n", 1),
    ("(float)s3[gi], (float)b3[gi], sum)",
     "tg_sb11(sbm, G_C, i0 + gi, sbase), tg_sb11(b3p, G_C, i0 + gi, bbase), sum)", 2),
]


def _sub(src, reps):
    for r in reps:
        old, new, n = r if len(r) == 3 else (*r, 1)
        assert src.count(old) == n, (old, src.count(old))
        src = src.replace(old, new)
    return src


_A11, _B11, _BC11 = _sub(TG._A, _RA), _sub(TG._B, _RB), _sub(TG._BC, _RB)


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(name="tierglu11_a", input_names=["x", "w3g", "w3u", "sbg", "sbu", "l1g", "l1u", "sblg", "sblu",
                                                                 "msb_slot", "lsb_slot"],
                                output_names=["h"], source=_A11, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(name="tierglu11_b", input_names=["h", "w3d", "sbd", "l1d", "sbld", "msb_slot", "lsb_slot"],
                                output_names=["y"], source=_B11, header=_HDR)


@lru_cache(None)
def _kbc():
    return mx.fast.metal_kernel(name="tierglu11_bc", input_names=["h", "w3d", "sbd", "l1d", "sbld", "msb_slot", "lsb_slot",
                                                                  "scores"],
                                output_names=["y"], source=_BC11, header=_HDR)


def tier_glu11(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", ra=4, rb=4, nsg_b=2, scores=None, ly=8, rbc=4, ld=0):
    """tier_glu on the 11-bit pools: same grids, same per-pair arithmetic"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = TG.plan_masks(plan)
    assert m2 == 0, "no 2-bit tier in the 11-bit prototype"
    assert msb_slot.shape == (M, R), (msb_slot.shape, M, R)
    N = msb["gate_proj"]["w3"].shape[1]
    D = msb["down_proj"]["w3"].shape[1]
    assert K % 512 == 0 and N % ra == 0 and D % (rb * nsg_b) == 0 and N % 64 == 0 and N * K // 64 == G == D * N // 64
    if r4:
        assert lsb_slot is not None and lsb_slot.shape == (M, r4), (None if lsb_slot is None else lsb_slot.shape, r4)
        ls, L = lsb_slot.astype(mx.int32).reshape(-1), lsb
    else:
        ls, L = mx.zeros((1,), mx.int32), {p: {"l1": msb[p]["w3"], "sb": msb[p]["sb"]} for p in TG.PROJ}   # unread
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in TG.PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", 0), ("R4_C", max(r4, 1)), ("G_C", G), ("SBW_C", SBW),
            ("SBB_C", SBB), ("PB_C", PB), ("SBL_C", SBL)]
    (h,) = _ka()(inputs=[x, g["w3"], u["w3"], g["sb"], u["sb"], L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["sb"],
                         L["up_proj"]["sb"], ms, ls],
                 template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                 grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
                 output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    if scores is not None:
        assert D % rbc == 0 and R * 32 <= 1024
        (y,) = _kbc()(inputs=[h, d["w3"], d["sb"], L["down_proj"]["l1"], L["down_proj"]["sb"], ms, ls,
                              scores.astype(x.dtype).reshape(-1)],
                      template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                      grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                      output_shapes=[(M, D)], output_dtypes=[x.dtype])
        return y
    (y,) = _kb()(inputs=[h, d["w3"], d["sb"], L["down_proj"]["l1"], L["down_proj"]["sb"], ms, ls],
                 template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                 grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                 output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
