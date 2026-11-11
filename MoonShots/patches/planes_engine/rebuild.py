"""GPU rebuild for the 3b path (BigMoE DESIGN_3bit.md): 4-bit-rank experts from the MSB-3 pool + the LSB-1 sidecar into
per-call 4-bit scratch rows, for gather_qmm(bits=4). One dispatch for all three projections and every pair, reading the
pools by slot (no gathered copies): per thread 32 weights = 3 words of 3-bit codes + 1 word of low bits -> 4 words
(q = 2m + l, the ext's spread8 bit math), stored as one 16-byte write; the thread of an even item also writes its group's
scale s3 * 0.5 (exact) and the LSB part's bias. Bit-exact by construction (integer ops, exact halving).
rebuild_slots(msb, lsb, msb_slot, lsb_slot) -> {proj: (w4 [P, out, K/8], s [P, out, K/64], b [P, out, K/64])}"""
from functools import lru_cache
import mlx.core as mx

PROJ = ("gate_proj", "up_proj", "down_proj")

_SRC = r"""
    const uint k = thread_position_in_grid.x;               // 32-weight item within the projection
    const uint p = thread_position_in_grid.y;               // pair (4-bit rank use)
    const uint j = thread_position_in_grid.z;               // projection
    if (k >= (uint)NI) return;
    const device uint* w3 = j == 0 ? w3g : (j == 1 ? w3u : w3d);
    const device uint* l1 = j == 0 ? l1g : (j == 1 ? l1u : l1d);
    const device bfloat16_t* s3 = j == 0 ? s3g : (j == 1 ? s3u : s3d);
    const device bfloat16_t* bb = j == 0 ? bg : (j == 1 ? bu : bd);
    device uint* o = j == 0 ? og : (j == 1 ? ou : od);
    device bfloat16_t* so = j == 0 ? sog : (j == 1 ? sou : sod);
    device bfloat16_t* bo = j == 0 ? bog : (j == 1 ? bou : bod);
    const uint sm = (uint)msb_slot[p], sl = (uint)lsb_slot[p];
    const device uint* a = w3 + (size_t)sm * (3 * NI) + 3 * k;
    const uint x0 = a[0], x1 = a[1], x2 = a[2];
    const uint l = l1[(size_t)sl * NI + k];
    uint4 r;
    r.x = spread8(x0 & 0xFFFFFFu, l);
    r.y = spread8((x0 >> 24) | ((x1 & 0xFFFFu) << 8), l >> 8);
    r.z = spread8((x1 >> 16) | ((x2 & 0xFFu) << 16), l >> 16);
    r.w = spread8(x2 >> 8, l >> 24);
    ((device uint4*)o)[(size_t)p * NI + k] = r;
    if ((k & 1u) == 0) {                                    // one thread per group of 64 weights
        const uint g = k >> 1;
        so[(size_t)p * (NI / 2) + g] = (bfloat16_t)((float)s3[(size_t)sm * (NI / 2) + g] * 0.5f);
        bo[(size_t)p * (NI / 2) + g] = bb[(size_t)sl * (NI / 2) + g];
    }
"""

_HDR = r"""
inline uint spread8(uint x, uint l) {                       // 24 code bits (8 x m) + 8 low bits -> 8 nibbles q = 2m + l
    uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
    a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
    a = (a & 0x07070707u) | ((a & 0x38383838u) << 1);
    uint y = l & 0xFFu;
    y = (y | (y << 12)) & 0x000F000Fu;
    y = (y | (y << 6)) & 0x03030303u;
    y = (y | (y << 3)) & 0x11111111u;
    return (a << 1) | y;
}
"""


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="planes_rebuild_slots",
        input_names=["w3g", "w3u", "w3d", "s3g", "s3u", "s3d", "l1g", "l1u", "l1d", "bg", "bu", "bd", "msb_slot", "lsb_slot"],
        output_names=["og", "ou", "od", "sog", "sou", "sod", "bog", "bou", "bod"],
        source=_SRC, header=_HDR)


def rebuild_slots(msb, lsb, msb_slot, lsb_slot, tg=256):
    """msb[proj] = {"w3": [C_m, out, K*3/32] u32, "s3": [C_m, out, K/64] bf16, ...}; lsb[proj] = {"l1": [C_l, out, K/32],
    "b": [C_l, out, K/64]}; msb_slot, lsb_slot: int32 [P]. Every projection must have the same weight count (the 125B's
    gate/up [640, 2560] and down [2560, 640] do)."""
    shp = {p: msb[p]["s3"].shape[1:] for p in PROJ}                     # (out, K/64)
    ni = {p: shp[p][0] * shp[p][1] * 2 for p in PROJ}                   # 32-weight items per projection
    NI = ni["gate_proj"]
    assert all(v == NI for v in ni.values()), ni
    P = msb_slot.size
    outs = _kernel()(
        inputs=[msb[p]["w3"] for p in PROJ] + [msb[p]["s3"] for p in PROJ] + [lsb[p]["l1"] for p in PROJ]
        + [lsb[p]["b"] for p in PROJ] + [msb_slot.astype(mx.int32).reshape(-1), lsb_slot.astype(mx.int32).reshape(-1)],
        template=[("NI", NI)],
        grid=(NI, P, 3), threadgroup=(tg, 1, 1),
        output_shapes=[(P, shp[p][0], shp[p][1] * 8) for p in PROJ] + [(P, *shp[p]) for p in PROJ] * 2,
        output_dtypes=[mx.uint32] * 3 + [mx.bfloat16] * 6)
    return {p: (outs[i], outs[3 + i], outs[6 + i]) for i, p in enumerate(PROJ)}
