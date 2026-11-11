"""p11: the 11-bit packed scales / biases of the native+p11 record layout (Leap, Tue 2026-11-03 17:2x; the orchestrator's
routing after Kamen's yes to the combined conversion; format pinned with BigMoE 17:2x; MoonShots M-27 / M-27b).
This is the reference: the kernels' p11 reads, the combined conversion tool and BigMoE's C++ unpack are checked
against it bit for bit.

Piece (one projection's s3, b3 or b: O x KG bf16 values, 25,600 for gate / up / down alike):
  - order: native first (per = 1: native[(t*KG + g)*32 + r] = rowmajor[(t*32 + r)*KG + g]); block b = t*KG + g
    holds the 32 rows r = 0..31; nb = O*KG/32 blocks (800);
  - code (11 bits) = sign << 10 | (exp - base) << 7 | mantissa7, exp = bf16 bits 7-14, base = the piece's minimum
    exponent; decode: bf16 = ((c & 0x400) << 5) | ((c & 0x3FF) + (base << 7)); no zero code, so a piece packs only
    if it has no zero, subnormal, inf or nan and its exponent span is <= 7;
  - words: block b = u32 words [11b, 11b + 11), LSB-first; value r at stream bits [11r, 11r + 11) = word 11b + (11r >> 5),
    shift (11r) & 31, spilling into the next word when the shift is > 21;
  - word 11*nb = base (bits 0-7, the rest 0); zero padding to 16 B. 25,600 values -> 8,804 words = 35,216 B.
Record (layout "native+p11"): MSB part per projection w3 (native) + s3 + b3 (p11), zero-padded to 16 KB; LSB part per
projection l1 (native) + b (p11), zero-padded to 16 KB. records3: 2,064,384 + 720,896 = 2,785,280 B per record.
Usage: [BigMoE venv] p11.py   (self-test on records3 layer 20 and the native sample; CPU only, reads ~3 GB)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))

PROJ = ("gate_proj", "up_proj", "down_proj")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
ALIGN = 16384
PADW = 9216                    # native+p11+pad: the MSB scale pieces' words (36,864 B = 9 x 4 KB)
_POS = [(11 * j >> 5, 11 * j & 31) for j in range(32)]


def _nm(o, kg, per):
    import tierglu_nax6n as NN                                     # lazy: tierglu_nax6n imports this module
    return NN.native_map(o, kg, per)


# The kernels' p11 read (tierglu_3cn / tierglu_nax6n with uint32 s3 / b3 / b pools): P11P wraps one expert's piece and
# returns, for a native element index, a float equal to (float) of the bf16 value, bit for bit.
P11_METAL = r"""
struct P11P {
    const device uint* w;
    uint base7;
    uint off;
    float operator[](int i) const {
        const uint j = off + (uint)i, bit = 11u * (j & 31u);
        const uint wi = 11u * (j >> 5) + (bit >> 5), sh = bit & 31u;
        uint c = w[wi] >> sh;
        if (sh > 21u) c |= w[wi + 1] << (32u - sh);
        c &= 0x7FFu;
        return as_type<float>((((c & 0x400u) << 5) | ((c & 0x3FFu) + base7)) << 16);
    }
};
inline P11P p11p(const device uint* pool, uint e, uint pw, uint nb, uint off) {
    P11P v;
    v.w = pool + (size_t)e * pw;
    v.base7 = v.w[11u * nb] << 7;
    v.off = off;
    return v;
}
// 3c-N: a simdgroup decodes its rows x groups once into threadgroup memory (t[g * rows + row] = v[row + 32 g]); P11T reads
// them back with the kernels' own index gi = row + 32 g. rows divides 32, so a lane keeps one row (one bit offset) and
// steps 32 / rows groups (blocks) at a time; two-word reads, no branch.
inline void p11_fill(threadgroup float* t, const P11P v, int rows, int groups, uint lane) {
    const int row = (int)lane % rows, gs = 32 / rows;
    const uint j = v.off + (uint)row + 32u * (lane / (uint)rows), bit = 11u * (j & 31u), sh = bit & 31u;
    uint wi = 11u * (j >> 5) + (bit >> 5);
    for (int g = (int)lane / rows; g < groups; g += gs, wi += 11u * (uint)gs) {
        const uint c = (uint)((((ulong)v.w[wi + 1] << 32) | (ulong)v.w[wi]) >> sh) & 0x7FFu;
        t[g * rows + row] = as_type<float>((((c & 0x400u) << 5) | ((c & 0x3FFu) + v.base7)) << 16);
    }
}
struct P11T {
    threadgroup const float* t;
    int rows;
    float operator[](int gi) const { return t[(gi >> 5) * rows + (gi & 31)]; }
};
// NAX6-N: a lane reads one row (off & 31) of every group: [32 g] -> block (off >> 5) + g, the same bit offset each time
struct P11R {
    const device uint* w;
    uint base7;
    uint wi0;
    uint sh;
    float operator[](int i) const {
        const uint wi = wi0 + 11u * ((uint)i >> 5);
        const uint c = (uint)((((ulong)w[wi + 1] << 32) | (ulong)w[wi]) >> sh) & 0x7FFu;
        return as_type<float>((((c & 0x400u) << 5) | ((c & 0x3FFu) + base7)) << 16);
    }
};
inline P11R p11r(const device uint* pool, uint e, uint pw, uint nb, uint off) {
    P11R v;
    v.w = pool + (size_t)e * pw;
    v.base7 = v.w[11u * nb] << 7;
    const uint bit = 11u * (off & 31u);
    v.sh = bit & 31u;
    v.wi0 = 11u * (off >> 5) + (bit >> 5);
    return v;
}
"""


def piece_words(nvals):
    return -(-(11 * (nvals // 32) + 1) // 4) * 4                  # u32 words incl. the base word, 16-B padded


def packable(v):
    """v: (P, n) uint16 bf16 bits -> (P,) bool: the piece packs losslessly at 11 bits."""
    ex = (v >> 7).astype(np.int32) & 0xFF
    bad = ((ex == 0) | (ex == 255)).any(1)
    return ~bad & (ex.max(1) - ex.min(1) <= 7)


def pack(v):
    """v: (P, n) uint16, native order -> (P, piece_words(n)) uint32. Raises if a piece does not pack."""
    v = np.ascontiguousarray(v, dtype=np.uint16)
    P, n = v.shape
    ok = packable(v)
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} of {P} pieces do not pack at 11 bits")
    ex = (v >> 7).astype(np.uint32) & 0xFF
    base = ex.min(1, keepdims=True)
    c = ((v.astype(np.uint32) >> 15) << 10) | ((ex - base) << 7) | (v.astype(np.uint32) & 0x7F)
    c = c.reshape(P, n // 32, 32)
    w = np.zeros((P, n // 32, 11), np.uint64)
    for j, (wi, sh) in enumerate(_POS):
        w[:, :, wi] |= c[:, :, j].astype(np.uint64) << np.uint64(sh)
    w &= 0xFFFFFFFF                                                # bits above 31 spill into the next word:
    for j, (wi, sh) in enumerate(_POS):
        if sh > 21:
            w[:, :, wi + 1] |= c[:, :, j].astype(np.uint64) >> np.uint64(32 - sh)
    out = np.zeros((P, piece_words(n)), np.uint32)
    out[:, :11 * (n // 32)] = w.reshape(P, -1).astype(np.uint32)
    out[:, 11 * (n // 32)] = base[:, 0]
    return out


def unpack(w, n):
    """w: (P, piece_words(n)) uint32 -> (P, n) uint16 bf16 bits, native order."""
    w = np.ascontiguousarray(w, dtype=np.uint32)
    P, nb = w.shape[0], n // 32
    blk = w[:, :11 * nb].reshape(P, nb, 11).astype(np.uint64)
    base = w[:, 11 * nb].astype(np.uint32)[:, None, None]
    c = np.empty((P, nb, 32), np.uint32)
    for j, (wi, sh) in enumerate(_POS):
        x = blk[:, :, wi] >> np.uint64(sh)
        if sh > 21:
            x |= blk[:, :, wi + 1] << np.uint64(32 - sh)
        c[:, :, j] = (x & 0x7FF).astype(np.uint32)
    return (((c & 0x400) << 5) | ((c & 0x3FF) + (base << 7))).astype(np.uint16).reshape(P, n)


def pieces(r):
    """The record's pieces from an index.json layer entry (any layout): (name, part, proj, off, size, dtype, per, packed)."""
    out = []
    p11 = r.get("layout") in ("native+p11", "native+p11+pad")
    for p, pr in enumerate(PROJ):
        for k, (nm, dt, per) in enumerate((("w3", np.uint32, 6), ("s3", np.uint16, 1), ("b3", np.uint16, 1))):
            out.append((nm, "msb", pr, r["msb_piece_off"][3 * p + k], r["msb_piece_size"][3 * p + k], dt, per, p11 and k > 0))
        for k, (nm, dt, per) in enumerate((("l1", np.uint32, 2), ("b", np.uint16, 1))):
            out.append((nm, "lsb", pr, r["lsb_piece_off"][2 * p + k], r["lsb_piece_size"][2 * p + k], dt, per, p11 and k > 0))
    return out


def layout_p11(r, pad=False):
    """The native+p11 index entry for a row-major (or native) records3 layer entry r. pad (BigMoE + MoonShots' M-29, Tue
    22:5x): "native+p11+pad", every piece 4 KB-aligned in the record for zero-copy slot reads. The MSB scale pieces are
    zero-padded to PADW words (36,864 B, so msb_sz is unchanged), and the LSB part holds the three l1 planes first, then
    the three b pieces unpadded (lsb_sz unchanged). The index arrays keep their logical order (per projection: w3, s3, b3
    / l1, b) with these physical offsets."""
    q = {k: v for k, v in r.items() if k not in ("recsz", "msb_sz", "lsb_sz", "msb_piece_off", "msb_piece_size",
                                                 "lsb_piece_off", "lsb_piece_size", "layout")}
    mo, ms, lo, ls = [], [], [], []
    off = 0
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        pw = 4 * (PADW if pad else piece_words(o * i // 64))
        for k, sz in enumerate((r["msb_piece_size"][3 * p], pw, pw)):
            mo.append(off); ms.append(sz); off += sz
    msb_sz = -(-off // ALIGN) * ALIGN
    if pad:
        lo, ls = [0] * 6, [0] * 6
        off = msb_sz
        for k in (0, 1):                                   # all l1 planes, then all b pieces
            for p, pr in enumerate(PROJ):
                o, i = SH[pr]
                sz = r["lsb_piece_size"][2 * p] if k == 0 else 4 * piece_words(o * i // 64)
                lo[2 * p + k], ls[2 * p + k] = off, sz
                off += sz
    else:
        off = msb_sz
        for p, pr in enumerate(PROJ):
            o, i = SH[pr]
            for k, sz in enumerate((r["lsb_piece_size"][2 * p], 4 * piece_words(o * i // 64))):
                lo.append(off); ls.append(sz); off += sz
    lsb_sz = -(-(off - msb_sz) // ALIGN) * ALIGN
    q.update(layout="native+p11+pad" if pad else "native+p11", recsz=msb_sz + lsb_sz, msb_sz=msb_sz, lsb_sz=lsb_sz,
             msb_piece_off=mo, msb_piece_size=ms, lsb_piece_off=lo, lsb_piece_size=ls)
    return q


def _piece_shape(pr, per, dt):
    o, i = SH[pr]
    kg = i // 64
    return o, kg, o * kg * per


def to_p11(raw, r, pad=False):
    """raw: (E, recsz) uint8 records of a ROW-MAJOR layer (entry r) -> (E, recsz') uint8 native+p11 records (pad:
    native+p11+pad; padding bytes are zero)."""
    q = layout_p11(r, pad)
    E = raw.shape[0]
    out = np.zeros((E, q["recsz"]), np.uint8)
    for (nm, part, pr, off, sz, dt, per, _), (_, _, _, off2, sz2, _, _, packed) in zip(pieces(r), pieces(q)):
        o, kg, n = _piece_shape(pr, per, dt)
        x = np.ascontiguousarray(raw[:, off:off + n * np.dtype(dt).itemsize]).view(dt).reshape(E, n)
        x = x[:, _nm(o, kg, per)]
        y = pack(x).view(np.uint8) if packed else np.ascontiguousarray(x).view(np.uint8)
        out[:, off2:off2 + y.shape[1]] = y.reshape(E, -1)
    return out, q


def from_p11(new, q, r):
    """new: (E, recsz') native+p11 records (entry q) -> (E, recsz) row-major records of entry r (zero padding)."""
    E = new.shape[0]
    out = np.zeros((E, r["recsz"]), np.uint8)
    for (nm, part, pr, off2, sz2, dt, per, packed), (_, _, _, off, sz, _, _, _) in zip(pieces(q), pieces(r)):
        o, kg, n = _piece_shape(pr, per, dt)
        if packed:
            x = unpack(np.ascontiguousarray(new[:, off2:off2 + 4 * piece_words(n)]).view(np.uint32), n)
        else:
            x = np.ascontiguousarray(new[:, off2:off2 + n * np.dtype(dt).itemsize]).view(dt).reshape(E, n)
        y = np.empty_like(x)
        y[:, _nm(o, kg, per)] = x
        out[:, off:off + n * np.dtype(dt).itemsize] = y.view(np.uint8).reshape(E, -1)
    return out


def self_test(L=20, chunk=64):
    import json, time
    RD = os.path.expanduser("~/BigMoE/models/bare/records3")
    assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
        "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
    ND = os.path.expanduser("~/Leap/data/native_sample")
    r = json.load(open(f"{RD}/index.json"))["layers"][str(L)]
    raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(r["E"], r["recsz"]))
    nat = np.memmap(f"{ND}/{r['file']}", dtype=np.uint8, mode="r", shape=(r["E"], r["recsz"])) if os.path.exists(f"{ND}/{r['file']}") else None
    t0 = time.time()
    rng = np.random.default_rng(0)
    # 1. the codec on synthetic edge cases: every sign / exponent offset / mantissa, base 1 and 247
    for base in (1, 120, 247):
        c = np.arange(2048, dtype=np.uint32)
        v = (((c & 0x400) << 5) | ((c & 0x3FF) + (base << 7))).astype(np.uint16)
        v = np.tile(v, 13)[:25600][None]
        rng.shuffle(v[0])
        assert np.array_equal(unpack(pack(v), 25600), v), base
    bad = np.full((1, 25600), 0x3F80, np.uint16); bad[0, 5] = 0
    assert not packable(bad)[0]
    # 2. a whole layer: row-major -> native+p11 -> row-major == the file, bitwise, chunk by chunk; the native pieces of
    #    the packed record == the native sample's, and its unpacked scales == the sample's
    q, ok_rt, ok_nat = None, True, True
    for e0 in range(0, r["E"], chunk):
        a = np.asarray(raw[e0:e0 + chunk])
        b, q = to_p11(a, r)
        ok_rt &= bool(np.array_equal(from_p11(b, q, r), a))
        if nat is not None:
            na = np.asarray(nat[e0:e0 + chunk])
            for (nm, part, pr, off, sz, dt, per, _), (_, _, _, off2, sz2, _, _, packed) in zip(pieces(r), pieces(q)):
                o, kg, n = _piece_shape(pr, per, dt)
                x = np.ascontiguousarray(na[:, off:off + n * np.dtype(dt).itemsize]).view(dt).reshape(len(na), n)
                if packed:
                    y = unpack(np.ascontiguousarray(b[:, off2:off2 + 4 * piece_words(n)]).view(np.uint32), n)
                else:
                    y = np.ascontiguousarray(b[:, off2:off2 + n * np.dtype(dt).itemsize]).view(dt).reshape(len(na), n)
                ok_nat &= bool(np.array_equal(x, y))
    print(f"p11 self-test layer {L}: codec edge cases ok; round trip {ok_rt}; == native sample {ok_nat if nat is not None else 'n/a'}; "
          f"recsz {r['recsz']} -> {q['recsz']} (msb {q['msb_sz']}, lsb {q['lsb_sz']}); {time.time() - t0:.1f} s", flush=True)
    return ok_rt and ok_nat


if __name__ == "__main__":
    sys.exit(0 if self_test(*(int(a) for a in sys.argv[1:2])) else 1)
