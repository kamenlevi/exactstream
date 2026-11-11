"""BigMoE's own reference of the records3 conversion (docs/PLAN_records3_convert.md; Leap's format pinned 11-03 17:2x), written
independently of Leap's tool so the two can be checked against each other. Layouts:
  native      every piece in NAX6-N's native order: chunk (t*KG + g)*32 + r = row-major chunk (t*32 + r)*KG + g (a chunk =
              one group of 64 weights: 6 w3 words, 2 l1 words, one s3 / b3 / b value); offsets and sizes as today.
  native+p11  native, and the s3 / b3 / b pieces as 11-bit codes: code = sign << 10 | (exp - base) << 7 | mantissa (base =
              the piece's minimum exponent; refused: zeros, subnormals, inf / nan, span > 7). Block b = t*KG + g = 11 u32
              words, code r at stream bits [11r, 11r + 11), LSB-first; word 8800 = base (bits 0-7); zero padding to 8804
              words (35,216 B). Record: MSB per projection w3 s3 b3, 16 KB-aligned (2,064,384), LSB per projection l1 b
              (720,896), record 2,785,280 B, padding zero.
Writes converted COPIES of some layers into a scratch directory (never in place): the engine's readers are tested on them.
usage: [BigMoE venv] taskpolicy -b python scripts/r3layout_ref.py SRC DST --layout native|native+p11 --layers 0 20 47"""
import argparse, json, os, sys
import numpy as np

PROJ = ("gate_proj", "up_proj", "down_proj")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}   # (O, I) per projection
PAGE = 16384
P11_WORDS = 8804                                        # 800 blocks x 11 words + the base word, padded to 16 B
PAD4K = 36864                                           # native+p11+pad: an MSB scale piece's 4 KB-aligned slot (9 x 4 KB)


def native(x, O, KG, per):
    """row-major [O, KG, per] (any dtype) -> native order"""
    return np.ascontiguousarray(x.reshape(O // 32, 32, KG, per).transpose(0, 2, 1, 3)).reshape(-1)


def rowmajor(y, O, KG, per):
    return np.ascontiguousarray(y.reshape(O // 32, KG, 32, per).transpose(0, 2, 1, 3)).reshape(-1)


def p11_pack(v, O, KG):
    """one row-major bf16 piece (uint16 [O * KG]) -> uint32 [8804]; raises if it does not fit"""
    n = native(v, O, KG, 1)
    ex = (n >> 7) & 0xFF
    if (ex == 0).any() or (ex == 255).any():
        raise ValueError("zero / subnormal / inf / nan in a scale piece")
    base = int(ex.min())
    if int(ex.max()) - base > 7:
        raise ValueError(f"exponent span {int(ex.max()) - base} > 7")
    code = ((n >> 15).astype(np.uint32) << 10) | ((ex - base).astype(np.uint32) << 7) | (n & 0x7F).astype(np.uint32)
    nb = n.size // 32
    bits = ((code.reshape(nb, 32)[:, :, None] >> np.arange(11, dtype=np.uint32)) & 1).astype(np.uint8).reshape(nb, 352)
    words = np.packbits(bits, axis=1, bitorder="little").view("<u4").reshape(-1)
    out = np.zeros(P11_WORDS, np.uint32)
    out[:words.size] = words
    out[11 * nb] = base
    return out


def p11_unpack(w, O, KG):
    """uint32 [8804] -> row-major bf16 piece (uint16 [O * KG])"""
    nb = O // 32 * KG
    base = int(w[11 * nb])
    assert base >> 8 == 0 and not w[11 * nb + 1:].any(), "p11 header / padding"
    bits = np.unpackbits(np.ascontiguousarray(w[:11 * nb]).view(np.uint8), bitorder="little").reshape(nb * 32, 11)
    c = (bits.astype(np.uint32) << np.arange(11, dtype=np.uint32)).sum(1).astype(np.uint32)
    n = (((c & 0x400) << 5) | ((c & 0x3FF) + (base << 7))).astype(np.uint16)
    return rowmajor(n, O, KG, 1)


def entry(r, layout):
    """the converted layer's index entry"""
    e = dict(r)
    e["layout"] = layout
    if layout == "native":
        return e
    pad = layout == "native+p11+pad"                    # (Leap + BigMoE 11-03 22:4x on MoonShots' M-29: zero-copy reads need
    sc = PAD4K if pad else 4 * P11_WORDS                # 4 KB-aligned pieces in the file and the pools; the MSB part keeps its
    m, mo, ms = 0, [], []                               # size: 3 x (614,400 + 2 x 36,864) = 2,064,384)
    for p in range(3):
        for sz in (r["msb_piece_size"][3 * p], sc, sc):
            mo.append(m); ms.append(sz); m += sz
    msb = -(-m // PAGE) * PAGE
    lo, ls, l = [0] * 6, [0] * 6, 0                     # the index keeps the per-projection order (l1, b) x 3
    order = [0, 2, 4, 1, 3, 5] if pad else range(6)     # pad: the three l1 planes first (4 KB-aligned), then the packed b's
    for k in order:
        lo[k] = msb + l; ls[k] = r["lsb_piece_size"][k] if k % 2 == 0 else 4 * P11_WORDS; l += ls[k]
    lsb = -(-l // PAGE) * PAGE
    e.update(recsz=msb + lsb, msb_sz=msb, lsb_sz=lsb, msb_piece_off=mo, msb_piece_size=ms, lsb_piece_off=lo,
             lsb_piece_size=ls)
    return e


def convert(rec, r, e):
    """one row-major record (uint8 [r['recsz']]) -> the layout of entry e"""
    p11 = e["layout"] in ("native+p11", "native+p11+pad")
    out = np.zeros(e["recsz"], np.uint8) if p11 else rec.copy()
    mo, lo = r["msb_piece_off"], r["lsb_piece_off"]
    if p11:                                             # today's padding must be zero (so the inverse is exact)
        ends = [(mo[-1] + r["msb_piece_size"][-1], r["msb_sz"]), (lo[-1] + r["lsb_piece_size"][-1], r["recsz"])]
        assert not any(rec[a:b].any() for a, b in ends), "nonzero padding in a row-major record"
    for p, pr in enumerate(PROJ):
        O, I = SH[pr]
        KG = I // 64
        for (src, dst, size, dt, per, sc) in (
                (mo[3 * p], e["msb_piece_off"][3 * p], r["msb_piece_size"][3 * p], np.uint32, 6, False),
                (mo[3 * p + 1], e["msb_piece_off"][3 * p + 1], r["msb_piece_size"][3 * p + 1], np.uint16, 1, True),
                (mo[3 * p + 2], e["msb_piece_off"][3 * p + 2], r["msb_piece_size"][3 * p + 2], np.uint16, 1, True),
                (lo[2 * p], e["lsb_piece_off"][2 * p], r["lsb_piece_size"][2 * p], np.uint32, 2, False),
                (lo[2 * p + 1], e["lsb_piece_off"][2 * p + 1], r["lsb_piece_size"][2 * p + 1], np.uint16, 1, True)):
            x = np.ascontiguousarray(rec[src:src + size]).view(dt)
            assert x.size == O * KG * per, (pr, x.size)
            y = p11_pack(x, O, KG) if (sc and p11) else native(x, O, KG, per)
            b = y.view(np.uint8)
            out[dst:dst + b.size] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layout", choices=("native", "native+p11", "native+p11+pad"), required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    a = ap.parse_args()
    S, D = os.path.realpath(a.src), os.path.realpath(a.dst)
    if S == D or D.startswith(S + os.sep) or os.path.basename(D).startswith("records3"):
        sys.exit("refused: the destination must be a scratch directory, not the records")
    ix = json.load(open(os.path.join(S, "index.json")))
    os.makedirs(D, exist_ok=True)
    need = sum(entry(ix["layers"][str(L)], a.layout)["recsz"] * ix["layers"][str(L)]["E"] for L in a.layers)
    st = os.statvfs(D)
    if st.f_bavail * st.f_frsize - need < 115e9:
        sys.exit(f"refused: free disk would fall under 115 GB ({need / 1e9:.1f} GB to write)")
    out_ix = {k: v for k, v in ix.items() if k != "layers"}
    out_ix["layers"] = {}
    for L in a.layers:
        r = ix["layers"][str(L)]
        e = entry(r, a.layout)
        with open(os.path.join(S, r["file"]), "rb") as fi, open(os.path.join(D, r["file"]), "wb") as fo:
            for x in range(r["E"]):
                rec = np.frombuffer(fi.read(r["recsz"]), np.uint8)
                fo.write(convert(rec, r, e).tobytes())
        out_ix["layers"][str(L)] = e
        print(f"layer {L}: {a.layout}, record {e['recsz']:,} B", flush=True)
    with open(os.path.join(D, "index.json"), "w") as f:
        json.dump(out_ix, f, indent=1)


if __name__ == "__main__":
    main()
