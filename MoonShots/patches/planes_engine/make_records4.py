"""records3/L{L:02d}.bin -> records4/L{L:02d}.bin: the 2+1+1 plane record (DESIGN_records4.md), lossless, with a built-in
verifier. From records3: q = 2m + l (merge_codes), s = s3 / 2, b = LSB b, b3 = records3's b3. Per projection:
  P2  w2 = MLX bits=2 packing of q >> 2 (value i at bits 2i, 16 per u32), s2 = 4s (bf16, exact), b2 = bf16(fp32(b) +
      fp32(s) * mean(q & 3))                              -> part 1 (padded to 16 KB)
  B3  c = (q >> 1) & 1 bits (bit j of byte i = weight 8i + j), b3          -> part 2
  B4  l = q & 1 bits, b                                                     -> part 3
Verifier (every expert, from the bytes written): q rebuilt from w2, c, l == q; s2 / 4 == s and s2 / 2 == s3 bitwise;
b3 == records3's; b == b; b2 == recomputed from the unpacked nibbles; padding zero; per layer MLX dequantize(bits=2) of w2
with unit scales == q >> 2. usage: make_records4.py --layers 0,17 --out DIR [--src ~/BigMoE/models/bare/records3]
[--chunk 32] [--min-free-gb 110]   (CPU + SSD; run with taskpolicy -b)"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_records3 as M

U32, U16, U8, G = np.uint32, np.uint16, np.uint8, 64


def layout4(shapes=M.SHAPES):
    """part offsets (from the record start) and sizes: P2 (w2, s2, b2) x 3, B3 (c, b3) x 3, B4 (l, b) x 3."""
    o, offs, sizes, parts = 0, [], [], []
    for name, pieces in (("P2", lambda out, K: (out * K // 4, out * K // G * 2, out * K // G * 2)),
                         ("B3", lambda out, K: (out * K // 8, out * K // G * 2)),
                         ("B4", lambda out, K: (out * K // 8, out * K // G * 2))):
        start = o
        for out, K in shapes:
            for n in pieces(out, K):
                offs.append(o); sizes.append(n); o += n
        o = M.pad(o)
        parts.append((name, start, o - start))
    return dict(recsz=o, piece_off=offs, piece_size=sizes, parts=parts)


def m2_pack(w):
    """4-bit words [..., n] (n even) -> u32 words [..., n/2] of q >> 2 in MLX's bits=2 stream (value i at bits 2i)."""
    y = (w >> 2) & U32(0x33333333)
    y = (y & U32(0x03030303)) | ((y & U32(0x30303030)) >> 2)          # per byte: 2 fields (4 bits)
    y = (y & U32(0x000F000F)) | ((y & U32(0x0F000F00)) >> 4)          # per half: 4 fields (8 bits)
    y = (y & U32(0xFF)) | ((y & U32(0x00FF0000)) >> 8)                # 8 fields (16 bits) per 4-bit word
    y = y.reshape(*w.shape[:-1], -1, 2)
    return y[..., 0] | (y[..., 1] << 16)


def bit_plane(w, k):
    """4-bit words [..., n] -> bytes [..., n]: bit j = bit k of nibble j (k = 0: l, k = 1: c)."""
    x = (w >> k) & U32(0x11111111)
    x = (x | (x >> 3)) & U32(0x03030303)
    x = (x | (x >> 6)) & U32(0x000F000F)
    x = (x | (x >> 12)) & U32(0xFF)
    return x.astype(U8)


def unplane(w2, c8, l8):
    """inverse (verifier): w2 [..., n/2] u32, c8 / l8 [..., n] bytes -> 4-bit words [..., n]."""
    h = np.stack([w2 & U32(0xFFFF), w2 >> 16], axis=-1).reshape(*w2.shape[:-1], -1)   # 16 bits = 8 fields per 4-bit word
    y = (h & U32(0xFF)) | ((h & U32(0xFF00)) << 8)
    y = (y & U32(0x000F000F)) | ((y & U32(0x00F000F0)) << 4)
    y = (y & U32(0x03030303)) | ((y & U32(0x0C0C0C0C)) << 2)          # 2-bit fields at nibble positions
    sp = lambda b: (lambda x: (x | (x << 3)) & U32(0x11111111))(
        (lambda x: (x | (x << 6)) & U32(0x03030303))((lambda x: (x | (x << 12)) & U32(0x000F000F))(b.astype(U32))))
    return (y << 2) | (sp(c8) << 1) | sp(l8)


def convert_chunk(r3, n, lay3, lay4):
    R3, R4 = lay3["recsz"], lay4["recsz"]
    mo, ms, lo, ls = lay3["msb_piece_off"], lay3["msb_piece_size"], lay3["lsb_piece_off"], lay3["lsb_piece_size"]
    po, ps = lay4["piece_off"], lay4["piece_size"]
    out = M.aligned(n * R4)
    out[:] = 0
    for p in range(3):
        q = M.merge_codes(M.view(r3, n, R3, mo[3 * p], ms[3 * p], U8, inner=3), M.view(r3, n, R3, lo[2 * p], ls[2 * p], U8))
        s3 = M.view(r3, n, R3, mo[3 * p + 1], ms[3 * p + 1], U16)
        b3 = M.view(r3, n, R3, mo[3 * p + 2], ms[3 * p + 2], U16)
        b = M.view(r3, n, R3, lo[2 * p + 1], ls[2 * p + 1], U16)
        sf = M.bf16_to_f32(s3) * np.float32(0.5)                          # s, exact
        c8, l8 = bit_plane(q, 1), bit_plane(q, 0)
        cnt = (2 * np.bitwise_count(c8.view(np.uint64)) + np.bitwise_count(l8.view(np.uint64))).astype(np.float32)
        b2 = M.f32_to_bf16(M.bf16_to_f32(b) + sf * (cnt / np.float32(G)))
        i2, i3, i4 = 3 * p, 9 + 2 * p, 15 + 2 * p
        M.view(out, n, R4, po[i2], ps[i2], U32)[...] = m2_pack(q)
        M.view(out, n, R4, po[i2 + 1], ps[i2 + 1], U16)[...] = M.f32_to_bf16(sf * np.float32(4))
        M.view(out, n, R4, po[i2 + 2], ps[i2 + 2], U16)[...] = b2
        M.view(out, n, R4, po[i3], ps[i3], U8)[...] = c8
        M.view(out, n, R4, po[i3 + 1], ps[i3 + 1], U16)[...] = b3
        M.view(out, n, R4, po[i4], ps[i4], U8)[...] = l8
        M.view(out, n, R4, po[i4 + 1], ps[i4 + 1], U16)[...] = b
    return out


def verify_chunk(r3, r4, n, lay3, lay4):
    R3, R4 = lay3["recsz"], lay4["recsz"]
    mo, ms, lo, ls = lay3["msb_piece_off"], lay3["msb_piece_size"], lay3["lsb_piece_off"], lay3["lsb_piece_size"]
    po, ps = lay4["piece_off"], lay4["piece_size"]
    bad = dict(q=0, s2=0, s3=0, b3=0, b=0, b2=0, pad=0)
    for p in range(3):
        q = M.merge_codes(M.view(r3, n, R3, mo[3 * p], ms[3 * p], U8, inner=3), M.view(r3, n, R3, lo[2 * p], ls[2 * p], U8))
        s3 = M.view(r3, n, R3, mo[3 * p + 1], ms[3 * p + 1], U16)
        i2, i3, i4 = 3 * p, 9 + 2 * p, 15 + 2 * p
        w2 = M.view(r4, n, R4, po[i2], ps[i2], U32)
        s2 = M.view(r4, n, R4, po[i2 + 1], ps[i2 + 1], U16)
        b2 = M.view(r4, n, R4, po[i2 + 2], ps[i2 + 2], U16)
        c8 = M.view(r4, n, R4, po[i3], ps[i3], U8)
        b3 = M.view(r4, n, R4, po[i3 + 1], ps[i3 + 1], U16)
        l8 = M.view(r4, n, R4, po[i4], ps[i4], U8)
        bb = M.view(r4, n, R4, po[i4 + 1], ps[i4 + 1], U16)
        bad["q"] += int(np.count_nonzero(unplane(w2, c8, l8) != q))
        s2f = M.bf16_to_f32(s2)
        bad["s2"] += int(np.count_nonzero(M.f32_to_bf16(s2f * np.float32(0.25)) != M.f32_to_bf16(M.bf16_to_f32(s3) * np.float32(0.5))))
        bad["s3"] += int(np.count_nonzero(M.f32_to_bf16(s2f * np.float32(0.5)) != s3))
        bad["b3"] += int(np.count_nonzero(b3 != M.view(r3, n, R3, mo[3 * p + 2], ms[3 * p + 2], U16)))
        bad["b"] += int(np.count_nonzero(bb != M.view(r3, n, R3, lo[2 * p + 1], ls[2 * p + 1], U16)))
        nib = ((q[..., None] >> (4 * np.arange(8, dtype=U32))) & U32(15)).reshape(n, -1)       # independent path for b2
        off = (nib & U32(3)).reshape(n, -1, G).sum(-1).astype(np.float32) / np.float32(G)
        ref = M.f32_to_bf16(M.bf16_to_f32(bb) + M.bf16_to_f32(s3) * np.float32(0.5) * off)
        bad["b2"] += int(np.count_nonzero(ref != b2))
    recs = np.ndarray((n, R4), U8, buffer=r4, strides=(R4, 1))
    for name, start, size in lay4["parts"]:
        used = [po[i] + ps[i] for i in range(len(po)) if start <= po[i] < start + size]
        bad["pad"] += int(np.count_nonzero(recs[:, max(used):start + size]))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.expanduser("~/BigMoE/models/bare/records3"))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--min-free-gb", type=float, default=110)
    a = ap.parse_args()
    sidx = json.load(open(os.path.join(a.src, "index.json")))
    layers = sorted(int(k) for k in sidx["layers"]) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    os.makedirs(a.out, exist_ok=True)
    lay3, lay4 = M.layout(), layout4()
    ipath = os.path.join(a.out, "index.json")
    oidx = json.load(open(ipath)) if os.path.exists(ipath) else {"align": M.ALIGN, "format": "records4", "layers": {}}
    import shutil
    ok_all = True
    for L in layers:
        E = sidx["layers"][str(L)]["E"]
        assert sidx["layers"][str(L)]["recsz"] == lay3["recsz"], L
        if shutil.disk_usage(a.out).free / 1e9 - E * lay4["recsz"] / 1e9 < a.min_free_gb:
            print(f"STOP layer {L}: disk floor {a.min_free_gb} GB", flush=True); ok_all = False; break
        src, dst = os.path.join(a.src, f"L{L:02d}.bin"), os.path.join(a.out, f"L{L:02d}.bin")
        t0 = time.time()
        fs, fo = M.open_nocache(src, os.O_RDONLY), M.open_nocache(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        buf = M.aligned(a.chunk * lay3["recsz"])
        for e0 in range(0, E, a.chunk):
            n = min(a.chunk, E - e0)
            M.pread_into(fs, buf[:n * lay3["recsz"]], e0 * lay3["recsz"])
            M.pwrite_all(fo, convert_chunk(buf, n, lay3, lay4), e0 * lay4["recsz"])
        os.fsync(fo); os.close(fo); os.close(fs)
        t1 = time.time()
        fs, fd = M.open_nocache(src, os.O_RDONLY), M.open_nocache(dst, os.O_RDONLY)
        b3_, b4_ = M.aligned(a.chunk * lay3["recsz"]), M.aligned(a.chunk * lay4["recsz"])
        bad = dict(q=0, s2=0, s3=0, b3=0, b=0, b2=0, pad=0)
        for e0 in range(0, E, a.chunk):
            n = min(a.chunk, E - e0)
            M.pread_into(fs, b3_[:n * lay3["recsz"]], e0 * lay3["recsz"]); M.pread_into(fd, b4_[:n * lay4["recsz"]], e0 * lay4["recsz"])
            for k, v in verify_chunk(b3_, b4_, n, lay3, lay4).items():
                bad[k] += v
        os.close(fs); os.close(fd)
        ok = all(v == 0 for v in bad.values())
        ok_all &= ok
        print(f"L{L:02d}: {'OK' if ok else 'FAIL'} convert {t1 - t0:.1f}s verify {time.time() - t1:.1f}s mismatches {bad}", flush=True)
        oidx["layers"][str(L)] = dict(file=f"L{L:02d}.bin", E=E, recsz=lay4["recsz"], parts=lay4["parts"],
                                      piece_off=lay4["piece_off"], piece_size=lay4["piece_size"], verified=bool(ok))
        json.dump(oidx, open(ipath, "w"), indent=1)
    print("ALL OK" if ok_all else "FAILURES", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
