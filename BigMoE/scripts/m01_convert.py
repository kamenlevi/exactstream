"""R-44 (a) M-01 converter (CPU / SSD, exact): records3/ -> records3m/, a fixed-width re-pack of records3's per-group
fields (MoonShots' M-01 SBX with fixed-width fields instead of rANS, after R-16's KILL at 24 GB). Code planes are copied
unchanged; s3 / b3 (MSB part) and b (LSB part) become (scripts/m01_fields.py measured the ranges):
  MSB, per projection of G groups: A = s3 sign | s3 mantissa (G B); B = b3 mantissa | s3 exponent offset bit 0 << 7 (G B);
    N = nibbles (G/2 B; group 2j low, 2j+1 high) = s3 exponent offset bits 1-2 | (b3 exponent - s3 exponent - d0) << 2.
    s3 exponent = base_s (per expert and projection) + a 3-bit offset; b3's sign is the opposite of s3's (checked).
  LSB, per projection: A = b sign | b mantissa (G B); E = 3-bit offsets of b's exponent to base_b (3G/8 B, 8 per 3 bytes,
    little-endian).
  Part = code planes (w3 x 3, or l1 x 3) | fields of projections 0, 1, 2 | 8-byte header ('M1', d0 x 3, base x 3; the LSB
  header's d0 bytes are 0) | zero pad to 16 KB.
The MSB and LSB parts each unpack on their own (the tier pools read them separately). For the 125B: MSB 2,162,688 ->
2,048,000 B (-7 blocks), LSB 770,048 -> 720,896 B (-3 blocks). Every expert is unpacked again (numpy, the ext's formula)
and compared with its records3 bytes before it is written. Resumable per layer; stops before the disk falls under 110 GB
free, or after --minutes (to fit a quiet-lock chunk).
usage: [BigMoE venv] taskpolicy -b python m01_convert.py [--layers 0-47] [--minutes 12]"""
import argparse, fcntl, json, os, shutil, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0-47")
ap.add_argument("--minutes", type=float, default=12.0)
ap.add_argument("--chunk", type=int, default=64)
args = ap.parse_args()
a, _, b = args.layers.partition("-")
LAYERS = list(range(int(a), int(b or a) + 1))
SRC = os.path.expanduser("~/BigMoE/models/bare/records3/")
DST = os.path.expanduser("~/BigMoE/models/bare/records3m/")
os.makedirs(DST, exist_ok=True)
IX3 = json.load(open(SRC + "index.json"))["layers"]
assert all(r.get("layout", "row-major") == "row-major" for r in IX3.values()), "records3 is converted (docs/PLAN_records3_convert.md): this script reads row-major records only"
IXP = DST + "index.json"
IXM = json.load(open(IXP)) if os.path.exists(IXP) else {"align": 16384, "format": "records3m", "src": "records3",
                                                           "layers": {}}
ALIGN, F_NOCACHE = 16384, 48
T0 = time.time()
up = lambda n: -(-n // ALIGN) * ALIGN


def plan(r):
    """sizes of one records3 layer -> M-01 part layout"""
    W3 = [r["msb_piece_size"][3 * p] for p in range(3)]
    G = [r["msb_piece_size"][3 * p + 1] // 2 for p in range(3)]
    L1 = [r["lsb_piece_size"][2 * p] for p in range(3)]
    assert all(g % 8 == 0 for g in G)
    msb_body = sum(W3) + sum(2 * g + g // 2 for g in G) + 8
    lsb_body = sum(L1) + sum(g + 3 * g // 8 for g in G) + 8
    return W3, G, L1, up(msb_body), up(lsb_body)


def fields(u):
    sign, ex, man = (u >> 15).astype(np.uint8), ((u >> 7) & 0xFF).astype(np.int32), (u & 0x7F).astype(np.uint8)
    return sign, ex, man


def pack(src, r):
    """[n, recsz3] uint8 records3 rows -> [n, recsz_m] uint8 M-01 rows (and the per-row header bytes)"""
    W3, G, L1, msz, lsz = plan(r)
    n = src.shape[0]
    out = np.zeros((n, msz + lsz), np.uint8)
    om, ol = sum(W3), msz + sum(L1)
    hm, hl = np.zeros((n, 8), np.uint8), np.zeros((n, 8), np.uint8)
    hm[:, 0], hm[:, 1], hl[:, 0], hl[:, 1] = ord("M"), ord("1"), ord("M"), ord("1")
    cm, cl = 0, msz
    for p in range(3):
        g = G[p]
        mo, lo = r["msb_piece_off"][3 * p:3 * p + 3], r["lsb_piece_off"][2 * p:2 * p + 2]
        out[:, cm:cm + W3[p]] = src[:, mo[0]:mo[0] + W3[p]]
        out[:, cl:cl + L1[p]] = src[:, lo[0]:lo[0] + L1[p]]
        cm += W3[p]; cl += L1[p]
        s3 = np.ascontiguousarray(src[:, mo[1]:mo[1] + 2 * g]).view(np.uint16)
        b3 = np.ascontiguousarray(src[:, mo[2]:mo[2] + 2 * g]).view(np.uint16)
        bb = np.ascontiguousarray(src[:, lo[1]:lo[1] + 2 * g]).view(np.uint16)
        ss, es, ms = fields(s3)
        sb3, eb3, mb3 = fields(b3)
        sb, eb, mb = fields(bb)
        assert (sb3 != ss).all(), "b3 sign equals s3 sign somewhere"
        base = es.min(1)
        os_ = es - base[:, None]
        assert (os_ <= 7).all() and (base >= 0).all(), "s3 exponent span > 7"
        d = eb3 - es
        d0 = d.min(1)
        dd = d - d0[:, None]
        assert (dd <= 3).all() and (d0 >= -128).all() and (d0 <= 127).all(), "b3 exponent offset span > 3"
        bbase = eb.min(1)
        ob = eb - bbase[:, None]
        assert (ob <= 7).all(), "b exponent span > 7"
        A = (ss << 7) | ms
        B = mb3 | ((os_ & 1).astype(np.uint8) << 7)
        nib = (((os_ >> 1) & 3) | (dd << 2)).astype(np.uint8)
        N = nib[:, 0::2] | (nib[:, 1::2] << 4)
        out[:, om:om + g], out[:, om + g:om + 2 * g], out[:, om + 2 * g:om + 2 * g + g // 2] = A, B, N
        om += 2 * g + g // 2
        Ab = (sb << 7) | mb
        o8 = ob.reshape(n, g // 8, 8).astype(np.uint32)
        w = (o8 << (3 * np.arange(8, dtype=np.uint32))).sum(2, dtype=np.uint32)
        E = np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], 2).reshape(n, 3 * g // 8).astype(np.uint8)
        out[:, ol:ol + g], out[:, ol + g:ol + g + 3 * g // 8] = Ab, E
        ol += g + 3 * g // 8
        hm[:, 2 + p] = (d0 & 0xFF).astype(np.uint8)
        hm[:, 5 + p] = base.astype(np.uint8)
        hl[:, 5 + p] = bbase.astype(np.uint8)
    out[:, om:om + 8], out[:, ol:ol + 8] = hm, hl
    assert om + 8 <= msz and ol + 8 <= msz + lsz
    return out


def unpack_check(out, src, r):
    """unpack every field of the M-01 rows the way the ext does and compare with the records3 bytes"""
    W3, G, L1, msz, lsz = plan(r)
    n = out.shape[0]
    om, ol, cm, cl = sum(W3), msz + sum(L1), 0, msz
    hm, hl = out[:, om + sum(2 * g + g // 2 for g in G):][:, :8], out[:, ol + sum(g + 3 * g // 8 for g in G):][:, :8]
    assert (hm[:, :2] == [ord("M"), ord("1")]).all() and (hl[:, :2] == [ord("M"), ord("1")]).all()
    for p in range(3):
        g = G[p]
        mo, lo = r["msb_piece_off"][3 * p:3 * p + 3], r["lsb_piece_off"][2 * p:2 * p + 2]
        assert (out[:, cm:cm + W3[p]] == src[:, mo[0]:mo[0] + W3[p]]).all()
        assert (out[:, cl:cl + L1[p]] == src[:, lo[0]:lo[0] + L1[p]]).all()
        cm += W3[p]; cl += L1[p]
        A, B = out[:, om:om + g].astype(np.uint32), out[:, om + g:om + 2 * g].astype(np.uint32)
        N = out[:, om + 2 * g:om + 2 * g + g // 2].astype(np.uint32)
        om += 2 * g + g // 2
        nb = np.stack([N & 0xF, N >> 4], 2).reshape(n, g)
        d0 = hm[:, 2 + p].astype(np.int8).astype(np.int32)[:, None]
        base = hm[:, 5 + p].astype(np.uint32)[:, None]
        es = base + ((B >> 7) | ((nb & 3) << 1))
        s3 = ((A & 0x80) << 8) | (es << 7) | (A & 0x7F)
        b3 = (((A & 0x80) ^ 0x80) << 8) | (((es.astype(np.int32) + d0 + (nb >> 2).astype(np.int32)) & 0xFF).astype(np.uint32) << 7) | (B & 0x7F)
        assert (s3.astype(np.uint16) == np.ascontiguousarray(src[:, mo[1]:mo[1] + 2 * g]).view(np.uint16)).all(), "s3"
        assert (b3.astype(np.uint16) == np.ascontiguousarray(src[:, mo[2]:mo[2] + 2 * g]).view(np.uint16)).all(), "b3"
        Ab = out[:, ol:ol + g].astype(np.uint32)
        E = out[:, ol + g:ol + g + 3 * g // 8].astype(np.uint32).reshape(n, g // 8, 3)
        ol += g + 3 * g // 8
        w = E[..., 0] | (E[..., 1] << 8) | (E[..., 2] << 16)
        ob = ((w[..., None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).reshape(n, g)
        eb = hl[:, 5 + p].astype(np.uint32)[:, None] + ob
        bb = ((Ab & 0x80) << 8) | (eb << 7) | (Ab & 0x7F)
        assert (bb.astype(np.uint16) == np.ascontiguousarray(src[:, lo[1]:lo[1] + 2 * g]).view(np.uint16)).all(), "b"


for L in LAYERS:
    r = IX3[str(L)]
    E, rs3 = r["E"], r["recsz"]
    W3, G, L1, msz, lsz = plan(r)
    rsm = msz + lsz
    dst = DST + r["file"]
    done = IXM["layers"].get(str(L), {})
    if done.get("verified") and os.path.exists(dst) and os.path.getsize(dst) == E * rsm:
        continue
    if (time.time() - T0) / 60 > args.minutes:
        print(f"time up before layer {L}", flush=True); break
    if shutil.disk_usage("/").free < 110e9 + E * rsm:
        print(f"stop at layer {L}: disk would fall under 110 GB free", flush=True); break
    fi = os.open(SRC + r["file"], os.O_RDONLY); fcntl.fcntl(fi, F_NOCACHE, 1)
    fo = os.open(dst + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644); fcntl.fcntl(fo, F_NOCACHE, 1)
    t = time.time()
    for e0 in range(0, E, args.chunk):
        n = min(args.chunk, E - e0)
        raw = bytearray(n * rs3)
        mv, got = memoryview(raw), 0
        while got < n * rs3:
            k = os.preadv(fi, [mv[got:]], e0 * rs3 + got)
            assert k > 0
            got += k
        src = np.frombuffer(raw, np.uint8).reshape(n, rs3)
        out = pack(src, r)
        unpack_check(out, src, r)
        b = out.tobytes()
        w = 0
        while w < len(b):
            w += os.write(fo, b[w:])
    os.close(fi); os.fsync(fo); os.close(fo)
    os.replace(dst + ".tmp", dst)
    IXM["layers"][str(L)] = {"file": r["file"], "E": E, "recsz": rsm, "msb_sz": msz, "lsb_sz": lsz, "verified": True,
                             "r3_recsz": rs3, "r3_msb_sz": r["msb_sz"]}
    json.dump(IXM, open(IXP + ".tmp", "w"), indent=1); os.replace(IXP + ".tmp", IXP)
    print(f"layer {L}: {E} x {rsm} B (MSB {msz}, LSB {lsz}; records3 {rs3}) verified, {time.time() - t:.1f} s, free "
          f"{shutil.disk_usage('/').free / 1e9:.0f} GB", flush=True)
print(f"done ({time.time() - T0:.0f} s); layers verified: {len(IXM['layers'])} / 48", flush=True)
