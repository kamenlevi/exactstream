"""R-44 (b) POM converter (CPU / SSD): records3/ -> records3p/, plane-separated records for precision on miss (MoonShots'
M-03): a miss whose uses are all at gate rank >= R2 reads only segment 1 (the top two code planes) and sits in its pool slot
as a 3-bit slot with plane 3 = 0 and the 2-plane bias, which is exactly the Eval suite's tier-2 math (evalsuite/model.py
tier_weights n = 2: codes q4 & 12 at scale s4, bias bf16(b4 + s4 * mean(q4 & 3)); s3 * 2m2 == s4 * 4m2 exactly). A later
use at rank < R2 reads segment 2 (plane 3 + b3) and upgrades the slot in place; a miss with a rank < R2 use reads both.
Per expert (16 KB blocks): seg1 = q2 planes x 3 (2-bit, 4 per byte, little-endian) | M-01 MSB fields with b3 replaced by
bias2 (A = s3 sign | s3 mantissa; B = bias2 mantissa | s3 exponent offset bit 0 << 7; N nibbles = s3 offset bits 1-2 |
(bias2 exp - s3 exp - d2) << 2; bias2's sign is the opposite of s3's) | 8-byte header ('P', '2', d2 x 3, base x 3);
seg2 = p3 planes x 3 (1 bit per weight, little-endian) | per projection B3 = b3 mantissa | dd bit 0 << 7 (G B) and dd bit 1
(G/8 B, little-endian) with dd = b3 exp - s3 exp - d3 | header ('P', '3', d3 x 3). No LSB part: full misses and the LSB
sidecars read records3m (the engine keeps both sets open). For the 125B: seg1 87 blocks, seg2 43 (records3m MSB 125).
Every expert is checked before it is written: full (seg1 + seg2) == records3's MSB bytes, 2-plane (seg1) == the suite's
tier-2 codes and bias computed from records3 (b4 and the LSB plane from its LSB part).
Resumable per layer; stops before the disk falls under 110 GB free, or after --minutes.
usage: [BigMoE venv] taskpolicy -b python m03_convert.py [--layers 0-47] [--minutes 12]"""
import argparse, fcntl, json, os, shutil, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0-47")
ap.add_argument("--minutes", type=float, default=12.0)
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--workers", type=int, default=4)            # layers in parallel (CPU-only jobs: <= 4 cores)
args = ap.parse_args()
a, _, b = args.layers.partition("-")
LAYERS = list(range(int(a), int(b or a) + 1))
SRC = os.path.expanduser("~/BigMoE/models/bare/records3/")
DST = os.path.expanduser("~/BigMoE/models/bare/records3p/")
os.makedirs(DST, exist_ok=True)
IX3 = json.load(open(SRC + "index.json"))["layers"]
assert all(r.get("layout", "row-major") == "row-major" for r in IX3.values()), "records3 is converted (docs/PLAN_records3_convert.md): this script reads row-major records only"
IXP = DST + "index.json"
IXO = json.load(open(IXP)) if os.path.exists(IXP) else {"align": 16384, "format": "records3p", "src": "records3",
                                                           "layers": {}}
ALIGN, F_NOCACHE = 16384, 48
T0 = time.time()
up = lambda n: -(-n // ALIGN) * ALIGN


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


def unpack3(b):
    """[n, 3k] uint8 MLX 3-bit stream -> [n, 8k] codes"""
    t = b.reshape(b.shape[0], -1, 3).astype(np.uint32)
    w = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
    return ((w[..., None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).reshape(b.shape[0], -1).astype(np.uint8)


def pack3(q):
    w = (q.reshape(q.shape[0], -1, 8).astype(np.uint32) << (3 * np.arange(8, dtype=np.uint32))).sum(2, dtype=np.uint32)
    return np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], 2).reshape(q.shape[0], -1).astype(np.uint8)


def pack2(q):
    w = q.reshape(q.shape[0], -1, 4).astype(np.uint8)
    return (w[..., 0] | (w[..., 1] << 2) | (w[..., 2] << 4) | (w[..., 3] << 6)).astype(np.uint8)


def unpack2(b):
    return ((b[..., None] >> (2 * np.arange(4, dtype=np.uint8))) & 3).reshape(b.shape[0], -1)


def plan(r):
    W3 = [r["msb_piece_size"][3 * p] for p in range(3)]
    G = [r["msb_piece_size"][3 * p + 1] // 2 for p in range(3)]
    NW = [w * 8 // 3 for w in W3]                      # weights per projection
    s1 = sum(n // 4 for n in NW) + sum(2 * g + g // 2 for g in G) + 8
    s2 = sum(n // 8 for n in NW) + sum(g + g // 8 for g in G) + 8
    return W3, G, NW, up(s1), up(s2)


def fields(u):
    return (u >> 15).astype(np.uint8), ((u >> 7) & 0xFF).astype(np.int32), (u & 0x7F).astype(np.uint8)


def convert(src, r):
    """[n, recsz3] records3 rows -> [n, seg1 + seg2] records3p rows (checked)"""
    W3, G, NW, S1, S2 = plan(r)
    n = src.shape[0]
    out = np.zeros((n, S1 + S2), np.uint8)
    o1, o2 = sum(n_ // 4 for n_ in NW), S1 + sum(n_ // 8 for n_ in NW)   # field offsets in seg1 / the record
    c1, c2 = 0, S1
    h1, h2 = np.zeros((n, 8), np.uint8), np.zeros((n, 8), np.uint8)
    h1[:, :2], h2[:, :2] = (ord("P"), ord("2")), (ord("P"), ord("3"))
    keep = []
    for p in range(3):
        g, nw = G[p], NW[p]
        mo, lo = r["msb_piece_off"][3 * p:3 * p + 3], r["lsb_piece_off"][2 * p:2 * p + 2]
        q3 = unpack3(src[:, mo[0]:mo[0] + W3[p]])
        q2, p3 = q3 >> 1, q3 & 1
        out[:, c1:c1 + nw // 4] = pack2(q2); c1 += nw // 4
        out[:, c2:c2 + nw // 8] = np.packbits(p3, axis=1, bitorder="little"); c2 += nw // 8
        s3 = np.ascontiguousarray(src[:, mo[1]:mo[1] + 2 * g]).view(np.uint16)
        b3 = np.ascontiguousarray(src[:, mo[2]:mo[2] + 2 * g]).view(np.uint16)
        b4 = np.ascontiguousarray(src[:, lo[1]:lo[1] + 2 * g]).view(np.uint16)
        l1 = np.unpackbits(src[:, lo[0]:lo[0] + nw // 8], axis=1, bitorder="little")
        # the suite's tier-2 bias: bf16(b4 + s4 * mean(q4 & 3)), q4 & 3 = 2 p3 + l, s4 = s3 / 2 (exact), float32 math
        s4 = np.where((s3 & 0x7FFF) == 0, s3, s3 - 0x80).astype(np.uint16)
        lsum = (2 * p3.astype(np.int32) + l1).reshape(n, g, 64).sum(2)
        bias2 = bf16(f32(b4) + f32(s4) * (lsum.astype(np.float32) / np.float32(64)))
        ss, es, ms = fields(s3)
        s2_, e2, m2 = fields(bias2)
        sb3, eb3, mb3 = fields(b3)
        assert (s2_ != ss).all() and (sb3 != ss).all(), "bias2 / b3 sign equals s3 sign somewhere"
        base = es.min(1); os_ = es - base[:, None]
        assert (os_ <= 7).all() and (base >= 0).all(), "s3 exponent span > 7"
        d2 = (e2 - es).min(1); dd2 = e2 - es - d2[:, None]
        d3 = (eb3 - es).min(1); dd3 = eb3 - es - d3[:, None]
        assert (dd2 <= 3).all() and (dd3 <= 3).all() and (np.abs(d2) < 100).all() and (np.abs(d3) < 100).all(), "offsets"
        A = (ss << 7) | ms
        B = m2 | ((os_ & 1).astype(np.uint8) << 7)
        nib = (((os_ >> 1) & 3) | (dd2 << 2)).astype(np.uint8)
        out[:, o1:o1 + g], out[:, o1 + g:o1 + 2 * g] = A, B
        out[:, o1 + 2 * g:o1 + 2 * g + g // 2] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        o1 += 2 * g + g // 2
        out[:, o2:o2 + g] = mb3 | ((dd3 & 1).astype(np.uint8) << 7)
        out[:, o2 + g:o2 + g + g // 8] = np.packbits((dd3 >> 1).astype(np.uint8), axis=1, bitorder="little")
        o2 += g + g // 8
        h1[:, 2 + p], h1[:, 5 + p] = (d2 & 0xFF).astype(np.uint8), base.astype(np.uint8)
        h2[:, 2 + p] = (d3 & 0xFF).astype(np.uint8)
        keep.append((q3, s3, b3, bias2))
    out[:, o1:o1 + 8], out[:, o2:o2 + 8] = h1, h2
    assert o1 + 8 <= S1 and o2 + 8 <= S1 + S2
    for p in range(3):                                        # the 2-plane slot (codes 2 m2 + 0, s3, bias2) and the full
        check_proj(out, p, *keep[p], (W3, G, NW, S1, S2), o1, o2)   # slot (q3, s3, b3), from the packed bytes + headers
    return out


def check_proj(out, p, q3, s3, b3, bias2, pl, h1o, h2o):
    """unpack projection p of the M-03 rows the way the ext does (fields + the written headers at h1o / h2o); compare
    with records3 (full slot) and the suite's tier-2 math (2-plane slot)"""
    W3, G, NW, S1, S2 = pl
    n = out.shape[0]
    c1 = sum(NW[i] // 4 for i in range(p)); c2 = S1 + sum(NW[i] // 8 for i in range(p))
    o1 = sum(NW[i] // 4 for i in range(3)) + sum(2 * G[i] + G[i] // 2 for i in range(p))
    o2 = S1 + sum(NW[i] // 8 for i in range(3)) + sum(G[i] + G[i] // 8 for i in range(p))
    g, nw = G[p], NW[p]
    q2 = unpack2(out[:, c1:c1 + nw // 4])
    p3 = np.unpackbits(out[:, c2:c2 + nw // 8], axis=1, bitorder="little")
    assert ((2 * q2 + p3) == q3).all(), "full codes"           # (the ext's 3-bit packing of 2 m2 [+ p3]: test_records3p)
    A, B = out[:, o1:o1 + g].astype(np.uint32), out[:, o1 + g:o1 + 2 * g].astype(np.uint32)
    N = out[:, o1 + 2 * g:o1 + 2 * g + g // 2].astype(np.uint32)
    nb = np.stack([N & 0xF, N >> 4], 2).reshape(n, g)
    assert (out[:, h1o:h1o + 2] == [ord("P"), ord("2")]).all() and (out[:, h2o:h2o + 2] == [ord("P"), ord("3")]).all()
    base = out[:, h1o + 5 + p].astype(np.uint32)[:, None]
    d2 = out[:, h1o + 2 + p].astype(np.int8).astype(np.int32)[:, None]
    d3 = out[:, h2o + 2 + p].astype(np.int8).astype(np.int32)[:, None]
    es = base + ((B >> 7) | ((nb & 3) << 1))
    s3u = ((A & 0x80) << 8) | (es << 7) | (A & 0x7F)
    b2u = (((A & 0x80) ^ 0x80) << 8) | (((es.astype(np.int32) + d2 + (nb >> 2).astype(np.int32)) & 0xFF).astype(np.uint32) << 7) | (B & 0x7F)
    assert (s3u.astype(np.uint16) == s3).all(), "s3"
    assert (b2u.astype(np.uint16) == bias2).all(), "bias2"
    B3 = out[:, o2:o2 + g].astype(np.uint32)
    hb = np.unpackbits(out[:, o2 + g:o2 + g + g // 8], axis=1, bitorder="little").astype(np.uint32)
    dd = (B3 >> 7) | (hb << 1)
    b3u = (((A & 0x80) ^ 0x80) << 8) | (((es.astype(np.int32) + d3 + dd.astype(np.int32)) & 0xFF).astype(np.uint32) << 7) | (B3 & 0x7F)
    assert (b3u.astype(np.uint16) == b3).all(), "b3"


def layer(L):
    """convert + check one layer into DST/L??.bin (via .tmp); returns its index entry"""
    r = IX3[str(L)]
    E, rs3 = r["E"], r["recsz"]
    W3, G, NW, S1, S2 = plan(r)
    rsp = S1 + S2
    dst = DST + r["file"]
    fi = os.open(SRC + r["file"], os.O_RDONLY); fcntl.fcntl(fi, F_NOCACHE, 1)
    fo = os.open(dst + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644); fcntl.fcntl(fo, F_NOCACHE, 1)
    t = time.time()

    def rd(fd, n, off):
        buf = bytearray(n); mv, got = memoryview(buf), 0
        while got < n:
            k = os.preadv(fd, [mv[got:]], off + got); assert k > 0; got += k
        return np.frombuffer(buf, np.uint8)

    for e0 in range(0, E, args.chunk):
        n = min(args.chunk, E - e0)
        src = rd(fi, n * rs3, e0 * rs3).reshape(n, rs3)
        out = convert(src, r)
        bts = out.tobytes(); w = 0
        while w < len(bts):
            w += os.write(fo, bts[w:])
    os.close(fi); os.fsync(fo); os.close(fo)
    os.replace(dst + ".tmp", dst)
    print(f"layer {L}: {E} x {rsp} B (seg1 {S1}, seg2 {S2}) verified, {time.time() - t:.1f} s, free "
          f"{shutil.disk_usage('/').free / 1e9:.0f} GB", flush=True)
    return L, {"file": r["file"], "E": E, "recsz": rsp, "seg1": S1, "seg2": S2, "verified": True, "r3_recsz": rs3,
               "r3_msb_sz": r["msb_sz"]}


if __name__ == "__main__":
    import multiprocessing as mp
    todo = []
    for L in LAYERS:
        r = IX3[str(L)]
        W3, G, NW, S1, S2 = plan(r)
        dst = DST + r["file"]
        done = IXO["layers"].get(str(L), {})
        if not (done.get("verified") and os.path.exists(dst) and os.path.getsize(dst) == r["E"] * (S1 + S2)):
            todo.append((L, r["E"] * (S1 + S2)))
    with mp.get_context("fork").Pool(args.workers) as pool:
        running = []
        while todo or running:
            while todo and len(running) < args.workers:
                L, need = todo[0]
                if (time.time() - T0) / 60 > args.minutes:
                    print(f"time up before layer {L}", flush=True); todo = []; break
                if shutil.disk_usage("/").free < 110e9 + need * (1 + len(running)):
                    print(f"stop at layer {L}: disk would fall under 110 GB free", flush=True); todo = []; break
                running.append(pool.apply_async(layer, (L,))); todo.pop(0)
            if running:
                L, ent = running.pop(0).get()           # in submission order; the index is written by this process only
                IXO["layers"][str(L)] = ent
                json.dump(IXO, open(IXP + ".tmp", "w"), indent=1); os.replace(IXP + ".tmp", IXP)
    print(f"done ({time.time() - T0:.0f} s); layers verified: {len(IXO['layers'])} / 48", flush=True)
