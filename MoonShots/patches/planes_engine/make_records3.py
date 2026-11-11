"""3-bit tier engine converter (BigMoE DESIGN_3bit.md, "Record file"): records/L{L}.bin -> records3/L{L:02d}.bin, plus the
built-in verifier. Each 4-bit code q = 2m + l (m = q >> 1, l = q & 1; groups of 64 along K; scale s, bias b):
  MSB part (offset 0, padded to 16 KB), per projection (gate, up, down): w3 = MLX bits=3 packing of m (little-endian
    bitstream, value i at bits 3i..3i+2, row-major [out, K]), s3 = 2s (bf16, exact), b3 = bf16(fp32(b) + fp32(s) * mean(l))
    per group (the values of planes125._tier_proj(n=3)).
  LSB part (offset MSB_SZ, padded to 16 KB), per projection: l1 (bit j of byte i = l of weight 8i + j), b (original bias).
Verifier, every expert of every converted layer, from the bytes on disk (F_NOCACHE): the rebuilt 4-bit words == the
originals bitwise; s3 / 2 == s bitwise (fp32 halving and BigMoE's exponent-decrement halve_bf16); LSB b == b; b3
recomputed from the rebuilt words == b3; padding zero. Per layer: an unpack/packbits reference for 4 random experts ==
their records byte for byte; MLX reads w3 as m (dequantize with unit scales); gather_qmm(bits=3) on the MSB part vs
planes125._tier_proj(n=3) on the 4-bit pieces, 10 random experts (allclose; they differ only by kernel).
Once per run: mx.quantize(bits=3) lays out codes exactly as w3.
usage: make_records3.py --layers 0,17|all --out DIR [--src DIR] [--chunk 32] [--min-free-gb 110] [--verify-only]
       [--device cpu|gpu] [--delete-src]       (CPU + SSD; the MLX checks run on --device, default cpu)
--delete-src removes records/L{L}.bin after its records3 layer passed every check (the full-conversion policy)."""
import argparse, fcntl, json, os, shutil, sys, time
import numpy as np

ALIGN = 16384
G = 64
U32, U16, U8 = np.uint32, np.uint16, np.uint8
SHAPES = ((640, 2560), (640, 2560), (2560, 640))          # [out, K] of gate, up, down (pieces w, s, b each): today's default
pad = lambda n: (n + ALIGN - 1) // ALIGN * ALIGN


def shapes_from_config(path):
    """[out, K] of gate, up, down from a model config.json (Qwen 4 day-0 kit, Sun 2026-11-08): (I, H), (I, H), (H, I) with
    I = moe_intermediate_size, H = hidden_size (text_config if nested); K must be a multiple of the group size 64"""
    c = json.load(open(path))
    t = c.get("text_config", c)
    I, H = int(t["moe_intermediate_size"]), int(t["hidden_size"])
    assert I % G == 0 and H % G == 0 and H % 8 == 0 and I % 8 == 0, (I, H)
    return ((I, H), (I, H), (H, I))


def layout(shapes=SHAPES):
    """records3 piece offsets (within the record) and sizes: MSB part (w3, s3, b3) x 3 from 0, LSB part (l1, b) x 3 from
    msb_sz."""
    mo, ms, lo, ls, o = [], [], [], [], 0
    for out, K in shapes:
        for n in (out * K * 3 // 8, out * K // G * 2, out * K // G * 2):
            mo.append(o); ms.append(n); o += n
    msb = o = pad(o)
    for out, K in shapes:
        for n in (out * K // 8, out * K // G * 2):
            lo.append(o); ls.append(n); o += n
    return dict(recsz=pad(o), msb_sz=msb, lsb_sz=pad(o) - msb, msb_piece_off=mo, msb_piece_size=ms,
                lsb_piece_off=lo, lsb_piece_size=ls)


def bf16_to_f32(h):
    return (h.astype(U32) << 16).view(np.float32)


def f32_to_bf16(f):                                          # round to nearest even, as MLX's astype(bfloat16)
    u = np.ascontiguousarray(f, np.float32).view(U32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(U16)


def split_codes(w):
    """4-bit words [..., n] -> (m24 [..., n] u32: m = q >> 1 of the word's 8 weights as 24 bits, value j at bits 3j,
    l8 [..., n] u8: bit j = q & 1 of weight j). The low 3 bytes of m24 are the word's 3 bytes of the bits=3 stream."""
    x = w & U32(0x11111111)
    x = (x | (x >> 3)) & U32(0x03030303)
    x = (x | (x >> 6)) & U32(0x000F000F)
    x = (x | (x >> 12)) & U32(0xFF)
    y = (w >> 1) & U32(0x77777777)
    y = (y & U32(0x07070707)) | ((y & U32(0x70707070)) >> 1)       # per byte: m_even bits 0-2, m_odd bits 3-5
    y = (y & U32(0x003F003F)) | ((y & U32(0x3F003F00)) >> 2)       # per 16-bit half: 12 bits
    y = (y & U32(0x00000FFF)) | ((y & U32(0x0FFF0000)) >> 4)       # 24 bits
    return y, x.astype(U8)


def merge_codes(w3b, l8):
    """Inverse of split_codes (the verifier's rebuild, as BigMoE's spread8): w3 bytes [..., n, 3] + l8 [..., n] -> words."""
    y = w3b[..., 0].astype(U32) | (w3b[..., 1].astype(U32) << 8) | (w3b[..., 2].astype(U32) << 16)
    y = (y & U32(0xFFF)) | ((y & U32(0xFFF000)) << 4)
    y = (y & U32(0x003F003F)) | ((y & U32(0x0FC00FC0)) << 2)
    y = (y & U32(0x07070707)) | ((y & U32(0x38383838)) << 1)
    x = l8.astype(U32)
    x = (x | (x << 12)) & U32(0x000F000F)
    x = (x | (x << 6)) & U32(0x03030303)
    x = (x | (x << 3)) & U32(0x11111111)
    return (y << 1) | x


def view(buf, n, rec, off, size, dt, inner=None):
    """the piece at `off` of n consecutive records in buf, as [n, size / itemsize] (or [n, size / 3, 3] bytes: inner=3)."""
    it = np.dtype(dt).itemsize
    if inner:
        return np.ndarray((n, size // inner, inner), dt, buffer=buf, offset=off, strides=(rec, inner, 1))
    return np.ndarray((n, size // it), dt, buffer=buf, offset=off, strides=(rec, it))


def convert_chunk(src, n, si, lay):
    """n source records (buffer of n * si recsz bytes) -> n records3 (np.uint8 [n * recsz])."""
    R, rs, po, ps = lay["recsz"], si["recsz"], si["piece_off"], si["piece_size"]
    mo, ms, lo, ls = lay["msb_piece_off"], lay["msb_piece_size"], lay["lsb_piece_off"], lay["lsb_piece_size"]
    out = aligned(n * R)
    out[:] = 0
    for p in range(3):
        w = view(src, n, rs, po[3 * p], ps[3 * p], U32)
        s = view(src, n, rs, po[3 * p + 1], ps[3 * p + 1], U16)
        b = view(src, n, rs, po[3 * p + 2], ps[3 * p + 2], U16)
        m24, l8 = split_codes(w)
        view(out, n, R, mo[3 * p], ms[3 * p], U8, inner=3)[...] = m24.view(U8).reshape(*m24.shape, 4)[..., :3]
        sf = bf16_to_f32(s)
        view(out, n, R, mo[3 * p + 1], ms[3 * p + 1], U16)[...] = f32_to_bf16(sf * np.float32(2))
        off = np.bitwise_count(l8.view(np.uint64)).astype(np.float32) / np.float32(G)   # l8: 8 bytes = one group
        view(out, n, R, mo[3 * p + 2], ms[3 * p + 2], U16)[...] = f32_to_bf16(bf16_to_f32(b) + sf * off)
        view(out, n, R, lo[2 * p], ls[2 * p], U8)[...] = l8
        view(out, n, R, lo[2 * p + 1], ls[2 * p + 1], U16)[...] = b
    return out


def verify_chunk(src, r3, n, si, lay):
    """mismatch counts over n experts: source records vs records3 bytes."""
    R, rs, po, ps = lay["recsz"], si["recsz"], si["piece_off"], si["piece_size"]
    mo, ms, lo, ls = lay["msb_piece_off"], lay["msb_piece_size"], lay["lsb_piece_off"], lay["lsb_piece_size"]
    bad = dict(w4=0, s3=0, s3_halve=0, b=0, b3=0, pad=0, nonfinite=0)
    for p in range(3):
        w = view(src, n, rs, po[3 * p], ps[3 * p], U32)
        s = view(src, n, rs, po[3 * p + 1], ps[3 * p + 1], U16)
        b = view(src, n, rs, po[3 * p + 2], ps[3 * p + 2], U16)
        w3 = view(r3, n, R, mo[3 * p], ms[3 * p], U8, inner=3)
        s3 = view(r3, n, R, mo[3 * p + 1], ms[3 * p + 1], U16)
        b3 = view(r3, n, R, mo[3 * p + 2], ms[3 * p + 2], U16)
        l8 = view(r3, n, R, lo[2 * p], ls[2 * p], U8)
        bl = view(r3, n, R, lo[2 * p + 1], ls[2 * p + 1], U16)
        q4 = merge_codes(w3, l8)
        bad["w4"] += int(np.count_nonzero(q4 != w))
        bad["s3"] += int(np.count_nonzero(f32_to_bf16(bf16_to_f32(s3) * np.float32(0.5)) != s))
        bad["s3_halve"] += int(np.count_nonzero(np.where((s3 & U16(0x7FFF)) == 0, s3, s3 - U16(0x80)) != s))
        bad["b"] += int(np.count_nonzero(bl != b))
        lsum = np.bitwise_count(q4 & U32(0x11111111)).reshape(n, -1, G // 8).sum(-1, dtype=U32)
        ref = f32_to_bf16(bf16_to_f32(b) + bf16_to_f32(s) * (lsum.astype(np.float32) / np.float32(G)))
        bad["b3"] += int(np.count_nonzero(ref != b3))
        for h in (s, b):
            bad["nonfinite"] += int(np.count_nonzero((h & U16(0x7F80)) == U16(0x7F80)))
    end_m, end_l = mo[-1] + ms[-1], lo[-1] + ls[-1]
    recs = np.ndarray((n, R), U8, buffer=r3, strides=(R, 1))
    bad["pad"] = int(np.count_nonzero(recs[:, end_m:lay["msb_sz"]])) + int(np.count_nonzero(recs[:, end_l:]))
    return bad


def reference_record(rec, si, lay):
    """one records3 record by plain unpack / packbits (independent of the bit tricks; BigMoE's test_records3 split)."""
    po, ps = si["piece_off"], si["piece_size"]
    out = np.zeros(lay["recsz"], U8)
    for p in range(3):
        w = np.frombuffer(rec, U32, ps[3 * p] // 4, po[3 * p])
        s = np.frombuffer(rec, U16, ps[3 * p + 1] // 2, po[3 * p + 1])
        b = np.frombuffer(rec, U16, ps[3 * p + 2] // 2, po[3 * p + 2])
        q = ((w[:, None] >> (4 * np.arange(8, dtype=U32))) & 15).reshape(-1)
        m, l = q >> 1, q & 1
        w3 = np.packbits(((m[:, None] >> np.arange(3, dtype=U32)) & 1).astype(U8).reshape(-1), bitorder="little")
        l1 = np.packbits(l.astype(U8), bitorder="little")
        sf = bf16_to_f32(s)
        b3 = f32_to_bf16(bf16_to_f32(b) + sf * (l.reshape(-1, G).sum(1).astype(np.float32) / G))
        for k, a in enumerate((w3, f32_to_bf16(2 * sf), b3)):
            o = lay["msb_piece_off"][3 * p + k]; out[o:o + a.nbytes] = a.view(U8)
        for k, a in enumerate((l1, b)):
            o = lay["lsb_piece_off"][2 * p + k]; out[o:o + a.nbytes] = a.view(U8)
    return out


def aligned(nbytes):
    raw = np.empty(nbytes + ALIGN, U8)
    o = (-raw.ctypes.data) % ALIGN
    return raw[o:o + nbytes]


def pread_into(fd, buf, off):
    mv, done = memoryview(buf), 0
    while done < len(mv):
        k = os.preadv(fd, [mv[done:]], off + done)
        if k <= 0:
            raise IOError(f"short read at {off + done}")
        done += k


def pwrite_all(fd, buf, off):
    mv, done = memoryview(buf), 0
    while done < len(mv):
        done += os.pwrite(fd, mv[done:], off + done)


def open_nocache(path, flags, mode=0o644):
    fd = os.open(path, flags, mode)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    return fd


def mlx_checks(L, src_path, dst_path, si, lay, rng, mx, P, shapes=SHAPES):
    """per-layer MLX checks: MLX reads w3 as m; gather_qmm(bits=3) on the MSB part vs planes125._tier_proj(n=3)."""
    E, rs, po, ps, R = si["E"], si["recsz"], si["piece_off"], si["piece_size"], lay["recsz"]
    ids = np.sort(rng.choice(E, 10, replace=False))
    fs, fd = open_nocache(src_path, os.O_RDONLY), open_nocache(dst_path, os.O_RDONLY)
    srcs, dsts = [], []
    for e in ids:
        a, c = aligned(rs), aligned(R)
        pread_into(fs, a, int(e) * rs); pread_into(fd, c, int(e) * R)
        srcs.append(a); dsts.append(c)
    os.close(fs); os.close(fd)
    res = {"experts": [int(e) for e in ids]}
    for p, (out, K) in enumerate(shapes):
        name = ("gate", "up", "down")[p]
        pc = lambda bufs, off, size, dt, shape: mx.array(np.stack([np.frombuffer(bb, dt, size // np.dtype(dt).itemsize, off)
                                                                     for bb in bufs]).reshape(len(bufs), *shape))
        w4 = pc(srcs, po[3 * p], ps[3 * p], U32, (out, K // 8))
        s4 = pc(srcs, po[3 * p + 1], ps[3 * p + 1], U16, (out, K // G)).view(mx.bfloat16)
        b4 = pc(srcs, po[3 * p + 2], ps[3 * p + 2], U16, (out, K // G)).view(mx.bfloat16)
        mo, ms = lay["msb_piece_off"][3 * p:3 * p + 3], lay["msb_piece_size"][3 * p:3 * p + 3]
        w3 = pc(dsts, mo[0], ms[0], U32, (out, K * 3 // 32))
        s3 = pc(dsts, mo[1], ms[1], U16, (out, K // G)).view(mx.bfloat16)
        b3 = pc(dsts, mo[2], ms[2], U16, (out, K // G)).view(mx.bfloat16)
        # (1) MLX's own bits=3 unpacking of w3 gives m = q >> 1 (unit scales, zero biases): exact small integers
        one, zero = mx.ones(s3.shape[1:], mx.float32), mx.zeros(s3.shape[1:], mx.float32)
        okm = True
        for k in range(2):
            mm = mx.dequantize(w3[k], one, zero, group_size=G, bits=3)
            q = (w4[k][..., None] >> mx.arange(0, 32, 4).astype(mx.uint32)) & 15
            okm &= bool(mx.array_equal(mm, (q.reshape(out, K) >> 1).astype(mx.float32)).item())
        # (2) gather_qmm(bits=3) on the MSB part vs planes125's masked-4-bit tier (same values, different kernel)
        x = (mx.random.normal((10, 1, K), key=mx.random.key(1000 * L + p)) * 0.5).astype(mx.bfloat16)
        idx = mx.arange(10).astype(mx.uint32)
        y3 = mx.gather_qmm(x, w3, s3, b3, rhs_indices=idx, transpose=True, group_size=G, bits=3)
        lin = type("Lin", (), dict(weight=w4, scales=s4, biases=b4, group_size=G))()
        yr = P._tier_proj(lin, x, mx.arange(10), idx, 3)
        # (3) the same bias as planes125 computes it on the MLX device (bitwise)
        wm, lsum = P._MASK[3](w4)
        off = lsum.reshape(*s4.shape, G // 8).sum(-1).astype(mx.float32) / G
        bias = (b4.astype(mx.float32) + s4.astype(mx.float32) * off).astype(mx.bfloat16)
        y3f, yrf = y3.astype(mx.float32), yr.astype(mx.float32)
        d = mx.abs(y3f - yrf)
        res[name] = dict(unpack_eq_m=okm, b3_eq_mlx=bool(mx.array_equal(bias.view(mx.uint16), b3.view(mx.uint16)).item()),
                         s3_eq_2s=bool(mx.array_equal((s4.astype(mx.float32) * 2).astype(mx.bfloat16).view(mx.uint16),
                                                      s3.view(mx.uint16)).item()),
                         max_rel=float((d.max() / mx.abs(yrf).max()).item()), frac_equal=float((d == 0).mean().item()),
                         allclose=bool(mx.allclose(y3f, yrf, rtol=1e-2, atol=1e-2 * mx.abs(yrf).max().item()).item()))
    return res


def quantize_layout_check(mx):
    """mx.quantize(bits=3) of W = m - 7 (every group holds -7 and 0, so scale 1, bias -7, codes m) == our packing of m."""
    rng = np.random.default_rng(5)
    m = rng.integers(0, 8, (64, 512)).astype(U32)
    m[:, ::G] = 0; m[:, 1::G] = 7
    W = mx.array(m.astype(np.float32) - 7).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(W, group_size=G, bits=3)
    q4 = ((m << 1).reshape(64, -1, 8) << (4 * np.arange(8, dtype=U32))).sum(-1, dtype=U32)   # 4-bit words of q = 2m
    m24, _ = split_codes(q4)
    ours = np.ascontiguousarray(m24.view(U8).reshape(*m24.shape, 4)[..., :3]).reshape(64, -1)
    sc_ok = bool(mx.all(sc == 1).item()) and bool(mx.all(bi == -7).item())
    return sc_ok and np.array_equal(np.array(wq).view(U8).reshape(64, -1), ours), sc_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.expanduser("~/BigMoE/models/bare/records"))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--min-free-gb", type=float, default=110)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--no-mlx", action="store_true")
    ap.add_argument("--delete-src", action="store_true")
    ap.add_argument("--config", default=os.path.expanduser("~/BigMoE/models/bare/config.json"),
                    help="model config.json the expert shapes come from (default: today's model)")
    a = ap.parse_args()
    SH = shapes_from_config(a.config)
    sidx = json.load(open(os.path.join(a.src, "index.json")))
    layers = sorted(int(k) for k in sidx["layers"]) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    os.makedirs(a.out, exist_ok=True)
    ipath = os.path.join(a.out, "index.json")
    oidx = json.load(open(ipath)) if os.path.exists(ipath) else {"align": ALIGN, "format": "records3", "layers": {}}
    lay = layout(SH)
    mx = P = None
    if not a.no_mlx:
        import mlx.core as mx
        mx.set_default_device(mx.cpu if a.device == "cpu" else mx.gpu)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planes125"))
        import planes125 as P
        ok, sc_ok = quantize_layout_check(mx)
        print(f"mx.quantize(bits=3) layout == w3 packing: {ok} (scale 1 / bias -7 as expected: {sc_ok})", flush=True)
    rng = np.random.default_rng(20261020)
    all_ok = True
    for L in layers:
        si = sidx["layers"][str(L)]
        E, rs = si["E"], si["recsz"]
        assert si["piece_size"] == [s for out, K in SH for s in (out * K // 2, out * K // G * 2, out * K // G * 2)], (L, SH)
        src_path, dst_path = os.path.join(a.src, si["file"]), os.path.join(a.out, f"L{L:02d}.bin")
        t0 = time.time()
        if not a.verify_only:
            free = shutil.disk_usage(a.out).free / 1e9
            need = E * lay["recsz"] / 1e9
            if free - need < a.min_free_gb:
                print(f"STOP layer {L}: {free:.1f} GB free, a layer needs {need:.2f} GB, floor {a.min_free_gb} GB", flush=True)
                all_ok = False
                break
            fs, fo = open_nocache(src_path, os.O_RDONLY), open_nocache(dst_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            buf = aligned(a.chunk * rs)
            for e0 in range(0, E, a.chunk):
                n = min(a.chunk, E - e0)
                pread_into(fs, buf[:n * rs], e0 * rs)
                pwrite_all(fo, convert_chunk(buf, n, si, lay), e0 * lay["recsz"])
            os.fsync(fo); os.close(fo); os.close(fs)
        t1 = time.time()
        assert os.path.getsize(dst_path) == E * lay["recsz"], (dst_path, os.path.getsize(dst_path))
        fs, fd = open_nocache(src_path, os.O_RDONLY), open_nocache(dst_path, os.O_RDONLY)
        bs, bd = aligned(a.chunk * rs), aligned(a.chunk * lay["recsz"])
        bad = dict(w4=0, s3=0, s3_halve=0, b=0, b3=0, pad=0, nonfinite=0)
        for e0 in range(0, E, a.chunk):
            n = min(a.chunk, E - e0)
            pread_into(fs, bs[:n * rs], e0 * rs); pread_into(fd, bd[:n * lay["recsz"]], e0 * lay["recsz"])
            for k, v in verify_chunk(bs, bd, n, si, lay).items():
                bad[k] += v
        ref_ok = True
        for e in rng.choice(E, 4, replace=False):
            pread_into(fs, bs[:rs], int(e) * rs); pread_into(fd, bd[:lay["recsz"]], int(e) * lay["recsz"])
            ref_ok &= np.array_equal(reference_record(bs[:rs], si, lay), bd[:lay["recsz"]])
        os.close(fs); os.close(fd)
        t2 = time.time()
        res = mlx_checks(L, src_path, dst_path, si, lay, rng, mx, P, shapes=SH) if mx is not None else {}
        t3 = time.time()
        ok = (all(v == 0 for k, v in bad.items() if k != "s3_halve") and ref_ok
              and all(r["unpack_eq_m"] and r["b3_eq_mlx"] and r["s3_eq_2s"] and r["allclose"] for k, r in res.items() if k != "experts"))
        all_ok &= ok
        print(f"L{L:02d}: {'OK' if ok else 'FAIL'} convert {t1 - t0:.1f}s verify {t2 - t1:.1f}s mlx {t3 - t2:.1f}s | mismatches {bad}"
              f" | reference records (4 experts) equal {ref_ok}", flush=True)
        if bad["s3_halve"]:
            print(f"  NOTE L{L:02d}: {bad['s3_halve']} scale groups where BigMoE's halve_bf16(s3) != s (subnormal s?)", flush=True)
        for k, r in res.items():
            print(f"  {k}: {r}", flush=True)
        oidx["layers"][str(L)] = dict(file=f"L{L:02d}.bin", E=E, src_recsz=rs, verified=bool(ok), mismatches=bad,
                                      halve_bf16_ok=bad["s3_halve"] == 0, **lay)
        json.dump(oidx, open(ipath, "w"), indent=1)
        if a.delete_src and ok and bad["s3_halve"] == 0 and not a.verify_only:
            os.remove(src_path)
            print(f"  removed {src_path}", flush=True)
    print("ALL OK" if all_ok else "FAILURES", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
