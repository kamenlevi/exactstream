"""W24-sp step (1), NewIdeas GO Mon 03:32: CPU audit of the exact scale / bias pack over ALL 24,576 experts (48 x 512).
Reads ONLY the scale / bias pieces of BigMoE's records3 (MSB part: s3 + b3 per projection, LSB part: b per projection = 9
pieces of 25,600 bf16 per expert, 450 KB of the 2.93 MB record; F_NOCACHE preads, one thread). Per piece: the bf16 exponent
span over its nonzero values, zeros / subnormals / inf-nan, sign use, and a pack + unpack round trip of the planar formats
  12-bit: plane A = sign + 7 mantissa bits (one byte), plane B = 4-bit exponent offset from a per-piece base (2 per byte)
  11-bit: the same with a 3-bit offset (MLX 3-bit packing, 8 values in 3 bytes)
offset 0 = (signed) zero, offsets 1..W = exponents base..base + W - 1 (W = 15 / 7), compared bitwise with the original
bytes. A piece whose nonzero exponents don't fit (span > 14 / > 6) or that holds a subnormal / inf / nan is an ESCAPE (kept
raw); for those the best window's value-level escape count is reported too. Quiet research mode: waits out
~/.local/share/quiet/PAUSE between layers; stops if pmset reports a thermal / performance warning.
usage: taskpolicy -b python sb_audit.py [first_layer=0] [last_layer=47] [experts=512]  -> stdout, data/sb_audit.npz"""
import fcntl, json, os, re, subprocess, sys, time
import numpy as np

D = "/Users/kamen/BigMoE/models/bare/records3/"
IX = json.load(open(D + "index.json"))["layers"]
L0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
L1 = int(sys.argv[2]) if len(sys.argv) > 2 else 47
NE = int(sys.argv[3]) if len(sys.argv) > 3 else 512
REC, G, CH = 2932736, 25600, 32                       # record bytes, values per piece, experts per chunk
TYPES = ("s3", "b3", "b")                             # piece index = type * 3 + projection
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


def hot():
    t = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return any(int(x) > 0 for x in re.findall(r"level set to (\d+)", t)) or \
        any(int(x) < 100 for x in re.findall(r"CPU_Speed_Limit\s*=\s*(\d+)", t))


def pack(b, k):                                       # b: (P, G) uint8 offsets < 2^k -> (P, G * k / 8) uint8
    if k == 4:
        b = b.reshape(len(b), -1, 2)
        return b[:, :, 0] | (b[:, :, 1] << 4)
    w = (b.reshape(len(b), -1, 8).astype(np.uint32) << (3 * np.arange(8, dtype=np.uint32))).sum(2, dtype=np.uint32)
    return np.stack([w & 0xFF, (w >> 8) & 0xFF, w >> 16], 2).astype(np.uint8).reshape(len(b), -1)


def unpack(p, k):
    if k == 4:
        return np.stack([p & 0xF, p >> 4], 2).reshape(len(p), -1)
    t = p.reshape(len(p), -1, 3).astype(np.uint32)
    w = t[:, :, 0] | (t[:, :, 1] << 8) | (t[:, :, 2] << 16)
    return ((w[:, :, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).astype(np.uint8).reshape(len(p), -1)


def best_window(e, W):                                # value-level: most nonzero normal exponents inside W consecutive
    h = np.bincount(e, minlength=256)
    c = np.convolve(h, np.ones(W, np.int64))[W - 1:]
    return len(e) - int(c.max())


def audit(v):                                         # v: (P, G) uint16 -> per-piece stats + round-trip mismatches
    e = ((v >> 7) & 0xFF).astype(np.int16)
    z = (v & 0x7FFF) == 0
    sub = (e == 0) & ~z
    spc = e == 255
    ok = ~z & ~sub & ~spc
    emin = np.where(ok, e, 999).min(1)
    emax = np.where(ok, e, -1).max(1)
    has = emax >= 0
    span = np.where(has, emax - emin, 0)
    neg = ((v >> 15) == 1) & ~z
    pos = ((v >> 15) == 0) & ~z
    r = dict(emin=np.where(has, emin, 0), emax=np.where(has, emax, 0), span=span, zeros=z.sum(1), sub=sub.sum(1),
             spc=spc.sum(1), both=neg.any(1) & pos.any(1))
    a = (((v >> 8) & 0x80) | (v & 0x7F)).astype(np.uint8)
    base = np.where(has, emin, 1)[:, None]
    for k in (4, 3):
        W = 2 ** k - 1
        fits = (span <= W - 1) & (r["sub"] == 0) & (r["spc"] == 0)
        off = np.where(z, 0, np.clip(e - base + 1, 0, W)).astype(np.uint8)
        b2 = unpack(pack(off, k), k).astype(np.int16)
        v2 = ((a.astype(np.uint16) & 0x80) << 8) | (a.astype(np.uint16) & 0x7F) | \
            np.where(b2 == 0, 0, (b2 + base - 1) << 7).astype(np.uint16)
        r[f"esc{k}"] = ~fits
        r[f"mism{k}"] = ((v2 != v) & fits[:, None]).sum(1)
        r[f"vesc{k}"] = np.array([0 if f else best_window(e[i][ok[i]], W) + int(sub[i].sum() + spc[i].sum())
                                  for i, f in enumerate(fits)])
    return r


def main():
    t_all = time.time()
    keys = ("emin", "emax", "span", "zeros", "sub", "spc", "both", "esc4", "mism4", "vesc4", "esc3", "mism3", "vesc3")
    out = {k: np.zeros((48, NE, 9), np.int32) for k in keys}
    print(f"W24-sp step (1) audit: layers {L0}-{L1}, {NE} experts each, 9 scale / bias pieces of {G} bf16 per expert; "
          f"start {time.strftime('%a %H:%M:%S')}", flush=True)
    last = L0 - 1
    for L in range(L0, L1 + 1):
        waited = 0.0
        while os.path.exists(PAUSE):
            time.sleep(5); waited += 5
        if hot():
            print(f"STOP: pmset reports a thermal / performance warning before layer {L}", flush=True); break
        t0 = time.time()
        ly = IX[str(L)]
        mo, ms, lo, ls = ly["msb_piece_off"], ly["msb_piece_size"], ly["lsb_piece_off"], ly["lsb_piece_size"]
        assert all(mo[3 * p + 2] == mo[3 * p + 1] + ms[3 * p + 1] and ms[3 * p + 1] == ms[3 * p + 2] == 2 * G
                   and ls[2 * p + 1] == 2 * G for p in range(3))
        fd = os.open(D + ly["file"], os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        for c0 in range(0, NE, CH):
            n = min(CH, NE - c0)
            buf = np.empty((n, 9, G), np.uint16)
            for i in range(n):
                o = (c0 + i) * REC
                for p in range(3):
                    sb = np.frombuffer(os.pread(fd, 4 * G, o + mo[3 * p + 1]), np.uint16)
                    buf[i, p], buf[i, 3 + p] = sb[:G], sb[G:]
                    buf[i, 6 + p] = np.frombuffer(os.pread(fd, 2 * G, o + lo[2 * p + 1]), np.uint16)
            r = audit(buf.reshape(n * 9, G))
            for k in keys:
                out[k][L, c0:c0 + n] = r[k].reshape(n, 9)
        os.close(fd)
        sp = out["span"][L]
        print(f"L{L:02d} span max s3 {sp[:, 0:3].max()} b3 {sp[:, 3:6].max()} b {sp[:, 6:9].max()} | escapes 12-bit "
              f"{out['esc4'][L].sum()} 11-bit {out['esc3'][L].sum()} | zeros {out['zeros'][L].sum()} subnormal "
              f"{out['sub'][L].sum()} inf/nan {out['spc'][L].sum()} | round-trip mismatches 12-bit {out['mism4'][L].sum()} "
              f"11-bit {out['mism3'][L].sum()} | {time.time() - t0:.0f} s" + (f" (waited {waited:.0f} s on PAUSE)" if waited
                                                                            else ""), flush=True)
        last = L
    if last < L0:
        return
    L, done = last, slice(L0, last + 1)
    np.savez_compressed(os.path.join(os.path.dirname(__file__), "..", "data", "sb_audit.npz"),
                        **{k: out[k][done].astype(np.int16 if k in ("emin", "emax", "span") else np.int32) for k in keys},
                        layers=np.arange(L0, L + 1))
    npc = (L + 1 - L0) * NE * 3
    print(f"\nSUMMARY over layers {L0}-{L}: {npc * 3} pieces ({npc} per type), {npc * 3 * G} values; "
          f"{time.time() - t_all:.0f} s; end {time.strftime('%a %H:%M:%S')}")
    for t, nm in enumerate(TYPES):
        s = slice(3 * t, 3 * t + 3)
        sp = out["span"][done, :, s].ravel()
        hist = np.bincount(sp, minlength=8)
        print(f"{nm:2s}: span histogram " + " ".join(f"{i}:{c}" for i, c in enumerate(hist) if c) +
              f" | both signs in {100 * out['both'][done, :, s].mean():.2f} % of pieces | zeros {out['zeros'][done, :, s].sum()}"
              f" (pieces with any {(out['zeros'][done, :, s] > 0).sum()}) | subnormal {out['sub'][done, :, s].sum()} inf/nan "
              f"{out['spc'][done, :, s].sum()} | exponent range {out['emin'][done, :, s].min()}-{out['emax'][done, :, s].max()}")
    for k, bits in ((4, 12), (3, 11)):
        esc = out[f"esc{k}"][done]
        print(f"{bits}-bit (offset {k} bits, window {2 ** k - 1} exponents + zero code): escaped pieces {esc.sum()} of "
              f"{esc.size} ({100 * esc.mean():.4f} %), layers with any {int((esc.reshape(esc.shape[0], -1).sum(1) > 0).sum())};"
              f" value-level escapes {out[f'vesc{k}'][done].sum()} (max per piece {out[f'vesc{k}'][done].max()} of {G}); "
              f"round-trip mismatches on packed pieces {out[f'mism{k}'][done].sum()}")


if __name__ == "__main__":
    main()
