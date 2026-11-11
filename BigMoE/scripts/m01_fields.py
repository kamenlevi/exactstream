"""R-44 (a) M-01 field widths (CPU / SSD only, exact, read-only): over all experts of some layers of records3, the bf16
layout of the per-group scale / bias fields, to fix the fixed-width fields of the M-01 record:
  MSB part: s3 (3-bit scale) and b3 (3-bit bias); LSB part: b (4-bit bias). Each part must unpack on its own (the tier
  pools read the MSB and LSB parts separately), so b cannot lean on s3 / b3.
Per layer: zeros / non-normals, sign split, the per-expert exponent span (max - min over normal values) for s3 and b, the
b3 exponent offset to s3's, and how many experts fit 2- / 3-bit exponent fields with no escapes.
usage: [BigMoE venv] taskpolicy -b python m01_fields.py [layers=0,6,18,24,36,47]"""
import fcntl, json, os, sys, time
import numpy as np

D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,6,18,24,36,47").split(",")]


def load(L):
    r = IX[str(L)]
    E, G = r["E"], r["msb_piece_size"][1] // 2
    fd = os.open(D3 + r["file"], os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    s3, b3, b = (np.empty((E, 3, G), np.uint16) for _ in range(3))
    for e in range(E):
        o = e * r["recsz"]
        for p in range(3):
            s3[e, p] = np.frombuffer(os.pread(fd, 2 * G, o + r["msb_piece_off"][3 * p + 1]), np.uint16)
            b3[e, p] = np.frombuffer(os.pread(fd, 2 * G, o + r["msb_piece_off"][3 * p + 2]), np.uint16)
            b[e, p] = np.frombuffer(os.pread(fd, 2 * G, o + r["lsb_piece_off"][2 * p + 1]), np.uint16)
    os.close(fd)
    return s3.reshape(E, -1), b3.reshape(E, -1), b.reshape(E, -1)


def span(u, name):
    """per-expert exponent span over normal values; non-normals (exp 0 or 255) counted apart"""
    ex = ((u >> 7) & 0xFF).astype(np.int32)
    zero = (u & 0x7FFF) == 0
    odd = ((ex == 0) & ~zero) | (ex == 255)
    big = np.where(zero | odd, -1, ex).max(1)
    small = np.where(zero | odd, 999, ex).min(1)
    sp = big - small
    neg = (u >> 15).mean()
    fit = {k: int((sp <= (1 << k) - 1 - (1 if zero.any() else 0)).sum()) for k in (2, 3)}
    return (f"{name}: zeros {int(zero.sum())}, denormal/inf/nan {int(odd.sum())}, negative {100 * neg:.1f} %, exp range "
            f"{int(small.min())}..{int(big.max())}, per-expert span max {int(sp.max())} (p99 {int(np.percentile(sp, 99))}), "
            f"experts fitting a 2-bit / 3-bit window: {fit[2]} / {fit[3]} of {u.shape[0]}")


t0 = time.time()
for L in LAYERS:
    s3, b3, b = load(L)
    print(f"L{L} ({time.time() - t0:.0f} s)")
    print("  " + span(s3, "s3"))
    print("  " + span(b, "b "))
    es, eb3 = ((s3 >> 7) & 0xFF).astype(np.int32), ((b3 >> 7) & 0xFF).astype(np.int32)
    zb3 = (b3 & 0x7FFF) == 0
    d = (eb3 - es)[~zb3]
    v, c = np.unique(d, return_counts=True)
    print(f"  b3: zeros {int(zb3.sum())}, negative {100 * (b3 >> 15).mean():.1f} %, exponent offset to s3: "
          + ", ".join(f"{a:+d} {100 * n / d.size:.3f} %" for a, n in zip(v, c)))
    # same sign as s3? (sign of b3 relative to s3 could save its bit)
    print(f"  b3 sign == s3 sign: {100 * ((b3 >> 15) == (s3 >> 15)).mean():.2f} %; b sign == b3 sign: "
          f"{100 * ((b >> 15) == (b3 >> 15)).mean():.2f} %", flush=True)
