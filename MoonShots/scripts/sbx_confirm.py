"""M-01 SBX confirmation (MoonShots Wed 15:4x; exact, CPU): the screen's 16.26 b (s3 + b3) and 2.58 b (b) came from plug-in
entropies of whole bf16 values over 32 experts. Here a concrete, trivially decodable format over ALL 512 experts of six layers:
  s3 = 7 raw mantissa bits + sign and exponent (rANS, one static table per layer); b3 = raw sign + 7 raw mantissa bits + exponent
  offset to s3's exponent (rANS); b = residual key(b) - key(bf16(b3 - s3/4)) (rANS, escape beyond +-15 -> 16 raw bits).
Code length = sum of -log2 p over the layer's own table (static rANS lands within ~0.1 % of it, plus 2 x 32-bit states per
stream); per-expert sizes, record = 16 KB blocks. Bitwise round trip of the transform (exponent / mantissa / sign / residual
-> bf16) checked on every value. PASS: mean layer <= -3 % bytes / token as in the screen. KILL: < -2 %.
usage: [BigMoE venv] taskpolicy -b python sbx_confirm.py [layers=0,6,18,24,36,47] -> stdout + data/sbx_confirm.txt"""
import fcntl, json, os, sys, time
import numpy as np

D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,6,18,24,36,47").split(",")]
G, E, ESC = 25600, 512, 15
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


def key(u):
    m = (u & 0x7FFF).astype(np.int32)
    return np.where(u & 0x8000, -m, m)


def unkey(k):
    return np.where(k < 0, 0x8000 | (-k), k).astype(np.uint16)


def bits(sym):
    """[E, n] symbols -> per-expert code length in bits under the layer's static table (all experts pooled)"""
    v, inv, c = np.unique(sym, return_inverse=True, return_counts=True)
    cost = -np.log2(c / c.sum())
    return cost[inv.reshape(sym.shape)].sum(1) + 64, len(v)


t0 = time.time()
tot = {}
for L in LAYERS:
    ix = IX[str(L)]
    fd = os.open(D3 + ix["file"], os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    s3, b3, b = (np.empty((E, 3, G), np.uint16) for _ in range(3))
    for e in range(E):
        o = e * ix["recsz"]
        for p in range(3):
            s3[e, p] = np.frombuffer(os.pread(fd, 51200, o + ix["msb_piece_off"][3 * p + 1]), np.uint16)
            b3[e, p] = np.frombuffer(os.pread(fd, 51200, o + ix["msb_piece_off"][3 * p + 2]), np.uint16)
            b[e, p] = np.frombuffer(os.pread(fd, 51200, o + ix["lsb_piece_off"][2 * p + 1]), np.uint16)
    os.close(fd)
    s3, b3, b = s3.reshape(E, -1), b3.reshape(E, -1), b.reshape(E, -1)
    es, ms = (s3 >> 7).astype(np.int32), s3 & 0x7F            # sign + exponent (some layers have negative scales)
    e3, m3, sg3 = ((b3 >> 7) & 0xFF).astype(np.int32), b3 & 0x7F, b3 >> 15
    d3 = e3 - (es & 0xFF)                                  # b3 exponent offset to s3 exponent
    r = key(b) - key(bf16(f32(b3) - f32(s3) / 4))
    esc = np.abs(r) > ESC
    # round trip of the transform
    s3r = ((es << 7) | ms).astype(np.uint16)
    b3r = ((sg3 << 15) | ((d3 + (es & 0xFF)) << 7) | m3).astype(np.uint16)
    br = unkey(np.where(esc, key(b), key(bf16(f32(b3r) - f32(s3r) / 4)) + r))
    assert (s3r == s3).all() and (b3r == b3).all() and (br == b).all(), "round trip"
    bs, ns = bits(es)
    bd, nd = bits(d3)
    br_, nr = bits(np.where(esc, ESC + 1, r))
    msb = bs + bd + 3 * G * (7 + 8)                        # + raw mantissas, sign
    lsb = br_ + esc.sum(1) * 16
    tot[L] = (msb, lsb)
    say(f"L{L}: s3 sign + exponent {bs.mean() / (3 * G):.3f} b ({ns} values), b3 exponent offset {bd.mean() / (3 * G):.3f} b ({nd}), "
        f"-> s3 + b3 {msb.mean() / (3 * G):.3f} b per group (screen 16.26; 22 today), per-expert max {msb.max() / (3 * G):.3f}; "
        f"b residual {lsb.mean() / (3 * G):.3f} b (screen 2.58; 11 today), max {lsb.max() / (3 * G):.3f}, escapes {100 * esc.mean():.3f} %; "
        f"round trip bitwise OK ({time.time() - t0:.0f} s)")

blk = lambda n: -(-n // 16384) * 16384
codes = round(1843200 * 2.7231 / 3)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W
W.MSB, W.LSB = 1900544, 720896
base = W.bytes_per_token("lfu", Cm=159, Cl=44)[0]
for nm, f in (("mean expert, mean layer", lambda a: np.mean([x.mean() for x in a])),
              ("worst expert of the worst layer", lambda a: max(x.max() for x in a))):
    m = blk(codes + int(f([v[0] for v in tot.values()]) / 8) + 48)
    l = blk(614400 + int(f([v[1] for v in tot.values()]) / 8))
    W.MSB, W.LSB = m, l
    v = W.bytes_per_token("lfu", Cm=159, Cl=44)[0]
    say(f"24 GB (C_m 159 / C_l 44): {nm}: MSB miss {m:,} / LSB miss {l:,} B -> {v / 1e6:.1f} MB / token vs {base / 1e6:.1f} "
        f"({100 * (v / base - 1):+.2f} %)")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/sbx_confirm.txt"), "w").write("\n".join(lines) + "\n")
