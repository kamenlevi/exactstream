"""M-01 SBX (MoonShots Wed 15:36; exact, CPU): shrink the per-group scale / bias bytes of the SSD copy (W24-ec + 11-bit today:
MSB miss 1,900,544 B incl. 211,200 B of s3 / b3, LSB miss 720,896 B incl. 105,600 B of the 4-bit bias b).
(a) The LSB record's bias b vs the MSB part: is b == bf16(b3 - s3 / 4) bitwise (the converter's b3 = b + s / 2)? Else the
    entropy of the residual in bf16 steps.
(b) s3 / b3 with context instead of 11 flat bits: s3 order-0, delta to the left group of the same row, to the same group of
    the row above, JPEG-LS MED of left / up / up-left (keys = the bf16 bit pattern, monotone, ~log scale); b3 = sign +
    key(|b3|) - key(s3) with the same contexts.
Static per-layer tables (plug-in entropy ~ what a table rANS reaches). Bytes per miss -> 24 GB replay (wall24 LFU, C_m 159 /
C_l 44, 16-row checks accepting 10; pools keep today's decoded layout, so only miss bytes change).
PASS (written first): >= -3 % SSD bytes / token vs W24-ec + 11-bit, rebuild bitwise. KILL: < -1 %.
usage: [BigMoE venv] taskpolicy -b python sbx.py [layers=6,24,42] [experts=32] -> stdout + data/sbx.txt"""
import fcntl, json, os, sys, time
import numpy as np

D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "6,24,42").split(",")]
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
G, SHAPES = 25600, ((640, 40), (640, 40), (2560, 10))      # groups per projection; gate / up / down as (rows, groups per row)
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def H(v):
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


def key(u):
    """monotone integer key of a bf16 bit pattern (sign-magnitude -> two's-complement-like order)"""
    m = (u & 0x7FFF).astype(np.int32)
    return np.where(u & 0x8000, -m, m)


def ctx(k):
    """residuals of a [rows, groups] key array under four predictors (first column / row fall back to the other neighbour)"""
    left = np.concatenate([k[:, :1] * 0 + np.roll(k[:, -1:], 1, 0), k[:, :-1]], 1)   # row start: previous row's last group
    up = np.concatenate([k[:1], k[:-1]], 0)
    ul = np.concatenate([up[:, :1], up[:, :-1]], 1)
    med = np.where(ul >= np.maximum(left, up), np.minimum(left, up),
                   np.where(ul <= np.minimum(left, up), np.maximum(left, up), left + up - ul))
    return {"order0": k, "left": k - left, "up": k - up, "MED": k - med}


t0 = time.time()
res = {}
for L in LAYERS:
    ix = IX[str(L)]
    ex = sorted(np.random.default_rng(L).choice(512, NE, replace=False).tolist())
    fd = os.open(D3 + ix["file"], os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)                                   # F_NOCACHE
    rd = lambda n, o: np.frombuffer(os.pread(fd, n, o), np.uint16)
    s3, b3, b = (np.empty((3, NE, G), np.uint16) for _ in range(3))
    for a, e in enumerate(ex):
        o = e * ix["recsz"]
        for p in range(3):
            s3[p, a] = rd(51200, o + ix["msb_piece_off"][3 * p + 1])
            b3[p, a] = rd(51200, o + ix["msb_piece_off"][3 * p + 2])
            b[p, a] = rd(51200, o + ix["lsb_piece_off"][2 * p + 1])
    os.close(fd)
    # (a) b from the MSB part
    pred = bf16(f32(b3) - f32(s3) / 4)
    exact = float((pred == b).mean())
    rb = key(b) - key(pred)
    vals, cnt = np.unique(rb, return_counts=True)
    top = ", ".join(f"{v:+d}: {100 * c / rb.size:.2f} %" for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1])[:4])
    hb = H(rb.ravel())
    say(f"L{L} ({NE} experts): (a) b == bf16(b3 - s3/4) in {100 * exact:.3f} % of groups; residual in bf16 steps {top}; "
        f"H = {hb:.3f} b (11-bit today)")
    # (b) s3 and b3 with contexts, per projection shape (gate / up pooled, down separate), bits per value
    hs, hb3 = {}, {}
    for nm, ps in (("gate/up", (0, 1)), ("down", (2,))):
        rs = {c: [] for c in ("order0", "left", "up", "MED")}
        r3 = {c: [] for c in rs}
        sg = []
        for p in ps:
            for a in range(NE):
                ks = key(s3[p, a]).reshape(SHAPES[p])
                k3 = key(b3[p, a] & 0x7FFF).reshape(SHAPES[p]) - ks          # magnitude offset to s3 (log ratio)
                sg.append((b3[p, a] >> 15).astype(np.int8))
                for c, v in ctx(ks).items():
                    rs[c].append(v.ravel())
                for c, v in ctx(k3).items():
                    r3[c].append(v.ravel())
        hs[nm] = {c: H(np.concatenate(v)) for c, v in rs.items()}
        hsg = H(np.concatenate(sg))
        hb3[nm] = {c: H(np.concatenate(v)) + hsg for c, v in r3.items()}
        say(f"  {nm}: s3 bits " + ", ".join(f"{c} {v:.3f}" for c, v in hs[nm].items()) +
            f" | b3 (sign {hsg:.3f} + |b3| vs s3) " + ", ".join(f"{c} {v:.3f}" for c, v in hb3[nm].items()))
    w = {"gate/up": 2 / 3, "down": 1 / 3}
    best_s = sum(w[k] * min(v.values()) for k, v in hs.items())
    best_b3 = sum(w[k] * min(v.values()) for k, v in hb3.items())
    res[L] = (best_s, best_b3, hb, exact)
    say(f"  L{L} best: s3 {best_s:.3f} + b3 {best_b3:.3f} = {best_s + best_b3:.3f} b per group (22 today); b {hb:.3f} (11 today)")

blk = lambda n: -(-n // 16384) * 16384
codes = round(1843200 * 2.7231 / 3)
worst = (max(r[0] + r[1] for r in res.values()), max(r[2] for r in res.values()))
mean = (np.mean([r[0] + r[1] for r in res.values()]), np.mean([r[2] for r in res.values()]))
say(f"bits per group over layers {LAYERS}: s3 + b3 mean {mean[0]:.3f} / worst {worst[0]:.3f}; b mean {mean[1]:.3f} / worst {worst[1]:.3f}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W
W.MSB, W.LSB = 1900544, 720896
base = W.bytes_per_token("lfu", Cm=159, Cl=44)[0]
say(f"24 GB exact stack (C_m 159 / C_l 44, W24-ec + 11-bit, MSB 1,900,544 / LSB 720,896): {base / 1e6:.1f} MB / token")
for nm, (sb, bb) in (("mean layer", mean), ("worst layer", worst)):
    for arm, m, l in (("(a) b from the MSB part only", 1900544, blk(614400 + round(3 * G * bb / 8))),
                      ("(b) s3 / b3 context-coded only", blk(codes + round(3 * G * sb / 8) + 48), 720896),
                      ("(a) + (b)", blk(codes + round(3 * G * sb / 8) + 48), blk(614400 + round(3 * G * bb / 8)))):
        W.MSB, W.LSB = m, l
        v = W.bytes_per_token("lfu", Cm=159, Cl=44)[0]
        say(f"  {nm}, {arm}: MSB miss {m:,} / LSB miss {l:,} B -> {v / 1e6:.1f} MB / token ({100 * (v / base - 1):+.2f} %)")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/sbx.txt"), "w").write("\n".join(lines) + "\n")
