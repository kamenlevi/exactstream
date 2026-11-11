"""Unit test of the BIGMOE_TIER GPU path (bigmoe/tier.py) on real experts of one layer, from the safetensors split in numpy
(the test-only stand-in for MoonShots' converter, as in test_records3.py):
1. rebuild4(w3, l1) == the original 4-bit words, bitwise (GPU kernel);
2. the 4-bit rank path (rebuilt w4, s3 * 0.5, LSB b) == gather_qmm on the original pieces, bitwise;
3. the 3-bit rank path (gather_qmm bits=3 on the MSB part) vs planes125._tier_proj(n=3) (the evaluated values).
Usage: test_tier.py [layer]   (small GPU work: a few gather_qmm calls)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes125"))
import mlx.core as mx
import bigmoe.tier as T

MD = os.path.expanduser("~/BigMoE/models/bare")
L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = 16
from bigmoe.stream import read_headers, PIECES          # source pieces: the safetensors (records/ is replaced)
_hdr = read_headers(MD)
_src = [_hdr[f"language_model.model.layers.{L}.mlp.switch_mlp.{a}.{b}"] for a, b in PIECES]
ps = [int(np.prod(sh[1:])) * (4 if dt == "U32" else 2) for _, _, dt, sh in _src]
po = [int(x) for x in np.cumsum([0] + ps)[:-1]]


def get_rec(e):
    out = b""
    for (path, off, _, _), n in zip(_src, ps):
        with open(path, "rb") as fh:
            fh.seek(off + e * n); out += fh.read(n)
    return out
shapes = [(640, 320), (640, 40), (640, 40), (640, 320), (640, 40), (640, 40), (2560, 80), (2560, 10), (2560, 10)]
names = ["gate_proj", "up_proj", "down_proj"]


def bf(h): return (h.astype(np.uint32) << 16).view(np.float32)
def tobf(f):
    u = f.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


W4 = {p: [] for p in names}; S = {p: [] for p in names}; B = {p: [] for p in names}
W3 = {p: [] for p in names}; S3 = {p: [] for p in names}; B3 = {p: [] for p in names}; L1 = {p: [] for p in names}
if True:
    for e in range(N):
        rec = get_rec(e)
        for i, p in enumerate(names):
            o, g = shapes[3 * i][0], shapes[3 * i + 1][1]
            w = np.frombuffer(rec[po[3 * i]:po[3 * i] + ps[3 * i]], np.uint32)
            s = np.frombuffer(rec[po[3 * i + 1]:po[3 * i + 1] + ps[3 * i + 1]], np.uint16)
            b = np.frombuffer(rec[po[3 * i + 2]:po[3 * i + 2] + ps[3 * i + 2]], np.uint16)
            q = ((w[:, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(-1)
            m, l = q >> 1, q & 1
            w3 = np.packbits(((m[:, None] >> np.arange(3, dtype=np.uint32)) & 1).astype(np.uint8).reshape(-1), bitorder="little").view(np.uint32)
            l1 = np.packbits(l.astype(np.uint8), bitorder="little").view(np.uint32)
            off = l.reshape(-1, 64).sum(1).astype(np.float32) / 64
            W4[p].append(w.reshape(o, -1)); S[p].append(s.reshape(o, g)); B[p].append(b.reshape(o, g))
            W3[p].append(w3.reshape(o, -1)); S3[p].append(tobf(2 * bf(s)).reshape(o, g))
            B3[p].append(tobf(bf(b) + bf(s) * off).reshape(o, g)); L1[p].append(l1.reshape(o, -1))
st = lambda d, dt=None: {p: (mx.array(np.stack(d[p])) if dt is None else mx.array(np.stack(d[p])).view(dt)) for p in names}
W4, W3, L1 = st(W4), st(W3), st(L1)
S, B, S3, B3 = st(S, mx.bfloat16), st(B, mx.bfloat16), st(S3, mx.bfloat16), st(B3, mx.bfloat16)
ok = True
for p in names:
    r = T.rebuild4(W3[p], L1[p]); mx.eval(r)
    good = mx.array_equal(r, W4[p]).item(); ok &= good
    print(f"1. rebuild4 {p}: {'bit-exact' if good else 'DIFFERS'}")

base = {p: {"w3": W3[p], "s3": S3[p], "b3": B3[p]} for p in names}
side = {p: {"l1": L1[p], "b": B[p]} for p in names}
pools = (base, side)
rng = np.random.default_rng(0)
for T_ in (1, 4):
    x = mx.array(rng.standard_normal((T_, 2560)).astype(np.float32)).astype(mx.bfloat16)
    xe = mx.expand_dims(x, (-2, -3))
    ids = mx.array(rng.choice(N, (T_, 10)).astype(np.int32))
    sm, sl = ids[:, :2], ids[:, :2]                       # sidecar slot == base slot here
    for p in ("gate_proj", "up_proj"):
        y4 = T._proj(xe, pools, p, (sm, sl), 4)
        ref = mx.gather_qmm(xe, W4[p], S[p], B[p], rhs_indices=sm, transpose=True, group_size=64, bits=4)
        good = mx.array_equal(y4, ref).item(); ok &= good
        print(f"2. T={T_} {p} 4-bit ranks: {'bit-exact' if good else 'DIFFERS ' + str(mx.abs(y4 - ref).max().item())}")
        y3 = T._proj(xe, pools, p, ids[:, 2:], 3)
        import planes125 as P125

        class Lin:                                          # the 4-bit QuantizedSwitchLinear fields _tier_proj reads
            weight, scales, biases, group_size = W4[p], S[p], B[p], 64
        sl3 = ids[:, 2:].reshape(-1)
        r3 = P125._tier_proj(Lin, xe, sl3, mx.arange(sl3.size).astype(mx.uint32).reshape(ids[:, 2:].shape), 3)
        d = mx.abs(y3.astype(mx.float32) - r3.astype(mx.float32)).max().item()
        rel = d / mx.abs(r3.astype(mx.float32)).max().item()
        exact = mx.array_equal(y3, r3).item()
        ok &= rel < 2e-2
        print(f"3. T={T_} {p} 3-bit ranks vs _tier_proj(3): {'bit-exact' if exact else f'max |d| {d:.3g} (rel {rel:.2g})'}")
import planes125 as P125
for p in names:                                             # the tier's weights themselves: dequantized, bitwise
    d3 = mx.dequantize(W3[p], S3[p], B3[p], group_size=64, bits=3)
    wm, lsum = P125._MASK[3](W4[p])
    off = lsum.reshape(*S[p].shape, 8).sum(-1).astype(mx.float32) / 64
    bias = (B[p].astype(mx.float32) + S[p].astype(mx.float32) * off).astype(S[p].dtype)
    dm = mx.dequantize(wm, S[p], bias, group_size=64, bits=4)
    good = mx.array_equal(d3, dm).item(); ok &= good
    print(f"4. {p} MSB-3 dequant == masked 4-bit dequant (planes125): {'bit-exact' if good else 'DIFFERS ' + str(mx.abs(d3.astype(mx.float32) - dm.astype(mx.float32)).max().item())}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
