"""Unit test of R-44 (b) POM + LP1 in the ext (run with BIGMOE_M01=1 BIGMOE_POM=4 BIGMOE_POM_LP1=1 BIGMOE_PF_JIT=1 and the
new ext, e.g. BIGMOE_EXT_BUILD=build_m01). For some layers, on the real records3m / records3p files:
(1) base pool with set_pom (R2 4, rows of 10 ranks): one ensure() with experts at ranks 0-3 (full reads) and 4-9 (2-plane
    reads): full slots == records3's MSB pieces; 2-plane slots == codes pack3(q3 & 6), s3, and the suite's tier-2 bias
    bf16(b4 + s4 * mean(q4 & 3)) (computed here from records3); then the 2-plane experts at rank 2 -> upgraded in place
    == records3's MSB pieces; pom_stats counts 2-plane reads / full reads / upgrades.
(2) sidecar with set_lp1 (rows of 2): an expert used only at rank 1 and not resident -> slot -2, nothing read; at rank 0 ->
    read, == records3's LSB pieces.
(3) GPU, tiny: the LP1-patched fused GLU phase kernels with a -2 sidecar slot at rank 1 == the same pairs computed with
    rank 1 at tier 3 (plan 4333333333), bitwise.
usage: BIGMOE_M01=1 BIGMOE_POM=4 BIGMOE_POM_LP1=1 BIGMOE_PF_JIT=1 BIGMOE_EXT_BUILD=build_m01 test_records3p.py [layers]"""
import fcntl, json, os, sys
import numpy as np
for k in ("BIGMOE_M01", "BIGMOE_POM", "BIGMOE_POM_LP1", "BIGMOE_PF_JIT"):
    assert os.environ.get(k), f"set {k}"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/" + os.environ.get("BIGMOE_EXT_BUILD", "build")))
import mlx.core as mx
import bigmoe_ext as ext

MD = os.path.expanduser("~/BigMoE/models/bare")
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,23,47").split(",")]
ix3 = json.load(open(os.path.join(MD, "records3", "index.json")))["layers"]
assert all(r.get("layout", "row-major") == "row-major" for r in ix3.values()), "records3 is converted (docs/PLAN_records3_convert.md): this script reads row-major records only"
ixm = json.load(open(os.path.join(MD, "records3m", "index.json")))["layers"]
ixp = json.load(open(os.path.join(MD, "records3p", "index.json")))["layers"]
R2, K = 4, 10
ok = True


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)


def unpack3(b):
    t = b.reshape(-1, 3).astype(np.uint32)
    w = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
    return ((w[:, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).reshape(-1).astype(np.uint8)


def pack3(q):
    w = (q.reshape(-1, 8).astype(np.uint32) << (3 * np.arange(8, dtype=np.uint32))).sum(1, dtype=np.uint32)
    return np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], 1).reshape(-1).astype(np.uint8)


def r3_expert(path, r, e):
    with open(path, "rb") as g:
        fcntl.fcntl(g.fileno(), 48, 1)
        g.seek(e * r["recsz"]); return np.frombuffer(g.read(r["recsz"]), np.uint8)


def expect(rec, r, two):
    """records3 bytes of one expert -> the 9 MSB pool pieces (full, or the 2-plane slot)"""
    out = []
    for p in range(3):
        mo, lo = r["msb_piece_off"][3 * p:3 * p + 3], r["lsb_piece_off"][2 * p:2 * p + 2]
        w3 = rec[mo[0]:mo[0] + r["msb_piece_size"][3 * p]]
        s3 = rec[mo[1]:mo[1] + r["msb_piece_size"][3 * p + 1]].view(np.uint16)
        b3 = rec[mo[2]:mo[2] + r["msb_piece_size"][3 * p + 2]].view(np.uint16)
        if not two:
            out += [w3.tobytes(), s3.tobytes(), b3.tobytes()]
            continue
        q3 = unpack3(w3)
        l1 = np.unpackbits(rec[lo[0]:lo[0] + r["lsb_piece_size"][2 * p]], bitorder="little")
        b4 = rec[lo[1]:lo[1] + r["lsb_piece_size"][2 * p + 1]].view(np.uint16)
        s4 = np.where((s3 & 0x7FFF) == 0, s3, s3 - 0x80).astype(np.uint16)
        lsum = (2 * (q3 & 1).astype(np.int32) + l1).reshape(-1, 64).sum(1)
        bias2 = bf16(f32(b4) + f32(s4) * (lsum.astype(np.float32) / np.float32(64)))
        out += [pack3(q3 & 6).tobytes(), s3.tobytes(), bias2.tobytes()]
    return out


for L in LAYERS:
    r3, rm, rp = ix3[str(L)], ixm[str(L)], ixp[str(L)]
    p3 = os.path.join(MD, "records3", r3["file"])
    rng = np.random.default_rng(300 + L)
    ids = [int(e) for e in rng.choice(r3["E"], 2 * K, replace=False)]
    sizes = r3["msb_piece_size"]
    # (1) base pool + POM
    P = [mx.zeros((3 * K, n // 2), mx.uint16) for n in sizes]; mx.eval(*P)
    Lx = ext.add_layer(3 * K, r3["E"], [0] * 9, [0] * 9, list(sizes), P)
    ext.set_records_part(Lx, os.path.join(MD, "records3m", rm["file"]), rm["recsz"], 0, rm["msb_sz"])
    ext.set_pom(Lx, os.path.join(MD, "records3p", rp["file"]), rp["recsz"], rp["seg1"], rp["seg2"], R2, K)
    s0 = ext.pom_stats()
    row = ids[:K]                                   # ranks 0-3 full, 4-9 2-plane
    sl = np.array(ext.ensure(Lx, mx.array(row, dtype=mx.int32)))
    s1 = ext.pom_stats()
    good = True
    for rank, (e, s) in enumerate(zip(row, sl)):
        exp = expect(r3_expert(p3, r3, e), r3, two=rank >= R2)
        good &= all(np.array(a[int(s)]).tobytes() == x for a, x in zip(P, exp))
    d = [b - a for a, b in zip(s0, s1)]
    good_c = d[0] == K - R2 and d[1] == R2 and d[2] == 0
    print(f"L{L}: rank 0-3 full / 4-9 2-plane slots == records3 / the suite's tier 2: {good}; stats 2-plane {d[0]} full "
          f"{d[1]} upgrades {d[2]}: {good_c}", flush=True)
    ok &= good and good_c
    # upgrade: the 2-plane experts at rank 2 (one row each, the others at ranks >= 4)
    good = True
    for e in row[R2:]:
        row2 = [ids[K], ids[K + 1], e] + ids[K + 2:K + 9]
        sl2 = np.array(ext.ensure(Lx, mx.array(row2, dtype=mx.int32)))
        exp = expect(r3_expert(p3, r3, e), r3, two=False)
        good &= all(np.array(a[int(sl2[2])]).tobytes() == x for a, x in zip(P, exp))
    d = [b - a for a, b in zip(s1, ext.pom_stats())]
    ok &= good and d[2] == K - R2
    print(f"L{L}: 2-plane slots upgraded in place == records3: {good}; upgrades {d[2]} (expected {K - R2})", flush=True)
    # (2) LP1 sidecar
    ls = r3["lsb_piece_size"]
    Q = [mx.zeros((8, n // 2), mx.uint16) for n in ls]; mx.eval(*Q)
    Ls = ext.add_layer(8, r3["E"], [0] * 6, [0] * 6, list(ls), Q)
    ext.set_records_part(Ls, os.path.join(MD, "records3m", rm["file"]), rm["recsz"], rm["msb_sz"], rm["recsz"] - rm["msb_sz"])
    ext.set_lp1(Ls, 2)
    a_, b_ = ids[0], ids[1]
    t0 = ext.pom_stats()
    o1 = np.array(ext.ensure(Ls, mx.array([a_, b_], dtype=mx.int32)))       # b_ only at rank 1: -2, nothing read
    o2 = np.array(ext.ensure(Ls, mx.array([b_, a_], dtype=mx.int32)))       # now at rank 0: read
    rec = r3_expert(p3, r3, b_)
    good = o1[0] >= 0 and o1[1] == -2 and o2[0] >= 0 and o2[1] == o1[0] and all(
        np.array(q[int(o2[0])]).tobytes() == rec[off:off + n].tobytes() for q, n, off in zip(Q, ls, r3["lsb_piece_off"]))
    d = [b - a for a, b in zip(t0, ext.pom_stats())]
    ok &= good and d[5] == 2 and d[6] == 1
    print(f"L{L}: LP1 rank-1-only miss -> -2 (no read), at rank 0 -> read == records3 LSB: {good}; rank-1 uses {d[5]}, "
          f"without LSB {d[6]}", flush=True)

# (3) GPU: the LP1-patched kernels (tiny random pools; same pairs with rank 1 at tier 3)
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
from bigmoe import tier as T
T._tier_glu()                                     # applies the LP1 patch (BIGMOE_POM_LP1)
import tierglu as TG
mx.set_default_device(mx.gpu)
rng = np.random.default_rng(5)
D, N, E_, M = 512, 256, 6, 2                      # hidden, expert width, pool slots, rows
mk = lambda *sh: mx.array(rng.integers(0, 2**32, sh, dtype=np.uint32))
sc = lambda *sh: mx.array((rng.standard_normal(sh) * 0.01).astype(np.float32)).astype(mx.bfloat16)
msb = {"gate_proj": {"w3": mk(E_, N, D * 3 // 32), "s3": sc(E_, N, D // 64), "b3": sc(E_, N, D // 64)},
       "up_proj": {"w3": mk(E_, N, D * 3 // 32), "s3": sc(E_, N, D // 64), "b3": sc(E_, N, D // 64)},
       "down_proj": {"w3": mk(E_, D, N * 3 // 32), "s3": sc(E_, D, N // 64), "b3": sc(E_, D, N // 64)}}
lsb = {"gate_proj": {"l1": mk(E_, N, D // 32), "b": sc(E_, N, D // 64)},
       "up_proj": {"l1": mk(E_, N, D // 32), "b": sc(E_, N, D // 64)},
       "down_proj": {"l1": mk(E_, D, N // 32), "b": sc(E_, D, N // 64)}}
x = mx.array(rng.standard_normal((M, D)).astype(np.float32)).astype(mx.bfloat16)
ms = mx.array(rng.integers(0, E_, (M, 10)), dtype=mx.int32)
ls = mx.array(rng.integers(0, E_, (M, 2)), dtype=mx.int32)
ph = mx.full((M * 10,), 2, mx.int32)
ls_lp = mx.concatenate([ls[:, :1], mx.full((M, 1), -2, mx.int32)], axis=1)
y_lp = TG.tier_glu_phase(x, msb, lsb, ms, ls_lp, ph, 2, plan="4433333333")
y_t3 = TG.tier_glu_phase(x, msb, lsb, ms, ls[:, :1], ph, 2, plan="4333333333")
mx.eval(y_lp, y_t3)
good = np.array_equal(np.array(y_lp.view(mx.uint16)), np.array(y_t3.view(mx.uint16)))
y_44 = TG.tier_glu_phase(x, msb, lsb, ms, ls, ph, 2, plan="4433333333"); mx.eval(y_44)
moved = not np.array_equal(np.array(y_44.view(mx.uint16)), np.array(y_lp.view(mx.uint16)))
ok &= good and moved
print(f"GPU: LP1 kernels, rank-1 sidecar slot -2 == rank 1 at tier 3 (bitwise): {good}; differs from the 4-bit rank 1: {moved}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
