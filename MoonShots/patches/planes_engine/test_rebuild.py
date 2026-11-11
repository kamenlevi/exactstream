"""rebuild.py check on real experts: MSB/LSB pools of 64 experts of one layer (built in memory with make_records3's
converter from records/), rebuild_slots vs (a) the original 4-bit pieces and (b) BigMoE's 3b path (gathers + rebuild4 +
s3 * 0.5 + LSB b), bitwise; then ms per call for P = 2..20 pairs (one layer's three projections).
usage: test_rebuild.py [layer]   (GPU: ~0.5 GB; quiet lock for the timings)"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import make_records3 as M
from rebuild import rebuild_slots, PROJ

L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N = 64
MD = os.path.expanduser("~/BigMoE/models/bare/records")
si = json.load(open(os.path.join(MD, "index.json")))["layers"][str(L)]
rs, po, ps = si["recsz"], si["piece_off"], si["piece_size"]
lay = M.layout()
fd = M.open_nocache(os.path.join(MD, si["file"]), os.O_RDONLY)
src = M.aligned(N * rs)
M.pread_into(fd, src, 0)
os.close(fd)
r3 = M.convert_chunk(src, N, si, lay)
R = lay["recsz"]


def piece(buf, rec, off, size, dt, shape):
    a = np.stack([np.frombuffer(buf, dt, size // np.dtype(dt).itemsize, e * rec + off) for e in range(N)])
    return mx.array(a.reshape(N, *shape))


orig, msb, lsb = {}, {}, {}
PERM_NP = np.random.default_rng(11).permutation(N)          # LSB slot j holds expert PERM[j]
PERM, INV = mx.array(PERM_NP), mx.array(np.argsort(PERM_NP).astype(np.int32))
for i, (p, (out, K)) in enumerate(zip(PROJ, M.SHAPES)):
    orig[p] = (piece(src, rs, po[3 * i], ps[3 * i], np.uint32, (out, K // 8)),
               piece(src, rs, po[3 * i + 1], ps[3 * i + 1], np.uint16, (out, K // 64)).view(mx.bfloat16),
               piece(src, rs, po[3 * i + 2], ps[3 * i + 2], np.uint16, (out, K // 64)).view(mx.bfloat16))
    mo, ms, lo, ls = (lay["msb_piece_off"][3 * i:3 * i + 3], lay["msb_piece_size"][3 * i:3 * i + 3],
                      lay["lsb_piece_off"][2 * i:2 * i + 2], lay["lsb_piece_size"][2 * i:2 * i + 2])
    msb[p] = {"w3": piece(r3, R, mo[0], ms[0], np.uint32, (out, K * 3 // 32)),
              "s3": piece(r3, R, mo[1], ms[1], np.uint16, (out, K // 64)).view(mx.bfloat16),
              "b3": piece(r3, R, mo[2], ms[2], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    lsb[p] = {"l1": piece(r3, R, lo[0], ls[0], np.uint32, (out, K // 32))[PERM],
              "b": piece(r3, R, lo[1], ls[1], np.uint16, (out, K // 64)).view(mx.bfloat16)[PERM]}
mx.eval(orig, msb, lsb)

import bigmoe.tier as T


def theirs(sm, sl):
    out = {}
    for p in PROJ:
        out[p] = (T.rebuild4(msb[p]["w3"][sm], lsb[p]["l1"][sl]),
                  msb[p]["s3"][sm] * mx.array(0.5, dtype=mx.bfloat16), lsb[p]["b"][sl])
    return out


rng = np.random.default_rng(1)
ok = True
for P in (2, 7, 20):
    sm = mx.array(rng.choice(N, P).astype(np.int32))
    sl = INV[sm]                                            # the same experts' LSB slots
    ours = rebuild_slots(msb, lsb, sm, sl)
    ref = theirs(sm, sl)
    for p in PROJ:
        w, s, b = ours[p]
        eq_ref = all(mx.array_equal(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a,
                                    c.view(mx.uint16) if c.dtype == mx.bfloat16 else c).item() for a, c in zip(ours[p], ref[p]))
        # vs the original pieces of the experts in the MSB slots (LSB slots name the same experts)
        eq_orig = (mx.array_equal(w, orig[p][0][sm]).item() and mx.array_equal(s.view(mx.uint16), orig[p][1][sm].view(mx.uint16)).item()
                   and mx.array_equal(b.view(mx.uint16), orig[p][2][sm].view(mx.uint16)).item())
        ok &= eq_ref and eq_orig
        print(f"P={P} {p}: == BigMoE 3b rebuild {eq_ref}; == original 4-bit pieces {eq_orig}", flush=True)
print("bitwise:", "PASS" if ok else "FAIL", flush=True)


def bench(fn, sm, sl, it=200):
    for _ in range(10):
        mx.eval(fn(sm, sl))
    t = time.perf_counter()
    for _ in range(it):
        mx.eval(fn(sm, sl))
    return (time.perf_counter() - t) / it * 1e3


for P in (2, 4, 8, 16, 20):
    sm = mx.array(rng.choice(N, P).astype(np.int32))
    sl = mx.array(rng.permutation(N)[:P].astype(np.int32))
    a = bench(lambda x, y: rebuild_slots(msb, lsb, x, y), sm, sl)
    b = bench(theirs, sm, sl)
    gb = P * (2 * 2.765e6) / (a * 1e-3) / 1e9
    print(f"P={P:2d}: rebuild_slots {a:.3f} ms ({gb:.0f} GB/s r+w) | BigMoE 3b gathers+rebuild4 {b:.3f} ms | x{b / a:.2f}", flush=True)
sys.exit(0 if ok else 1)
