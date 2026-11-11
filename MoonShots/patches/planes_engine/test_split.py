"""Out-of-order experts check: tier_glu_split (hit pairs, then missed pairs, then the LY=8 combine) == tier_glu(...,
scores=) bitwise, for random hit masks (incl. all-hit and all-miss), M = 1, 2, 4, 8, plans 4433333333 and 4433332222
(the 2-bit tier's bias pool is a stand-in here: both paths read the same one). Pools: 64 experts of records3/L{L}.bin.
usage: test_split.py [layer]   (GPU: seconds)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx
import make_records3 as M
from tierglu import tier_glu, tier_glu_split, tier_glu_phase, combine_phase, PROJ

L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NE = 64
lay = M.layout()
R3 = lay["recsz"]
fd = M.open_nocache(os.path.expanduser(f"~/BigMoE/models/bare/records3/L{L:02d}.bin"), os.O_RDONLY)
r3 = M.aligned(NE * R3)
M.pread_into(fd, r3, 0)
os.close(fd)


def piece(off, size, dt, shape):
    return mx.array(np.stack([np.frombuffer(r3, dt, size // np.dtype(dt).itemsize, e * R3 + off) for e in range(NE)]).reshape(NE, *shape))


msb, lsb = {}, {}
for i, (p, (out, K)) in enumerate(zip(PROJ, M.SHAPES)):
    mo, ms = lay["msb_piece_off"][3 * i:3 * i + 3], lay["msb_piece_size"][3 * i:3 * i + 3]
    lo, ls = lay["lsb_piece_off"][2 * i:2 * i + 2], lay["lsb_piece_size"][2 * i:2 * i + 2]
    msb[p] = {"w3": piece(mo[0], ms[0], np.uint32, (out, K * 3 // 32)),
              "s3": piece(mo[1], ms[1], np.uint16, (out, K // 64)).view(mx.bfloat16),
              "b3": piece(mo[2], ms[2], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    lsb[p] = {"l1": piece(lo[0], ls[0], np.uint32, (out, K // 32)),
              "b": piece(lo[1], ls[1], np.uint16, (out, K // 64)).view(mx.bfloat16)}
mx.eval(msb, lsb)
b2 = {p: msb[p]["b3"] for p in PROJ}
rng = np.random.default_rng(3)
ok = True
for plan in ("4433333333", "4433332222"):
    for Mr in (1, 2, 4, 8):
        x = mx.array((rng.standard_normal((Mr, 2560)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
        sm = mx.array(np.stack([rng.choice(NE, 10, replace=False) for _ in range(Mr)]).astype(np.int32))
        sl = mx.array(np.stack([rng.choice(NE, 2, replace=False) for _ in range(Mr)]).astype(np.int32))
        w = mx.softmax(mx.array(rng.standard_normal((Mr, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        ref = tier_glu(x, msb, lsb, sm, sl, plan=plan, b2=b2, scores=w)
        res = []
        for frac in (1.0, 0.0, 0.9, 0.5):
            hit = rng.random((Mr, 10)) < frac
            y = tier_glu_split(x, msb, lsb, sm, sl, hit, w, plan=plan, b2=b2)
            n = int((y.view(mx.uint16) != ref.view(mx.uint16)).sum().item())
            ok &= n == 0
            res.append(f"hit {frac:.1f}: {n}")
            # engine form: device phase flags, phase-1 slots -1 where not resident (must never be read)
            ph = mx.array(np.where(hit, 1, 2).astype(np.int32))
            sm1 = mx.where(ph == 1, sm, -1)
            y1 = tier_glu_phase(x, msb, lsb, sm1, sl, ph, 1, plan=plan, b2=b2)
            y2 = tier_glu_phase(x, msb, lsb, sm, sl, ph, 2, plan=plan, b2=b2)
            yp = combine_phase(y1, y2, ph, w)
            n2 = int((yp.view(mx.uint16) != ref.view(mx.uint16)).sum().item())
            ok &= n2 == 0
            res[-1] += f"/{n2}"
        print(f"plan {plan} M={Mr}: mismatches split/phase vs fused  " + " | ".join(res), flush=True)
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
