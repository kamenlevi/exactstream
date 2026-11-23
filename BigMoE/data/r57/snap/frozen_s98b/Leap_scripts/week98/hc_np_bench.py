"""hc_np_bench.py (Leap, Mon 16 Nov 2026 14:2x; CHECK25): the HC read's first kernel (K1C) splits HCD = 10,240 into NP K
parts, one threadgroup each: at NP 20 that is 20 threadgroups (one per GPU core) streaming a 3.3 MB 8-bit mixer, likely
too little memory parallelism (read32 measured 6.3-8.1 ms per check vs a ~2.4 ms byte floor). Times read32 per check
(96 reads on cold random mixers, as hc8r32.bench) at NP 20 / 40 / 80 / 160, 'ind' and 'chain', and the numeric drift vs
NP 20 (NP changes the fp32 partial-sum grouping: a FAST-track change, not bitwise; adoption = EvalSuite's call).
usage: [BigMoE venv] hc_np_bench.py   (MEM ~1 GB; quiet for clean numbers)"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts/week98"))
import hc8r32 as H32

BF = mx.bfloat16
HC, D, LR, NREAD, ROUNDS = 4, 2560, 320, 96, 7
Ps = []
for i in range(NREAD):
    k = mx.random.split(mx.random.key(1000 + i), 6)
    wq = mx.random.randint(0, 2 ** 31 - 1, (LR, HC * D // 4), key=k[0]).astype(mx.uint32)
    uq = mx.random.randint(0, 2 ** 31 - 1, (HC * D, LR // 4), key=k[1]).astype(mx.uint32)
    ws = mx.random.uniform(1e-4, 2e-4, (LR, HC * D // 64), key=k[2]).astype(BF)
    us = mx.random.uniform(1e-4, 2e-4, (HC * D, LR // 64), key=k[3]).astype(BF)
    Ps.append(dict(wq=wq, ws=ws, wb=-128 * ws, uq=uq, us=us, ub=-128 * us,
                   wi=(mx.random.normal((HC, HC * D), key=k[4]) * 0.02).astype(BF),
                   wn=mx.ones((HC * D,), BF), hc=HC, d=D, lr=LR, eps=1e-6))
mx.eval([v for P in Ps for v in P.values() if isinstance(v, mx.array)])
h32 = (mx.random.normal((32, HC * D), key=mx.random.key(7)) * 0.8).astype(BF)
mx.eval(h32)
NPS = [20, 40, 80, 160]
ref = [H32.read32(P, h32, NP=20) for P in Ps[:8]]
mx.eval(ref)
print(f"HC NP BENCH {time.strftime('%a %d %b %H:%M:%S')}: read32 on {NREAD} cold 8-bit mixers (HC {HC}, D {D}, LR {LR}); "
      f"bytes per check ~{NREAD * 2 * LR * HC * D / 1e9:.2f} GB (8-bit codes) -> floor {NREAD * 2 * LR * HC * D / 280e6:.2f} ms")
for NP in NPS:
    out = [H32.read32(P, h32, NP=NP) for P in Ps[:8]]
    mx.eval(out)
    rel = max(float((mx.max(mx.abs(o[0].astype(mx.float32) - r[0].astype(mx.float32))) /
                     mx.max(mx.abs(r[0].astype(mx.float32)))).item()) for o, r in zip(out, ref))
    bit = all(bool(mx.array_equal(o[0], r[0]).item()) for o, r in zip(out, ref))
    res = {}
    for md in ("ind", "chain"):
        ts = []
        for r_ in range(ROUNDS):
            mx.synchronize()
            t0 = time.perf_counter()
            if md == "chain":
                h = h32
                for P in Ps:
                    y = H32.read32(P, h, NP=NP)[0]
                    h = h32 + (y[:1, :1] * 0).astype(BF)
                mx.eval(h)
            else:
                mx.eval([H32.read32(P, h32, NP=NP)[0] for P in Ps])
            mx.synchronize()
            ts.append(time.perf_counter() - t0)
        res[md] = 1e3 * float(np.median(ts[1:]))
    print(f"  NP {NP:3d} ({NP} K1C threadgroups): ind {res['ind']:6.2f} ms, chain {res['chain']:6.2f} ms per check; mixed vs NP 20: "
          f"{'bitwise' if bit else f'max rel {rel:.2e}'}")
