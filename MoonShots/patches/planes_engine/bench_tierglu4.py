"""records4 tier GLU (tierglu4) speed vs records3's (tierglu) and stock 4-bit, chained 48 layers, synthetic pools:
per plan, ms per layer at M = 1, 2, 4, 8, and GB/s of the plane bytes each plan reads (tier 2: P2 1.536 MB; tier 3:
+ B3 0.768; tier 4: + B4 0.768; records3: MSB 2.150 + LSB 0.768 for 4-bit ranks). usage: bench_tierglu4.py   (quiet lock)"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx
from tierglu import tier_glu
from tierglu4 import tier_glu4

K, N, G = 2560, 640, 64
C, NLP, LAYERS, REPS = 96, 6, 48, 5
SH = {"gate_proj": (N, K), "up_proj": (N, K), "down_proj": (K, N)}
rb = lambda s: mx.random.randint(0, 2 ** 31, s, dtype=mx.int32).astype(mx.uint32) * 2
rs = lambda s, a: (mx.random.normal(s) * a).astype(mx.bfloat16)
P3, P4 = [], []
for _ in range(NLP):
    msb = {p: {"w3": rb((C, o, k * 3 // 32)), "s3": rs((C, o, k // G), 2e-3), "b3": rs((C, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    lsb = {p: {"l1": rb((C, o, k // 32)), "b": rs((C, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    p2 = {p: {"w2": rb((C, o, k // 16)), "s2": rs((C, o, k // G), 4e-3), "b2": rs((C, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    b3 = {p: {"c": rb((C, o, k // 32)), "b3": rs((C, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    b4 = {p: {"l": rb((C, o, k // 32)), "b": rs((C, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    mx.eval(msb, lsb, p2, b3, b4)
    P3.append((msb, lsb)); P4.append((p2, b3, b4))
rng = np.random.default_rng(0)
BYTES4 = {"2": 1.536, "3": 2.304, "4": 3.072}


def timed(fn, M, plan):
    t = np.array([int(c) for c in plan])
    R = []
    for _ in range(LAYERS):
        sm = mx.array(np.stack([rng.choice(C, 10, replace=False) for _ in range(M)]).astype(np.int32))
        R.append((sm, mx.where(mx.array(t >= 3)[None], sm, -1), mx.where(mx.array(t == 4)[None], sm, -1)))
    mx.eval(R)
    best = 1e9
    for rep in range(REPS + 1):
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        t0 = time.perf_counter()
        for li in range(LAYERS):
            y = fn(x, li % NLP, *R[li])
            x = y.sum(axis=-2).astype(mx.bfloat16)
        mx.eval(x)
        if rep:
            best = min(best, time.perf_counter() - t0)
    return best / LAYERS * 1e3


for plan in os.environ.get("PLANS", "4433333333,4422222222,4222222222").split(","):   # env PLANS / MS (MoonShots 3B)
    for M in [int(v) for v in os.environ.get("MS", "1,2,4,8").split(",")]:
        r4 = lambda x, li, sm, cs, ls: tier_glu4(x, *P4[li], sm, cs, ls, plan=plan)
        ms4 = timed(r4, M, plan)
        mb = M * sum(BYTES4[c] for c in plan)                      # plane MB read per layer (distinct slots ~ all)
        line = f"plan {plan} M={M}: records4 {ms4:.3f} ms ({1e3 * ms4 / (10 * M):.2f} us/pair; {mb / ms4:.0f} GB/s of {mb:.1f} MB)"
        if set(plan) <= set("34"):                                # records3 runs any 3 / 4-bit plan
            r3 = lambda x, li, sm, cs, ls: tier_glu(x, *P3[li], sm, sm[:, :2], plan=plan)
            m3 = timed(r3, M, plan); line += f" | records3 {m3:.3f} ms ({1e3 * m3 / (10 * M):.2f} us/pair)"
        print(line, flush=True)
