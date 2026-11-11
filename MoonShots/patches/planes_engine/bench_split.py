"""Out-of-order experts cost model at decode/verify shapes (chained 48 layers, synthetic pools like bench_tierglu.py):
  fused     tier_glu(..., scores=): today's critical path after the spin (all pairs)
  hits      tier_glu_pairs on the hit pairs (would run during the spin: must fit in the stall, p50 ~0.6 ms/layer)
  misses+c  tier_glu_pairs on the missed pairs + the combine kernel: the critical path after the spin with OoO
for miss fractions 10 % and 30 % of the pairs. usage: bench_split.py   (GPU ~2 GB; quiet lock, daytime)"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx
import tierglu as TG

K, N, G = 2560, 640, 64
C_M, C_L, NLP, LAYERS, REPS = 96, 48, 6, 48, 5
SH = {"gate_proj": (N, K), "up_proj": (N, K), "down_proj": (K, N)}
rb = lambda s: mx.random.randint(0, 2 ** 31, s, dtype=mx.int32).astype(mx.uint32) * 2
rs = lambda s, a: (mx.random.normal(s) * a).astype(mx.bfloat16)
POOLS = []
for _ in range(NLP):
    msb = {p: {"w3": rb((C_M, o, k * 3 // 32)), "s3": rs((C_M, o, k // G), 2e-3), "b3": rs((C_M, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    lsb = {p: {"l1": rb((C_L, o, k // 32)), "b": rs((C_L, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    mx.eval(msb, lsb)
    POOLS.append((msb, lsb))
rng = np.random.default_rng(0)


def timed(fn):
    best = 1e9
    for rep in range(REPS + 1):
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        t = time.perf_counter()
        for li in range(LAYERS):
            x = fn(x, li)
        mx.eval(x)
        if rep:
            best = min(best, time.perf_counter() - t)
    return best / LAYERS * 1e3


for M in (1, 2, 4, 8):
    R = [(mx.array(np.stack([rng.choice(C_M, 10, replace=False) for _ in range(M)]).astype(np.int32)),
          mx.array(np.stack([rng.choice(C_L, 2, replace=False) for _ in range(M)]).astype(np.int32)),
          mx.softmax(mx.array(rng.standard_normal((M, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)) for _ in range(LAYERS)]
    mx.eval(R)
    out = [f"fused {timed(lambda x, li: TG.tier_glu(x, *POOLS[li % NLP], R[li][0], R[li][1], scores=R[li][2])):.3f}"]
    for mf in (0.1, 0.3):
        nm = max(1, round(mf * M * 10))
        perm = rng.permutation(M * 10)
        hi, mi = np.sort(perm[nm:]).astype(np.int32), np.sort(perm[:nm]).astype(np.int32)
        loc = np.empty(M * 10, np.int32); loc[hi] = np.arange(hi.size); loc[mi] = -(np.arange(mi.size) + 1)
        HI, MI, LOC = mx.array(hi), mx.array(mi), mx.array(loc)
        yh0 = mx.zeros((hi.size, K), mx.bfloat16)

        def hits(x, li):
            y = TG.tier_glu_pairs(x, *POOLS[li % NLP], R[li][0], R[li][1], HI)
            return y[:M] if y.shape[0] >= M else mx.broadcast_to(y[:1], (M, K))

        def misses_c(x, li):
            ym = TG.tier_glu_pairs(x, *POOLS[li % NLP], R[li][0], R[li][1], MI)
            (o,) = TG._kc()(inputs=[yh0, ym, LOC, R[li][2].reshape(-1)], template=[("T", mx.bfloat16), ("D_C", K), ("NR_C", 10), ("LY_C", 8)],
                            grid=(K, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, K)], output_dtypes=[mx.bfloat16])
            return o

        out.append(f"miss {mf:.0%}: hits {timed(hits):.3f} | misses+combine {timed(misses_c):.3f}")
    print(f"M={M}: ms/layer  " + "  ".join(out), flush=True)
