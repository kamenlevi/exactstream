"""NewIdeas R-52 price: MLX fresh-allocation cost on this Mac (Thu 10-29).
Arm F = the shipped plain path's pattern: mx.take of N expert slots from a pool into a NEW buffer, eval, free, mx.clear_cache()
(the next take must allocate and fault in fresh pages). Arm R = the same take without clear_cache (MLX reuses the cached buffer).
F - R = the allocation / page-fault cost per call; reported per GB. Sizes: 118 slots (~346 MB, R-50's 16-row scratch),
48 (~141 MB, a PF_GROUP 48 group), 36 (~106 MB, a 5-7-row check). Order F R R F per size (drift control)."""
import time, json, sys
import numpy as np
import mlx.core as mx

SLOT = 2932736                      # one expert record: MSB 2,162,688 + LSB 770,048 B
C = 144
pool = mx.zeros((C, SLOT // 4), dtype=mx.uint32)
mx.eval(pool)
rng = np.random.default_rng(0)
out = {"slot_bytes": SLOT, "C": C, "mlx": mx.__version__, "sizes": {}}

def run(arm, idxs, n):
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        a = mx.take(pool, idxs[i % len(idxs)], axis=0)
        mx.eval(a)
        del a
        if arm == "F":
            mx.clear_cache()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts

for n_slots in (118, 48, 36):
    idxs = [mx.array(np.sort(rng.choice(C, n_slots, replace=False)).astype(np.int32)) for _ in range(8)]
    run("R", idxs, 6)                                  # warm the kernel and the cache
    r = {"F": [], "R": []}
    for arm in ("F", "R", "R", "F", "F", "R"):
        ts = run(arm, idxs, 40)
        r[arm] += ts[3:]                               # drop each block's first calls (arm switch)
    gb = n_slots * SLOT / 1e9
    med = {k: float(np.median(v)) for k, v in r.items()}
    mean = {k: float(np.mean(v)) for k, v in r.items()}
    slow = [t for t in r["F"] if t > 2 * med["R"]]
    out["sizes"][n_slots] = {"GB": round(gb, 4), "n": len(r["F"]), "F_ms": round(med["F"], 3), "R_ms": round(med["R"], 3),
                             "F_minus_R_ms": round(med["F"] - med["R"], 3),
                             "alloc_ms_per_GB": round((med["F"] - med["R"]) / gb, 2),
                             "F_mean_ms": round(mean["F"], 3), "R_mean_ms": round(mean["R"], 3),
                             "mean_alloc_ms_per_GB": round((mean["F"] - mean["R"]) / gb, 2),
                             "copy_ms_per_GB": round(mean["R"] / gb, 2),
                             "F_slow_share": round(len(slow) / len(r["F"]), 3),
                             "F_slow_mean_ms": round(float(np.mean(slow)) if slow else 0.0, 3),
                             "F_p10_p90": [round(float(np.percentile(r["F"], q)), 3) for q in (10, 90)],
                             "R_p10_p90": [round(float(np.percentile(r["R"], q)), 3) for q in (10, 90)]}
    mx.clear_cache()
print(json.dumps(out, indent=1))
