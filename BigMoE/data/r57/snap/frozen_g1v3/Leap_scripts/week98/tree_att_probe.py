"""tree_att_probe.py (Leap, Sun 15 Nov 2026 14:5x; TREE16 L2 step 1): which K partitioning does MLX's split-K NAX nn GEMM
use for the exact SDPA's P·V (probs [1, 1, 2, 16, K] @ v [1, 1, 1, K, 256], the fallback's matmul inside MTPLX's 2-head
chunks)? tree_att.pv_replica at candidate partition sizes (fixed sizes 128..4096, or K split into 1 / 2 / 4 / 8 / 16 / 32
partitions of 32-aligned size) vs mx.matmul on the same arrays, bitwise, per key count K (the run lengths ref2's classes
produce: below 128, the 128-511 band, the 512-key buckets). Also 32 rows (the one-call 32-row SDPA's P·V).
Line: the candidate(s) that match at every K give MLX's rule; no candidate -> the replica's per-step arithmetic differs
(look at the NAX tile shapes) before any tail work.
usage: [BigMoE venv] tree_att_probe.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts/week98"))
import tree_att as TA

BF = mx.bfloat16
KS = (96, 128, 384, 512, 1024, 1536, 2048, 2560, 3072, 4608)


def say(s):
    print(s, flush=True)


def cands(K):
    out = {f"ps{p}": p for p in (128, 256, 512, 1024, 2048, 4096)}
    for n in (1, 2, 4, 8, 16, 32):
        ps = -(-K // n)
        ps = -(-ps // 32) * 32
        out[f"np{n}"] = ps
    return out


t0 = time.time()
say(f"TREE-ATT-PROBE {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}")
summary = {}
for rows in (16, 32):
    for K in KS:
        res = {}
        for seed in (1, 2):
            k1, k2 = mx.random.split(mx.random.key(1000 * K + 10 * rows + seed))
            sc = mx.random.normal((1, 1, 2, rows, K), key=k1) * 3.0
            probs = mx.softmax(sc, axis=-1, precise=True).astype(BF)
            v = mx.random.normal((1, 1, 1, K, 256), key=k2).astype(BF)
            ref = mx.matmul(probs, v).reshape(2 * rows, 256)
            a = probs.reshape(2 * rows, K)
            for name, ps in cands(K).items():
                got = TA.pv_replica(a, v.reshape(K, 256), ps)
                eq = bool(mx.all(got == ref).item())
                res[name] = res.get(name, True) and eq
        ok = [n for n, e in res.items() if e]
        summary[(rows, K)] = set(ok)
        say(f"  rows {rows:2d} K {K:5d}: bitwise with {', '.join(ok) if ok else 'NONE'}")
for rows in (16, 32):
    common = set.intersection(*[summary[(rows, K)] for K in KS])
    say(f"rows {rows}: candidates bitwise at every K: {sorted(common) if common else 'none (the rule depends on K: see the lines)'}")
say(f"done in {time.time() - t0:.0f} s")
