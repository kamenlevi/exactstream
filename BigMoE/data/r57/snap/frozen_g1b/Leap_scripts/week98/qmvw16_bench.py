"""QMVW16 (CHECK55 16-row lever (a); Leap, Sun 15 Nov 2026 11:4x): REF2's 16-row calls on the small shapes run MLX's own
qmv_wide (k_lanes 8; one32 probe). Per-row bits depend only on k_lanes, so the same kernel launched with fewer vectors per
threadgroup (more threadgroups, more parallel weight streams) keeps every bit. Is it faster than MLX's own launch at 16
rows? Per shape: bits (vs MLX's padded 16-row call) and GPU time per call for MLX's call vs vecs_per_tg 1 / 2 / 4 / 8 /
16, interleaved, many calls per timed graph.
usage: [BigMoE venv] qmvw16_bench.py   (MEM < 1 GB; share lock beside nothing heavy, or a quiet ticket for clean ms)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts/week98"))
import one32

SHAPES = [("router (8-bit)", 512, 2560, 8, 48), ("shared gate|up|gate (K-73)", 1281, 2560, 4, 48),
          ("att k / v_proj", 512, 2560, 4, 24), ("gdn b|a", 96, 2560, 4, 36), ("indexer qk", 640, 2560, 4, 12),
          ("8-bit 640", 640, 2560, 8, 12)]
VPTS = (1, 2, 4, 8, 16)
REP = 64


def say(s):
    print(s, flush=True)


def timed(fn, xs):
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn(x) for x in xs]
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / len(xs)


say(f"QMVW16 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; {REP} calls per timed graph, 7 interleaved rounds")
tot_mlx, tot_best = 0.0, 0.0
for name, N, K, bits, per_check in SHAPES:
    mx.random.seed(N + K + bits)
    wq, s, b = mx.quantize(mx.random.normal((N, K), dtype=mx.bfloat16) * 0.02, group_size=64, bits=bits)
    xs = [mx.random.normal((16, K), dtype=mx.bfloat16, key=mx.random.key(i)) * 0.7 for i in range(REP)]
    mx.eval(wq, s, b, xs)
    q = lambda x: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
    forms = {"mlx": q}
    for v in VPTS:
        forms[f"vpt{v}"] = (lambda v_: lambda x: one32.qmvw32(x, wq, s, b, bits, 8, v_, rows=16))(v)
    ok = {}
    for f, fn in forms.items():
        if f == "mlx":
            continue
        eq = mx.all(fn(xs[0]) == q(xs[0]))
        ok[f] = bool(eq.item())
    T = {f: [] for f in forms}
    for r in range(7):
        order = list(forms)[r % len(forms):] + list(forms)[:r % len(forms)]
        for f in order:
            T[f].append(timed(forms[f], xs))
    med = {f: 1e6 * float(np.median(T[f][1:])) for f in forms}
    best = min((f for f in forms if f != "mlx"), key=lambda f: med[f])
    tot_mlx += med["mlx"] * per_check / 1e3
    tot_best += med[best] * per_check / 1e3
    say(f"  {name:28s} N {N:5d} {bits}-bit: mlx {med['mlx']:6.1f} us | " + " ".join(
        f"{f} {med[f]:6.1f}{'' if ok[f] else ' (BITS DIFFER)'}" for f in forms if f != "mlx")
        + f" | best {best} {med['mlx'] / med[best]:.2f}x")
    del wq, s, b, xs
say(f"per 16-row check on the engine (calls per check as listed): MLX {tot_mlx:.2f} ms vs best launch {tot_best:.2f} ms "
    f"({tot_mlx - tot_best:+.2f} ms); isolated-call times, an upper bound on what the check sees")
