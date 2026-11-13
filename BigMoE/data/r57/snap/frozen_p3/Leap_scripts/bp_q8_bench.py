"""BP-2a (Leap, Wed 2026-11-04 16:3x; TABLE_350 'BP-2'): the drafter's projections at its real shapes and row counts, bf16 vs
quantized, to price an 8-bit drafter (half of bf16's bytes) before building a kernel: does MLX's own quantized_matmul at
16 rows run near its bytes floor at 8 bits?
Shapes (DFlash2: hidden 2560, 32 q heads / 8 kv heads x 128, ffn 7680; fc 12800 -> 2560 on the 8 new context rows):
q 4096x2560, k / v 1024x2560, o 2560x4096, gate / up 7680x2560, down 2560x7680 at 16 rows; fc 2560x12800 at 8 rows.
Arms: bf16 x @ W.T (nn.Linear), bf16 x @ Wt (R-78's form), MLX quantized_matmul 8-bit g64 and 4-bit g64, and Leap's qmm16c
(4-bit 16-row NAX kernel, REF2's). Per shape: us per call (20 independent calls per eval, median of 9 rounds) and GB/s on
the weight bytes; then the per-layer and 5-layer totals.
usage: [BigMoE venv] bp_q8_bench.py -> ~/Leap/data/bp_q8_bench.txt (GPU, MEM ~1.5 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))

OUT = os.path.expanduser("~/Leap/data/bp_q8_bench.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nBP-2a {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
try:
    import qmm26
    QMM16C = getattr(qmm26, "qmm16c", None)
except Exception:
    QMM16C = None
if QMM16C is None:
    try:
        import ref2
        QMM16C = ref2.qmm16c
    except Exception:
        QMM16C = None
SHAPES = [("q", 16, 4096, 2560, 1), ("k+v", 16, 1024, 2560, 2), ("o", 16, 2560, 4096, 1), ("gate+up", 16, 7680, 2560, 2),
          ("down", 16, 2560, 7680, 1), ("fc (ctx)", 8, 2560, 12800, 0)]
mx.random.seed(1)
tot = {}
for name, M, N, K, per_layer in SHAPES:
    W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    Wt = mx.contiguous(W.T)
    x = (mx.random.normal((1, M, K)) * 0.5).astype(mx.bfloat16)
    q8 = mx.quantize(W, group_size=64, bits=8)
    q4 = mx.quantize(W, group_size=64, bits=4)
    mx.eval(W, Wt, x, *q8, *q4)
    arms = {"bf16 x@W.T": (lambda: x @ W.T, N * K * 2),
            "bf16 x@Wt": (lambda: x @ Wt, N * K * 2),
            "q8 mlx": (lambda: mx.quantized_matmul(x, *q8, transpose=True, group_size=64, bits=8), N * K * 1.0625),
            "q4 mlx": (lambda: mx.quantized_matmul(x, *q4, transpose=True, group_size=64, bits=4), N * K * 0.5625)}
    if QMM16C is not None and M == 16:
        arms["q4 qmm16c"] = (lambda: QMM16C(x.reshape(M, K), *q4), N * K * 0.5625)
    for f, _ in arms.values():
        mx.eval([f() for _ in range(4)])
    T = {k: [] for k in arms}
    for r in range(9):
        for k in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval([arms[k][0]() for _ in range(20)]); mx.synchronize()
            T[k].append((time.perf_counter() - t0) / 20)
    row = []
    for k, (f, b) in arms.items():
        t = float(np.median(T[k]))
        row.append(f"{k} {1e6 * t:.1f} us ({b / t / 1e9:.0f} GB/s)")
        tot.setdefault(k, [0.0, 0.0])
        mult = per_layer if per_layer else 0
        tot[k][0] += 1e3 * t * mult * 5
        tot[k][1] += 1e3 * t if per_layer == 0 else 0.0
    say(f"{name:9s} M {M:2d} N {N:5d} K {K:5d}: " + "; ".join(row))
say("per propose (5 layers' projections at 16 rows + fc at 8 rows): " + ", ".join(
    f"{k} {v[0] + (v[1] if v[1] else tot['bf16 x@W.T'][1]):.2f} ms" for k, v in tot.items()))
