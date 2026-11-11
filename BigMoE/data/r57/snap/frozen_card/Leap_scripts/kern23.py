"""L1-23 (Leap, Sat 2026-10-24, pre-registered in ~/Leap/TABLE_350.md; MoonShots' L1-01b): bit-exact 16-row kernels for
the 125B's dense layers. Random weights with the 125B's dense shapes and per-check counts (MoonShots' data/b8mr/
denseinv.txt; ~3.4 GB, so every call reads cold weights), per shape:
  MLX 1 row (decode), MLX 16 rows (natural), MLX chunked as the exact path does today (4-bit 12 + 4; bf16 5-row chunks),
  qmm16c configurations (ks, nt) padded to 16 rows, the floor at 280 GB/s;
  bitwise: real rows equal across row counts 1, 2, 5, 8, 12, 16 and across padding contents (zeros / random), for
  qmm16c and for MLX's 16-row call (padded).
Then the best batch-invariant option per shape and the per-check totals.
Usage: [BigMoE venv] kern23.py -> ~/Leap/data/kern23.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
from qmm16 import qmm16c

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

# (name, kind, N, K, count per 16-row check, MLX exact chunk today)
SHAPES = [("in_proj_qkv", "q4", 10240, 2560, 36, 12), ("o_proj+out_proj", "q4", 2560, 6144, 48, 12),
          ("lm_head", "q4", 248320, 2560, 1, 12), ("in_proj_z", "q4", 6144, 2560, 36, 12),
          ("q_proj", "q4", 12288, 2560, 12, 12), ("shared gate/up", "q4", 640, 2560, 96, 16),
          ("shared down", "q4", 2560, 640, 48, 16), ("k/v_proj", "q4", 512, 2560, 24, 16),
          ("in_proj_a/b", "q4", 48, 2560, 72, 16), ("shared_expert_gate", "q4", 1, 2560, 48, 16),
          ("HC mix up", "bf16", 10240, 320, 97, 5), ("HC mix down", "bf16", 320, 10240, 97, 15),
          ("ple key", "bf16", 10240, 2560, 1, 5), ("ple value", "bf16", 2560, 2560, 1, 5),
          ("block_inject", "bf16", 4, 10240, 96, 16)]
CFGS = [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)]
mx.random.seed(23)
say("L1-23: dense layers of a 16-row check, random weights at the 125B's shapes and counts (cold), ms per check")
say(f"{'shape':20s} {'N x K':>13s} {'n':>3s} | floor | MLX 1 | MLX 16 | MLX exact today | qmm16c best (ks, nt) | "
    f"16-row padding-independent: MLX / qmm16c | best exact")
tot = {"floor": 0.0, "m1": 0.0, "today": 0.0, "best": 0.0}
sub = {k: {"floor": 0.0, "m1": 0.0, "today": 0.0, "best": 0.0} for k in ("q4", "bf16")}
for name, kind, N, K, cnt, chunk in SHAPES:
    if kind == "q4":
        Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(cnt)]
        nbytes = sum(a.nbytes for a in Ws[0])
        mlx = lambda x, W: mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4)
    else:
        Ws = [(mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16) for _ in range(cnt)]
        nbytes = Ws[0].nbytes
        mlx = lambda x, W: x @ W.T
    mx.eval(Ws)
    X = (mx.random.normal((32, K)) * 0.5).astype(mx.bfloat16)
    floor = cnt * nbytes / 280e9 * 1e3
    t1 = timeit(lambda: [mlx(X[:1], W) for W in Ws])
    t16 = timeit(lambda: [mlx(X[:16], W) for W in Ws])
    tch = timeit(lambda: [mx.concatenate([mlx(X[i:min(16, i + chunk)], W) for i in range(0, 16, chunk)]) for W in Ws])
    # padding independence of MLX's 16-row call: real rows of padded calls vs the all-real 16-row call
    def indep(fn):
        ref = bits(fn(X[:16], Ws[0]))
        for M in (1, 2, 5, 8, 12):
            for pad in (mx.zeros((16 - M, K), mx.bfloat16), X[16:32 - M]):
                if not np.array_equal(bits(fn(mx.concatenate([X[:M], pad]), Ws[0]))[:M], ref[:M]):
                    return False
        return True
    ok_mlx = indep(mlx)
    best_c, tc, ok_c = None, None, None
    if kind == "q4":
        Np = -(-N // 64) * 64
        if Np != N:                                    # pad output columns with zero weights (columns are independent)
            Wq = [tuple(mx.concatenate([a, mx.zeros((Np - N,) + a.shape[1:], a.dtype)]) for a in W) for W in Ws]
            mx.eval(Wq)
        else:
            Wq = Ws
        for ks, nt in CFGS:
            if Np % (32 * nt) or K % (64 * ks):
                continue
            t = timeit(lambda: [qmm16c(X[:16], *W, ks=ks, nt=nt) for W in Wq])
            if tc is None or t < tc:
                best_c, tc = (ks, nt), t
        ks, nt = best_c
        ok_c = indep(lambda x, W: qmm16c(x, *(Wq[0] if W is Ws[0] else W), ks=ks, nt=nt)[:, :N])
        # accuracy vs fp32 on one instance (max |err|): MLX vs qmm16c
        ref = X[:16].astype(mx.float32) @ mx.dequantize(*Ws[0], group_size=64, bits=4).astype(mx.float32).T
        e_m = float(mx.abs(mlx(X[:16], Ws[0]).astype(mx.float32) - ref).max())
        e_c = float(mx.abs(qmm16c(X[:16], *Wq[0], ks=ks, nt=nt)[:, :N].astype(mx.float32) - ref).max())
    opts = [t for t, ok in ((t16, ok_mlx), (tc, ok_c)) if t is not None and ok]
    best = min(opts) if opts else tch
    for d in (tot, sub[kind]):
        d["floor"] += floor; d["m1"] += t1; d["today"] += tch; d["best"] += best
    qtxt = f"{tc:6.2f} {best_c}" if tc is not None else "     -       "
    acc = f"; max|err| vs fp32 MLX {e_m:.3g} / qmm16c {e_c:.3g}" if kind == "q4" else ""
    say(f"{name:20s} {N:6d} x{K:5d} {cnt:3d} | {floor:5.2f} | {t1:5.2f} | {t16:6.2f} | {tch:6.2f} (c{chunk}) | {qtxt} | "
        f"{'yes' if ok_mlx else 'NO '} / {('yes' if ok_c else 'NO ') if ok_c is not None else '-  '} | {best:6.2f}{acc}")
    del Ws
for k, d in sub.items():
    say(f"{k} subtotal: floor {d['floor']:.1f} ms; MLX 1-row {d['m1']:.1f}; exact today {d['today']:.1f}; "
        f"best batch-invariant {d['best']:.1f}")
say("(8-bit router / indexer, ~1.7 ms today, not in this set.) L1-23 lines: kill if the 4-bit best > 25 ms, pass <= 15 ms")
say(f"TOTAL per 16-row check: floor {tot['floor']:.1f} ms; MLX 1-row decode {tot['m1']:.1f}; exact today (chunked) "
    f"{tot['today']:.1f}; best batch-invariant {tot['best']:.1f}")
open(os.path.expanduser("~/Leap/data/kern23.txt"), "w").write("\n".join(lines) + "\n")
