"""ONE32-BENCH (CHECK55 step 1 after C7b; Leap, Sun 15 Nov 2026 11:4x): which 17-32-row form is fastest per shape, with
the weights cold as in a real check? C7b: the paired one-weight-pass forms LOST 0.22 x A16 on the slice (halved / 8x
fewer threadgroups). Per shape this streams the engine's per-check set of DISTINCT weights (48-layer counts: GDN qkv|z x
36, out / o_proj x 48, ...; the head once at 318 MB), so each weight is read cold, and times every form on the same set:
  qmm16c shapes: split (2 x qmm16c, v32's path), qmm16c2 (tiles on the grid's y), qmm16c2i (tiles interleaved),
                 qmm16c2a (both tiles per simdgroup, one weight pass);
  small shapes:  split (2 x MLX's padded 16-row call), qmv_wide k_lanes 8 at 2 / 4 / 8 / 16 / 32 vectors per threadgroup;
  HC S4 (the 324 x 10240 bf16 mixer GEMM of each HC read, x 96): split (2 x leapk.s4_partials), s4_partials2,
                 s4_partials2i.
Inputs are 32 rows; independent calls in one graph per form; 7 interleaved rounds, median; reported as ms per 32-row
check for that shape (all its calls), with the 16-row form's time (1 x qmm16c / MLX / s4) for the R per shape.
Bits are one32_test / one32_hc_test's (every form here is bitwise per row there).
usage: [BigMoE venv] one32_bench.py   (MEM ~3 GB; quiet ticket for clean ms)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98m/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98m/Leap_scripts/week98"))
from qmm16 import qmm16c
import leapk as LK
import one32

BF = mx.bfloat16
C16 = [("gdn qkv|z", 16384, 2560, 36), ("out_proj / o_proj", 2560, 6144, 48), ("q_proj", 12288, 2560, 12),
       ("shared down", 2560, 640, 48), ("ple key", 10240, 2560, 1), ("ple value", 2560, 2560, 1),
       ("lm_head", 248320, 2560, 1)]
SMALL = [("router (8-bit)", 512, 2560, 8, 48), ("shared gate|up|gate", 1281, 2560, 4, 48), ("k / v_proj", 512, 2560, 4, 24),
         ("gdn b|a", 96, 2560, 4, 36), ("indexer qk", 640, 2560, 4, 12), ("8-bit 640", 640, 2560, 8, 12)]
ROUNDS = 7


def say(s):
    print(s, flush=True)


def bench(forms, ws, x):
    """forms: name -> f(x, W); ws: the per-check weight list; returns name -> ms for one pass over ws"""
    T = {f: [] for f in forms}
    names = list(forms)
    for r in range(ROUNDS):
        order = names[r % len(names):] + names[:r % len(names)]
        for f in order:
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval([forms[f](x, W) for W in ws])
            mx.synchronize()
            T[f].append(time.perf_counter() - t0)
    return {f: 1e3 * float(np.median(T[f][1:])) for f in forms}


def pad16(x):
    return x if x.shape[0] == 16 else mx.concatenate([x, mx.zeros((16 - x.shape[0], x.shape[1]), x.dtype)], 0)


t00 = time.time()
say(f"ONE32-BENCH {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; {ROUNDS} interleaved rounds, median of the last "
    f"{ROUNDS - 1}; ms per 32-row check for each shape's calls (cold distinct weights)")
tot = {"split": 0.0, "best": 0.0, "16": 0.0}
x_all = mx.random.normal((32, 6144), dtype=BF, key=mx.random.key(5)) * 0.7
for name, N, K, cnt in C16:
    ws = []
    for i in range(cnt):
        mx.random.seed(1000 + i)
        ws.append(mx.quantize(mx.random.normal((N, K), dtype=BF) * 0.02, group_size=64, bits=4))
        mx.eval(ws[-1])
    x = x_all[:, :K]
    forms = {"16": lambda x_, W: qmm16c(x_[:16], *W),
             "split": lambda x_, W: mx.concatenate([qmm16c(x_[:16], *W), qmm16c(x_[16:], *W)], 0),
             "c2": lambda x_, W: one32.qmm16c2(x_, *W), "c2i": lambda x_, W: one32.qmm16c2i(x_, *W),
             "c2a": lambda x_, W: one32.qmm16c2a(x_, *W)}
    t = bench(forms, ws, x)
    best = min(("c2", "c2i", "c2a"), key=lambda f: t[f])
    tot["split"] += t["split"]; tot["best"] += min(t["split"], t[best]); tot["16"] += t["16"]
    say(f"  {name:20s} N {N:6d} K {K:5d} x{cnt:2d}: 16-row {t['16']:6.3f} | split {t['split']:6.3f} | c2 {t['c2']:6.3f} | "
        f"c2i {t['c2i']:6.3f} | c2a {t['c2a']:6.3f} ms -> best {best} (R32 {min(t['split'], t[best]) / t['16']:.2f})")
    del ws
for name, N, K, bits, cnt in SMALL:
    ws = []
    for i in range(cnt):
        mx.random.seed(2000 + i)
        ws.append(mx.quantize(mx.random.normal((N, K), dtype=BF) * 0.02, group_size=64, bits=bits))
        mx.eval(ws[-1])
    x = x_all[:, :K]
    q = lambda x_, W: mx.quantized_matmul(x_, *W, transpose=True, group_size=64, bits=bits)
    forms = {"16": lambda x_, W: q(x_[:16], W), "split": lambda x_, W: mx.concatenate([q(x_[:16], W), q(x_[16:], W)], 0)}
    for v in (2, 4, 8, 16, 32):
        forms[f"vpt{v}"] = (lambda v_: lambda x_, W: one32.qmvw32(x_, *W, bits, 8, v_))(v)
    t = bench(forms, ws, x)
    cand = [f for f in forms if f.startswith("vpt")]
    best = min(cand, key=lambda f: t[f])
    tot["split"] += t["split"]; tot["best"] += min(t["split"], t[best]); tot["16"] += t["16"]
    say(f"  {name:20s} N {N:6d} K {K:5d} {bits}b x{cnt:2d}: 16-row {t['16']:6.3f} | split {t['split']:6.3f} | " +
        " | ".join(f"{f} {t[f]:6.3f}" for f in cand) + f" ms -> best {best} (R32 {min(t['split'], t[best]) / t['16']:.2f})")
    del ws
ws = []
for i in range(96):
    mx.random.seed(3000 + i)
    ws.append((mx.random.normal((324, 10240), dtype=BF) * 0.01,))
    mx.eval(ws[-1])
xh = mx.random.normal((1, 32, 10240), dtype=BF, key=mx.random.key(9))
forms = {"16": lambda x_, W: LK.s4_partials(x_[:, :16], W[0])[0],
         "split": lambda x_, W: mx.concatenate([LK.s4_partials(x_[:, :16], W[0])[0], LK.s4_partials(x_[:, 16:], W[0])[0]], 1),
         "s4x2t": lambda x_, W: one32.s4_partials2(x_, W[0])[0], "s4x2i": lambda x_, W: one32.s4_partials2i(x_, W[0])[0]}
t = bench(forms, ws, xh)
best = min(("s4x2t", "s4x2i"), key=lambda f: t[f])
tot["split"] += t["split"]; tot["best"] += min(t["split"], t[best]); tot["16"] += t["16"]
say(f"  {'HC S4 (mixer GEMM)':20s} N    324 K 10240 x96: 16-row {t['16']:6.3f} | split {t['split']:6.3f} | s4x2t "
    f"{t['s4x2t']:6.3f} | s4x2i {t['s4x2i']:6.3f} ms -> best {best} (R32 {min(t['split'], t[best]) / t['16']:.2f})")
say(f"TOTAL these calls per check: 16-row {tot['16']:.2f} ms | 32-row split {tot['split']:.2f} | 32-row best forms "
    f"{tot['best']:.2f} ({tot['best'] - tot['split']:+.2f} ms vs split) ({time.time() - t00:.0f} s; peak "
    f"{mx.get_peak_memory() / 1e9:.1f} GB)")
