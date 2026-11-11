"""R-68 INPROJ screen (Leap, Sun 2026-11-01; shared GPU, rough): MTPLX_FUSED_GDN_INPROJ's one concatenated quantized matmul
vs the four GDN input projections, both through REF2's 16-row routing (ref2._qmm: qmm16c for N >= 1024, MLX's own 16-row
call padded otherwise). Synthetic 4-bit g64 weights at the 125B's GDN shapes: qkv 10240, z 6144, b 48, a 48 rows x 2560.
Per path: ms per call (8 chained calls per eval, best of 3 medians) and x 36 GDN layers per check; bitwise check of the
qkv / z parts (expected equal: qmm16c's per-column order does not depend on N) and of b / a (expected to differ).
Usage: [BigMoE venv] r68_inproj_test.py -> ~/Leap/data/r68_inproj.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
ref2._ORIG.setdefault("qmm", mx.quantized_matmul)
ref2._ON[0] = True
qmm = ref2._qmm
mx.random.seed(0)
K = 2560
shapes = {"qkv": 10240, "z": 6144, "b": 48, "a": 48}
W = {}
for nm, n in shapes.items():
    w = (mx.random.normal((n, K)) * 0.02).astype(mx.bfloat16)
    W[nm] = mx.quantize(w, group_size=64, bits=4)
mx.eval(W)
PAD = int(os.environ.get("R68_PAD", "32"))            # rows of zeros so N % 64 == 0 (16480 + 32 = 16512 -> qmm16c)
zp = mx.quantize(mx.zeros((max(PAD, 64), K), mx.bfloat16), group_size=64, bits=4)
fw = mx.concatenate([W[n][0] for n in shapes] + ([zp[0][:PAD]] if PAD else []))
fs = mx.concatenate([W[n][1] for n in shapes] + ([zp[1][:PAD]] if PAD else []))
fb = mx.concatenate([W[n][2] for n in shapes] + ([zp[2][:PAD]] if PAD else []))
mx.eval(fw, fs, fb)
x = (mx.random.normal((1, 16, K)) * 0.5).astype(mx.bfloat16)
def sep():
    return [qmm(x, *W[n], transpose=True, group_size=64, bits=4) for n in shapes]
def fused():
    y = qmm(x, fw, fs, fb, transpose=True, group_size=64, bits=4)
    o = np.cumsum([0] + list(shapes.values()))
    return [y[..., o[i]:o[i + 1]] for i in range(4)]
a, b = sep(), fused()
mx.eval(a, b)
eqs = {n: bool(mx.all(a[i].view(mx.uint16) == b[i].view(mx.uint16)).item()) for i, n in enumerate(shapes)}
say(f"R-68 (pad {PAD}): REF2 routing counts {ref2.counts()}; bitwise separate == fused per part: {eqs}")
def bench(f, chain=8, n=40):
    for _ in range(5):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / chain
ts_, tf_ = [], []
for _ in range(3):
    ts_.append(bench(sep)); tf_.append(bench(fused))
s_, f_ = 1e3 * min(ts_), 1e3 * min(tf_)
say(f"separate 4 calls {s_:.3f} ms, fused 1 call {f_:.3f} ms per GDN layer ({s_ / f_:.2f}x); x 36 layers: {36 * (s_ - f_):+.2f} ms "
    f"per check saved (chained, so dispatch gaps are mostly hidden: a lower bound)")
open(os.path.expanduser(f"~/Leap/data/r68_inproj_pad{PAD}.txt"), "w").write("\n".join(lines) + "\n")
