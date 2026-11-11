"""HC-K2C GPU check (Leap, Fri 2026-11-06 08:5x): Fable B's hc_k2c (~/FableGod/fableb/kernels/hc_k2c.py, a DRAFT, read-only
use) against hc8r's kernel 7 (k2b, the live HC8-1P gate) with the live call's shapes (LR 320, HC 4, NP 16, NC 352, D
2560, eps 1e-6; part [16, 16, 352] fp32, ssq [16, 16] fp32).
  bits: mix1 / inject (bf16) max relative error and the share that differ; rinv / xsum (fp32) max relative error
  time: dependent chains of 96 calls (a check's HC reads), round-robin over the two kernels, median of 9 rounds
Line (Fable B's): PASS <= 6 us per call chained, KILL > 8; bits FAST (HC8 is a FAST read). -> ~/Leap/data/hc_k2c_check.txt"""
import importlib.util, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
import mlx.core as mx
import hc8r

_spec = importlib.util.spec_from_file_location("fb_hc_k2c", os.path.expanduser("~/FableGod/fableb/kernels/hc_k2c.py"))
FB = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FB)
OUT = os.path.expanduser("~/Leap/data/hc_k2c_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nHC-K2C check {time.strftime('%a %H:%M:%S')} ({os.environ.get('K2C_COND', 'share lock')}; MLX {mx.__version__})")
EPS, LR, HC, NP, D = 1e-06, 320, 4, 16, 2560
NT = (LR + HC + 31) // 32
NC = NT * 32
K2B = hc8r._kern(7, EPS)
K2C = mx.fast.metal_kernel(name="fb_hc_k2c", input_names=FB.IN, output_names=FB.OUT,
                           source=f"constexpr float EPS = {EPS!r}f;\n" + FB.SRC, header=FB.HDR)
BF = mx.bfloat16
rng = np.random.default_rng(4)


def call(K, part, ssq):
    return K(inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NC), ("D", D)],
             grid=(16 * NC, 1, 1), threadgroup=(NC, 1, 1),
             output_shapes=[(16, LR), (16, HC), (16, HC), (16, LR // 64)], output_dtypes=[BF, BF, mx.float32, mx.float32])


def inputs():
    part = mx.array(rng.standard_normal((NP, 16, NC)).astype(np.float32) * 4.0)
    ssq = mx.array(rng.uniform(100.0, 4000.0, (NP, 16)).astype(np.float32))
    mx.eval(part, ssq)
    return part, ssq


ok = True
for trial in range(4):
    part, ssq = inputs()
    a, b = call(K2B, part, ssq), call(K2C, part, ssq)
    mx.eval(a, b)
    errs = []
    for x, y, nm in zip(a, b, ("mix1", "inject", "rinv", "xsum")):
        X, Y = np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))
        e = float(np.abs(Y - X).max() / max(1e-30, np.abs(X).max()))
        errs.append(f"{nm} {e:.1e} ({(X != Y).mean() * 100:.1f} % differ)")
        ok &= e <= 1e-2
    say(f"  trial {trial}: " + "; ".join(errs))
clock = time.perf_counter
A = [inputs() for _ in range(96)]


def chain(K):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for part, ssq in A:
            s2 = ssq + (prev * 0)
            if calls:
                o = call(K, part, s2)
                prev = o[2].reshape(-1)[:1]
            else:
                prev = s2.reshape(-1)[:1]
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(A)


chain(K2B); chain(K2C)
rb, rc = [], []
for r in range(9):
    for which in ((0, 1) if r % 2 == 0 else (1, 0)):
        (rb if which == 0 else rc).append(chain(K2B if which == 0 else K2C))
tb, tc = float(np.median(rb)), float(np.median(rc))
say(f"  chained over 96 calls: k2b {tb:.2f} us, k2c {tc:.2f} us per call ({96 * (tb - tc) / 1e3:+.2f} ms per check saved)")
v = "PASS" if ok and tc <= 6.0 else ("KILL" if tc > 8.0 else "BETWEEN")
say(f"RESULT {v} (line: PASS <= 6 us chained and bits ok, KILL > 8; k2c {tc:.2f} us, bits {'ok' if ok else 'BAD'})")
