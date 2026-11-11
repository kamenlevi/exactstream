"""K3C diagnostic (Leap, Fri 2026-11-06 09:4x): is hc8r's kernel 8 (k3c, the HC8 up-mix: 27.4 us per call chained, 143 GB/s, 96
calls per check) bound by bytes or by latency? Dependent chains of 48 calls at the live call's shapes (LR 320, HC 4, D 2560,
8-bit uq [10240, 80] uint32, us / ub [10240, 5] bf16), cold (48 distinct weight copies, 184 MB) vs warm (one copy, 48 times:
3.8 MB stays in the SLC). warm ~ cold -> latency-bound (prologue / tail / few threadgroups): restructuring pays.
warm << cold -> bytes: little left. -> ~/Leap/data/k3c_diag.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
import mlx.core as mx
import hc8r

OUT = os.path.expanduser("~/Leap/data/k3c_diag.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nK3C diagnostic {time.strftime('%a %H:%M:%S')} ({os.environ.get('K3_COND', 'share lock')}; MLX {mx.__version__})")
LR, HC, D = 320, 4, 2560
HCD = HC * D
BF = mx.bfloat16
rng = np.random.default_rng(3)
K3 = hc8r._kern(8, None)
clock = time.perf_counter


def weights():
    uq = mx.array(rng.integers(0, 2 ** 32, (HCD, LR // 4), dtype=np.uint64).astype(np.uint32))
    us = mx.array((rng.random((HCD, LR // 64)) * 0.01).astype(np.float32)).astype(BF)
    ub = mx.array((rng.random((HCD, LR // 64)) * 0.01 - 0.005).astype(np.float32)).astype(BF)
    wn = mx.array(rng.random(HCD).astype(np.float32)).astype(BF)
    mx.eval(uq, us, ub, wn)
    return uq, us, ub, wn


def acts():
    mix1 = mx.array(rng.standard_normal((16, LR)).astype(np.float32)).astype(BF)
    hw = mx.array(rng.standard_normal((16, HCD)).astype(np.float32)).astype(BF)
    rinv = mx.array(rng.random((16, HC)).astype(np.float32) + 0.5)
    xsum = mx.array(rng.standard_normal((16, LR // 64)).astype(np.float32))
    mx.eval(mix1, hw, rinv, xsum)
    return mix1, hw, rinv, xsum


def call(w, a, mix1):
    uq, us, ub, wn = w
    _, hw, rinv, xsum = a
    (y,) = K3(inputs=[mix1, uq, us, ub, hw, rinv, wn, xsum], template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC)],
              grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1), output_shapes=[(16, D)], output_dtypes=[BF])
    return y


def chain(W, A):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for w, a in zip(W, A):
            m = a[0] + (prev * 0).astype(BF)
            if calls:
                prev = call(w, a, m).reshape(-1)[:1].astype(mx.float32)
            else:
                prev = m.reshape(-1)[:1].astype(mx.float32)
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(W)


Wc = [weights() for _ in range(48)]
A = [acts() for _ in range(48)]
Ww = [Wc[0]] * 48
mb = (HCD * LR + 2 * HCD * (LR // 64) * 2 + HCD * 2 + 16 * HCD * 2) / 1e6
chain(Wc, A); chain(Ww, A)
rc, rw = [], []
for r in range(9):
    for which in ((0, 1) if r % 2 == 0 else (1, 0)):
        (rc if which == 0 else rw).append(chain(Wc if which == 0 else Ww, A))
tc, tw = float(np.median(rc)), float(np.median(rw))
say(f"  k3c chained over 48 calls ({mb:.2f} MB per call incl. hw): cold {tc:.1f} us ({mb * 1e-3 / max(1e-9, tc * 1e-6):.0f} GB/s), "
    f"warm {tw:.1f} us; warm / cold {tw / tc:.2f} -> {'latency-bound: restructure' if tw / tc > 0.7 else 'bytes-bound'}")
