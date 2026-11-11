"""PF-D v2 stage bench (Sun 8 Nov 18:0x, no model): one layer's tG pieces on synthetic pools of the real shapes. In-run tG was
~20 ms per layer (0.96 s per prompt) against the kernels' ~6 ms in pfd_unit.py: time the gathers, kernels, numpy -> MLX side
copies and the scatter into a zero scratch separately (each stage evaluated on its own)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import pflm  # noqa: E402

H, I, CM, CL = 2560, 640, 256, 136
nA, nB, nC = 48, 68, 36                                     # one chat layer's mean rows (pde1: A 2303 / B 3273 / C 1750 per 48)
rows3, ins = [I, I, H], [H, H, I]
shapes4 = []
for O, K in zip(rows3, ins):
    shapes4 += [((O, K // 8), mx.uint32), ((O, K // 64), mx.bfloat16), ((O, K // 64), mx.bfloat16)]
t = dict(rows3=rows3, shapes4=shapes4)
rng = np.random.default_rng(0)
P = ("gate_proj", "up_proj", "down_proj")
base = {p: {"w3": mx.array(rng.integers(0, 2**32, (CM, rows3[i], 6 * (ins[i] // 64)), dtype=np.uint32)),
            "s3": mx.array(rng.integers(0, 2**31, (CM, 9216), dtype=np.uint32))} for i, p in enumerate(P)}
side = {p: {"l1": mx.array(rng.integers(0, 2**32, (CL, rows3[i], 2 * (ins[i] // 64)), dtype=np.uint32)),
            "b": mx.array(rng.integers(0, 2**31, (CL, 8804), dtype=np.uint32))} for i, p in enumerate(P)}
for p in P:                                                 # p11 headers (the base exponent word) in range
    for d, k in ((base, "s3"), (side, "b")):
        a = np.array(d[p][k]); a[:, 11 * 800] = 100; d[p][k] = mx.array(a)
mx.eval(*[a for d in (base, side) for p in P for a in d[p].values()])
lsbB = {p: (rng.integers(0, 2**32, (nB, rows3[i], 2 * (ins[i] // 64)), dtype=np.uint32),
            rng.integers(0, 2**31, (nB, 8804), dtype=np.uint32)) for i, p in enumerate(P)}
gi = mx.array(rng.choice(CM, nA + nB, replace=False).astype(np.int32))
si = mx.array(rng.choice(CL, nA, replace=False).astype(np.int32))
N = nA + nB + nC
ix = mx.array(np.sort(rng.choice(N, nA + nB, replace=False)).astype(np.int32))


def stage(name, fn, reps=5):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
        best = min(best, time.perf_counter() - t0)
    print(f"  {name:34s} {best * 1e3:7.2f} ms")
    return out


print(f"one layer: A {nA}, B {nB}, C {nC} rows (scratch {N} rows)")
zs = stage("scratch zeros (9 pieces)", lambda: [mx.zeros((N, *sh), dt) for sh, dt in shapes4])
g = stage("gathers (w3, s3, side l1 / b)", lambda: [mx.take(base[p]["w3"], gi, axis=0) for p in P]
          + [mx.take(base[p]["s3"], gi, axis=0) for p in P] + [mx.take(side[p]["l1"], si, axis=0) for p in P]
          + [mx.take(side[p]["b"], si, axis=0) for p in P])
cp = stage("numpy -> MLX side copies (B)", lambda: [mx.array(lsbB[p][0]) for p in P] + [mx.array(lsbB[p][1]) for p in P])
w3s, s3s, l1A, bA = g[0:3], g[3:6], g[6:9], g[9:12]
l1s = [mx.concatenate([l1A[i], cp[i]], axis=0) for i in range(3)]
bs = [mx.concatenate([bA[i], cp[3 + i]], axis=0) for i in range(3)]
mx.eval(*l1s, *bs)
reb = stage("kernels (w4 x3, p11 x6)", lambda: pflm.rebuild_parts(t, w3s, s3s, l1s, bs))


def scat():
    s = [mx.zeros((N, *sh), dt) for sh, dt in shapes4]
    mx.eval(*s)
    for i in range(9):
        s[i][ix] = reb[i]
    return s


stage("zeros + scatter (9 pieces)", scat)


def whole():
    gg = ([mx.take(base[p]["w3"], gi, axis=0) for p in P], [mx.take(base[p]["s3"], gi, axis=0) for p in P])
    l1 = [mx.concatenate([mx.take(side[p]["l1"], si, axis=0), mx.array(lsbB[p][0])], axis=0) for p in P]
    b = [mx.concatenate([mx.take(side[p]["b"], si, axis=0), mx.array(lsbB[p][1])], axis=0) for p in P]
    r = pflm.rebuild_parts(t, gg[0], gg[1], l1, b)
    s = [mx.zeros((N, *sh), dt) for sh, dt in shapes4]
    mx.eval(*s)
    for i in range(9):
        s[i][ix] = r[i]
    return s


stage("WHOLE (as in prefill_call, + zeros)", whole)
rm = np.zeros((N, 3), np.int32)                              # v3: the same rows by rowmap, the pools read in place
kinds = rng.permutation(np.array([0] * nA + [1] * nB + [2] * nC))
ia = ib = ic = 0
gnp, snp = np.array(gi), np.array(si)
for k in range(N):
    if kinds[k] == 0:
        rm[k] = (0, gnp[ia], snp[ia]); ia += 1
    elif kinds[k] == 1:
        rm[k] = (1, gnp[nA + ib], ib); ib += 1
    else:
        rm[k] = (2, ic, 0); ic += 1
scrC = [mx.zeros((nC, *sh), dt) for sh, dt in shapes4]
mx.eval(*scrC)
bl1s = [mx.array(lsbB[p][0]) for p in P]
bbs = [mx.array(lsbB[p][1]) for p in P]
mx.eval(*bl1s, *bbs)
stage("v3 assemble (2 kernels x 3, pools in place)", lambda: pflm.assemble(t, rm, base, side, bl1s, bbs, scrC))
stage("v3 WHOLE (C zeros + copies + assemble)", lambda: (lambda sC: pflm.assemble(t, rm, base, side,
      [mx.array(lsbB[p][0]) for p in P], [mx.array(lsbB[p][1]) for p in P], sC))(
      [mx.zeros((nC, *sh), dt) for sh, dt in shapes4]))
