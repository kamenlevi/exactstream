"""PF-KBENCH (Leap, Sat 2026-11-07 23:5x; Kamen's MUST: prefill >= 500 tok/s; BigMoE's asks 1-3): MLX's 4-bit g64 kernels at
the 125B's prefill shapes and M = rows per call, random weights (the shapes and the kernel selection are what matter):
  dense  mx.quantized_matmul: head 2560 -> 248,320; GDN in_proj qkv|z 2560 -> 16,384; out_proj 6,144 -> 2,560;
         attention q 2560 -> 12,288; shared gate|up 2560 -> 1,280; shared down 640 -> 2,560
  experts mx.gather_qmm, sorted rhs (mlx_lm SwitchGLU's form): 512 experts, top 10, gate|up 2560 -> 1,280, down 640 -> 2,560
Per (kernel, M): ms per call (median of 5 after 2 warm-ups) and TFLOPS. Bits: row 0's output computed inside calls of
different M (the same input row placed first) must be equal for prefill to be batch-invariant across chunk sizes.
usage: python pf_kbench.py -> ~/Leap/data/pf_kbench.txt"""
import os
import time

import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/pf_kbench.txt")
MS = [int(m) for m in os.environ.get("PFK_MS", "1,16,64,256,1024,2048,4096,8192").split(",")]


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def tmed(fn, reps=5):
    for _ in range(2):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def qw(rng, n, k):
    w = mx.array((rng.standard_normal((n, k)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    mx.eval(wq, s, b)
    return wq, s, b


def dense(rng):
    shapes = [("head", 2560, 248320), ("gdn_inproj", 2560, 16384), ("out_proj", 6144, 2560), ("att_q", 2560, 12288),
              ("shared_gu", 2560, 1280), ("shared_down", 640, 2560)]
    for name, K, N in shapes:
        wq, s, b = qw(rng, N, K)
        X = mx.array(rng.standard_normal((max(MS), K)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(X)
        row0, line = None, []
        for M in MS:
            if name == "head" and M > 2048:
                line.append(f"M {M}: skip (logits {M * N * 2 / 1e9:.1f} GB)")
                continue
            x = X[:M]
            dt = tmed(lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4))
            y0 = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)[0]
            mx.eval(y0)
            same = None if row0 is None else bool(mx.array_equal(y0, row0))
            row0 = y0 if row0 is None else row0
            line.append(f"M {M}: {1e3 * dt:.3f} ms {2 * M * N * K / dt / 1e12:.2f} TF" + ("" if same is None else f" row0{'==' if same else '!='}"))
        say(f"  dense {name:12s} K {K:5d} N {N:6d} | " + " | ".join(line))
        del wq, s, b


def experts(rng):
    E, top, D, I = 512, 10, 2560, 640
    wq1, s1, b1 = qw(rng, E * 2 * I, D)
    wq1, s1, b1 = (a.reshape(E, 2 * I, -1) for a in (wq1, s1, b1))
    wq2, s2, b2 = qw(rng, E * D, I)
    wq2, s2, b2 = (a.reshape(E, D, -1) for a in (wq2, s2, b2))
    mx.eval(wq1, s1, b1, wq2, s2, b2)
    Mmax = max(MS)
    X = mx.array(rng.standard_normal((Mmax, D)).astype(np.float32)).astype(mx.bfloat16)
    ids_all = np.stack([rng.choice(E, top, replace=False) for _ in range(Mmax)]).astype(np.uint32)
    mx.eval(X)

    def run(M):
        ids = ids_all[:M].reshape(-1)
        order = np.argsort(ids, kind="stable")
        xs = X[:M][mx.array((order // top).astype(np.int32))][:, None, :]      # rows sorted by expert (SwitchGLU's form)
        ri = mx.array(ids[order])
        h = mx.gather_qmm(xs, wq1, s1, b1, rhs_indices=ri, transpose=True, group_size=64, bits=4, sorted_indices=True)
        g, u = mx.split(h, 2, axis=-1)
        a = (g * mx.sigmoid(g)) * u
        y = mx.gather_qmm(a, wq2, s2, b2, rhs_indices=ri, transpose=True, group_size=64, bits=4, sorted_indices=True)
        inv = np.argsort(order)
        return y[mx.array(inv.astype(np.int32))].reshape(M, top, D), order

    row0, line = None, []
    for M in MS:
        if M < 16:
            continue
        dt = tmed(lambda: run(M)[0])
        y = run(M)[0][0]
        mx.eval(y)
        same = None if row0 is None else bool(mx.array_equal(y, row0))
        row0 = y if row0 is None else row0
        fl = 2 * M * top * (D * 2 * I + I * D)
        line.append(f"M {M}: {1e3 * dt:.2f} ms {fl / dt / 1e12:.2f} TF" + ("" if same is None else f" row0{'==' if same else '!='}"))
    say(f"  experts gather_qmm (512 x [1280x2560 + 2560x640], top 10, sorted) | " + " | ".join(line))


if __name__ == "__main__":
    say(f"\nPF-KBENCH {time.strftime('%a %H:%M:%S')} ({os.environ.get('GS_COND', 'quiet lock')}; MLX {mx.__version__}); M {MS}")
    rng = np.random.default_rng(1)
    dense(rng)
    experts(rng)
