"""Mend item (c) measurement (Leap, Sat 2026-10-24): MLX's per-op cost on the CPU side (Python graph building + Metal
encoding + launch), the share a native per-check loop could remove. A chain of n small ops on [16, 2560] bf16 (GPU work
negligible): wall time per op for (1) eager graph build + eval, (2) build only, (3) the same chain through mx.compile,
and (4) small matmuls [16, 2560] x [2560, 256]. A 125B verify check runs ~2,000-3,000 kernels (48 layers x ~40-60), so
ops per check x cost per op = the per-check overhead to compare with a ~60 ms (kernel-optimised) check.
Usage: [BigMoE venv] ovh27.py -> ~/Leap/data/ovh27.txt"""
import os, time
import mlx.core as mx

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

x0 = (mx.random.normal((16, 2560)) * 0.1).astype(mx.bfloat16)
w = (mx.random.normal((256, 2560)) * 0.02).astype(mx.bfloat16)
mx.eval(x0, w)

def chain(x, n):
    for i in range(n):
        x = x * 1.0001 + 0.0001 if i % 2 else mx.tanh(x)
    return x

def mm_chain(x, n):
    acc = []
    for i in range(n):
        acc.append(x @ w.T)
    return mx.stack(acc)

cchain = {}
def compiled(n):
    if n not in cchain:
        cchain[n] = mx.compile(lambda x: chain(x, n))
    return cchain[n]

for n in (500, 2000):
    best = {}
    for rep in range(6):
        t0 = time.perf_counter(); y = chain(x0, n); t1 = time.perf_counter(); mx.eval(y); t2 = time.perf_counter()
        f = compiled(n); t3 = time.perf_counter(); z = f(x0); mx.eval(z); t4 = time.perf_counter()
        t5 = time.perf_counter(); m = mm_chain(x0, n // 4); mx.eval(m); t6 = time.perf_counter()
        if rep:
            for k, v in (("build", t1 - t0), ("eager", t2 - t0), ("compiled", t4 - t3), ("matmul", t6 - t5)):
                best[k] = min(best.get(k, 1e9), v)
    say(f"  chain of {n} ops: eager build+eval {best['eager'] / n * 1e6:.1f} us per op (build alone "
        f"{best['build'] / n * 1e6:.1f}); mx.compile {best['compiled'] / n * 1e6:.1f} us per op; {n // 4} small matmuls "
        f"{best['matmul'] / (n // 4) * 1e6:.1f} us each")
say("  per-check overhead estimate = ops per check (~2,000-3,000) x us per op; compare with a ~60 ms check")
open(os.path.expanduser("~/Leap/data/ovh27.txt"), "w").write("\n".join(lines) + "\n")
