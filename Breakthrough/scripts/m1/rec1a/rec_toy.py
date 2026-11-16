"""rec_toy.py (Breakthrough, MODULE 1a): mechanics test of the recorder on a toy MLX graph (a 32 x 2560 @ 2560 x 1024 bf16
matmul, softmax, sigmoid). (a) a replay with the recorded inputs reproduces the recorded output bit for bit; (b) new inputs
written in place into the recorded input array, then a replay == MLX's own forward on them, bit for bit. Tiny memory."""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlrec as R
R.load()
import mlx.core as mx
import numpy as np

mx.set_cache_limit(0)
rng = np.random.default_rng(20261116)
x0 = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
W = (mx.array(rng.standard_normal((2560, 1024)).astype(np.float32)) * 0.02).astype(mx.bfloat16)
mx.eval(x0, W)


def f(x):
    h = x @ W
    return mx.softmax(h.astype(mx.float32), axis=-1) * 2 + mx.sigmoid(h).astype(mx.float32)


mx.synchronize()
R.start()
y = f(x0)
mx.eval(y)
mx.synchronize()
R.stop()
print("record:", R.counts())
y_rec = np.array(y).copy()
rc, st = R.replay(0)
print(f"(a) replay, same inputs: rc {rc} {st}; bitwise == recorded: {np.array_equal(np.array(y), y_rec)}")


def bits(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint16)


for k in range(3):
    x1 = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x1)
    ref = f(x1)
    mx.eval(ref)
    ref_np = np.array(ref).copy()
    bits(x0)[:] = bits(x1)
    rc, st = R.replay(0)
    out = np.array(y)
    print(f"(b{k}) new inputs: rc {rc}; bitwise == MLX: {np.array_equal(out, ref_np)} (max |diff| "
          f"{np.abs(out - ref_np).max():.3g}); cpu encode {st['cpu_encode_us']:.1f} us, commit+wait "
          f"{st['cpu_commit_wait_us']:.1f} us, gpu {st['gpu_us']:.1f} us, {st['dispatches']} dispatches")
rc, st = R.replay(1)
print(f"serial replay: rc {rc}; bitwise == last MLX: {np.array_equal(np.array(y), ref_np)}; {st}")

# ---- ICB form: build from the record, then new inputs in place -> one execute vs MLX
n, err = R.build_icb()
print(f"ICB build: {n} commands {('(error: ' + err + ')') if n < 0 else ''}")
if n > 0:
    for k in range(3):
        x1 = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(x1)
        ref = f(x1)
        mx.eval(ref)
        ref_np = np.array(ref).copy()
        bits(x0)[:] = bits(x1)
        np.frombuffer(memoryview(y).cast("B"), dtype=np.uint8)[:] = 0
        rc, st = R.replay_icb()
        out = np.array(y)
        print(f"(icb{k}) rc {rc}; bitwise == MLX: {np.array_equal(out, ref_np)}; cpu encode {st['cpu_encode_us']:.1f} us, "
              f"commit+wait {st['cpu_commit_wait_us']:.1f} us, gpu {st['gpu_us']:.1f} us, {st['dispatches']} commands")
