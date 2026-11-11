"""BW probe (Leap, Mon 2026-11-02; for C2's floor): the M5 Pro's achieved read bandwidth for a streaming kernel (uint4 loads,
XOR-fold, one word out per threadgroup) over 1 / 2 / 4 GB, and MLX's mx.sum over a bf16 array, quiet GPU. Median of 9.
Usage: [BigMoE venv] bw_probe.py -> ~/Leap/data/bw_probe.txt"""
import os, time
import numpy as np
import mlx.core as mx
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
SRC = r"""
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint per = N_C / (G_C * 256u);
    const device uint4* p = (const device uint4*)a + (size_t)tg * per * 256u;
    uint4 acc = uint4(0);
    for (uint i = 0; i < per; ++i) acc ^= p[i * 256u + t];
    threadgroup uint s[256];
    s[t] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { uint v = 0; for (int j = 0; j < 256; ++j) v ^= s[j]; out[tg] = v; }
"""
k = mx.fast.metal_kernel(name="bw_read", input_names=["a"], output_names=["out"], source=SRC)
for gb in (1, 2, 4):
    n4 = gb * (1 << 30) // 16                            # uint4 elements
    a = mx.random.randint(0, 1 << 30, (n4 * 4,), dtype=mx.uint32)
    mx.eval(a)
    for G in (1024, 4096):
        f = lambda: k(inputs=[a], template=[("N_C", n4), ("G_C", G)], grid=(G * 256, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(G,)], output_dtypes=[mx.uint32])[0]
        for _ in range(3):
            mx.eval(f())
        ts = []
        for _ in range(9):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
        say(f"read kernel {gb} GB, {G} threadgroups: {gb * (1 << 30) / np.median(ts) / 1e9:.0f} GB/s (best {gb * (1 << 30) / min(ts) / 1e9:.0f})")
    b = a.view(mx.bfloat16)
    for _ in range(3):
        mx.eval(b.sum())
    ts = []
    for _ in range(9):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(b.sum()); ts.append(time.perf_counter() - t0)
    say(f"mx.sum over {gb} GB bf16: {gb * (1 << 30) / np.median(ts) / 1e9:.0f} GB/s")
    del a, b
    mx.clear_cache()
open(os.path.expanduser("~/Leap/data/bw_probe.txt"), "w").write("\n".join(lines) + "\n")
