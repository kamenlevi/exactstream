"""Does the Apple GPU lose bandwidth to address translation when a decode step reads many
scattered chunks (experts) instead of one contiguous region?
One kernel reads the same total bytes from a 6 GB buffer in chunks of C bytes, either in order
or at random chunk offsets. Random 3 MB chunks = what an MoE expert gather does."""
import time, sys
import numpy as np
import mlx.core as mx

TOTAL_GB = 6.0
READ_MB = 1024                                  # bytes read per pass
buf = mx.random.randint(0, 2**31, (int(TOTAL_GB * 1e9) // 16, 4)).astype(mx.uint32)   # random, so no zero-page shortcut
mx.eval(buf)
NE = buf.shape[0]

SRC = """
    // each threadgroup reads one chunk starting at offs[tg]. buf is bound as uint32, so a chunk
    // is CH*4 bytes (not CH*16); the printed GB/s divide by 4 for that.
    uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    uint base = offs[tg];
    uint4 acc = 0;
    for (uint i = t; i < CH; i += 256) acc ^= buf[base + i];
    acc.x ^= acc.y ^ acc.z ^ acc.w;
    acc.x = simd_sum(acc.x);
    if (t == 0) out[tg] = acc.x;
"""
kern = mx.fast.metal_kernel(name="chunkread", input_names=["buf", "offs"], output_names=["out"], source=SRC,
                            ensure_row_contiguous=False)
rng = np.random.default_rng(0)
print(f"{READ_MB} MB per pass from a {TOTAL_GB:.0f} GB buffer, one kernel")
for ch_kb in (16, 64, 256, 1024, 3072):
    ch = ch_kb * 1024 // 16
    n = READ_MB * 1024 // ch_kb
    res = {}
    for pat in ("seq", "rand"):
        if pat == "seq":
            start = rng.integers(0, NE - n * ch)
            offs = start + np.arange(n) * ch
        else:
            offs = rng.choice(NE // ch, n, replace=False) * ch
        o = mx.array(offs.astype(np.uint32))

        def run():
            mx.eval(kern(inputs=[buf, o], template=[("CH", ch)], grid=(n * 256, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(n,)], output_dtypes=[mx.uint32])[0])
        for _ in range(3):
            run()
        ts = []
        for _ in range(15):
            t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
        res[pat] = READ_MB / 1024 / sorted(ts)[7]
    print(f"chunk {ch_kb:>5} KB: sequential {res["seq"]*1.074/4:6.0f} GB/s, random {res["rand"]*1.074/4:6.0f} GB/s "
          f"({res['rand']/res['seq']:.3f}x)")
