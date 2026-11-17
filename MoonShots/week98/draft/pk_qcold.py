"""PK: cold (DRAM) rate of the stock 8-bit qmv_fast vs prefetched variants (MoonShots, Tue 17 Nov 2026).
Each timed call follows a 512 MB flush read in the same command stream; the GEMV's own time = (flush + gemv) - flush.
usage: ~/BigMoE/.venv/bin/python -I -B pk_qcold.py"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/draft"))
import mlx.core as mx                                      # noqa: E402
import pk_qpf                                              # noqa: E402

FL = mx.random.randint(0, 2 ** 31, (128 * 1024 * 1024,), dtype=mx.uint32)   # 512 MB
fk = mx.fast.metal_kernel(name="pk_flush", input_names=["a"], output_names=["o"], source="""
  uint gid = thread_position_in_grid.x, gsz = threads_per_grid.x; uint4 acc = 0;
  const device uint4* p = (const device uint4*)a;
  for (uint i = gid; i < 33554432u; i += gsz) acc ^= p[i];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) o[gid & 255u] = 1u;""")


def flush():
    return fk(inputs=[FL], grid=(65536, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(256,)], output_dtypes=[mx.uint32])[0]


def med(f, n=9):
    ts = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(f())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2]


for name, N, K, C in (("q_proj", 12288, 2560, 16), ("o_proj", 2560, 6144, 32), ("fc", 2560, 2560, 64)):
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    W = []
    for c in range(C):                                     # C distinct copies: each call reads cold weights
        wq, sc, bi = mx.quantize(mx.random.normal((N, K)).astype(mx.bfloat16) * 0.05, group_size=64, bits=8)
        W.append((wq, sc, bi))
    mx.eval(W)
    mb = (W[0][0].nbytes + W[0][1].nbytes + W[0][2].nbytes) / 1e6
    row = []
    rs = mx.fast.metal_kernel(name=f"pkc_rowstream_{N}_{K}", input_names=["w", "scales", "biases", "x"], output_names=["y"],
                              source=f"""
  // NOT stock order: one simdgroup per row; lanes read the row's bytes as contiguous uint4 (stream-friendly)
  const uint sgid = simdgroup_index_in_threadgroup, ln = thread_index_in_simdgroup;
  const uint row = threadgroup_position_in_grid.x * 8u + sgid;
  const device uint4* wr = (const device uint4*)((const device uint8_t*)w + size_t(row) * {K}u);
  float acc = 0.0f;
  for (uint c = ln; c < {K // 16}u; c += 32u) {{
    uint4 q = wr[c];
    uint k0 = c * 16u;
    float s = float(scales[row * {K // 64}u + k0 / 64u]), b = float(biases[row * {K // 64}u + k0 / 64u]);
    float d = 0.0f, xs = 0.0f;
    for (uint u = 0; u < 4u; ++u) for (uint by = 0; by < 4u; ++by) {{
      float xv = float(x[k0 + u * 4u + by]);
      d += xv * float((q[u] >> (8u * by)) & 0xFFu); xs += xv; }}
    acc += s * d + b * xs;
  }}
  acc = simd_sum(acc);
  if (ln == 0u) y[row] = bfloat16_t(acc);""")
    def runrs():
        return [rs(inputs=[wq, sc, bi, x], grid=(N // 8 * 256, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(1, N)], output_dtypes=[mx.bfloat16])[0] for wq, sc, bi in W]
    t = med(runrs) / C
    row = [f"ROW-STREAM (not bitwise) {t * 1e6:6.1f} us ({mb / t / 1e3:5.1f} GB/s)"]
    for fn, pfd in (("qmv_fast_impl", 0),):
        k = mx.fast.metal_kernel(name=f"pkc2_{fn}{pfd}_{N}_{K}", input_names=["w", "scales", "biases", "x"],
                                 output_names=["y"], header=pk_qpf.header(max(pfd, 1)),
                                 source=f"{fn}<bfloat16_t, 64, 8>(w, scales, biases, x, y, {K}, {N}, threadgroup_position_in_grid, "
                                 "simdgroup_index_in_threadgroup, thread_index_in_simdgroup);")
        def run():
            return [k(inputs=[wq, sc, bi, x], grid=(32, N // 8 * 2, 1), threadgroup=(32, 2, 1),
                      output_shapes=[(1, N)], output_dtypes=[mx.bfloat16])[0] for wq, sc, bi in W]
        t = med(run) / C
        row.insert(0, f"{fn.replace('qmv_fast_', '')}{pfd if pfd else ''} {t * 1e6:6.1f} us ({mb / t / 1e3:5.1f} GB/s)")
    print(f"{name:7s} {mb:5.1f} MB cold x {C}: " + "; ".join(row), flush=True)
    del W
