"""PK: qmv_fast_impl with a prefetch (MoonShots, Tue 17 Nov 2026). The shipped quantized.h text, unchanged, plus, at the
top of each K-block iteration, one uint32 load per row of the lane's weight bytes PFD blocks ahead, folded into a dummy
accumulator. The arithmetic is the shipped text's, so the bits are too; the test below checks that against MLX's own
mx.quantized_matmul on every PK shape and times stock vs prefetched (standalone, rough).
usage: ~/BigMoE/.venv/bin/python -I -B pk_qpf.py [--pfd 2]"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/bet7"))
import mlx.core as mx                                      # noqa: E402
from fl_qmatch import QHEADER                              # noqa: E402


def _pf_text(pfd):
    s = QHEADER
    a = s.index("template <typename T, int group_size, int bits>\nMETAL_FUNC void qmv_fast_impl(")
    b = s.index("template <typename T, int group_size, int bits>", a + 10)
    f = s[a:b].replace("METAL_FUNC void qmv_fast_impl(", "METAL_FUNC void qmv_fast_pf_impl(")
    loop = "  for (int k = 0; k < in_vec_size; k += block_size) {\n"
    assert f.count(loop) == 1
    f = f.replace(loop, "  uint pfacc = 0u;\n" + loop + (
        "    if (k + %d * block_size < in_vec_size) {\n"
        "      for (int row = 0; row < results_per_simdgroup; row++)\n"
        "        pfacc ^= *(const device uint32_t*)(ws + row * in_vec_size_w + %d * (block_size * bytes_per_pack / pack_factor));\n"
        "    }\n") % (pfd, pfd))
    tail = "  for (int row = 0; row < results_per_simdgroup; row++) {\n    result[row] = simd_sum(result[row]);"
    assert f.count(tail) == 1
    f = f.replace(tail, "  if (pfacc == 0x9e3779b9u && in_vec_size < 0) y[0] = static_cast<T>(0);\n" + tail)
    return f


def header(pfd=2):
    return QHEADER + "\n" + _pf_text(pfd)


SHAPES = [  # (name, N, K, bits, group): every qmv_fast shape in a PK pop
    ("q_proj", 12288, 2560, 8, 64), ("k/v_proj", 512, 2560, 8, 64), ("o_proj", 2560, 6144, 8, 64),
    ("fc_emb/hid", 2560, 2560, 8, 64), ("hc_down", 320, 10240, 8, 64), ("router", 512, 2560, 8, 64),
    ("se_gate/up", 640, 2560, 8, 64), ("expert gate/up", 640, 2560, 4, 32), ("head 36864", 36864, 2560, 4, 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfd", type=int, default=2)
    a = ap.parse_args()
    H = header(a.pfd)
    mx.random.seed(1)
    for name, N, K, bits, gs in SHAPES:
        w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.05
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
        x = (mx.random.normal((1, K)) * mx.exp(mx.random.normal((1, K)))).astype(mx.bfloat16)   # heavy tails
        ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=gs, bits=bits)
        out = {}
        for fn in ("qmv_fast_impl", "qmv_fast_pf_impl"):
            k = mx.fast.metal_kernel(
                name=f"pkq_{fn}_{N}_{K}_{bits}_{gs}", input_names=["w", "scales", "biases", "x"], output_names=["y"],
                header=H, source=f"{fn}<bfloat16_t, {gs}, {bits}>(w, scales, biases, x, y, {K}, {N}, "
                "threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);")

            def call():
                return k(inputs=[wq, sc, bi, x], grid=(32, N // 8 * 2, 1), threadgroup=(32, 2, 1),
                         output_shapes=[(1, N)], output_dtypes=[mx.bfloat16])[0]
            y = call()
            mx.eval(y)
            ts = []
            for _ in range(7):
                mx.synchronize()
                t0 = time.perf_counter()
                ys = [call() for _ in range(50)]
                mx.eval(ys)
                mx.synchronize()
                ts.append((time.perf_counter() - t0) / 50)
            out[fn] = (y, sorted(ts)[3])
        eq_s = bool(mx.array_equal(out["qmv_fast_impl"][0], ref).item())
        eq_p = bool(mx.array_equal(out["qmv_fast_pf_impl"][0], ref).item())
        mb = (wq.nbytes + sc.nbytes + bi.nbytes) / 1e6
        ts_, tp_ = out["qmv_fast_impl"][1], out["qmv_fast_pf_impl"][1]
        print(f"{name:16s} {N:6d} x {K:5d} {bits}b g{gs}: {mb:6.2f} MB; stock == MLX {eq_s}, prefetched == MLX {eq_p}; "
              f"stock {ts_ * 1e6:7.1f} us ({mb / ts_ / 1e3:5.1f} GB/s), prefetched {tp_ * 1e6:7.1f} us "
              f"({mb / tp_ / 1e3:5.1f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
