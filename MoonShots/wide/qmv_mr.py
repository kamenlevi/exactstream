"""L1-01 prototype: multi-row qmv. MLX's qmv_fast (quantized.h, 0.32.2; 4-bit, group 64, K % 512 == 0, N % 8 == 0) gives a
threadgroup ONE input row and 8 output rows, so the weights are re-read for every row. Here a threadgroup takes up to MR
input rows and the same 8 output rows. Per 512-value K block each thread loads its 4 rows' weight words, scales and biases
once, then runs qmv's own load_vector / qdot arithmetic (tierglu's verbatim copies: tg_load_vector, tg_qdot4 on
register-held weight words) for each input row. Same operations in the same order per (row, output) -> bitwise equal to
MLX's qmv at any M, i.e. to today's decode / verify reference wherever MLX itself runs qmv.
qmv_mr(x [M, K] bf16, w, scales, biases, mr=16) -> y [M, N] bf16.
usage: qmv_mr.py [shapes...]   (GPU, ~1 min; quiet lock): bit checks vs MLX (per-row M=1 calls and plain M), us per call at
M = 1, 4, 8, 12, 16 over 48 weight instances (DRAM) for qmv_mr (mr 16 / 8), MLX plain, MLX in 5-row chunks."""
import os, sys, time
from functools import lru_cache
import mlx.core as mx
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Breakthrough/v3f"))
from tierglu import _HDR                                         # tg_load_vector / tg_qdot4: MLX's expressions verbatim

_SRC = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint3 tid = threadgroup_position_in_grid;
    constexpr int values_per_thread = 16, block_size = 512, results_per_simdgroup = 4;
    const int in_vec_size_w = K / 2, in_vec_size_g = K / 64;
    const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
    const int m0 = tid.x * MR;
    const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w + simd_lid * 8;
    const device T* sc = scales + out_row * in_vec_size_g + simd_lid / 4;
    const device T* bi = biases + out_row * in_vec_size_g + simd_lid / 4;
    const device T* xb = x + m0 * K + simd_lid * values_per_thread;
    float result[MR][results_per_simdgroup];
    for (int m = 0; m < MR; m++) for (int r = 0; r < results_per_simdgroup; r++) result[m][r] = 0;
    float x_thread[values_per_thread];
    for (int k = 0; k < K; k += block_size) {
        uint16_t wr[results_per_simdgroup][4];
        float s[results_per_simdgroup], b[results_per_simdgroup];
        for (int r = 0; r < results_per_simdgroup; r++) {
            const device uint16_t* wl = (const device uint16_t*)(ws + r * in_vec_size_w);
            for (int i = 0; i < 4; i++) wr[r][i] = wl[i];
            s[r] = sc[r * in_vec_size_g];
            b[r] = bi[r * in_vec_size_g];
        }
        for (int m = 0; m < MR; m++) {
            if (m0 + m >= M) break;
            float sum = tg_load_vector<T, float, values_per_thread, 4>(xb + m * K, x_thread);
            for (int r = 0; r < results_per_simdgroup; r++)
                result[m][r] += tg_qdot4<float, values_per_thread>(wr[r], x_thread, s[r], b[r], sum);
        }
        ws += block_size / 2;
        sc += block_size / 64;
        bi += block_size / 64;
        xb += block_size;
    }
    for (int m = 0; m < MR; m++) {
        if (m0 + m >= M) break;
        for (int r = 0; r < results_per_simdgroup; r++) {
            float v = simd_sum(result[m][r]);
            if (simd_lid == 0) y[(m0 + m) * N + out_row + r] = static_cast<T>(v);
        }
    }
"""


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(name="qmv_mr", input_names=["x", "w", "scales", "biases"], output_names=["y"],
                                source=_SRC, header=_HDR)


def qmv_mr(x, w, scales, biases, mr=16):
    M, K = x.shape
    N = w.shape[0]
    assert K % 512 == 0 and N % 8 == 0 and w.shape[1] * 8 == K, (M, K, N, w.shape)
    (y,) = _kernel()(inputs=[x, w, scales, biases], template=[("T", x.dtype), ("MR", mr), ("K", K), ("N", N), ("M", M)],
                     grid=(64 * ((M + mr - 1) // mr), N // 8, 1), threadgroup=(64, 1, 1), output_shapes=[(M, N)],
                     output_dtypes=[x.dtype])
    return y


def compile_graph():                                              # patches/sol_layer/compile_check.py (no GPU)
    w, s, b = mx.quantize(mx.zeros((64, 1024), mx.bfloat16), group_size=64, bits=4)
    for mr in (16, 8):
        qmv_mr(mx.zeros((5, 1024), mx.bfloat16), w, s, b, mr=mr)


SHAPES = {"in_proj": (16480, 2560), "out_proj": (2560, 6144), "router_sh": (1792, 2560), "lm_head": (248320, 2560)}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SHAPES)
    for name in names:
        N, K = SHAPES[name]
        NI = 48 if N * K < 100e6 else 2                             # instances round-robin (lm_head: 2 x 318 MB)
        W = [mx.quantize((mx.random.normal((N, K), key=mx.random.key(i)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
             for i in range(NI)]
        mx.eval(W)
        X = (mx.random.normal((16, K), key=mx.random.key(99)) * 0.5).astype(mx.bfloat16)
        mlx = lambda x, i=0: mx.quantized_matmul(x, *W[i], transpose=True, group_size=64, bits=4)
        R1 = mx.concatenate([mlx(X[r:r + 1]) for r in range(16)])  # MLX at M = 1 per row (qmv: the decode reference)
        mx.eval(R1); R1 = np.array(R1.view(mx.uint16))
        line = [f"{name} {N}x{K}:"]
        for mr in (16, 8):
            ok = all(np.array_equal(np.array(qmv_mr(X[:M], *W[0], mr=mr).view(mx.uint16)), R1[:M]) for M in range(1, 17))
            line.append(f"mr{mr} == MLX M=1 rows (M 1..16): {ok}")
        plain_eq = [M for M in range(1, 17) if np.array_equal(np.array(mlx(X[:M]).view(mx.uint16)), R1[:M])]
        line.append(f"MLX plain == its M=1 rows at M {plain_eq}")
        for M in (1, 4, 8, 12, 16):
            cells = []
            for lab, f in (("mr16", lambda i, x: [qmv_mr(x, *W[i], mr=16)]), ("mr8", lambda i, x: [qmv_mr(x, *W[i], mr=8)]),
                           ("mlx", lambda i, x: [mlx(x, i)]),
                           ("mlx_c5", lambda i, x: [mlx(x[a:a + 5], i) for a in range(0, x.shape[0], 5)])):
                best = 1e9
                for rep in range(6):
                    outs = [f(i % NI, X[:M]) for i in range(48)]
                    t0 = time.perf_counter(); mx.eval(outs)
                    if rep:
                        best = min(best, time.perf_counter() - t0)
                cells.append(f"{lab} {best / 48 * 1e6:.0f}")
            line.append(f"M={M}: " + ", ".join(cells) + " us")
        print("\n  ".join(line), flush=True)
