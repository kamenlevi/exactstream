"""Round 31 candidate (v2b, not bit-identical): chunk-streaming small-M 4-bit matvec, y[m, n] = sum_k x[m, k] W[n, k] for
M = 1..4, MLX affine 4-bit g64 layout (w [N, K/8] u32, scales / biases [N, K/64]). Each simdgroup owns R consecutive
rows; its lanes stream the rows' bytes as contiguous 16-byte chunks (32 values each; a simdgroup reads 512 contiguous
bytes per step: fully coalesced, many loads in flight), x is read from device memory (cached), each lane accumulates R x M partial sums in fp32 and one simd_sum per (row, m) finishes the dot products.
The arithmetic order differs from MLX's (per-lane chunking, one reduction), so outputs differ in the last bits."""
from functools import lru_cache
import mlx.core as mx

_SRC = """
    constexpr int K = K_C, M = M_C, R = R_C, G = G_C;
    constexpr int CPR = K / 32;                                   // 16-byte chunks per row
    constexpr int STEPS = (R * CPR + 31) / 32;
    const uint nthr = threads_per_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = nthr / 32;
    const int row0 = (threadgroup_position_in_grid.x * nsg + sg) * R;
    if (row0 >= N_C) return;
    const device uint4* wb = (const device uint4*)w + (size_t)row0 * CPR;
    const device T* sb = sc + (size_t)row0 * (K / G);
    const device T* bb = bi + (size_t)row0 * (K / G);
    float acc[R][M];
    for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (int s = 0; s < STEPS; ++s) {
        const int c = lane + 32 * s;                              // flat chunk index over R rows
        if (c < R * CPR) {
            const uint4 q = wb[c];
            const int r = c / CPR, cc = c % CPR;
            const int g = (cc * 32) / G;
            const float scale = (float)sb[r * (K / G) + g], bias = (float)bb[r * (K / G) + g];
            const uint words[4] = {q.x, q.y, q.z, q.w};
            for (int m = 0; m < M; ++m) {
                const device T* xv = x + m * K + cc * 32;
                float dq = 0.0f, xsum = 0.0f;
                for (int wi = 0; wi < 4; ++wi) {
                    const uint wd = words[wi];
                    for (int v = 0; v < 8; ++v) {
                        const float xx = (float)xv[wi * 8 + v];
                        dq += (float)((wd >> (4 * v)) & 0xF) * xx;
                        xsum += xx;
                    }
                }
                const float contrib = scale * dq + bias * xsum;
                for (int rr = 0; rr < R; ++rr) acc[rr][m] += (rr == r) ? contrib : 0.0f;
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        for (int m = 0; m < M; ++m) {
            const float tot = simd_sum(acc[r][m]);
            if (lane == 0 && row0 + r < N_C) y[(size_t)m * N_C + row0 + r] = (T)tot;
        }
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="qmv_cs", input_names=["x", "w", "sc", "bi"], output_names=["y"], source=_SRC,
                                ensure_row_contiguous=True)


def qmv(x, w, s, b, rows_per_sg=4, sgs=8, group_size=64):
    """x [M, K] bf16 (1 <= M <= 4) -> [M, N] bf16."""
    M, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    tg_rows = rows_per_sg * sgs
    (y,) = _k()(inputs=[x, w, s, b], template=[("T", x.dtype), ("K_C", K), ("M_C", M), ("R_C", rows_per_sg),
                                               ("G_C", group_size), ("N_C", N)],
                grid=(((N + tg_rows - 1) // tg_rows) * 32 * sgs, 1, 1), threadgroup=(32 * sgs, 1, 1),
                output_shapes=[(M, N)], output_dtypes=[x.dtype])
    return y
