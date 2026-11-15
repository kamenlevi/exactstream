"""L1-24 (Leap, Sat 2026-10-24): routed experts of a verify check read each DISTINCT expert once, with MLX's own per-row
arithmetic, so every (row, expert) output stays bit-identical to the engine's path (MLX's per-pair gather_qmv: gate/up
= qmv_fast, down (K 640) = qmv_impl with its clamped tail; exp24.py found the engine's three expert paths bitwise equal
to it at any row count). MoonShots' qmv_mr (wide/qmv_mr.py) did this for dense layers (bitwise, but slower than MLX's
16-row qmm there); here the rows of one expert are few (~2.2 at 16 rows) and MLX's sorted path runs at 56 % of the floor.
Kernel A: one threadgroup = one expert segment x 8 gate/up output rows; per 512-value K block the 4 weight rows of each
simdgroup are loaded once, then load_vector / qdot run for each of the segment's rows (MLX 0.32.2 expressions verbatim).
Kernel B: the same for down_proj with qmv_impl's 256-value blocks and tail. Between them MLX's nn.silu(gate) * up.
Segments are built on the GPU (argsort, cumsum, scatter-min): no host sync.
experts_gmr(x [R, K], gu (w, s, b), dn (w, s, b), ids [R, k]) -> [R, k, D], R <= 16."""
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn

_HDR = r"""
// MLX 0.32.2 quantized.h load_vector / load_vector_safe / qdot / qdot_safe for bits = 4, verbatim except that qdot reads
// its weight words from registers (as MoonShots' qmv_mr / Breakthrough's tierglu, measured bitwise equal to MLX)
template <typename T, typename U, int values_per_thread>
inline U g_load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < values_per_thread; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}
template <typename T, typename U, int values_per_thread>
inline U g_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  U sum = 0;
  for (int i = 0; i < N; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
  return sum;
}
template <typename U, int values_per_thread>
inline U g_qdot(const thread uint16_t* ws, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
template <typename U, int values_per_thread>
inline U g_qdot_safe(const thread uint16_t* ws, const thread U* x_thread, U scale, U bias, U sum, int N) {
  U accum = 0;
  for (int i = 0; i < (N / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

# A: gate/up, MLX qmv_fast_impl (values_per_thread 16, block 512; K % 512 == 0, N % 8 == 0)
_A = r"""
    constexpr int V = 16, BLK = 512, RPS = 4;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int s = threadgroup_position_in_grid.x;
    const int p0 = seg_start[s];
    if (p0 >= P) return;
    const int p1 = seg_start[s + 1];
    const int cnt = p1 - p0;
    const int e = sexp[p0];
    const int KW = K / 2, KG = K / 64;                       // bytes and groups per weight row
    const int out_row = threadgroup_position_in_grid.y * 8 + simd_gid * RPS;
    const device uint8_t* ws = (const device uint8_t*)w + ((size_t)e * N + out_row) * KW + simd_lid * 8;
    const device T* sc = scales + ((size_t)e * N + out_row) * KG + simd_lid / 4;
    const device T* bi = biases + ((size_t)e * N + out_row) * KG + simd_lid / 4;
    float result[MR][RPS];
    for (int m = 0; m < MR; m++) for (int r = 0; r < RPS; r++) result[m][r] = 0;
    float x_thread[V];
    for (int k = 0; k < K; k += BLK) {
        uint16_t wr[RPS][V / 4];
        float sv[RPS], bv[RPS];
        for (int r = 0; r < RPS; r++) {
            const device uint16_t* wl = (const device uint16_t*)(ws + r * KW);
            for (int i = 0; i < V / 4; i++) wr[r][i] = wl[i];
            sv[r] = sc[r * KG];
            bv[r] = bi[r * KG];
        }
        for (int m = 0; m < MR; m++) {
            if (m >= cnt) break;
            const device T* xr = x + (size_t)prow[p0 + m] * K + k + simd_lid * V;
            float sum = g_load_vector<T, float, V>(xr, x_thread);
            for (int r = 0; r < RPS; r++) result[m][r] += g_qdot<float, V>(wr[r], x_thread, sv[r], bv[r], sum);
        }
        ws += BLK / 2;
        sc += BLK / 64;
        bi += BLK / 64;
    }
    for (int m = 0; m < MR; m++) {
        if (m >= cnt) break;
        for (int r = 0; r < RPS; r++) {
            float v = simd_sum(result[m][r]);
            if (simd_lid == 0) y[(size_t)(p0 + m) * N + out_row + r] = static_cast<T>(v);
        }
    }
"""

# B: down, MLX qmv_impl (values_per_thread 8, block 256, clamped tail; N % 8 == 0 so used_out_row == out_row)
_B = r"""
    constexpr int V = 8, BLK = 256, RPS = 4;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int s = threadgroup_position_in_grid.x;
    const int p0 = seg_start[s];
    if (p0 >= P) return;
    const int p1 = seg_start[s + 1];
    const int cnt = p1 - p0;
    const int e = sexp[p0];
    const int KW = K / 2, KG = K / 64;
    const int out_row = threadgroup_position_in_grid.y * 8 + simd_gid * RPS;
    const device uint8_t* ws = (const device uint8_t*)w + ((size_t)e * N + out_row) * KW + simd_lid * 4;
    const device T* sc = scales + ((size_t)e * N + out_row) * KG + simd_lid / 8;
    const device T* bi = biases + ((size_t)e * N + out_row) * KG + simd_lid / 8;
    float result[MR][RPS];
    for (int m = 0; m < MR; m++) for (int r = 0; r < RPS; r++) result[m][r] = 0;
    float x_thread[V];
    int k = 0;
    for (; k < K - BLK; k += BLK) {
        uint16_t wr[RPS][V / 4];
        float sv[RPS], bv[RPS];
        for (int r = 0; r < RPS; r++) {
            const device uint16_t* wl = (const device uint16_t*)(ws + r * KW);
            for (int i = 0; i < V / 4; i++) wr[r][i] = wl[i];
            sv[r] = sc[r * KG];
            bv[r] = bi[r * KG];
        }
        for (int m = 0; m < MR; m++) {
            if (m >= cnt) break;
            const device T* xr = x + (size_t)(p0 + m) * K + k + simd_lid * V;
            float sum = g_load_vector<T, float, V>(xr, x_thread);
            for (int r = 0; r < RPS; r++) result[m][r] += g_qdot<float, V>(wr[r], x_thread, sv[r], bv[r], sum);
        }
        ws += BLK / 2;
        sc += BLK / 64;
        bi += BLK / 64;
    }
    const int remaining = clamp((int)(K - k - simd_lid * V), 0, V);
    if (remaining > 0) {
        uint16_t wr[RPS][V / 4];
        float sv[RPS], bv[RPS];
        for (int r = 0; r < RPS; r++) {
            const device uint16_t* wl = (const device uint16_t*)(ws + r * KW);
            for (int i = 0; i < V / 4; i++) wr[r][i] = wl[i];
            sv[r] = sc[r * KG];
            bv[r] = bi[r * KG];
        }
        for (int m = 0; m < MR; m++) {
            if (m >= cnt) break;
            const device T* xr = x + (size_t)(p0 + m) * K + k + simd_lid * V;
            float sum = g_load_vector_safe<T, float, V>(xr, x_thread, remaining);
            for (int r = 0; r < RPS; r++)
                result[m][r] += g_qdot_safe<float, V>(wr[r], x_thread, sv[r], bv[r], sum, remaining);
        }
    }
    for (int m = 0; m < MR; m++) {
        if (m >= cnt) break;
        for (int r = 0; r < RPS; r++) {
            float v = simd_sum(result[m][r]);
            if (simd_lid == 0) y[(size_t)(p0 + m) * N + out_row + r] = static_cast<T>(v);
        }
    }
"""


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(name="leap_gmr_a", input_names=["x", "w", "scales", "biases", "prow", "sexp", "seg_start"],
                                output_names=["y"], source=_A, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(name="leap_gmr_b", input_names=["x", "w", "scales", "biases", "sexp", "seg_start"],
                                output_names=["y"], source=_B, header=_HDR)


def segments(ids):
    """pairs sorted by expert: order, sorted experts, row per sorted pair, segment starts (sentinel P past the last)"""
    k = ids.shape[-1]
    flat = ids.flatten().astype(mx.int32)
    P = flat.shape[0]
    order = mx.argsort(flat)
    sexp = flat[order]
    prow = (order // k).astype(mx.int32)
    first = mx.concatenate([mx.array([True]), sexp[1:] != sexp[:-1]])
    sid = mx.cumsum(first.astype(mx.int32)) - 1
    tgt = mx.where(first, sid, P + 1)
    seg = mx.full((P + 2,), P, mx.int32).at[tgt].minimum(mx.arange(P, dtype=mx.int32))
    return order, sexp, prow, seg, P


def experts_gmr(x, gu, dn, ids, mr=16):
    """[R, k, D] routed-expert outputs (before the score-weighted sum), bit-identical to the per-pair gather_qmv path"""
    R, K = x.shape
    assert R <= mr
    order, sexp, prow, seg, P = segments(ids)
    gw, gs, gb = gu
    dw, dsc, db = dn
    N, D, KD = gw.shape[1], dw.shape[1], dw.shape[2] * 8
    assert K % 512 == 0 and N % 8 == 0 and D % 8 == 0 and KD % 64 == 0
    (g,) = _ka()(inputs=[x, gw, gs, gb, prow, sexp, seg],
                 template=[("T", x.dtype), ("K", K), ("N", N), ("MR", mr), ("P", P)],
                 grid=(64 * P, N // 8, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up                                    # [P, N / 2], the engine's chain
    (y,) = _kb()(inputs=[h, dw, dsc, db, sexp, seg],
                 template=[("T", x.dtype), ("K", KD), ("N", D), ("MR", mr), ("P", P)],
                 grid=(64 * P, D // 8, 1), threadgroup=(64, 1, 1), output_shapes=[(P, D)], output_dtypes=[x.dtype])
    inv = mx.argsort(order)
    return y[inv].reshape(R, ids.shape[-1], D)
