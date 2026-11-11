"""Round 26 PoC: fused routed-expert GLU for M = 1..4 rows x top-k, two dispatches per MoE layer, 4-bit affine g64.
  A  h[p, j] = silu(gate_e . x_m) * (up_e . x_m)      p = m*k + kk, e = slot of (m, kk); fused gate_up pack [S, 2N, K]
  B  y[m, d] = sum_kk w[m, kk] * (down_e[d, :] . h[p, :])   fixed expert order, fp32 accumulation, one bf16 write
Speed-first (a "v2b" candidate): fp32 accumulation everywhere, h rounded to bf16 like the stock chain's intermediate.
Layouts are MLX's affine 4-bit (little-endian nibbles, w = scale * q + bias), so the pools BigMoE streams work as is.
A: one simdgroup per (pair, RJ units): 2*RJ matrix rows; lane l covers values [16l + 512b, +16) for b = 0..K/512-1
   (MLX's qmv_fast split), x loaded once per simdgroup and reused across rows; all loads of a row issued up front.
B: one simdgroup per group of 4 output rows (4 x 640 values = 160 chunks of 16 values, 5 per lane, contiguous bytes);
   per-row partials reduced with simd_sum; the expert loop (fixed order) accumulates w * dot in registers."""
from functools import lru_cache
import mlx.core as mx

_A = """
    constexpr int K = K_C, N = N_C, G = G_C, RJ = RJ_C, NB = K / 512;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint units_per_tg = nsg * RJ;
    const uint p = threadgroup_position_in_grid.y;              // pair = m * topk + kk
    const uint j0 = threadgroup_position_in_grid.x * units_per_tg + sg * RJ;
    if (j0 >= (uint)N) return;
    const uint m = p / TOPK_C;
    const uint e = slots[p];
    // x: 16 values per block per lane, as float, plus their sum (bias term)
    float xv[NB][16];
    float xs[NB];
    const device T* xr = x + (size_t)m * K + lane * 16;
    for (int b = 0; b < NB; ++b) {
        float s = 0.0f;
        for (int v = 0; v < 16; ++v) { float t = (float)xr[b * 512 + v]; xv[b][v] = t; s += t; }
        xs[b] = s;
    }
    const size_t rowstride_w = K / 8;                          // uint32 words per row
    const size_t rowstride_g = K / G;
    float acc[2 * RJ];
    for (int r = 0; r < 2 * RJ; ++r) {
        const uint row = (r < RJ) ? (j0 + r) : (N + j0 + r - RJ);
        const device uint2* wr = (const device uint2*)(w + ((size_t)e * 2 * N + row) * rowstride_w) + lane;   // 8 B per lane per block
        const device T* sr = sc + ((size_t)e * 2 * N + row) * rowstride_g;
        const device T* br = bi + ((size_t)e * 2 * N + row) * rowstride_g;
        uint2 q[NB];
        for (int b = 0; b < NB; ++b) q[b] = wr[b * 32];        // 512 values = 256 B = 32 x uint2 per block
        float a = 0.0f;
        for (int b = 0; b < NB; ++b) {
            const int gidx = (b * 512 + lane * 16) / G;
            float dq = 0.0f;
            for (int v = 0; v < 8; ++v) dq += (float)((q[b].x >> (4 * v)) & 0xF) * xv[b][v];
            for (int v = 0; v < 8; ++v) dq += (float)((q[b].y >> (4 * v)) & 0xF) * xv[b][8 + v];
            a += (float)sr[gidx] * dq + (float)br[gidx] * xs[b];
        }
        acc[r] = simd_sum(a);
    }
    if (lane < (uint)RJ) {
        const float g = acc[lane], u = acc[RJ + lane];
        const float gb = (float)(T)g, ub = (float)(T)u;        // round like the stock chain's bf16 gate_up output
        const float hval = gb / (1.0f + metal::exp(-gb)) * ub;
        h[(size_t)p * N + j0 + lane] = (T)hval;
    }
"""

_B = """
    constexpr int D = D_C, N = N_C, G = G_C, NCH = N / 16;      // 16-value chunks per down row (40 for N = 640)
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint m = threadgroup_position_in_grid.y;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * 4;
    if (d0 >= (uint)D) return;
    const size_t rowstride_w = N / 8, rowstride_g = N / G;
    float y4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int kk = 0; kk < TOPK_C; ++kk) {
        const uint p = m * TOPK_C + kk;
        const uint e = slots[p];
        const float wk = route_w[p];
        const device uint2* wbase = (const device uint2*)(w + ((size_t)e * D + d0) * rowstride_w);   // 4 rows, contiguous
        const device T* sbase = sc + ((size_t)e * D + d0) * rowstride_g;
        const device T* bbase = bi + ((size_t)e * D + d0) * rowstride_g;
        const device T* hp = h + (size_t)p * N;
        uint2 q[5];
        for (int i = 0; i < 5; ++i) q[i] = wbase[lane + 32 * i];
        float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 5; ++i) {
            const uint c = lane + 32 * i;                       // flat chunk over 4 rows x NCH chunks
            const uint r = c / NCH, cc = c % NCH;
            const device T* hv = hp + cc * 16;
            float dq = 0.0f, hs = 0.0f;
            for (int v = 0; v < 8; ++v) { float t = (float)hv[v]; hs += t; dq += (float)((q[i].x >> (4 * v)) & 0xF) * t; }
            for (int v = 0; v < 8; ++v) { float t = (float)hv[8 + v]; hs += t; dq += (float)((q[i].y >> (4 * v)) & 0xF) * t; }
            const int gidx = (cc * 16) / G;
            const float contrib = (float)sbase[r * rowstride_g + gidx] * dq + (float)bbase[r * rowstride_g + gidx] * hs;
            for (int rr = 0; rr < 4; ++rr) part[rr] += (r == (uint)rr) ? contrib : 0.0f;
        }
        for (int rr = 0; rr < 4; ++rr) y4[rr] += wk * simd_sum(part[rr]);
    }
    if (lane < 4) y[(size_t)m * D + d0 + lane] = (T)y4[lane];
"""


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(name="glu_a", input_names=["x", "w", "sc", "bi", "slots"], output_names=["h"], source=_A)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(name="glu_b", input_names=["h", "w", "sc", "bi", "slots", "route_w"], output_names=["y"], source=_B)


def glu(x, gu, dn, slots, route_w, group_size=64, rj=4, sg_a=8, sg_b=8):
    """x [M, K] bf16; gu = (w [S, 2N, K/8] u32, scales, biases); dn = (w [S, D, N/8], scales, biases);
    slots uint32 [M*topk] (row-major); route_w float32 [M*topk]. Returns y [M, D] bf16 (routed experts only)."""
    gw, gs, gb = gu
    dw, ds, db = dn
    M, K = x.shape
    N = gw.shape[1] // 2
    D = dw.shape[1]
    topk = slots.size // M
    assert K % 512 == 0 and N % (rj * sg_a) == 0 and D % (4 * sg_b) == 0 and N == 640
    (h,) = _ka()(inputs=[x, gw, gs, gb, slots],
                 template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("G_C", group_size), ("RJ_C", rj), ("TOPK_C", topk)],
                 grid=(N // rj * 32, M * topk, 1), threadgroup=(32 * sg_a, 1, 1),
                 output_shapes=[(M * topk, N)], output_dtypes=[x.dtype])
    (y,) = _kb()(inputs=[h, dw, ds, db, slots, route_w],
                 template=[("T", x.dtype), ("D_C", D), ("N_C", N), ("G_C", group_size), ("TOPK_C", topk)],
                 grid=(D // 4 * 32, M, 1), threadgroup=(32 * sg_b, 1, 1),
                 output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return y
