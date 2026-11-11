"""Roofline-engine Metal kernels (eps = rms_norm_eps = 1e-6 is a source constant: MLX templates take no floats; non-exact: fp32 accumulation, any reduction order). Shapes are the 125B's
(engine/MODEL_NOTES.md); M = rows per call (1-8). Lineage: patches/sol_layer (speed-of-light layer, measured 2026-10-21)."""
from functools import lru_cache
import mlx.core as mx

HDR = r"""
inline float silu_f(float x) { return x / (1.0f + metal::exp(-x)); }
inline float sigm_f(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
"""

# HC read, down part: threadgroup = 4 simdgroups, simdgroup k owns HC group k (2560 columns); one pass gives the group RMS
# and the partial dots for RB rows of W [324 = 320 low-rank | 4 inject, 10240]; out y [M, 324] fp32 = silu(./4) | 2 sigm(./4)
HCD = r"""
    constexpr int RB = RB_C, M = M_C, NROW = 324;
    const uint lane = thread_index_in_simdgroup, k = simdgroup_index_in_threadgroup;
    const uint n0 = threadgroup_position_in_grid.x * RB;
    threadgroup float part[4][RB][M];
    float acc[RB][M], ss[M];
    for (int m = 0; m < M; ++m) { ss[m] = 0.0f; for (int r = 0; r < RB; ++r) acc[r][m] = 0.0f; }
    float inv[M];
    if (FAITH) {                                                  // pass 1: the group RMS (MLX rms_norm, then * weight)
        for (int j = 0; j < 10; ++j) {
            const uint c = k * 320 + j * 32 + lane;
            for (int m = 0; m < M; ++m) {
                const uint4 x4 = ((const device uint4*)(xin + (size_t)m * 10240))[c];
                const thread bfloat16_t* xb = (const thread bfloat16_t*)&x4;
                for (int i = 0; i < 8; ++i) { const float v = (float)xb[i]; ss[m] += v * v; }
            }
        }
        for (int m = 0; m < M; ++m) inv[m] = metal::rsqrt(simd_sum(ss[m]) / 2560.0f + 1e-6f);
    }
    for (int j = 0; j < 10; ++j) {
        const uint c = k * 320 + j * 32 + lane;
        const uint4 nw4 = ((const device uint4*)nw)[c];
        const thread bfloat16_t* nv = (const thread bfloat16_t*)&nw4;
        float xv[M][8];
        for (int m = 0; m < M; ++m) {
            const uint4 x4 = ((const device uint4*)(xin + (size_t)m * 10240))[c];
            const thread bfloat16_t* xb = (const thread bfloat16_t*)&x4;
            for (int i = 0; i < 8; ++i) {
                const float v = (float)xb[i];
                if (FAITH) xv[m][i] = (float)bfloat16_t((float)bfloat16_t(v * inv[m]) * (float)nv[i]);
                else { ss[m] += v * v; xv[m][i] = v * (float)nv[i]; }
            }
        }
        for (int r = 0; r < RB; ++r) {
            if (n0 + r >= (uint)NROW) break;
            const uint4 w4 = ((const device uint4*)(W + (size_t)(n0 + r) * 10240))[c];
            const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
            for (int m = 0; m < M; ++m) { float a = 0.0f; for (int i = 0; i < 8; ++i) a += (float)wv[i] * xv[m][i]; acc[r][m] += a; }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float iv = FAITH ? inv[m] : metal::rsqrt(simd_sum(ss[m]) / 2560.0f + 1e-6f);
        if (threadgroup_position_in_grid.x == 0 && lane == 0) inv_out[m * 4 + k] = iv;
        for (int r = 0; r < RB; ++r) { const float a = simd_sum(acc[r][m]) * (FAITH ? 1.0f : iv); if (lane == 0) part[k][r][m] = a; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = k * 32 + lane;
    if (t < (uint)(RB * M)) {
        const uint r = t / M, m = t % M, n = n0 + r;
        if (n < (uint)NROW) {
            const float a = part[0][r][m] + part[1][r][m] + part[2][r][m] + part[3][r][m];
            if (FAITH) {                                          // bf16 Linear output, /4 (exact), MLX sigmoid / silu = x * sigmoid(x)
                const float d4 = (float)bfloat16_t(a) / 4.0f;
                const float sg = (float)bfloat16_t(sigm_f(d4));
                y[(size_t)m * NROW + n] = (n < 320) ? (float)bfloat16_t(d4 * sg) : (float)bfloat16_t(2.0f * sg);
            } else {
                y[(size_t)m * NROW + n] = (n < 320) ? silu_f(a / 4.0f) : 2.0f * sigm_f(a / 4.0f);
            }
        }
    }
"""

# HC read, up part: simdgroup = 2 output dims x 4 groups = 8 rows of W_up [10240, 320]; 4 lanes per row; mixed [M, 2560]
HCU = r"""
    constexpr int M = M_C;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint r = lane / 4, q = lane % 4, kk = r % 4, j = r / 4;
    const uint d = sgi * 2 + j, n = kk * 2560 + d;
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    for (int i = 0; i < 10; ++i) {
        const uint c = q + 4 * i;
        const uint4 w4 = ((const device uint4*)(W + (size_t)n * 320))[c];
        const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
        for (int m = 0; m < M; ++m) {
            const device float* xm = yin + (size_t)m * 324 + c * 8;
            float a = 0.0f;
            for (int e = 0; e < 8; ++e) a += (float)wv[e] * xm[e];
            acc[m] += a;
        }
    }
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        a += simd_shuffle_xor(a, 1); a += simd_shuffle_xor(a, 2);
        float term;
        if (FAITH) {
            const float u = (float)bfloat16_t(sigm_f((float)bfloat16_t(a)));
            const float nn = (float)bfloat16_t((float)bfloat16_t((float)stream[(size_t)m * 10240 + n] * inv_in[m * 4 + kk]) * (float)nw[n]);
            term = (float)bfloat16_t(u * nn);
        } else {
            term = sigm_f(a) * (float)stream[(size_t)m * 10240 + n] * inv_in[m * 4 + kk] * (float)nw[n];
        }
        term += simd_shuffle_xor(term, 4); term += simd_shuffle_xor(term, 8);
        if ((lane & 15u) == 0) mixed[(size_t)m * 2560 + d] = static_cast<bfloat16_t>(term * 0.25f);
    }
"""

# HC write (+ optional nothing else): out[m, k*2560+n] = stream + inject[m, k] * o[m, n]
HCW = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= (uint)(M_C * 10240)) return;
    const uint m = i / 10240, c = i % 10240;
    const float prod = y[m * 324 + 320 + c / 2560] * (float)o[m * 2560 + c % 2560];
    out[i] = static_cast<bfloat16_t>((float)stream[i] + (FAITH ? (float)bfloat16_t(prod) : prod));
"""

# GDN prework after the in_proj GEMV: causal conv4 over [state (3 rows); M rows] + silu for q|k|v, sum-based l2norm per
# head for q, k (q also / sqrt(128)); z, b, a pass through. One threadgroup = 128 channels = one head for q|k.
GEPI = r"""
    constexpr int M = M_C, NIN = 16480;
    const uint t = thread_position_in_threadgroup.x, n = threadgroup_position_in_grid.x * 128 + t;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float hsum[M][4];
    float v[M];
    for (int m = 0; m < M; ++m) v[m] = n < (uint)NIN ? (float)pre[m * NIN + n] : 0.0f;
    if (n < 10240u) {
        float xw[M + 3];
        for (int i = 0; i < 3; ++i) xw[i] = (float)cst[i * 10240 + n];
        for (int m = 0; m < M; ++m) xw[3 + m] = v[m];
        for (int i = 0; i < 3; ++i) cst_out[i * 10240 + n] = static_cast<bfloat16_t>(xw[M + i]);
        for (int m = 0; m < M; ++m) {
            float cv = 0.0f;
            for (int i = 0; i < 4; ++i) cv += (float)cw[n * 4 + i] * xw[m + i];
            if (FAITH) { const float c = (float)bfloat16_t(cv); v[m] = (float)bfloat16_t(c * (float)bfloat16_t(sigm_f(c))); }
            else v[m] = silu_f(cv);
        }
    }
    const bool qk = threadgroup_position_in_grid.x < 32u;
    if (qk) for (int m = 0; m < M; ++m) { const float s2 = simd_sum(v[m] * v[m]); if (lane == 0) hsum[m][sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (qk) for (int m = 0; m < M; ++m) {
        const float r = metal::rsqrt(hsum[m][0] + hsum[m][1] + hsum[m][2] + hsum[m][3] + 1e-6f);
        if (FAITH) {                                              // MTPLX: q = bf16(1/sqrt(128)) * bf16(l2n(q)), k = bf16(l2n(k))
            const float l = (float)bfloat16_t(v[m] * r);
            v[m] = n < 2048u ? (float)bfloat16_t(0.08837890625f * l) : l;
        } else {
            v[m] *= r * (n < 2048u ? 0.08838834764831845f : 1.0f);
        }
    }
    if (n < (uint)NIN) for (int m = 0; m < M; ++m) gin[m * NIN + n] = static_cast<bfloat16_t>(v[m]);
"""

# gated delta recurrence per v-head (one threadgroup of 1024 = 128 v rows x 8 lanes of 16 k columns), rows in order over M,
# then y = rmsnorm(o; w) * sigmoid(z). State [48, 128 (v), 128 (k)] fp32 (MTPLX's [Hv, Dv, Dk]).
REC = r"""
    constexpr int M = M_C, NIN = 16480;
    const uint h = threadgroup_position_in_grid.x, kh = h / 3;
    const uint t = thread_position_in_threadgroup.x, row = t / 8, cg = (t % 8) * 16;
    const device float* St = state + (size_t)h * 16384 + row * 128 + cg;
    device float* So = state_out + (size_t)h * 16384 + row * 128 + cg;
    float s[16];
    for (int i = 0; i < 16; ++i) s[i] = St[i];
    threadgroup float ov[M][128];
    const float A = metal::exp((float)A_log[h]), dtb = (float)dt_bias[h];
    for (int m = 0; m < M; ++m) {
        const device bfloat16_t* g_ = gin + (size_t)m * NIN;
        const float bb = (float)g_[16384 + h], aa = (float)g_[16432 + h];
        float sp = aa + dtb;
        if (FAITH) sp = (float)bfloat16_t(sp);
        float spl = sp > 20.0f ? sp : metal::log(1.0f + metal::exp(sp));
        if (FAITH) spl = (float)bfloat16_t(spl);
        const float g = metal::exp(-A * spl);
        const float beta = FAITH ? (float)bfloat16_t(sigm_f(bb)) : sigm_f(bb);
        float kx[16], qx[16];
        for (int i = 0; i < 16; ++i) { kx[i] = (float)g_[2048 + kh * 128 + cg + i]; qx[i] = (float)g_[kh * 128 + cg + i]; }
        float sk = 0.0f;
        for (int i = 0; i < 16; ++i) { s[i] *= g; sk += s[i] * kx[i]; }
        sk += simd_shuffle_xor(sk, 1); sk += simd_shuffle_xor(sk, 2); sk += simd_shuffle_xor(sk, 4);
        const float dlt = beta * ((float)g_[4096 + h * 128 + row] - sk);
        float o = 0.0f;
        for (int i = 0; i < 16; ++i) { s[i] += dlt * kx[i]; o += s[i] * qx[i]; }
        o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2); o += simd_shuffle_xor(o, 4);
        if ((t % 8) == 0) ov[m][row] = FAITH ? (float)bfloat16_t(o) : o;
    }
    for (int i = 0; i < 16; ++i) So[i] = s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 128) {
        for (int m = 0; m < M; ++m) {
            float ss = 0.0f;
            for (int i = 0; i < 128; ++i) ss += ov[m][i] * ov[m][i];
            const float inv = metal::rsqrt(ss / 128.0f + 1e-6f);
            const float z = (float)gin[(size_t)m * NIN + 10240 + h * 128 + t];
            const float xn = FAITH ? (float)bfloat16_t(ov[m][t] * inv * (float)nwg[t]) : ov[m][t] * inv * (float)nwg[t];
            y[(size_t)m * 6144 + h * 128 + t] = static_cast<bfloat16_t>(xn * sigm_f(z));
        }
    }
"""

SPEC = dict(
    hcd=("re_hcd", HCD, ["xin", "W", "nw"], ["y", "inv_out"]),
    hcu=("re_hcu", HCU, ["yin", "W", "nw", "stream", "inv_in"], ["mixed"]),
    hcw=("re_hcw", HCW, ["stream", "y", "o"], ["out"]),
    gepi=("re_gepi", GEPI, ["pre", "cw", "cst"], ["gin", "cst_out"]),
    rec=("re_rec", REC, ["gin", "state", "A_log", "dt_bias", "nwg"], ["y", "state_out"]),
)


import os
FAITH = int(os.environ.get("RLE_FAITH", "1"))               # 1: round to bf16 where MTPLX does (HC read/write), 0: plain fp32


@lru_cache(None)
def K(name):
    n, src, ins, outs = SPEC[name]
    return mx.fast.metal_kernel(name=n, input_names=ins, output_names=outs, source=src, header=HDR)


def hc_read(stream, Wd, Wu, nw, M, rb=2):
    """HC read of stream [M, 10240] bf16: returns (y [M, 324] fp32 = silu low-rank | inject, mixed [M, 2560] bf16)"""
    y, inv = K("hcd")(inputs=[stream, Wd, nw], template=[("RB_C", rb), ("M_C", M), ("FAITH", FAITH)],
                      grid=(-(-324 // rb) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(M, 324), (M, 4)],
                      output_dtypes=[mx.float32, mx.float32])
    (mixed,) = K("hcu")(inputs=[y, Wu, nw, stream, inv], template=[("M_C", M), ("FAITH", FAITH)], grid=(1280 * 32, 1, 1),
                        threadgroup=(128, 1, 1), output_shapes=[(M, 2560)], output_dtypes=[mx.bfloat16])
    return y, mixed


def hc_write(stream, y, o, M):
    """stream + inject (y[:, 320:324]) x o [M, 2560]"""
    return K("hcw")(inputs=[stream, y, o], template=[("M_C", M), ("FAITH", FAITH)], grid=(M * 10240, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(M, 10240)], output_dtypes=[mx.bfloat16])[0]


def gdn_prework(pre, cw, cst, M):
    return K("gepi")(inputs=[pre, cw, cst], template=[("M_C", M), ("FAITH", FAITH)], grid=(129 * 128, 1, 1), threadgroup=(128, 1, 1),
                     output_shapes=[(M, 16480), (3, 10240)], output_dtypes=[mx.bfloat16, mx.bfloat16])


def gdn_recurrence(gin, state, A_log, dt_bias, nwg, M):
    return K("rec")(inputs=[gin, state, A_log, dt_bias, nwg], template=[("M_C", M), ("FAITH", FAITH)], grid=(48 * 1024, 1, 1),
                    threadgroup=(1024, 1, 1), output_shapes=[(M, 6144), (48 * 16384,)], output_dtypes=[mx.bfloat16, mx.float32])
