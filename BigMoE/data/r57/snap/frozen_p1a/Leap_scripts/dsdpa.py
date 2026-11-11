"""DSDPA (Leap, Wed 2026-11-04 17:4x; TABLE_350 'DSDPA'): split-K ("Flash-Decoding", Dao et al. 2023; published) attention
for the DFlash2 drafter's block: q 32 heads x 16 rows, K / V 8 heads (GQA 4) x (ctx_len cache rows + the 16 block rows),
head_dim 128, mask = the cache rows inside the 2,048 window | the whole block (is_causal false). MLX's SDPA runs this on
32 threadgroups that walk every key serially (~64 GB/s; DSDPA-0: 0.46 / 0.83 ms per 5 layers at ctx 1,008 / 2,054).
Kernel A: one threadgroup per (kv head, key chunk), 4 simdgroups = the 4 q heads of the group; per 32-key step
S = Q K^T and O += P V on the matrix units (MPP matmul2d 16 x 32 x 16, qmm16c's fragment maps; P in bf16, its row sums
taken from the rounded values), online softmax in fp32, partial (m, l, o) per chunk. K / V rows come from the cache and
the block's own buffers (no concat). Kernel B merges the chunks and writes o_proj's input layout [1, 16, 32 x 128].
Not bitwise vs MLX's SDPA (summation order, bf16 P): a drafter-only change, exact-safe (drafts are verified).
attend(q [1, 32, 16, 128], kc / vc [1, 8, ctx_len, 128], kp / vp [1, 8, 16, 128], ctx_len) -> [1, 16, 4096]
install() / uninstall(): DFlashAttention.__call__ (draft_fused's memoized-mask version underneath) takes attend for
B 1, L 16, sliding non-causal layers of this shape; everything else (L 17 under R57_D17, other shapes) falls back."""
from functools import lru_cache
import mlx.core as mx

COUNTS = {"dsdpa": 0, "fallback": 0}
_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_SRC_A = r"""
    constexpr int D = 128, L = 16;
    constexpr float NEG = -1e30f;
    constexpr float SCALE = 0.08838834764831845f;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int c = threadgroup_position_in_grid.x;
    const int hk = threadgroup_position_in_grid.y;
    const int h = hk * G + sg;
    const int ctx_len = prm[0];
    const int Nc = prm[1];
    const int N = ctx_len + L;
    const int CK = ((N + NC - 1) / NC + 31) / 32 * 32;
    const int k_begin = c * CK;
    const int k_end = min(N, k_begin + CK);
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    using CTA = metal::remove_addrspace_t<decltype(ct_a)>;
    using CTB = metal::remove_addrspace_t<decltype(ct_b)>;
    auto ct_s = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o0 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o1 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o2 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o3 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();

    T qv[8][8];                                         // A fragments of Q: run t, dims 16 t + 4 fq + j; rows fm / fm + 8
    {
        const device T* q0 = q + ((size_t)h * L + fm) * D + 4 * fq;
        const device T* q1 = q + ((size_t)h * L + fm + 8) * D + 4 * fq;
        for (short t = 0; t < 8; t++)
            for (short j = 0; j < 4; j++) { qv[t][j] = q0[16 * t + j]; qv[t][4 + j] = q1[16 * t + j]; }
    }
    float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
    for (short i = 0; i < 16; i++) { ct_o0[i] = 0.0f; ct_o1[i] = 0.0f; ct_o2[i] = 0.0f; ct_o3[i] = 0.0f; }

#define DSDPA_PV(NT, CT)                                                                          \
    for (short r = 0; r < 4; r++) {                                                               \
        const short base = (r >> 1) * 8 + (r & 1) * 4;                                            \
        const int dn = NT * 32 + (r >> 1) * 16 + fm + (r & 1) * 8;                                \
        for (short j = 0; j < 4; j++) ct_b[base + j] = vok[u][j] ? vr[u][j][dn] : static_cast<T>(0.0f); \
    }                                                                                             \
    op.run(ct_a, ct_b, CT);

    for (int kb = k_begin; kb < k_end; kb += 32) {
        const device T* kr[4];                          // B rows of K^T: keys kb + (r >> 1) * 16 + fm + (r & 1) * 8
        for (short r = 0; r < 4; r++) {
            const int n = min(kb + (r >> 1) * 16 + fm + (r & 1) * 8, N - 1);
            kr[r] = (n < ctx_len ? kc + ((size_t)hk * Nc + n) * D : kp + ((size_t)hk * L + (n - ctx_len)) * D) + 4 * fq;
        }
        for (short i = 0; i < 16; i++) ct_s[i] = 0.0f;
        for (short t = 0; t < 8; t++) {
            for (short j = 0; j < 8; j++) ct_a[j] = qv[t][j];
            for (short r = 0; r < 4; r++) {
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++) ct_b[base + j] = kr[r][16 * t + j];
            }
            op.run(ct_a, ct_b, ct_s);
        }
        float s[16];                                    // S element i: row fm + 8 ((i & 7) >> 2), key (i >> 3) 16 + 4 fq + (i & 3)
        float r0 = NEG, r1 = NEG;
        for (short i = 0; i < 16; i++) {
            const int row = fm + 8 * ((i & 7) >> 2);
            const int n = kb + (i >> 3) * 16 + 4 * fq + (i & 3);
            const bool vis = n < N && (n >= ctx_len || n > ctx_len + row - SW);
            s[i] = vis ? ct_s[i] * SCALE : NEG;
            if ((i & 7) < 4) r0 = max(r0, s[i]); else r1 = max(r1, s[i]);
        }
        r0 = max(r0, simd_shuffle_xor(r0, 1)); r0 = max(r0, simd_shuffle_xor(r0, 8));
        r1 = max(r1, simd_shuffle_xor(r1, 1)); r1 = max(r1, simd_shuffle_xor(r1, 8));
        const float mn0 = max(m0, r0), mn1 = max(m1, r1);
        const float a0 = exp(m0 - mn0), a1 = exp(m1 - mn1);
        T pv[16];
        float ps0 = 0.0f, ps1 = 0.0f;
        for (short i = 0; i < 16; i++) {
            const bool top = (i & 7) < 4;
            const float p = s[i] <= 0.5f * NEG ? 0.0f : exp(s[i] - (top ? mn0 : mn1));
            pv[i] = static_cast<T>(p);
            const float pr_ = static_cast<float>(pv[i]);
            if (top) ps0 += pr_; else ps1 += pr_;
        }
        ps0 += simd_shuffle_xor(ps0, 1); ps0 += simd_shuffle_xor(ps0, 8);
        ps1 += simd_shuffle_xor(ps1, 1); ps1 += simd_shuffle_xor(ps1, 8);
        l0 = l0 * a0 + ps0; l1 = l1 * a1 + ps1; m0 = mn0; m1 = mn1;
        for (short i = 0; i < 16; i++) {
            const float a = (i & 7) < 4 ? a0 : a1;
            ct_o0[i] *= a; ct_o1[i] *= a; ct_o2[i] *= a; ct_o3[i] *= a;
        }
        const device T* vr[2][4];                        // V rows for run u, slot j: key kb + 16 u + 4 fq + j
        bool vok[2][4];
        for (short u = 0; u < 2; u++)
            for (short j = 0; j < 4; j++) {
                const int n0 = kb + 16 * u + 4 * fq + j;
                vok[u][j] = n0 < N;
                const int n = min(n0, N - 1);
                vr[u][j] = n < ctx_len ? vc + ((size_t)hk * Nc + n) * D : vp + ((size_t)hk * L + (n - ctx_len)) * D;
            }
        for (short u = 0; u < 2; u++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = pv[8 * u + j]; ct_a[4 + j] = pv[8 * u + 4 + j]; }
            DSDPA_PV(0, ct_o0)
            DSDPA_PV(1, ct_o1)
            DSDPA_PV(2, ct_o2)
            DSDPA_PV(3, ct_o3)
        }
    }

    const int pr = (h * NC + c) * L;
    if (fq == 0) { mo[pr + fm] = m0; mo[pr + fm + 8] = m1; lo[pr + fm] = l0; lo[pr + fm + 8] = l1; }
#define DSDPA_ST(NT, CT)                                                                          \
    for (short i = 0; i < 16; i++) {                                                              \
        const int row = fm + 8 * ((i & 7) >> 2);                                                  \
        const int dim = NT * 32 + (i >> 3) * 16 + 4 * fq + (i & 3);                               \
        oo[(size_t)(pr + row) * D + dim] = CT[i];                                                 \
    }
    DSDPA_ST(0, ct_o0)
    DSDPA_ST(1, ct_o1)
    DSDPA_ST(2, ct_o2)
    DSDPA_ST(3, ct_o3)
"""

_SRC_A2 = r"""
    constexpr int D = 128, L = 16, KS = 136;
    constexpr float NEG = -1e30f;
    threadgroup T Ks[32 * KS];
    threadgroup T Vs[32 * KS];
    const uint tid = thread_index_in_threadgroup;
    constexpr float SCALE = 0.08838834764831845f;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int c = threadgroup_position_in_grid.x;
    const int hk = threadgroup_position_in_grid.y;
    const int h = hk * G + sg;
    const int ctx_len = prm[0];
    const int Nc = prm[1];
    const int N = ctx_len + L;
    const int CK = ((N + NC - 1) / NC + 31) / 32 * 32;
    const int k_begin = c * CK;
    const int k_end = min(N, k_begin + CK);
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    using CTA = metal::remove_addrspace_t<decltype(ct_a)>;
    using CTB = metal::remove_addrspace_t<decltype(ct_b)>;
    auto ct_s = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o0 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o1 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o2 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();
    auto ct_o3 = op.template get_destination_cooperative_tensor<CTA, CTB, float>();

    T qv[8][8];                                         // A fragments of Q: run t, dims 16 t + 4 fq + j; rows fm / fm + 8
    {
        const device T* q0 = q + ((size_t)h * L + fm) * D + 4 * fq;
        const device T* q1 = q + ((size_t)h * L + fm + 8) * D + 4 * fq;
        for (short t = 0; t < 8; t++)
            for (short j = 0; j < 4; j++) { qv[t][j] = q0[16 * t + j]; qv[t][4 + j] = q1[16 * t + j]; }
    }
    float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
    for (short i = 0; i < 16; i++) { ct_o0[i] = 0.0f; ct_o1[i] = 0.0f; ct_o2[i] = 0.0f; ct_o3[i] = 0.0f; }

#define DSDPA2_PV(NT, CT)                                                                          \
    for (short r = 0; r < 4; r++) {                                                               \
        const short base = (r >> 1) * 8 + (r & 1) * 4;                                            \
        const int dn = NT * 32 + (r >> 1) * 16 + fm + (r & 1) * 8;                                \
        for (short j = 0; j < 4; j++) ct_b[base + j] = Vs[(16 * u + 4 * fq + j) * KS + dn];   \
    }                                                                                             \
    op.run(ct_a, ct_b, CT);

    for (int kb = k_begin; kb < k_end; kb += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);             // the last tile is consumed
        for (short w = 0; w < 4; w++) {                              // 32 keys x 128 dims of K and V: 512 16-byte pieces each
            const int ci = (int)tid + 128 * w;
            const int kl = ci >> 4, d8 = (ci & 15) * 8;
            const int n = kb + kl;
            uint4 kv4 = uint4(0), vv4 = uint4(0);
            if (n < N) {
                const size_t ro = n < ctx_len ? ((size_t)hk * Nc + n) * D : ((size_t)hk * L + (n - ctx_len)) * D;
                kv4 = *(const device uint4*)((n < ctx_len ? kc : kp) + ro + d8);
                vv4 = *(const device uint4*)((n < ctx_len ? vc : vp) + ro + d8);
            }
            *(threadgroup uint4*)(Ks + kl * KS + d8) = kv4;
            *(threadgroup uint4*)(Vs + kl * KS + d8) = vv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) ct_s[i] = 0.0f;
        for (short t = 0; t < 8; t++) {
            for (short j = 0; j < 8; j++) ct_a[j] = qv[t][j];
            for (short r = 0; r < 4; r++) {
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                const threadgroup T* kr = Ks + ((r >> 1) * 16 + fm + (r & 1) * 8) * KS + 16 * t + 4 * fq;
                for (short j = 0; j < 4; j++) ct_b[base + j] = kr[j];
            }
            op.run(ct_a, ct_b, ct_s);
        }
        float s[16];                                    // S element i: row fm + 8 ((i & 7) >> 2), key (i >> 3) 16 + 4 fq + (i & 3)
        float r0 = NEG, r1 = NEG;
        for (short i = 0; i < 16; i++) {
            const int row = fm + 8 * ((i & 7) >> 2);
            const int n = kb + (i >> 3) * 16 + 4 * fq + (i & 3);
            const bool vis = n < N && (n >= ctx_len || n > ctx_len + row - SW);
            s[i] = vis ? ct_s[i] * SCALE : NEG;
            if ((i & 7) < 4) r0 = max(r0, s[i]); else r1 = max(r1, s[i]);
        }
        r0 = max(r0, simd_shuffle_xor(r0, 1)); r0 = max(r0, simd_shuffle_xor(r0, 8));
        r1 = max(r1, simd_shuffle_xor(r1, 1)); r1 = max(r1, simd_shuffle_xor(r1, 8));
        const float mn0 = max(m0, r0), mn1 = max(m1, r1);
        const float a0 = exp(m0 - mn0), a1 = exp(m1 - mn1);
        T pv[16];
        float ps0 = 0.0f, ps1 = 0.0f;
        for (short i = 0; i < 16; i++) {
            const bool top = (i & 7) < 4;
            const float p = s[i] <= 0.5f * NEG ? 0.0f : exp(s[i] - (top ? mn0 : mn1));
            pv[i] = static_cast<T>(p);
            const float pr_ = static_cast<float>(pv[i]);
            if (top) ps0 += pr_; else ps1 += pr_;
        }
        ps0 += simd_shuffle_xor(ps0, 1); ps0 += simd_shuffle_xor(ps0, 8);
        ps1 += simd_shuffle_xor(ps1, 1); ps1 += simd_shuffle_xor(ps1, 8);
        l0 = l0 * a0 + ps0; l1 = l1 * a1 + ps1; m0 = mn0; m1 = mn1;
        for (short i = 0; i < 16; i++) {
            const float a = (i & 7) < 4 ? a0 : a1;
            ct_o0[i] *= a; ct_o1[i] *= a; ct_o2[i] *= a; ct_o3[i] *= a;
        }
        for (short u = 0; u < 2; u++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = pv[8 * u + j]; ct_a[4 + j] = pv[8 * u + 4 + j]; }
            DSDPA2_PV(0, ct_o0)
            DSDPA2_PV(1, ct_o1)
            DSDPA2_PV(2, ct_o2)
            DSDPA2_PV(3, ct_o3)
        }
    }

    const int pr = (h * NC + c) * L;
    if (fq == 0) { mo[pr + fm] = m0; mo[pr + fm + 8] = m1; lo[pr + fm] = l0; lo[pr + fm + 8] = l1; }
#define DSDPA2_ST(NT, CT)                                                                          \
    for (short i = 0; i < 16; i++) {                                                              \
        const int row = fm + 8 * ((i & 7) >> 2);                                                  \
        const int dim = NT * 32 + (i >> 3) * 16 + 4 * fq + (i & 3);                               \
        oo[(size_t)(pr + row) * D + dim] = CT[i];                                                 \
    }
    DSDPA2_ST(0, ct_o0)
    DSDPA2_ST(1, ct_o1)
    DSDPA2_ST(2, ct_o2)
    DSDPA2_ST(3, ct_o3)
"""

_SRC_B = r"""
    constexpr int D = 128, L = 16, HQ = 32;
    const uint idx = thread_position_in_grid.x;
    const int d = idx % D, row = (idx / D) % L, h = idx / (D * L);
    float M = -1e30f;
    for (int c = 0; c < NC; c++) M = max(M, mo[(h * NC + c) * L + row]);
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < NC; c++) {
        const int p = (h * NC + c) * L + row;
        const float w = exp(mo[p] - M);
        num += w * oo[(size_t)p * D + d];
        den += w * lo[p];
    }
    out[((size_t)row * HQ + h) * D + d] = static_cast<T>(num / den);
"""


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(name="leap_dsdpa_a", input_names=["q", "kc", "vc", "kp", "vp", "prm"],
                                output_names=["mo", "lo", "oo"], source=_SRC_A, header=_HDR)


@lru_cache(None)
def _ka2():
    return mx.fast.metal_kernel(name="leap_dsdpa_a2", input_names=["q", "kc", "vc", "kp", "vp", "prm"],
                                output_names=["mo", "lo", "oo"], source=_SRC_A2, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(name="leap_dsdpa_b", input_names=["mo", "lo", "oo"], output_names=["out"], source=_SRC_B)


_PRM = {}


def attend(q, kc, vc, kp, vp, ctx_len, sw=2048, nc=8, v=2):
    """q [1, 32, 16, 128], kc / vc [1, 8, Nc >= ctx_len, 128] (rows 0 .. ctx_len - 1 used), kp / vp [1, 8, 16, 128]
    -> [1, 16, 4096] (o_proj's input)"""
    assert q.shape == (1, 32, 16, 128) and kp.shape == vp.shape == (1, 8, 16, 128) and kc.shape[2] >= ctx_len, \
        (q.shape, kc.shape, kp.shape, ctx_len)
    key = (int(ctx_len), int(kc.shape[2]))
    prm = _PRM.get(key)
    if prm is None:
        if len(_PRM) > 4096:
            _PRM.clear()
        prm = _PRM[key] = mx.array(list(key), mx.int32)
    t = q.dtype
    mo, lo, oo = (_ka2() if v == 2 else _ka())(inputs=[q, kc, vc, kp, vp, prm], template=[("T", t), ("NC", nc), ("SW", sw), ("G", 4)],
                       grid=(nc * 128, 8, 1), threadgroup=(128, 1, 1),
                       output_shapes=[(32, nc, 16), (32, nc, 16), (32, nc, 16, 128)],
                       output_dtypes=[mx.float32, mx.float32, mx.float32])
    (out,) = _kb()(inputs=[mo, lo, oo], template=[("T", t), ("NC", nc)], grid=(32 * 16 * 128, 1, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(1, 16, 4096)], output_dtypes=[t])
    COUNTS["dsdpa"] += 1
    return out


_ORIG = {}
NC = [8]


def install(nc=8):
    """DFlashAttention.__call__ -> attend where it applies (install draft_fused first: its memoized-mask call stays the
    fallback). Inert until called; uninstall() restores the previous call."""
    import dflash_model_mlx as DF
    NC[0] = int(nc)
    if _ORIG:
        return
    prev = _ORIG["att"] = DF.DFlashAttention.__call__

    def att(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        if not (B == 1 and L == 16 and self.is_sliding and not self.is_causal and self.sliding_window == 2048
                and self.n_heads == 32 and self.n_kv_heads == 8 and x.dtype == mx.bfloat16):
            COUNTS["fallback"] += 1
            return prev(self, x, x_ctx, rope, cache)
        S = x_ctx.shape[1]
        keep_ctx = self.sliding_window - 1
        if S > keep_ctx:
            skip = S - keep_ctx
            x_ctx = x_ctx[:, skip:]
            S = x_ctx.shape[1]
            cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        out = attend(queries, keys, values, prop_keys, prop_values, keys.shape[2], nc=NC[0])
        return self.o_proj(out)

    DF.DFlashAttention.__call__ = att


def uninstall():
    import dflash_model_mlx as DF
    if _ORIG:
        DF.DFlashAttention.__call__ = _ORIG.pop("att")
