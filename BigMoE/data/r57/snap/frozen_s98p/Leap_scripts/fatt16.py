"""R-66 FATT16 (Leap, Sun 2026-11-01; REF2c candidate): one fused kernel for a 16-row attention call at the 125B's QSA shape
(head_dim 256, which MLX 0.32's fused SDPA does not take: it falls back to scale, QK^T, select, softmax, PV; R-65b).
Flash-attention (FlashAttention-2 style online softmax) on the M5's matrix units (MPP matmul2d 16 x 32 x 16):
- one simdgroup per query head (GQA: kv head = h / (Hq / Hkv)); the keys in fixed 32-key blocks [32 b, 32 b + 32), in order;
- S = Q K^T per block (16 MMAs over the 256 dims), * scale, masked (bool mask, keys >= N) -> -inf; online softmax per row
  in fp32 (row max / sum over the 4 lanes of a row: xor 1, then 8); P as bf16; O += P V (8 output tiles x 2 MMAs);
- a masked key adds exactly 0 to the row sum and 0 * V to O, and a block with nothing visible leaves (m, l, O) unchanged
  (factor exp(0) = 1), so a row's bits depend only on its own query and visible keys: not on N, padding, or the other
  rows (INV by construction; r66_fatt_test.py checks it), which would also retire REF2b's key buckets.
NOT bit-identical to MLX's fallback (different summation): a reference change (REF2c).
fatt16(q [1, Hq, 16, 256], k [1, Hkv, N, 256], v [1, Hkv, N, 256], scale, mask [.., 16, N] bool) -> [1, Hq, 16, 256]"""
from functools import lru_cache
import mlx.core as mx

_HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"

_SRC = r"""
    constexpr int D = D_C, HQ = HQ_C, HK = HK_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.x;
    const uint kvh = h / (HQ / HK);
    const int N = n_keys[0];
    const short fm0 = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const short fn0 = (((lane >> 2) & 2) | (lane & 1)) * 4;
    const device T* qh = q + (size_t)h * 16 * D;
    const device T* kh = k + (size_t)kvh * N * D;
    const device T* vh = v + (size_t)kvh * N * D;
    const float sc = scale[0];
    constexpr auto dqk = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dpv = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<dqk, metal::execution_simdgroup> opqk;
    mpp::tensor_ops::matmul2d<dpv, metal::execution_simdgroup> oppv;
    auto qa = opqk.template get_left_input_cooperative_tensor<T, T, float>();
    auto kb = opqk.template get_right_input_cooperative_tensor<T, T, float>();
    auto ss = opqk.template get_destination_cooperative_tensor<decltype(qa), decltype(kb), float>();
    auto pa = oppv.template get_left_input_cooperative_tensor<T, T, float>();
    auto vb = oppv.template get_right_input_cooperative_tensor<T, T, float>();
    auto o0 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o1 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o2 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o3 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o4 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o5 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o6 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    auto o7 = oppv.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
    for (int i = 0; i < 16; ++i) { o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; o4[i] = 0.0f; o5[i] = 0.0f;
                                   o6[i] = 0.0f; o7[i] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;   // rows fm0, fm0 + 8
    const int nb = (N + 31) / 32;
    for (int b = 0; b < nb; ++b) {
        const int kb0 = 32 * b;
        for (int i = 0; i < 16; ++i) ss[i] = 0.0f;
        for (int s = 0; s < D / 16; ++s) {
            const int d0 = 16 * s + fn0;
            const vec<T, 4> q0 = *(const device vec<T, 4>*)(qh + (size_t)fm0 * D + d0);
            const vec<T, 4> q1 = *(const device vec<T, 4>*)(qh + (size_t)(fm0 + 8) * D + d0);
            for (int j = 0; j < 4; ++j) { qa[j] = q0[j]; qa[4 + j] = q1[j]; }
            for (int t = 0; t < 4; ++t) {                             // keys fm0 + 8 t of this block
                const int key = kb0 + fm0 + 8 * t;
                vec<T, 4> kk = vec<T, 4>(0);
                if (key < N) kk = *(const device vec<T, 4>*)(kh + (size_t)key * D + d0);
                for (int j = 0; j < 4; ++j) kb[4 * t + j] = kk[j];
            }
            opqk.run(qa, kb, ss);
        }
        // scale, mask, online softmax (element i: row fm0 + 8 ((i >> 2) & 1), key kb0 + 16 (i >> 3) + fn0 + (i & 3))
        float x[16];
        float bm0 = -INFINITY, bm1 = -INFINITY;
        for (int i = 0; i < 16; ++i) {
            const int row = fm0 + 8 * ((i >> 2) & 1), key = kb0 + 16 * (i >> 3) + fn0 + (i & 3);
            const bool vis = key < N && mask[(size_t)row * mstride + key];
            x[i] = vis ? ss[i] * sc : -INFINITY;
            if ((i >> 2) & 1) bm1 = max(bm1, x[i]); else bm0 = max(bm0, x[i]);
        }
        bm0 = max(bm0, simd_shuffle_xor(bm0, 1)); bm0 = max(bm0, simd_shuffle_xor(bm0, 8));
        bm1 = max(bm1, simd_shuffle_xor(bm1, 1)); bm1 = max(bm1, simd_shuffle_xor(bm1, 8));
        const float n0 = max(m0, bm0), n1 = max(m1, bm1);
        const float f0 = (n0 == -INFINITY) ? 1.0f : metal::exp(m0 - n0);
        const float f1 = (n1 == -INFINITY) ? 1.0f : metal::exp(m1 - n1);
        float s0 = 0.0f, s1 = 0.0f;
        for (int i = 0; i < 16; ++i) {
            const bool r1 = (i >> 2) & 1;
            const float p = x[i] == -INFINITY ? 0.0f : metal::exp(x[i] - (r1 ? n1 : n0));
            x[i] = p;
            if (r1) s1 += p; else s0 += p;
        }
        s0 += simd_shuffle_xor(s0, 1); s0 += simd_shuffle_xor(s0, 8);
        s1 += simd_shuffle_xor(s1, 1); s1 += simd_shuffle_xor(s1, 8);
        l0 = l0 * f0 + s0; l1 = l1 * f1 + s1; m0 = n0; m1 = n1;
        for (int i = 0; i < 16; ++i) {
            const float f = ((i >> 2) & 1) ? f1 : f0;
            o0[i] *= f; o1[i] *= f; o2[i] *= f; o3[i] *= f; o4[i] *= f; o5[i] *= f; o6[i] *= f; o7[i] *= f;
        }
        // O += P V: P's C layout is the A layout of the two 16-key steps; V [key][dim] as the un-transposed right operand
        for (int ks = 0; ks < 2; ++ks) {
            for (int j = 0; j < 8; ++j) pa[j] = static_cast<T>(x[8 * ks + j]);
            #define PV_TILE(OT, ACC) { \
                for (int t = 0; t < 2; ++t) { \
                    const int key = kb0 + 16 * ks + fm0 + 8 * t; \
                    vec<T, 4> va = vec<T, 4>(0), vc = vec<T, 4>(0); \
                    if (key < N) { va = *(const device vec<T, 4>*)(vh + (size_t)key * D + 32 * OT + fn0); \
                                   vc = *(const device vec<T, 4>*)(vh + (size_t)key * D + 32 * OT + 16 + fn0); } \
                    for (int j = 0; j < 4; ++j) { vb[4 * t + j] = va[j]; vb[8 + 4 * t + j] = vc[j]; } \
                } \
                oppv.run(pa, vb, ACC); }
            PV_TILE(0, o0) PV_TILE(1, o1) PV_TILE(2, o2) PV_TILE(3, o3)
            PV_TILE(4, o4) PV_TILE(5, o5) PV_TILE(6, o6) PV_TILE(7, o7)
        }
    }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    device T* oh = out + (size_t)h * 16 * D;
    #define OUT_TILE(OT, ACC) \
        for (int i = 0; i < 16; ++i) { \
            const int row = fm0 + 8 * ((i >> 2) & 1), col = 32 * OT + 16 * (i >> 3) + fn0 + (i & 3); \
            oh[(size_t)row * D + col] = static_cast<T>(ACC[i] * (((i >> 2) & 1) ? i1 : i0)); }
    OUT_TILE(0, o0) OUT_TILE(1, o1) OUT_TILE(2, o2) OUT_TILE(3, o3)
    OUT_TILE(4, o4) OUT_TILE(5, o5) OUT_TILE(6, o6) OUT_TILE(7, o7)
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="fatt16", input_names=["q", "k", "v", "mask", "scale", "n_keys"], output_names=["out"],
                                source=_SRC, header=_HDR)


def fatt16(q, k, v, scale, mask):
    """q [1, Hq, 16, D], k / v [1, Hkv, N, D] (bf16), mask broadcastable to [16, N] (bool) -> [1, Hq, 16, D]"""
    B, Hq, S, D = q.shape
    Hk, N = k.shape[1], k.shape[2]
    assert B == 1 and S == 16 and D % 32 == 0 and Hq % Hk == 0, (q.shape, k.shape)
    m = mx.broadcast_to(mask, mask.shape[:-2] + (16, N)).reshape(-1, 16, N)[0] if mask.ndim > 2 else mask
    m = m.astype(mx.bool_)
    (o,) = _k()(inputs=[q, k, v, m, mx.array([scale], mx.float32), mx.array([N], mx.int32)],
                template=[("T", q.dtype), ("D_C", D), ("HQ_C", Hq), ("HK_C", Hk), ("mstride", N)],
                grid=(32 * Hq, 1, 1), threadgroup=(32, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])
    return o
