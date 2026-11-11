"""B6 FATT16-S (Leap, Tue 2026-11-03 00:1x; Fable B's mission B6, ~/FableGod/fableb/LONG_CONTEXT.md section 5; TABLE_350
'B6'): R-66's FATT16 (fatt16.py) with its one latency-bound walk per head split over KS_C simdgroups, FAST-K:
- a threadgroup = one 16-row tile = KS_C simdgroups; simdgroup s walks the ABSOLUTE 32-key blocks b = s, s + KS_C, ...
  (interleaved: contiguous quarters of the N keys would move with N and break the N-invariance by construction) and
  keeps its own (m, l, O) per row with FATT16's online softmax, unchanged;
- a block whose 16 x 32 mask entries are all false is skipped (it would leave (m, l, O) unchanged anyway: exact);
- the partials merge in a FIXED order (((0 + 1) + 2) + 3) through threadgroup memory with the standard rescale
  (an empty partial: factor 0; its partner's factor exp(0) = 1);
- RR_C real rows per head: 16, or 8 (two query heads of the same kv head stacked in one 16-row tile: chat's checks).
A row's bits depend only on its own query, its visible keys and their absolute positions: not on N, padding, the other
rows or which head it is stacked with (INV; b6_fatt_split.py checks it). Not bit-identical to MLX or to FATT16 (the
merge changes the summation order): a FAST-K reference change.
fatt16s(q [1, Hq, R, 256], k / v [1, Hkv, N, 256], scale, mask [R, N] bool, ks=4) -> [1, Hq, R, 256], R in (16, 8)"""
from functools import lru_cache
import mlx.core as mx
from fatt16 import _HDR

_SRC = r"""
    constexpr int D = D_C, HQ = HQ_C, HK = HK_C, RR = RR_C, KS = KS_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = thread_position_in_threadgroup.x / 32;
    const uint g = threadgroup_position_in_grid.x;
    const int h0 = (int)g * (16 / RR);                               // the tile's first head (RR 8: heads h0, h0 + 1)
    const uint kvh = h0 / (HQ / HK);
    const int N = n_keys[0];
    const short fm0 = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const short fn0 = (((lane >> 2) & 2) | (lane & 1)) * 4;
    // tile row r -> (head h0 + r / RR, query row r % RR); this lane's rows are fm0 and fm0 + 8
    const int hA = h0 + fm0 / RR, rA = fm0 % RR, hB = h0 + (fm0 + 8) / RR, rB = (fm0 + 8) % RR;
    const device T* qA = q + ((size_t)hA * RR + rA) * D;
    const device T* qB = q + ((size_t)hB * RR + rB) * D;
    const size_t kst = (size_t)k_strides[2], vst = (size_t)v_strides[2];   // the cache's slices: no copy of K / V
    const device T* kh = k + (size_t)kvh * (size_t)k_strides[1];
    const device T* vh = v + (size_t)kvh * (size_t)v_strides[1];
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
    for (int b = (int)sg; b < nb; b += KS) {
        const int kb0 = 32 * b;
        bool any = false;                                           // B6 (b): skip a block no row sees
        for (int i = 0; i < 16; ++i) {
            const int key = kb0 + 16 * (i >> 3) + fn0 + (i & 3);
            any |= key < N && mask[(size_t)(((i >> 2) & 1) ? rB : rA) * mstride + key];
        }
        if (!simd_any(any)) continue;
        for (int i = 0; i < 16; ++i) ss[i] = 0.0f;
        for (int s = 0; s < D / 16; ++s) {
            const int d0 = 16 * s + fn0;
            const vec<T, 4> q0 = *(const device vec<T, 4>*)(qA + d0);
            const vec<T, 4> q1 = *(const device vec<T, 4>*)(qB + d0);
            for (int j = 0; j < 4; ++j) { qa[j] = q0[j]; qa[4 + j] = q1[j]; }
            for (int t = 0; t < 4; ++t) {                             // keys fm0 + 8 t of this block
                const int key = kb0 + fm0 + 8 * t;
                vec<T, 4> kk = vec<T, 4>(0);
                if (key < N) kk = *(const device vec<T, 4>*)(kh + (size_t)key * kst + d0);
                for (int j = 0; j < 4; ++j) kb[4 * t + j] = kk[j];
            }
            opqk.run(qa, kb, ss);
        }
        // scale, mask, online softmax (element i: row fm0 + 8 ((i >> 2) & 1), key kb0 + 16 (i >> 3) + fn0 + (i & 3))
        float x[16];
        float bm0 = -INFINITY, bm1 = -INFINITY;
        for (int i = 0; i < 16; ++i) {
            const bool r1 = (i >> 2) & 1;
            const int key = kb0 + 16 * (i >> 3) + fn0 + (i & 3);
            const bool vis = key < N && mask[(size_t)(r1 ? rB : rA) * mstride + key];
            x[i] = vis ? ss[i] * sc : -INFINITY;
            if (r1) bm1 = max(bm1, x[i]); else bm0 = max(bm0, x[i]);
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
        for (int ks = 0; ks < 2; ++ks) {
            for (int j = 0; j < 8; ++j) pa[j] = static_cast<T>(x[8 * ks + j]);
            #define PV_TILE(OT, ACC) { \
                for (int t = 0; t < 2; ++t) { \
                    const int key = kb0 + 16 * ks + fm0 + 8 * t; \
                    vec<T, 4> va = vec<T, 4>(0), vc = vec<T, 4>(0); \
                    if (key < N) { va = *(const device vec<T, 4>*)(vh + (size_t)key * vst + 32 * OT + fn0); \
                                   vc = *(const device vec<T, 4>*)(vh + (size_t)key * vst + 32 * OT + 16 + fn0); } \
                    for (int j = 0; j < 4; ++j) { vb[4 * t + j] = va[j]; vb[8 + 4 * t + j] = vc[j]; } \
                } \
                oppv.run(pa, vb, ACC); }
            PV_TILE(0, o0) PV_TILE(1, o1) PV_TILE(2, o2) PV_TILE(3, o3)
            PV_TILE(4, o4) PV_TILE(5, o5) PV_TILE(6, o6) PV_TILE(7, o7)
        }
    }
    // B6 (a): the fixed-order merge, partial s into s + 1, through threadgroup memory (lane-for-lane: same layout)
    threadgroup float tO[8 * 16 * 32];
    threadgroup float tM[4 * 32];
    #define FOR_O(F) F(0, o0) F(1, o1) F(2, o2) F(3, o3) F(4, o4) F(5, o5) F(6, o6) F(7, o7)
    for (int st = 0; st < KS - 1; ++st) {
        if ((int)sg == st) {
            tM[lane] = m0; tM[32 + lane] = m1; tM[64 + lane] = l0; tM[96 + lane] = l1;
            #define PUT_O(OT, ACC) for (int i = 0; i < 16; ++i) tO[(OT * 16 + i) * 32 + lane] = ACC[i];
            FOR_O(PUT_O)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)sg == st + 1) {
            const float ma0 = tM[lane], ma1 = tM[32 + lane], la0 = tM[64 + lane], la1 = tM[96 + lane];
            const float n0 = max(ma0, m0), n1 = max(ma1, m1);
            const float fa0 = ma0 == -INFINITY ? 0.0f : metal::exp(ma0 - n0), fb0 = m0 == -INFINITY ? 0.0f : metal::exp(m0 - n0);
            const float fa1 = ma1 == -INFINITY ? 0.0f : metal::exp(ma1 - n1), fb1 = m1 == -INFINITY ? 0.0f : metal::exp(m1 - n1);
            l0 = la0 * fa0 + l0 * fb0; l1 = la1 * fa1 + l1 * fb1; m0 = n0; m1 = n1;
            #define MERGE_O(OT, ACC) for (int i = 0; i < 16; ++i) { const bool r1 = (i >> 2) & 1; \
                ACC[i] = tO[(OT * 16 + i) * 32 + lane] * (r1 ? fa1 : fa0) + ACC[i] * (r1 ? fb1 : fb0); }
            FOR_O(MERGE_O)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if ((int)sg == KS - 1) {
        const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
        device T* oA = out + ((size_t)hA * RR + rA) * D;
        device T* oB = out + ((size_t)hB * RR + rB) * D;
        #define OUT_TILE(OT, ACC) \
            for (int i = 0; i < 16; ++i) { \
                const bool r1 = (i >> 2) & 1; \
                const int col = 32 * OT + 16 * (i >> 3) + fn0 + (i & 3); \
                (r1 ? oB : oA)[col] = static_cast<T>(ACC[i] * (r1 ? i1 : i0)); }
        FOR_O(OUT_TILE)
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="fatt16s", input_names=["q", "k", "v", "mask", "scale", "n_keys"],
                                output_names=["out"], source=_SRC, header=_HDR, ensure_row_contiguous=False)


def fatt16s(q, k, v, scale, mask, ks=4):
    """q [1, Hq, R, D] (R 16, or 8: head pairs stacked), k / v [1, Hkv, N, D] (bf16), mask broadcastable to [R, N] (bool)
    -> [1, Hq, R, D]"""
    B, Hq, R, D = q.shape
    Hk, N = k.shape[1], k.shape[2]
    assert B == 1 and R in (16, 8) and D % 32 == 0 and Hq % Hk == 0 and (Hq // Hk) % (16 // R) == 0, (q.shape, k.shape)
    m = mx.broadcast_to(mask, mask.shape[:-2] + (R, N)).reshape(-1, R, N)[0] if mask.ndim > 2 else mask
    m = mx.contiguous(m.astype(mx.bool_))                           # k / v may be strided (last dim contiguous)
    (o,) = _k()(inputs=[mx.contiguous(q), k, v, m, mx.array([scale], mx.float32), mx.array([N], mx.int32)],
                template=[("T", q.dtype), ("D_C", D), ("HQ_C", Hq), ("HK_C", Hk), ("RR_C", R), ("KS_C", ks), ("mstride", N)],
                grid=(32 * ks * (Hq * R // 16), 1, 1), threadgroup=(32 * ks, 1, 1),
                output_shapes=[q.shape], output_dtypes=[q.dtype])
    return o
