"""B6-G FATT16-G, the GQA-packed FATT16 (Leap, Tue 2026-11-03 15:1x; B6's next lever, routed by the orchestrator 15:0x;
pre-registered in ~/Leap/TABLE_350.md 'B6-G'; FAST-K). B6 found fatt16s bound by K / V re-reads: the 12 query heads of a
kv head each read the same 32-key blocks (2 heads stacked: 1.66x). Published fix (TensorRT-LLM's XQA, FlashInfer's GQA
head-group fusion): load each K / V block once and serve every query head of that kv head from it.
- Threadgroup (kv head, head group, key split s): HG_C simdgroups, one per 16-row tile (R 16: a query head; R 8: two
  heads of the kv head stacked, as fatt16s). Per absolute 32-key block b = s, s + KS_C, ...: skip it if no row sees a key
  in it (the mask rows are the same for every tile, so the whole threadgroup agrees); else load its K and V rows once
  into threadgroup memory (2 x 16 KB = the M5 Pro's 32 KB), then every tile runs FATT16's QK^T / online softmax / PV
  unchanged, from threadgroup memory.
- Each (split, tile) writes its partial (O unnormalized, m, l); a second kernel folds the KS_C partials of each (head,
  row) in split order 0, 1, ... (an empty partial: factor 0, its partner's exp(0) = 1) and normalizes.
A row's bits depend only on its query, its visible keys and their absolute positions (the split is b mod KS_C, the merge
order fixed): batch-, N- and stacking-invariant like fatt16s, but not bit-identical to it (other split).
fatt16g(q [1, Hq, R, 256], k / v [1, Hkv, N, 256] (strided OK), scale, mask [R, N] bool, hg=None, ks=8) -> [1, Hq, R, 256]"""
from functools import lru_cache
import mlx.core as mx
from fatt16 import _HDR

_SRC = r"""
    constexpr int D = D_C, HQ = HQ_C, HK = HK_C, RR = RR_C, HG = HG_C, KS = KS_C, REC = D + 4;
    constexpr int TPK = (HQ / HK) * RR / 16;                          // 16-row tiles per kv head
    constexpr int NGK = TPK / HG;                                     // head groups per kv head
    const ushort lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int sg = (int)(tid / 32);
    const int g = (int)threadgroup_position_in_grid.x;
    const int s = g % KS, hgi = (g / KS) % NGK, kvh = g / (KS * NGK);
    const int h0 = kvh * (HQ / HK) + (hgi * HG + sg) * (16 / RR);     // the tile's first head
    const int N = n_keys[0];
    const short fm0 = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const short fn0 = (((lane >> 2) & 2) | (lane & 1)) * 4;
    const int hA = h0 + fm0 / RR, rA = fm0 % RR, hB = h0 + (fm0 + 8) / RR, rB = (fm0 + 8) % RR;
    const device T* qA = q + ((size_t)hA * RR + rA) * D;
    const device T* qB = q + ((size_t)hB * RR + rB) * D;
    const size_t kst = (size_t)k_strides[2], vst = (size_t)v_strides[2];
    const device T* kh = k + (size_t)kvh * (size_t)k_strides[1];
    const device T* vh = v + (size_t)kvh * (size_t)v_strides[1];
    const float sc = scale[0];
    threadgroup T tK[32 * D];
    threadgroup T tV[32 * D];
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
    for (int b = s; b < nb; b += KS) {
        const int kb0 = 32 * b;
        bool any = false;                                           // the same mask rows in every tile: uniform
        for (int i = 0; i < 16; ++i) {
            const int key = kb0 + 16 * (i >> 3) + fn0 + (i & 3);
            any |= key < N && mask[(size_t)(((i >> 2) & 1) ? rB : rA) * mstride + key];
        }
        if (!simd_any(any)) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);            // the previous block's readers are done
        for (int i = (int)tid; i < 32 * D / 8; i += 32 * HG) {      // the block's K and V rows, 8 values per load
            const int kr = i / (D / 8), c8 = i % (D / 8), key = kb0 + kr;
            uint4 kv = uint4(0), vv = uint4(0);
            if (key < N) { kv = *(const device uint4*)(kh + (size_t)key * kst + 8 * c8);
                           vv = *(const device uint4*)(vh + (size_t)key * vst + 8 * c8); }
            ((threadgroup uint4*)tK)[i] = kv;
            ((threadgroup uint4*)tV)[i] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 16; ++i) ss[i] = 0.0f;
        for (int st = 0; st < D / 16; ++st) {
            const int d0 = 16 * st + fn0;
            const vec<T, 4> q0 = *(const device vec<T, 4>*)(qA + d0);
            const vec<T, 4> q1 = *(const device vec<T, 4>*)(qB + d0);
            for (int j = 0; j < 4; ++j) { qa[j] = q0[j]; qa[4 + j] = q1[j]; }
            for (int t = 0; t < 4; ++t) {                             // keys fm0 + 8 t of this block
                const vec<T, 4> kk = *(const threadgroup vec<T, 4>*)(tK + (fm0 + 8 * t) * D + d0);
                for (int j = 0; j < 4; ++j) kb[4 * t + j] = kk[j];
            }
            opqk.run(qa, kb, ss);
        }
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
                    const int kr = 16 * ks + fm0 + 8 * t; \
                    const vec<T, 4> va = *(const threadgroup vec<T, 4>*)(tV + kr * D + 32 * OT + fn0); \
                    const vec<T, 4> vc = *(const threadgroup vec<T, 4>*)(tV + kr * D + 32 * OT + 16 + fn0); \
                    for (int j = 0; j < 4; ++j) { vb[4 * t + j] = va[j]; vb[8 + 4 * t + j] = vc[j]; } \
                } \
                oppv.run(pa, vb, ACC); }
            PV_TILE(0, o0) PV_TILE(1, o1) PV_TILE(2, o2) PV_TILE(3, o3)
            PV_TILE(4, o4) PV_TILE(5, o5) PV_TILE(6, o6) PV_TILE(7, o7)
        }
    }
    // this (split, tile)'s partial: O unnormalized at [0, D), m at D, l at D + 1 of each row record
    device float* pA = part + ((size_t)(s * HQ + hA) * RR + rA) * REC;
    device float* pB = part + ((size_t)(s * HQ + hB) * RR + rB) * REC;
    #define PART_TILE(OT, ACC) \
        for (int i = 0; i < 16; ++i) { \
            const int col = 32 * OT + 16 * (i >> 3) + fn0 + (i & 3); \
            (((i >> 2) & 1) ? pB : pA)[col] = ACC[i]; }
    PART_TILE(0, o0) PART_TILE(1, o1) PART_TILE(2, o2) PART_TILE(3, o3)
    PART_TILE(4, o4) PART_TILE(5, o5) PART_TILE(6, o6) PART_TILE(7, o7)
    if (fn0 == 0) { pA[D] = m0; pA[D + 1] = l0; pB[D] = m1; pB[D + 1] = l1; }
"""

_MERGE = r"""
    // one thread per (head, row, 4 dims): fold the KS_C partials in split order, normalize
    constexpr int D = D_C, REC = D + 4;
    const uint t = thread_position_in_grid.x;
    const int c4 = (int)(t % (D / 4)), hr = (int)(t / (D / 4));
    if (hr >= HQ_C * RR_C) return;
    float M = -INFINITY, L = 0.0f;
    float4 O = float4(0.0f);
    for (int s = 0; s < KS_C; ++s) {
        const device float* pp = part + ((size_t)s * HQ_C * RR_C + hr) * REC;
        const float ms = pp[D], ls = pp[D + 1];
        const float n = max(M, ms);
        const float fa = M == -INFINITY ? 0.0f : metal::exp(M - n), fb = ms == -INFINITY ? 0.0f : metal::exp(ms - n);
        L = L * fa + ls * fb;
        O = O * fa + *(const device float4*)(pp + 4 * c4) * fb;
        M = n;
    }
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    for (int j = 0; j < 4; ++j) out[(size_t)hr * D + 4 * c4 + j] = static_cast<T>(O[j] * inv);
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="fatt16g", input_names=["q", "k", "v", "mask", "scale", "n_keys"],
                                output_names=["part"], source=_SRC, header=_HDR, ensure_row_contiguous=False)


@lru_cache(None)
def _km():
    return mx.fast.metal_kernel(name="fatt16g_merge", input_names=["part"], output_names=["out"], source=_MERGE)


def fatt16g(q, k, v, scale, mask, hg=None, ks=8):
    """q [1, Hq, R, D] (R 16, or 8: head pairs stacked), k / v [1, Hkv, N, D] (bf16, may be strided), mask
    broadcastable to [R, N] (bool) -> [1, Hq, R, D]; hg tiles per threadgroup (default: every tile of the kv head), ks
    key splits (threadgroups per head group)"""
    B, Hq, R, D = q.shape
    Hk, N = k.shape[1], k.shape[2]
    tpk = (Hq // Hk) * R // 16
    hg = tpk if hg is None else hg
    assert B == 1 and R in (16, 8) and D % 32 == 0 and Hq % Hk == 0 and (Hq // Hk) % (16 // R) == 0, (q.shape, k.shape)
    assert tpk % hg == 0 and 32 * hg <= 1024 and 2 * 32 * D * 2 <= 32768, (tpk, hg, D)
    m = mx.broadcast_to(mask, mask.shape[:-2] + (R, N)).reshape(-1, R, N)[0] if mask.ndim > 2 else mask
    m = mx.contiguous(m.astype(mx.bool_))
    tmpl = [("T", q.dtype), ("D_C", D), ("HQ_C", Hq), ("HK_C", Hk), ("RR_C", R), ("KS_C", ks)]
    (part,) = _k()(inputs=[mx.contiguous(q), k, v, m, mx.array([scale], mx.float32), mx.array([N], mx.int32)],
                   template=tmpl + [("HG_C", hg), ("mstride", N)],
                   grid=(32 * hg * Hk * (tpk // hg) * ks, 1, 1), threadgroup=(32 * hg, 1, 1),
                   output_shapes=[(ks, Hq, R, D + 4)], output_dtypes=[mx.float32])
    nt = Hq * R * D // 4
    (o,) = _km()(inputs=[part], template=tmpl, grid=((nt + 255) // 256 * 256, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[q.shape], output_dtypes=[q.dtype])
    return o
