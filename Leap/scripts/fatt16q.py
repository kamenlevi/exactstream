"""KV8 (Leap, Tue 2026-11-03 21:3x; TABLE_350 'KV8'; FAST-Q): fs4 (fatt16s.py: FATT16 + a KS-way split of the key walk
over absolute 32-key blocks, fixed-order merge, all-masked blocks skipped) over an int8 K / V cache.
- The cache (quant_kv, done on the cache write): K, V int8 [1, Hk, Np, D] (Np = N rounded up to 32, zero rows past N),
  smoothed (the channel mean subtracted: K's shifts every row's scores by one constant, which softmax cancels exactly;
  V's is added back at the output, since the probabilities sum to 1), one scale per (kv head, token): ksc / vsc
  [Hk, Np] float32; vmu [Hk, D] float32.
- MODE 0, the storage form (llama.cpp's q8_0 / Strata): K and V converted to T for fs4's matmul2d (16, 32, 16); the
  scores times K's per-token scale; V's scale folded into P after the row sums.
- MODE 1, the MMA form (SageAttention-style): Q in int8 per (head, row) (quant_q, timed with the call); QK^T as
  int8 x int8 -> int32 on matmul2d (16, 32, 32), 8 runs per block instead of 16. Its int32 score tile has fs4's float
  layout (kv8_layout_probe.py), so fs4's online softmax runs unchanged. scores = acc x s_q x s_k x scale; PV as MODE 0.
fatt16q(q, cache, scale, mask, mode, qq=None, ks=4) -> [1, Hq, R, D] (T = q.dtype); quant_kv(k, v) -> cache;
quant_q(q) -> (q8, qsc)"""
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
    const int N = n_keys[0], NP = (N + 31) / 32 * 32;
    const short fm0 = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const short fn0 = (((lane >> 2) & 2) | (lane & 1)) * 4;
    const int hA = h0 + fm0 / RR, rA = fm0 % RR, hB = h0 + (fm0 + 8) / RR, rB = (fm0 + 8) % RR;
    const device char* kc = (const device char*)k8 + (size_t)kvh * NP * D;
    const device char* vc = (const device char*)v8 + (size_t)kvh * NP * D;
    const device float* ksh = ksc + (size_t)kvh * NP;
    const device float* vsh = vsc + (size_t)kvh * NP;
    const float sc = scale[0];
#if MODE == 0
    const device T* qA = q + ((size_t)hA * RR + rA) * D;
    const device T* qB = q + ((size_t)hB * RR + rB) * D;
    constexpr auto dqk = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<dqk, metal::execution_simdgroup> opqk;
    auto qa = opqk.template get_left_input_cooperative_tensor<T, T, float>();
    auto kb = opqk.template get_right_input_cooperative_tensor<T, T, float>();
    auto ss = opqk.template get_destination_cooperative_tensor<decltype(qa), decltype(kb), float>();
#else
    const device char* q8A = (const device char*)q8 + ((size_t)hA * RR + rA) * D;
    const device char* q8B = (const device char*)q8 + ((size_t)hB * RR + rB) * D;
    const float qsA = qsc[hA * RR + rA], qsB = qsc[hB * RR + rB];
    constexpr auto dqk = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<dqk, metal::execution_simdgroup> opqk;
    auto qa = opqk.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto kb = opqk.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ss = opqk.template get_destination_cooperative_tensor<decltype(qa), decltype(kb), int32_t>();
#endif
    constexpr auto dpv = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<dpv, metal::execution_simdgroup> oppv;
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
        for (int i = 0; i < 16; ++i) ss[i] = 0;
#if MODE == 0
        for (int s = 0; s < D / 16; ++s) {                          // fs4's QK^T, K converted from int8
            const int d0 = 16 * s + fn0;
            const vec<T, 4> q0 = *(const device vec<T, 4>*)(qA + d0);
            const vec<T, 4> q1 = *(const device vec<T, 4>*)(qB + d0);
            for (int j = 0; j < 4; ++j) { qa[j] = q0[j]; qa[4 + j] = q1[j]; }
            for (int t = 0; t < 4; ++t) {                             // keys fm0 + 8 t (rows past N are zeros)
                const char4 kk = *(const device char4*)(kc + (size_t)(kb0 + fm0 + 8 * t) * D + d0);
                for (int j = 0; j < 4; ++j) kb[4 * t + j] = static_cast<T>((float)kk[j]);
            }
            opqk.run(qa, kb, ss);
        }
#else
        for (int s = 0; s < D / 32; ++s) {                          // int8 x int8 -> int32, 32 dims per run
            const int d0 = 32 * s + fn0;                              // qa: dim 16 (i >> 3) + fn0 + (i & 3), row A / B
            const char4 a0 = *(const device char4*)(q8A + d0), a1 = *(const device char4*)(q8A + d0 + 16);
            const char4 b0 = *(const device char4*)(q8B + d0), b1 = *(const device char4*)(q8B + d0 + 16);
            for (int j = 0; j < 4; ++j) { qa[j] = (int8_t)a0[j]; qa[4 + j] = (int8_t)b0[j]; qa[8 + j] = (int8_t)a1[j];
                                          qa[12 + j] = (int8_t)b1[j]; }
            for (int t = 0; t < 4; ++t) {                             // kb: key fm0 + 8 t, dim 16 (i >> 4) + fn0 + (i & 3)
                const device char* kr = kc + (size_t)(kb0 + fm0 + 8 * t) * D + d0;
                const char4 c0 = *(const device char4*)kr, c1 = *(const device char4*)(kr + 16);
                for (int j = 0; j < 4; ++j) { kb[4 * t + j] = (int8_t)c0[j]; kb[16 + 4 * t + j] = (int8_t)c1[j]; }
            }
            opqk.run(qa, kb, ss);
        }
#endif
        // per-token K scales (keys kb0 + fn0 .. + 3 and kb0 + 16 + fn0 .. + 3 for this lane)
        const float4 kv0 = *(const device float4*)(ksh + kb0 + fn0), kv1 = *(const device float4*)(ksh + kb0 + 16 + fn0);
        float x[16];
        float bm0 = -INFINITY, bm1 = -INFINITY;
        for (int i = 0; i < 16; ++i) {
            const bool r1 = (i >> 2) & 1;
            const int key = kb0 + 16 * (i >> 3) + fn0 + (i & 3);
            const bool vis = key < N && mask[(size_t)(r1 ? rB : rA) * mstride + key];
#if MODE == 0
            const float sv = (float)ss[i] * sc;
#else
            const float sv = (float)ss[i] * (r1 ? qsB : qsA) * sc;
#endif
            x[i] = vis ? sv * ((i >> 3) ? kv1[i & 3] : kv0[i & 3]) : -INFINITY;
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
        const float4 vv0 = *(const device float4*)(vsh + kb0 + fn0), vv1 = *(const device float4*)(vsh + kb0 + 16 + fn0);
        for (int ks = 0; ks < 2; ++ks) {
            for (int j = 0; j < 8; ++j) pa[j] = static_cast<T>(x[8 * ks + j] * (ks ? vv1[j & 3] : vv0[j & 3]));   // V's scale into P
            #define PV_TILE(OT, ACC) { \
                for (int t = 0; t < 2; ++t) { \
                    const device char* vr = vc + (size_t)(kb0 + 16 * ks + fm0 + 8 * t) * D + 32 * OT + fn0; \
                    const char4 va = *(const device char4*)vr, vz = *(const device char4*)(vr + 16); \
                    for (int j = 0; j < 4; ++j) { vb[4 * t + j] = static_cast<T>((float)va[j]); \
                                                  vb[8 + 4 * t + j] = static_cast<T>((float)vz[j]); } \
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
        const float u0 = l0 > 0.0f ? 1.0f : 0.0f, u1 = l1 > 0.0f ? 1.0f : 0.0f;   // V's mean, back (rows that saw keys)
        device T* oA = out + ((size_t)hA * RR + rA) * D;
        device T* oB = out + ((size_t)hB * RR + rB) * D;
        const device float* mu = vmu + (size_t)kvh * D;
        #define OUT_TILE(OT, ACC) \
            for (int i = 0; i < 16; ++i) { \
                const bool r1 = (i >> 2) & 1; \
                const int col = 32 * OT + 16 * (i >> 3) + fn0 + (i & 3); \
                (r1 ? oB : oA)[col] = static_cast<T>(ACC[i] * (r1 ? i1 : i0) + mu[col] * (r1 ? u1 : u0)); }
        FOR_O(OUT_TILE)
    }
"""


@lru_cache(None)
def _k(mode):
    return mx.fast.metal_kernel(name=f"fatt16q{mode}",
                                input_names=["q", "q8", "qsc", "k8", "v8", "ksc", "vsc", "vmu", "mask", "scale", "n_keys"],
                                output_names=["out"], source=_SRC, header=f"#define MODE {mode}\n" + _HDR,
                                ensure_row_contiguous=True)


def _q8(x, axis=-1):
    s = mx.max(mx.abs(x), axis=axis) / 127.0
    xi = mx.clip(mx.round(x / mx.expand_dims(mx.maximum(s, 1e-30), axis)), -127, 127).astype(mx.int8)
    return xi, s


def quant_kv(k, v):
    """k / v [1, Hk, N, D] -> the int8 cache dict (the cache write's work: smoothing, per-token scales, padding)"""
    N = k.shape[2]
    pad = (N + 31) // 32 * 32 - N
    kf, vf = k[0].astype(mx.float32), v[0].astype(mx.float32)
    kmu, vmu = mx.mean(kf, axis=1, keepdims=True), mx.mean(vf, axis=1, keepdims=True)
    k8, ksc = _q8(kf - kmu)
    v8, vsc = _q8(vf - vmu)
    if pad:
        k8, v8 = (mx.pad(a, [(0, 0), (0, pad), (0, 0)]) for a in (k8, v8))
        ksc, vsc = (mx.pad(a, [(0, 0), (0, pad)]) for a in (ksc, vsc))
    c = {"k8": k8[None], "v8": v8[None], "ksc": ksc, "vsc": vsc, "vmu": vmu[:, 0, :], "N": N}
    mx.eval(c["k8"], c["v8"], c["ksc"], c["vsc"], c["vmu"])
    return c


def quant_q(q):
    """q [1, Hq, R, D] -> (q8 int8 [1, Hq, R, D], qsc float32 [Hq * R]): one scale per (head, row)"""
    q8, s = _q8(q.astype(mx.float32))
    return q8, s.reshape(-1)


def fatt16q(q, c, scale, mask, mode, ks=4):
    """q [1, Hq, R, D] (bf16), c = quant_kv(k, v), mask [R, N] bool -> [1, Hq, R, D]"""
    B, Hq, R, D = q.shape
    Hk, N = c["k8"].shape[1], c["N"]
    assert B == 1 and R in (16, 8) and D % 32 == 0 and Hq % Hk == 0 and (Hq // Hk) % (16 // R) == 0, (q.shape, Hk)
    m = mx.contiguous(mask.astype(mx.bool_))
    if mode == 1:
        q8, qsc = quant_q(q)
    else:
        q8, qsc = mx.zeros((1,), mx.int8), mx.zeros((1,), mx.float32)
    (o,) = _k(mode)(inputs=[q, q8, qsc, c["k8"], c["v8"], c["ksc"], c["vsc"], c["vmu"], m, mx.array([scale], mx.float32),
                            mx.array([N], mx.int32)],
                    template=[("T", q.dtype), ("D_C", D), ("HQ_C", Hq), ("HK_C", Hk), ("RR_C", R), ("KS_C", ks), ("mstride", N)],
                    grid=(32 * ks * (Hq * R // 16), 1, 1), threadgroup=(32 * ks, 1, 1),
                    output_shapes=[q.shape], output_dtypes=[q.dtype])
    return o
