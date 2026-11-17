"""HC8-1P (Leap, Wed 2026-11-04; BUILD_DAY_1104 #1 FAST; TABLE_350 'HC8-1P'): the 16-row hyper-connection read for the
FAST track with 8-bit mixers (the HC8 recipe: MLX affine 8-bit g64 of input_mix_weight_down / up; block_inject bf16), in
three dispatches that read the streams once:
  K1 (down): one simdgroup per (32-column tile, stream-aligned K partition). The pending write h' = h + bo x inj with the
     reference's bf16 ops (tile 0 stores it as `written` and sums h'^2 per row); A = bf16(h' * w_norm) (the norm weight
     before the GEMM, 1 / rms after it); a code MMA on the 8-bit codes (bf16(code) is exact), y = s * P + b * X per
     64-group (qmm16c's form with 8-bit codes); tile 10 = the 4 inject columns with bf16 weights. Out: fp32 partials
     [NP, 16, 352] and sums of squares [NP, 16].
  K2 (gate): rinv = rsqrt(mean_s(h'^2) + eps) per (row, stream); c = sum_p partial_p * rinv_{stream(p)} (fixed order);
     mix1 = bf16(silu(c / 4)); inject = bf16(2 sigmoid(logit / 4)).
  K3 (up + tail): one threadgroup per 32 channels, one simdgroup per stream: up = mix1 @ W_up8^T (code MMA, K 320), then
     mixed = (1/4) sum_s sigmoid(up_s) * h'_s * rinv_s * w_s in fp32 (streams in order), one bf16 rounding.
Not bitwise vs REF2-T (8-bit mixers; the norm moved after the GEMM; fp32 gates): a FAST read, judged by same-scores.
prep(g) -> P (the 8-bit mixers of one GatedResidual); read(P, h, bo=None, inj=None, NP=16) -> (mixed [16, D],
written [16, HC * D], inject [16, HC]); emulate(...) = the same math in fp32 MLX ops (the correctness reference)."""
from functools import lru_cache
import mlx.core as mx

BF = mx.bfloat16
_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_MMA = r"""
    const uint lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));         // fragment rows fm, fm + 8
    const short fq = ((qid & 2) | (lane & 1));                // fragment columns 4 fq .. 4 fq + 3 (and + 16)
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc[16];
    for (short i = 0; i < 16; i++) acc[i] = 0.0f;
"""

# K1: grid NT * NP simdgroups (threadgroup = 1 simdgroup); NC = NT * 32 partial columns
_K1 = _MMA + r"""
    constexpr int G = 64, NC = NT * 32, KG = HCD / G;
    const int tg = int(threadgroup_position_in_grid.x);
    const int tile = tg % NT, part = tg / NT;
    const int n0 = tile * 32;
    const int kp0 = part * KP;
    const int st = kp0 / D;                                   // the stream this partition lies in
    const bool itile = n0 >= LR;                              // the inject tile (bf16 weights, 4 valid columns)
    float ss0 = 0.0f, ss1 = 0.0f;
    float i0 = 0.0f, i1 = 0.0f;
    if (PEND) { i0 = float(injp[fm * HC + st]); i1 = float(injp[(fm + 8) * HC + st]); }
    for (int k0 = kp0; k0 < kp0 + KP; k0 += G) {
        const int g = k0 / G;
        const int ka = k0 + 16 * fq;
        T a0[16], a1[16];
        float xs0 = 0.0f, xs1 = 0.0f;
        for (short j = 0; j < 16; j++) {
            T h0 = h[(size_t)fm * HCD + ka + j], h1 = h[(size_t)(fm + 8) * HCD + ka + j];
            if (PEND) {
                const int d = (ka + j) - st * D;
                h0 = static_cast<T>(float(h0) + float(static_cast<T>(float(bo[(size_t)fm * D + d]) * i0)));
                h1 = static_cast<T>(float(h1) + float(static_cast<T>(float(bo[(size_t)(fm + 8) * D + d]) * i1)));
            }
            if (tile == 0) {
                hw[(size_t)fm * HCD + ka + j] = h0;
                hw[(size_t)(fm + 8) * HCD + ka + j] = h1;
                ss0 += float(h0) * float(h0);
                ss1 += float(h1) * float(h1);
            }
            const float w_ = float(wn[ka + j]);
            a0[j] = static_cast<T>(float(h0) * w_);
            a1[j] = static_cast<T>(float(h1) * w_);
            xs0 += float(a0[j]);
            xs1 += float(a1[j]);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        if (!itile) {
            uint4 q[4];
            for (short r = 0; r < 4; r++) {
                const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                q[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + ka);
            }
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const uint word = q[r][s];
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short bk = 0; bk < 2; bk++)
                for (short j = 0; j < 4; j++) {
                    const int n = n0 + 16 * bk + 4 * fq + j;
                    const float s_ = float(ws[(size_t)n * KG + g]), b_ = float(wb[(size_t)n * KG + g]);
                    const short ia = bk * 8 + j, ib = bk * 8 + 4 + j;
                    acc[ia] += s_ * ct_c[ia] + b_ * xs0;
                    acc[ib] += s_ * ct_c[ib] + b_ * xs1;
                }
        } else {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const int nl = (r >> 1) * 16 + fm + (r & 1) * 8;
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++)
                        ct_b[base + j] = nl < HC ? wi[(size_t)nl * HCD + ka + 4 * s + j] : T(0);
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short i = 0; i < 16; i++) acc[i] += ct_c[i];
        }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        part_out[((size_t)part * 16 + m) * NC + n] = acc[i];
    }
    if (tile == 0) {
        ss0 += simd_shuffle_xor(ss0, 1); ss0 += simd_shuffle_xor(ss0, 8);
        ss1 += simd_shuffle_xor(ss1, 1); ss1 += simd_shuffle_xor(ss1, 8);
        if (fq == 0) { ssq[part * 16 + fm] = ss0; ssq[part * 16 + fm + 8] = ss1; }
    }
"""

_K2 = r"""
    constexpr int N = LR + HC, PPS = NP / HC;
    const uint i = thread_position_in_grid.x;
    if (i >= 16 * N) return;
    const uint r = i / N, n = i % N;
    float rv[HC];
    for (int s = 0; s < HC; s++) {
        float ss = 0.0f;
        for (int p = s * PPS; p < (s + 1) * PPS; p++) ss += ssq[p * 16 + r];
        rv[s] = metal::precise::rsqrt(ss / float(D) + EPS);
    }
    float c = 0.0f;
    for (int p = 0; p < NP; p++) c += part_in[((size_t)p * 16 + r) * NC + n] * rv[p / PPS];
    if (n < LR) {
        const float z = c / float(HC);
        mix1[r * LR + n] = static_cast<T>(z / (1.0f + metal::precise::exp(-z)));
    } else {
        inject[r * HC + (n - LR)] = static_cast<T>(2.0f / (1.0f + metal::precise::exp(-c / float(HC))));
    }
    if (n == 0) for (int s = 0; s < HC; s++) rinv[r * HC + s] = rv[s];
"""

_K3 = _MMA + r"""
    constexpr int G = 64, KG = LR / G;
    const int sg = int(simdgroup_index_in_threadgroup);       // the stream
    const int ch0 = int(threadgroup_position_in_grid.x) * 32;
    const int c0 = sg * D + ch0;                              // this simdgroup's first up column
    for (int g = 0; g < KG; g++) {
        const int k0 = g * G, ka = k0 + 16 * fq;
        uint4 q[4];
        for (short r = 0; r < 4; r++) {
            const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + ka);
        }
        T a0[16], a1[16];
        float xs0 = 0.0f, xs1 = 0.0f;
        for (short j = 0; j < 16; j++) {
            a0[j] = mix1[fm * LR + ka + j]; a1[j] = mix1[(fm + 8) * LR + ka + j];
            xs0 += float(a0[j]); xs1 += float(a1[j]);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
            for (short r = 0; r < 4; r++) {
                const uint word = q[r][s];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
            }
            op.run(ct_a, ct_b, ct_c);
        }
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const int n = c0 + 16 * bk + 4 * fq + j;
                const float s_ = float(us[(size_t)n * KG + g]), b_ = float(ub[(size_t)n * KG + g]);
                const short ia = bk * 8 + j, ib = bk * 8 + 4 + j;
                acc[ia] += s_ * ct_c[ia] + b_ * xs0;
                acc[ib] += s_ * ct_c[ib] + b_ * xs1;
            }
    }
    threadgroup float tp[HC * 16 * 32];
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
        const int col = c0 + nl;
        const float gate = 1.0f / (1.0f + metal::precise::exp(-acc[i]));
        const float nv = float(hw[(size_t)m * HCD + col]) * rinv[m * HC + sg] * float(wn[col]);
        tp[(sg * 16 + m) * 32 + nl] = gate * nv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = thread_index_in_threadgroup;
    for (uint o = t; o < 16 * 32; o += 32 * HC) {
        const uint m = o / 32, nl = o % 32;
        float sum = 0.0f;
        for (int s = 0; s < HC; s++) sum += tp[(s * 16 + m) * 32 + nl];
        mixed[(size_t)m * D + ch0 + nl] = static_cast<T>(sum / float(HC));
    }
"""


# K1 v2 (post-hoc, after the BETWEEN screen: K1 was 38.5 us per read, all 11 tiles rebuilding A from scalar loads): one
# threadgroup per K partition, NT simdgroups (one per column tile). Per 64-group, threads 0..127 build A once (8 values
# each: uint4 loads of h / bo / w_norm, the pending write, A = bf16(h' * w)) into threadgroup memory (double-buffered),
# store written, sum h'^2 and the per-row group sums X; one barrier; each simdgroup then runs its MMAs with A from
# threadgroup memory and its codes prefetched one group ahead. Same arithmetic as K1 per output (bitwise the same partials
# up to the fp32 order of X and the h'^2 sums).
_K1B = _MMA + r"""
    constexpr int G = 64, NC = NT * 32, KG = HCD / G, NG = KP / G;
    const int part = int(threadgroup_position_in_grid.x);
    const int tile = int(simdgroup_index_in_threadgroup);
    const uint t = thread_index_in_threadgroup;
    const int n0 = tile * 32;
    const int kp0 = part * KP;
    const int st = kp0 / D;
    const bool itile = n0 >= LR;
    threadgroup T As[2][16 * G];
    threadgroup float Xs[2][16];
    // the A builders: t < 128 -> row t / 8, values (t % 8) * 8 .. + 7 of the group
    const int ar = int(t) / 8, ak = (int(t) % 8) * 8;
    float ss = 0.0f;
    const float ia = (PEND && t < 128) ? float(injp[ar * HC + st]) : 0.0f;
    // codes of this lane's 4 B rows, one group ahead
    uint4 q[4], qn[4];
    if (!itile) {
        for (short r = 0; r < 4; r++) {
            const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            q[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + kp0 + 16 * fq);
        }
    }
    for (int gi = 0; gi < NG; gi++) {
        const int k0 = kp0 + gi * G, g = k0 / G, buf = gi & 1;
        if (t < 128) {
            const int k = k0 + ak;
            const uint4 hv = *(const device uint4*)(h + (size_t)ar * HCD + k);
            const uint4 wv = *(const device uint4*)(wn + k);
            thread const T* hp = (thread const T*)&hv;
            thread const T* wp = (thread const T*)&wv;
            T hn[8], an[8];
            float xs = 0.0f;
            if (PEND) {
                const uint4 bv = *(const device uint4*)(bo + (size_t)ar * D + (k - st * D));
                thread const T* bp = (thread const T*)&bv;
                for (short j = 0; j < 8; j++)
                    hn[j] = static_cast<T>(float(hp[j]) + float(static_cast<T>(float(bp[j]) * ia)));
            } else {
                for (short j = 0; j < 8; j++) hn[j] = hp[j];
            }
            for (short j = 0; j < 8; j++) {
                ss += float(hn[j]) * float(hn[j]);
                an[j] = static_cast<T>(float(hn[j]) * float(wp[j]));
                xs += float(an[j]);
            }
            *(device uint4*)(hw + (size_t)ar * HCD + k) = *(thread uint4*)hn;
            *(threadgroup uint4*)(&As[buf][ar * G + ak]) = *(thread uint4*)an;
            xs += simd_shuffle_xor(xs, 1); xs += simd_shuffle_xor(xs, 2); xs += simd_shuffle_xor(xs, 4);
            if ((t & 7) == 0) Xs[buf][ar] = xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!itile && gi + 1 < NG) {
            for (short r = 0; r < 4; r++) {
                const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + k0 + G + 16 * fq);
            }
        }
        const float xs0 = Xs[buf][fm], xs1 = Xs[buf][fm + 8];
        threadgroup const T* a0 = &As[buf][fm * G + 16 * fq];
        threadgroup const T* a1 = &As[buf][(fm + 8) * G + 16 * fq];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        if (!itile) {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const uint word = q[r][s];
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short bk = 0; bk < 2; bk++)
                for (short j = 0; j < 4; j++) {
                    const int n = n0 + 16 * bk + 4 * fq + j;
                    const float s_ = float(ws[(size_t)n * KG + g]), b_ = float(wb[(size_t)n * KG + g]);
                    const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                    acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                    acc[i1] += s_ * ct_c[i1] + b_ * xs1;
                }
            for (short r = 0; r < 4; r++) q[r] = qn[r];
        } else {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const int nl = (r >> 1) * 16 + fm + (r & 1) * 8;
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++)
                        ct_b[base + j] = nl < HC ? wi[(size_t)nl * HCD + k0 + 16 * fq + 4 * s + j] : T(0);
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short i = 0; i < 16; i++) acc[i] += ct_c[i];
        }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        part_out[((size_t)part * 16 + m) * NC + n] = acc[i];
    }
    if (t < 128) {
        ss += simd_shuffle_xor(ss, 1); ss += simd_shuffle_xor(ss, 2); ss += simd_shuffle_xor(ss, 4);
        if ((t & 7) == 0) ssq[part * 16 + ar] = ss;
    }
"""

# K3 v2 (post-hoc): mix1 and its per-(row, group) sums staged once per threadgroup in threadgroup memory; the codes of
# the next group prefetched; the tail's h' / w_norm loads issued before the GEMM.
_K3B = _MMA + r"""
    constexpr int G = 64, KG = LR / G;
    const int sg = int(simdgroup_index_in_threadgroup);
    const uint t = thread_index_in_threadgroup;
    const int ch0 = int(threadgroup_position_in_grid.x) * 32;
    const int c0 = sg * D + ch0;
    threadgroup T Ms[16 * LR];
    threadgroup float Xs[16 * KG];
    threadgroup float tp[HC * 16 * 32];
    uint4 q[4], qn[4];
    for (short r = 0; r < 4; r++) {
        const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
        q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);
    }
    // the tail's operands, loaded before the GEMM: rows fm / fm + 8, columns 4 fq .. + 3 and 16 + 4 fq .. + 3
    T hv[16], wv[16];
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int col = c0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        hv[i] = hw[(size_t)m * HCD + col];
        wv[i] = wn[col];
    }
    const float r0 = rinv[fm * HC + sg], r1 = rinv[(fm + 8) * HC + sg];
    for (uint o = t; o < 16 * LR / 8; o += 32 * HC)
        *(threadgroup uint4*)(&Ms[o * 8]) = *(const device uint4*)(mix1 + o * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < 16 * KG; o += 32 * HC) {
        const uint m = o / KG, gg = o % KG;
        float xs = 0.0f;
        for (int k = 0; k < G; k++) xs += float(Ms[m * LR + gg * G + k]);
        Xs[o] = xs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < KG; g++) {
        const int k0 = g * G;
        if (g + 1 < KG) {
            for (short r = 0; r < 4; r++) {
                const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + k0 + G + 16 * fq);
            }
        }
        threadgroup const T* a0 = &Ms[fm * LR + k0 + 16 * fq];
        threadgroup const T* a1 = &Ms[(fm + 8) * LR + k0 + 16 * fq];
        const float xs0 = Xs[fm * KG + g], xs1 = Xs[(fm + 8) * KG + g];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
            for (short r = 0; r < 4; r++) {
                const uint word = q[r][s];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
            }
            op.run(ct_a, ct_b, ct_c);
        }
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const int n = c0 + 16 * bk + 4 * fq + j;
                const float s_ = float(us[(size_t)n * KG + g]), b_ = float(ub[(size_t)n * KG + g]);
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                acc[i1] += s_ * ct_c[i1] + b_ * xs1;
            }
        for (short r = 0; r < 4; r++) q[r] = qn[r];
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
        const float gate = 1.0f / (1.0f + metal::precise::exp(-acc[i]));
        const float nv = float(hv[i]) * (((i & 7) >> 2) ? r1 : r0) * float(wv[i]);
        tp[(sg * 16 + m) * 32 + nl] = gate * nv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < 16 * 32; o += 32 * HC) {
        const uint m = o / 32, nl = o % 32;
        float sum = 0.0f;
        for (int s = 0; s < HC; s++) sum += tp[(s * 16 + m) * 32 + nl];
        mixed[(size_t)m * D + ch0 + nl] = static_cast<T>(sum / float(HC));
    }
"""
_K1C = _MMA + r"""
    constexpr int G = 64, NC = NT * 32, KG = HCD / G, NG = KP / G;
    const int part = int(threadgroup_position_in_grid.x);
    const int tile = int(simdgroup_index_in_threadgroup);
    const uint t = thread_index_in_threadgroup;
    const int n0 = tile * 32;
    const int kp0 = part * KP;
    const int st = kp0 / D;
    const bool itile = n0 >= LR;
    threadgroup T As[2][16 * G];
    threadgroup float Xs[2][16];
    // the A builders: t < 128 -> row t / 8, values (t % 8) * 8 .. + 7 of the group
    const int ar = int(t) / 8, ak = (int(t) % 8) * 8;
    float ss = 0.0f;
    const float ia = (PEND && t < 128) ? float(injp[ar * HC + st]) : 0.0f;
    uint4 hv_n = uint4(0), wv_n = uint4(0), bv_n = uint4(0);
    if (t < 128) {
        hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);
        wv_n = *(const device uint4*)(wn + kp0 + ak);
        if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));
    }
    // codes of this lane's 4 B rows, one group ahead
    uint4 q[4], qn[4];
    if (!itile) {
        for (short r = 0; r < 4; r++) {
            const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            q[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + kp0 + 16 * fq);
        }
    }
    for (int gi = 0; gi < NG; gi++) {
        const int k0 = kp0 + gi * G, g = k0 / G, buf = gi & 1;
        if (t < 128) {
            const int k = k0 + ak;
            const uint4 hv = hv_n, wv = wv_n;
            const uint4 bv_c = bv_n;
            if (gi + 1 < NG) {
                hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);
                wv_n = *(const device uint4*)(wn + k + G);
                if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));
            }
            thread const T* hp = (thread const T*)&hv;
            thread const T* wp = (thread const T*)&wv;
            T hn[8], an[8];
            float xs = 0.0f;
            if (PEND) {
                const uint4 bv = bv_c;
                thread const T* bp = (thread const T*)&bv;
                for (short j = 0; j < 8; j++)
                    hn[j] = static_cast<T>(float(hp[j]) + float(static_cast<T>(float(bp[j]) * ia)));
            } else {
                for (short j = 0; j < 8; j++) hn[j] = hp[j];
            }
            for (short j = 0; j < 8; j++) {
                ss += float(hn[j]) * float(hn[j]);
                an[j] = static_cast<T>(float(hn[j]) * float(wp[j]));
                xs += float(an[j]);
            }
            *(device uint4*)(hw + (size_t)ar * HCD + k) = *(thread uint4*)hn;
            *(threadgroup uint4*)(&As[buf][ar * G + ak]) = *(thread uint4*)an;
            xs += simd_shuffle_xor(xs, 1); xs += simd_shuffle_xor(xs, 2); xs += simd_shuffle_xor(xs, 4);
            if ((t & 7) == 0) Xs[buf][ar] = xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!itile && gi + 1 < NG) {
            for (short r = 0; r < 4; r++) {
                const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + k0 + G + 16 * fq);
            }
        }
        const float xs0 = Xs[buf][fm], xs1 = Xs[buf][fm + 8];
        threadgroup const T* a0 = &As[buf][fm * G + 16 * fq];
        threadgroup const T* a1 = &As[buf][(fm + 8) * G + 16 * fq];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        if (!itile) {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const uint word = q[r][s];
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short bk = 0; bk < 2; bk++)
                for (short j = 0; j < 4; j++) {
                    const int n = n0 + 16 * bk + 4 * fq + j;
                    const float s_ = float(ws[(size_t)n * KG + g]), b_ = float(wb[(size_t)n * KG + g]);
                    const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                    acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                    acc[i1] += s_ * ct_c[i1] + b_ * xs1;
                }
            for (short r = 0; r < 4; r++) q[r] = qn[r];
        } else {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const int nl = (r >> 1) * 16 + fm + (r & 1) * 8;
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++)
                        ct_b[base + j] = nl < HC ? wi[(size_t)nl * HCD + k0 + 16 * fq + 4 * s + j] : T(0);
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short i = 0; i < 16; i++) acc[i] += ct_c[i];
        }
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        part_out[((size_t)part * 16 + m) * NC + n] = acc[i];
    }
    if (t < 128) {
        ss += simd_shuffle_xor(ss, 1); ss += simd_shuffle_xor(ss, 2); ss += simd_shuffle_xor(ss, 4);
        if ((t & 7) == 0) ssq[part * 16 + ar] = ss;
    }
"""
_K1D = _MMA + r"""
    constexpr int G = 64, NC = NT * 32, KG = HCD / G, NG = KP / G;
    constexpr int NTG = (NT + TPT - 1) / TPT;
    const int part = int(threadgroup_position_in_grid.x) / NTG;
    const int tbase = (int(threadgroup_position_in_grid.x) % NTG) * TPT;
    const int tile = tbase + int(simdgroup_index_in_threadgroup);
    const bool live = tile < NT;                                // the last threadgroup may hold fewer tiles
    const uint t = thread_index_in_threadgroup;
    const int n0 = tile * 32;
    const int kp0 = part * KP;
    const int st = kp0 / D;
    const bool itile = n0 >= LR || !live;
    threadgroup T As[2][16 * G];
    threadgroup float Xs[2][16];
    // the A builders: t < 128 -> row t / 8, values (t % 8) * 8 .. + 7 of the group
    const int ar = int(t) / 8, ak = (int(t) % 8) * 8;
    float ss = 0.0f;
    const float ia = (PEND && t < 128) ? float(injp[ar * HC + st]) : 0.0f;
    uint4 hv_n = uint4(0), wv_n = uint4(0), bv_n = uint4(0);
    if (t < 128) {
        hv_n = *(const device uint4*)(h + (size_t)ar * HCD + kp0 + ak);
        wv_n = *(const device uint4*)(wn + kp0 + ak);
        if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (kp0 + ak - st * D));
    }
    // codes of this lane's 4 B rows, one group ahead
    uint4 q[4], qn[4];
    if (!itile) {
        for (short r = 0; r < 4; r++) {
            const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            q[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + kp0 + 16 * fq);
        }
    }
    for (int gi = 0; gi < NG; gi++) {
        const int k0 = kp0 + gi * G, g = k0 / G, buf = gi & 1;
        if (t < 128) {
            const int k = k0 + ak;
            const uint4 hv = hv_n, wv = wv_n;
            const uint4 bv_c = bv_n;
            if (gi + 1 < NG) {
                hv_n = *(const device uint4*)(h + (size_t)ar * HCD + k + G);
                wv_n = *(const device uint4*)(wn + k + G);
                if (PEND) bv_n = *(const device uint4*)(bo + (size_t)ar * D + (k + G - st * D));
            }
            thread const T* hp = (thread const T*)&hv;
            thread const T* wp = (thread const T*)&wv;
            T hn[8], an[8];
            float xs = 0.0f;
            if (PEND) {
                const uint4 bv = bv_c;
                thread const T* bp = (thread const T*)&bv;
                for (short j = 0; j < 8; j++)
                    hn[j] = static_cast<T>(float(hp[j]) + float(static_cast<T>(float(bp[j]) * ia)));
            } else {
                for (short j = 0; j < 8; j++) hn[j] = hp[j];
            }
            for (short j = 0; j < 8; j++) {
                ss += float(hn[j]) * float(hn[j]);
                an[j] = static_cast<T>(float(hn[j]) * float(wp[j]));
                xs += float(an[j]);
            }
            if (tbase == 0) *(device uint4*)(hw + (size_t)ar * HCD + k) = *(thread uint4*)hn;
            *(threadgroup uint4*)(&As[buf][ar * G + ak]) = *(thread uint4*)an;
            xs += simd_shuffle_xor(xs, 1); xs += simd_shuffle_xor(xs, 2); xs += simd_shuffle_xor(xs, 4);
            if ((t & 7) == 0) Xs[buf][ar] = xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!itile && gi + 1 < NG) {
            for (short r = 0; r < 4; r++) {
                const int n = n0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)wq + (size_t)n * HCD + k0 + G + 16 * fq);
            }
        }
        const float xs0 = Xs[buf][fm], xs1 = Xs[buf][fm + 8];
        threadgroup const T* a0 = &As[buf][fm * G + 16 * fq];
        threadgroup const T* a1 = &As[buf][(fm + 8) * G + 16 * fq];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        if (!itile) {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const uint word = q[r][s];
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short bk = 0; bk < 2; bk++)
                for (short j = 0; j < 4; j++) {
                    const int n = n0 + 16 * bk + 4 * fq + j;
                    const float s_ = float(ws[(size_t)n * KG + g]), b_ = float(wb[(size_t)n * KG + g]);
                    const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                    acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                    acc[i1] += s_ * ct_c[i1] + b_ * xs1;
                }
            for (short r = 0; r < 4; r++) q[r] = qn[r];
        } else if (live) {
            for (short s = 0; s < 4; s++) {
                for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
                for (short r = 0; r < 4; r++) {
                    const int nl = (r >> 1) * 16 + fm + (r & 1) * 8;
                    const short base = (r >> 1) * 8 + (r & 1) * 4;
                    for (short j = 0; j < 4; j++)
                        ct_b[base + j] = nl < HC ? wi[(size_t)nl * HCD + k0 + 16 * fq + 4 * s + j] : T(0);
                }
                op.run(ct_a, ct_b, ct_c);
            }
            for (short i = 0; i < 16; i++) acc[i] += ct_c[i];
        }
    }
    if (live)
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
            part_out[((size_t)part * 16 + m) * NC + n] = acc[i];
        }
    if (t < 128 && tbase == 0) {
        ss += simd_shuffle_xor(ss, 1); ss += simd_shuffle_xor(ss, 2); ss += simd_shuffle_xor(ss, 4);
        if ((t & 7) == 0) ssq[part * 16 + ar] = ss;
    }
"""

_K2B = r"""
    constexpr int N = LR + HC, PPS = NP / HC, KG3 = LR / 64;
    const uint r = threadgroup_position_in_grid.x;
    const uint n = thread_index_in_threadgroup;
    threadgroup T ms[LR];
    if (n < N) {
        float rv[HC];
        for (int s = 0; s < HC; s++) {
            float ss = 0.0f;
            for (int p = s * PPS; p < (s + 1) * PPS; p++) ss += ssq[p * 16 + r];
            rv[s] = metal::precise::rsqrt(ss / float(D) + EPS);
        }
        float c = 0.0f;
        for (int p = 0; p < NP; p++) c += part_in[((size_t)p * 16 + r) * NC + n] * rv[p / PPS];
        if (n < LR) {
            const float z = c / float(HC);
            const T m = static_cast<T>(z / (1.0f + metal::precise::exp(-z)));
            mix1[r * LR + n] = m;
            ms[n] = m;
        } else {
            inject[r * HC + (n - LR)] = static_cast<T>(2.0f / (1.0f + metal::precise::exp(-c / float(HC))));
        }
        if (n == 0) for (int s = 0; s < HC; s++) rinv[r * HC + s] = rv[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n < KG3) {
        float xs = 0.0f;
        for (int k = 0; k < 64; k++) xs += float(ms[n * 64 + k]);
        xsum[r * KG3 + n] = xs;
    }
"""

_K3C = _MMA + r"""
    constexpr int G = 64, KG = LR / G;
    const int sg = int(simdgroup_index_in_threadgroup);
    const uint t = thread_index_in_threadgroup;
    const int ch0 = int(threadgroup_position_in_grid.x) * 32;
    const int c0 = sg * D + ch0;
    threadgroup T Ms[16 * LR];
    threadgroup float tp[HC * 16 * 32];
    uint4 q[4], qn[4];
    for (short r = 0; r < 4; r++) {
        const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
        q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);
    }
    // the tail's operands, loaded before the GEMM: rows fm / fm + 8, columns 4 fq .. + 3 and 16 + 4 fq .. + 3
    T hv[16], wv[16];
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int col = c0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        hv[i] = hw[(size_t)m * HCD + col];
        wv[i] = wn[col];
    }
    const float r0 = rinv[fm * HC + sg], r1 = rinv[(fm + 8) * HC + sg];
    for (uint o = t; o < 16 * LR / 8; o += 32 * HC)
        *(threadgroup uint4*)(&Ms[o * 8]) = *(const device uint4*)(mix1 + o * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < KG; g++) {
        const int k0 = g * G;
        if (g + 1 < KG) {
            for (short r = 0; r < 4; r++) {
                const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + k0 + G + 16 * fq);
            }
        }
        threadgroup const T* a0 = &Ms[fm * LR + k0 + 16 * fq];
        threadgroup const T* a1 = &Ms[(fm + 8) * LR + k0 + 16 * fq];
        const float xs0 = xsum[fm * KG + g], xs1 = xsum[(fm + 8) * KG + g];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
            for (short r = 0; r < 4; r++) {
                const uint word = q[r][s];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
            }
            op.run(ct_a, ct_b, ct_c);
        }
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const int n = c0 + 16 * bk + 4 * fq + j;
                const float s_ = float(us[(size_t)n * KG + g]), b_ = float(ub[(size_t)n * KG + g]);
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                acc[i1] += s_ * ct_c[i1] + b_ * xs1;
            }
        for (short r = 0; r < 4; r++) q[r] = qn[r];
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
        const float gate = 1.0f / (1.0f + metal::precise::exp(-acc[i]));
        const float nv = float(hv[i]) * (((i & 7) >> 2) ? r1 : r0) * float(wv[i]);
        tp[(sg * 16 + m) * 32 + nl] = gate * nv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < 16 * 32; o += 32 * HC) {
        const uint m = o / 32, nl = o % 32;
        float sum = 0.0f;
        for (int s = 0; s < HC; s++) sum += tp[(s * 16 + m) * 32 + nl];
        mixed[(size_t)m * D + ch0 + nl] = static_cast<T>(sum / float(HC));
    }
"""


@lru_cache(None)
def _kern(which, eps):
    if which == 1:
        return mx.fast.metal_kernel(name="hc8r_k1", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1, header=_HDR)
    if which == 2:
        return mx.fast.metal_kernel(name=f"hc8r_k2_{eps!r}".replace(".", "_").replace("-", "m"),
                                    input_names=["part_in", "ssq"], output_names=["mix1", "inject", "rinv"],
                                    source=f"constexpr float EPS = {eps!r}f;\n" + _K2)
    if which == 9:
        return mx.fast.metal_kernel(name="hc8r_k1d", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1D, header=_HDR)
    if which == 6:
        return mx.fast.metal_kernel(name="hc8r_k1c", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1C, header=_HDR)
    if which == 7:
        return mx.fast.metal_kernel(name=f"hc8r_k2b_{eps!r}".replace(".", "_").replace("-", "m"),
                                    input_names=["part_in", "ssq"], output_names=["mix1", "inject", "rinv", "xsum"],
                                    source=f"constexpr float EPS = {eps!r}f;\n" + _K2B)
    if which == 8:
        return mx.fast.metal_kernel(name="hc8r_k3c", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn", "xsum"],
                                    output_names=["mixed"], source=_K3C, header=_HDR)
    if which == 4:
        return mx.fast.metal_kernel(name="hc8r_k1b", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1B, header=_HDR)
    if which == 5:
        return mx.fast.metal_kernel(name="hc8r_k3b", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn"],
                                    output_names=["mixed"], source=_K3B, header=_HDR)
    return mx.fast.metal_kernel(name="hc8r_k3", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn"],
                                output_names=["mixed"], source=_K3, header=_HDR)


def prep(g, bits=8, group_size=64):
    """the 8-bit mixers of one GatedResidual (the HC8 recipe), its bf16 inject weight and norm weight"""
    wq, ws, wb = mx.quantize(g.input_mix_weight_down.weight.astype(BF), group_size=group_size, bits=bits)
    uq, us, ub = mx.quantize(g.input_mix_weight_up.weight.astype(BF), group_size=group_size, bits=bits)
    P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=g.block_inject_weight.weight.astype(BF),
             wn=g.hc_norm.weight.astype(BF), hc=int(g.hc_count), d=int(g.hidden_size),
             lr=int(g.input_mix_weight_down.weight.shape[0]), eps=float(g.hc_norm.eps))
    mx.eval([v for v in P.values() if isinstance(v, mx.array)])
    return P


def read(P, h, bo=None, inj=None, NP=16, v=1):
    """v 1: the pre-registered kernels; v 2: K1 / K3 v2 (A built once per partition, codes prefetched; ref2's hc8_fast);
    v 4: v2's arithmetic, bitwise (K1 builders prefetch the next group's inputs; K2 per row, with the mix1 group sums;
    K3 without its serial sums)"""
    """h [16, HC * D] bf16 (the hyper input), the pending (bo [16, D], inj [16, HC]) or None ->
    (mixed [16, D], written [16, HC * D], inject [16, HC])"""
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    assert D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and h.shape == (16, HCD)
    pend = bo is not None
    if v == 5:
        TPT = 4
        NTG = (NT + TPT - 1) // TPT
        part, ssq, hw = _kern(9, None)(
            inputs=[h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
            template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT),
                      ("PEND", int(pend)), ("TPT", TPT)],
            grid=(32 * TPT * NTG * NP, 1, 1), threadgroup=(32 * TPT, 1, 1),
            output_shapes=[(NP, 16, NT * 32), (NP, 16), (16, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
        v = 4
    else:
        part, ssq, hw = _kern({1: 1, 2: 4, 4: 6}[v], None)(
            inputs=[h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
            template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT),
                      ("PEND", int(pend))],
            grid=(32 * NT * NP, 1, 1), threadgroup=(32 if v == 1 else 32 * NT, 1, 1),
            output_shapes=[(NP, 16, NT * 32), (NP, 16), (16, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
    if v == 4:
        mix1, inject, rinv, xsum = _kern(7, P["eps"])(
            inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", D)],
            grid=(16 * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1),
            output_shapes=[(16, LR), (16, HC), (16, HC), (16, LR // 64)], output_dtypes=[BF, BF, mx.float32, mx.float32])
        (mixed,) = _kern(8, None)(
            inputs=[mix1, P["uq"], P["us"], P["ub"], hw, rinv, P["wn"], xsum],
            template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC)],
            grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1),
            output_shapes=[(16, D)], output_dtypes=[BF])
        return mixed, hw, inject
    mix1, inject, rinv = _kern(2, P["eps"])(
        inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", D)],
        grid=(16 * (LR + HC), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(16, LR), (16, HC), (16, HC)], output_dtypes=[BF, BF, mx.float32])
    (mixed,) = _kern(3 if v == 1 else 5, None)(
        inputs=[mix1, P["uq"], P["us"], P["ub"], hw, rinv, P["wn"]],
        template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC)],
        grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1),
        output_shapes=[(16, D)], output_dtypes=[BF])
    return mixed, hw, inject


def _deq(q, s, b, group_size=64):
    """exact fp32 dequantization of MLX affine 8-bit (s * code + b, no bf16 rounding)"""
    N = q.shape[0]
    codes = mx.view(q, mx.uint8).reshape(N, -1).astype(mx.float32)
    K = codes.shape[1]
    return (codes.reshape(N, K // group_size, group_size) * s.astype(mx.float32)[..., None]
            + b.astype(mx.float32)[..., None]).reshape(N, K)


def emulate(P, h, bo=None, inj=None):
    """read()'s math in fp32 MLX ops (default stream): the correctness reference for the kernels"""
    HC, D, LR = P["hc"], P["d"], P["lr"]
    f32 = mx.float32
    if bo is not None:
        prod = (bo[:, None, :].astype(f32) * inj[:, :, None].astype(f32)).astype(BF)
        hw = (h.reshape(16, HC, D).astype(f32) + prod.astype(f32)).astype(BF).reshape(16, HC * D)
    else:
        hw = h
    w = P["wn"].astype(f32)
    a = (hw.astype(f32) * w).astype(BF).astype(f32)
    rinv = mx.rsqrt(mx.mean(hw.astype(f32).reshape(16, HC, D) ** 2, axis=-1) + P["eps"])      # [16, HC]
    Wd = _deq(P["wq"], P["ws"], P["wb"])
    Wi = P["wi"].astype(f32)
    c = sum((a[:, s * D:(s + 1) * D] @ Wd[:, s * D:(s + 1) * D].T) * rinv[:, s:s + 1] for s in range(HC))
    lg = sum((a[:, s * D:(s + 1) * D] @ Wi[:, s * D:(s + 1) * D].T) * rinv[:, s:s + 1] for s in range(HC))
    z = c / HC
    mix1 = (z * mx.sigmoid(z)).astype(BF)
    inject = (2.0 * mx.sigmoid(lg / HC)).astype(BF)
    up = mix1.astype(f32) @ _deq(P["uq"], P["us"], P["ub"]).T                                  # [16, HC * D]
    nv = hw.astype(f32).reshape(16, HC, D) * rinv[:, :, None] * w.reshape(HC, D)
    mixed = mx.mean(mx.sigmoid(up.reshape(16, HC, D)) * nv, axis=1).astype(BF)
    return mixed, hw, inject


def hc8_install(model, bits=8, group_size=64, ple=False, keep=False):
    """the HC8 recipe (Breakthrough round 28 v3n/hc8.py; the map's hc8w / hc8w_mix): every trunk GatedResidual's
    input_mix_weight_down / up (and, with ple, the PLE layer's key / value projections) -> MLX affine 8-bit g64
    nn.QuantizedLinear (from_linear); the MTP head's modules are left alone; block_inject stays bf16. A FAST-track model
    change: prefill and every read then use the 8-bit mixers (MTPLX's eager chain, or ref2's hc8_fast read at <= 16
    rows). Returns (swapped, MB before, MB after, restore); restore is None unless keep=True (keeping it holds the bf16
    originals in memory, so the RAM saving only happens with keep=False)."""
    import mlx.nn as nn
    from mtplx.models import qwen4_exp as Q
    TM = getattr(model, "language_model", model)
    undo, before, after = [], 0, 0
    for name, mod in TM.named_modules():
        if name.startswith("mtp"):
            continue
        tg = (["input_mix_weight_down", "input_mix_weight_up"] if isinstance(mod, Q.GatedResidual) else
              ["key_proj", "value_proj"] if ple and isinstance(mod, Q.PLELayer) else [])
        for attr in tg:
            lin = getattr(mod, attr)
            if not isinstance(lin, nn.Linear) or lin.weight.shape[1] % group_size:
                continue
            q = nn.QuantizedLinear.from_linear(lin, group_size, bits)
            mx.eval(q.parameters())
            before += lin.weight.nbytes
            after += sum(v.nbytes for k, v in q.items() if isinstance(v, mx.array))
            setattr(mod, attr, q)
            undo.append((mod, attr, lin) if keep else None)
    def restore():
        for mod, attr, lin in undo:
            setattr(mod, attr, lin)
    return len(undo), before / 1e6, after / 1e6, (restore if keep else None)
