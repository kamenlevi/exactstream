"""gdn_w.py (Fable B, DENSE_REBUILD, Fri 2026-11-06): DRAFT of kernel W for a GDN layer. NOT RUN ON A GPU.
One dispatch for everything in a GDN layer that needs no cross-head sum:
  the four input projections (qkv | z | a | b, 4-bit g64)  ->  conv + silu + q / k l2norm  ->  the gated-delta recurrence
  ->  the gated RMS norm.   Output y [16, 6144] is out_proj's input; out_proj itself moves into the next HC read's kernel
  (hc_d.py), because it is the only step of the layer that sums over heads.
Why one dispatch is possible: the layer splits into 16 independent pipelines, one per key head (1 key head feeds 3 value
heads). Nothing in a pipeline waits for another pipeline, so there is no global sync point inside the layer body; the three
phase changes are threadgroup barriers, which only the 1,024 threads of one pipeline wait on.
Layout: grid 16 threadgroups (z = key head) x 1,024 threads (32 simdgroups).
  A   simdgroup i = tile i (32 output columns) of the head's packed weight stream, full K (40 quant groups):
        tiles 0-3 q, 4-7 k, 8-19 v (3 heads), 20-31 z (3 heads). Hot loop = Leap's qmm16n (qmm26.py) at KS 1, verbatim.
      Tile 32 (a | b, 6 real columns) is split over K by simdgroups 0-7 (5 quant groups each) after their own tile.
  A2  gates g / beta per (row, value head); conv + silu (+ l2norm, scale for q / k): Leap's GDNB phase A.
  B   the recurrence, fp32 state in registers: simdgroup sg holds state rows sg + 32 j (j < 12) of the head group's
      3 x 128 rows; mlx_lm's loop body per row, as in GDNB.
  C   the gated norm: GDNB phase C.
Weights: ONE stream per layer in the order the simdgroups read it, [16 heads][33 tiles][40 groups][32 lanes][8 words]
(pack_gdn below); scales / biases [16][33][40][32]. 21.6 + 2.7 MB per layer, read once, front to back per simdgroup.
Bits: phase A = qmm16n's arithmetic per tile at KS 1; A2 / B / C = GDNB's expressions. The a / b tile's K split and KS 1
differ from today's exact arm where it runs other (ks, nt), so this is a FAST kernel until Leap shows otherwise.
Bitwise variant: today's two dispatches (R-83 in_proj at the reference (ks, nt) + GDNB) are already bitwise == REF2-T.
Threadgroup memory: qs 4 + ks 4 + vs 12 + ab partials 4.1 + gates 0.3 = 24.4 KB at bf16 (limit 32 KB)."""
import numpy as np

HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
struct WSig { template <typename T> T operator()(T x) thread {                 // MLX Sigmoid, precise exp (GDNB GbSigmoid)
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x))); return (x < 0) ? y : 1 - y; } };
template <typename T> inline T w_log1p(T x) { return metal::precise::log(static_cast<T>(1) + x); }   // Leap: swap for MLX utils.h log1p
"""

SRC = r"""
    constexpr int G = 64, K_C = 2560, KG = K_C / G, NTP = 33, C = 10240, QK = 2048, HD = 128, VW = 6144, NHV = 48;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint hk = threadgroup_position_in_grid.z;            // key head = pipeline
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));         // op fragment rows fm, fm + 8
    const short fq = ((qid & 2) | (lane & 1));                // op fragment columns 4 fq .. 4 fq + 3 (and + 16)

    threadgroup T qs[16 * HD];
    threadgroup T ks[16 * HD];
    threadgroup T vs[16 * 3 * HD];
    threadgroup float abp[8 * 16 * 8];                        // a | b tile: [K part][row][column 0..7]
    threadgroup float gt[16 * 3];
    threadgroup T bt[16 * 3];

    // ---------------- A: in_proj, code MMA (qmm26.qmm16n at KS 1; x padded to 16 rows by the caller) ----------------
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;
    for (short pass = 0; pass < 2; pass++) {
        if (pass == 1 && sg >= 8) break;
        const int tile = pass == 0 ? (int)sg : 32;
        const int g0 = pass == 0 ? 0 : (int)sg * 5, g1 = pass == 0 ? KG : (int)sg * 5 + 5;
        const size_t tb = ((size_t)hk * NTP + tile) * KG;
        const device uint4* wq = (const device uint4*)(wn + (tb * 32 + lane) * 8);
        const device T* sc_ = sn + tb * 32 + 4 * fq;
        const device T* bi_ = bn + tb * 32 + 4 * fq;
        float acc[16];
        for (short i = 0; i < 16; i++) acc[i] = 0.0f;
        for (int g = g0; g < g1; g++) {
            const int k0 = g * G;
            const uint4 q0 = wq[(size_t)g * 64];
            const uint4 q1 = wq[(size_t)g * 64 + 1];
            const uint wv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
            float xs0 = 0.0f, xs1 = 0.0f;
            const device T* x0 = xr0 + k0;
            const device T* x1 = xr1 + k0;
            for (short st = 0; st < 4; st++) {
                for (short j = 0; j < 4; j++) {
                    const T a0 = x0[4 * st + j], a1 = x1[4 * st + j];
                    ct_a[j] = a0; ct_a[4 + j] = a1;
                    xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
                }
                for (short p = 0; p < 2; p++) {
                    const uint word = wv[st * 2 + p];
                    for (short j = 0; j < 4; j++) {
                        const uint two = ((word >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                        ct_b[p * 8 + j] = as_type<T>((ushort)(two & 0xFFFFu));
                        ct_b[p * 8 + 4 + j] = as_type<T>((ushort)(two >> 16));
                    }
                }
                op.run(ct_a, ct_b, ct_c);
            }
            xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
            xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
            const device T* scg = sc_ + (size_t)g * 32;
            const device T* big = bi_ + (size_t)g * 32;
            for (short bk = 0; bk < 2; bk++)
                for (short j = 0; j < 4; j++) {
                    const float s_ = static_cast<float>(scg[16 * bk + j]);
                    const float b_ = static_cast<float>(big[16 * bk + j]) - 128.0f * s_;
                    const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                    acc[i0] = acc[i0] + (s_ * ct_c[i0] + b_ * xs0);
                    acc[i1] = acc[i1] + (s_ * ct_c[i1] + b_ * xs1);
                }
        }
        if (pass == 0) {
            for (short i = 0; i < 16; i++) {
                const int m = fm + 8 * ((i & 7) >> 2);
                const int col = tile * 32 + (i >> 3) * 16 + 4 * fq + (i & 3);      // column in the head's packed order
                const T val = static_cast<T>(acc[i]);
                if (col < 2 * HD) qkv_raw[(size_t)m * C + (col < HD ? 0 : QK - HD) + hk * HD + col] = val;
                else if (col < 5 * HD) qkv_raw[(size_t)m * C + 2 * QK + hk * 3 * HD + (col - 2 * HD)] = val;
                else z_raw[(size_t)m * VW + hk * 3 * HD + (col - 5 * HD)] = val;
            }
        } else if (fq < 2) {
            for (short j = 0; j < 4; j++) {                    // columns 4 fq + j of block 0, rows fm and fm + 8
                abp[((int)sg * 16 + fm) * 8 + 4 * fq + j] = acc[j];
                abp[((int)sg * 16 + fm + 8) * 8 + 4 * fq + j] = acc[4 + j];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // ---------------- A2: gates per (row, value head); conv + silu (+ l2norm for q / k) (GDNB phase A) ----------------
    if (tid < (uint)(S_C * 3)) {
        const uint s = tid / 3, jh = tid % 3, hv = hk * 3 + jh;
        float av = 0.0f, bv = 0.0f;
        for (short kp = 0; kp < 8; kp++) { av += abp[(kp * 16 + s) * 8 + jh]; bv += abp[(kp * 16 + s) * 8 + 3 + jh]; }
        const T a_ = static_cast<T>(av), b_ = static_cast<T>(bv);
        a_out[s * NHV + hv] = a_; b_out[s * NHV + hv] = b_;
        const T sx = a_ + dtb[hv];                              // compute_g: exp(-exp(A_log) * softplus(a + dt_bias))
        const T mx_ = metal::max(sx, static_cast<T>(0)), mn_ = metal::min(sx, static_cast<T>(0));
        const T sp = mx_ + w_log1p(metal::precise::exp(mn_ - mx_));
        const float e = metal::precise::exp(static_cast<float>(alog[hv]));
        gt[s * 3 + jh] = metal::precise::exp((-e) * static_cast<float>(sp));
        bt[s * 3 + jh] = WSig()(b_);
    }
    for (uint task = sg; task < (uint)(S_C * 5); task += 32) {
        const uint s = task / 5, part = task % 5;               // part 0 q, 1 k, 2..4 v of value heads 3 hk + part - 2
        const uint gch = part == 0 ? hk * HD : (part == 1 ? QK + hk * HD : 2 * QK + (hk * 3 + part - 2) * HD);
        const uint c0 = gch + lane * 4;
        T sv[4];
        for (int i = 0; i < 4; ++i) {
            const uint c = c0 + i;
            float acc = 0.0f;
            for (int t = 0; t < 4; ++t) {
                const int st = (int)s + t;
                const float xv = st < 3 ? (float)state[st * C + c] : (float)qkv_raw[(st - 3) * C + c];
                acc = acc + (float)cw[c * 4 + t] * xv;
            }
            const T co = static_cast<T>(acc);
            sv[i] = static_cast<T>(co * WSig()(co));
        }
        if (part >= 2) {
            for (int i = 0; i < 4; ++i) {
                vs[(s * 3 + part - 2) * HD + lane * 4 + i] = sv[i];
                v_out[s * VW + (hk * 3 + part - 2) * HD + lane * 4 + i] = sv[i];
            }
        } else {
            float xf[4], p = 0.0f;
            for (int i = 0; i < 4; ++i) xf[i] = (float)sv[i];
            for (int i = 0; i < 4; ++i) { const float sq = xf[i] * xf[i]; p += sq; }
            const float tot = simd_sum(p);
            const float r = metal::precise::rsqrt(tot + 1e-6f);
            for (int i = 0; i < 4; ++i) {
                const T nb = static_cast<T>(xf[i] * r);
                if (part == 0) {
                    const T qv = static_cast<T>(scale[0] * nb);
                    qs[s * HD + lane * 4 + i] = qv; q_out[s * QK + hk * HD + lane * 4 + i] = qv;
                } else {
                    ks[s * HD + lane * 4 + i] = nb; k_out[s * QK + hk * HD + lane * 4 + i] = nb;
                }
            }
        }
    }
    for (uint i = tid; i < 3u * 640u; i += 1024u) {            // the new conv state: rows S, S + 1, S + 2 of [state; qkv]
        const uint j = i / 640, cc = i % 640;
        const uint c = cc < 2 * HD ? (cc < HD ? 0 : QK - HD) + hk * HD + cc : 2 * QK + hk * 3 * HD + (cc - 2 * HD);
        const int st = S_C + (int)j;
        state_out[j * C + c] = st < 3 ? state[st * C + c] : qkv_raw[(st - 3) * C + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------- B: the recurrence, rows in order (mlx_lm gated_delta_step's loop body per state row) ----------------
    constexpr int RPS = 12;                                    // 384 state rows of the 3 value heads over 32 simdgroups
    float sr[RPS][4];
    for (int j = 0; j < RPS; ++j) {
        const uint r = sg + 32 * j;
        const uint base = ((hk * 3 + r / HD) * HD + r % HD) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) sr[j][i] = static_cast<float>(dstate[base + i]);
    }
    for (int s = 0; s < S_C; ++s) {
        const threadgroup T* k_ = ks + s * HD;
        const threadgroup T* q_ = qs + s * HD;
        for (int j = 0; j < RPS; ++j) {
            const uint r = sg + 32 * j, jh = r / HD, dv = r % HD;
            const float gv = gt[s * 3 + jh];
            const T bv = bt[s * 3 + jh];
            float kv_mem = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] * gv;
                kv_mem += sr[j][i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);
            auto delta = (vs[(s * 3 + jh) * HD + dv] - kv_mem) * bv;
            float out = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] + k_[s_idx] * delta;
                out += sr[j][i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (lane == 0) y[s * VW + (hk * 3 + jh) * HD + dv] = static_cast<T>(out);   // parked in y until phase C
        }
    }
    for (int j = 0; j < RPS; ++j) {
        const uint r = sg + 32 * j;
        const uint base = ((hk * 3 + r / HD) * HD + r % HD) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) dstate_out[base + i] = static_cast<float>(sr[j][i]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---------------- C: the gated norm per (row, value head) (GDNB phase C) ----------------
    for (uint task = sg; task < (uint)(S_C * 3); task += 32) {
        const uint s = task / 3, hv = hk * 3 + task % 3;
        float acc = 0.0f;
        float hv4[4];
        for (int i = 0; i < 4; ++i) { hv4[i] = (float)y[s * VW + hv * HD + lane * 4 + i]; acc += hv4[i] * hv4[i]; }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / 128.0f + epsv[0]);
        for (int i = 0; i < 4; ++i) {
            const T xn = static_cast<T>(hv4[i] * inv);
            const T xw = static_cast<T>(w[lane * 4 + i] * xn);
            const float gz = WSig()((float)z_raw[s * VW + hv * HD + lane * 4 + i]);
            y[s * VW + hv * HD + lane * 4 + i] = static_cast<T>(gz * (float)xw);
        }
    }
"""

IN = ["x", "wn", "sn", "bn", "state", "cw", "scale", "dtb", "alog", "dstate", "w", "epsv"]
OUT = ["y", "qkv_raw", "z_raw", "state_out", "dstate_out", "q_out", "k_out", "v_out", "a_out", "b_out"]
TYPES = {"wn": "uint", "dstate": "float", "epsv": "float", "dstate_out": "float"}

# ------------------------------------------------------------------ the packed weight stream (numpy, CPU-checkable)
LANE = np.arange(32)
FM = ((LANE >> 2) & 4) | ((LANE >> 1) & 3)
FQ = ((LANE >> 2) & 2) | (LANE & 1)


def head_rows(hk):
    """row indices of the head's 1,056 packed columns into the stacked matrix [qkv (10240); z (6144); a (48); b (48)];
    -1 = zero padding"""
    q = hk * 128 + np.arange(128)
    k = 2048 + hk * 128 + np.arange(128)
    v = 4096 + hk * 384 + np.arange(384)
    z = 10240 + hk * 384 + np.arange(384)
    a = 16384 + hk * 3 + np.arange(3)
    b = 16432 + hk * 3 + np.arange(3)
    return np.concatenate([q, k, v, z, a, b, -np.ones(26, np.int64)])


def repack_native(w, s, b):
    """Leap's qmm26 / ref2._dn_repack layout from MLX's [N, K/8] uint32 codes + [N, K/64] scales / biases:
    wn [N/32, K/64, 32, 8]: lane l's word 2 st + p = codes of rows 16 p + fm(l) (low 16 bits) and 16 p + fm(l) + 8 (high),
    K = 64 g + 16 fq(l) + 4 st + j at nibble j; sn / bn [N/32, K/64, 32]"""
    N, KW = w.shape
    T, KGn = N // 32, KW // 8
    w3 = w.reshape(T, 32, KGn, 8).transpose(0, 2, 1, 3)                       # [tile, group, row, word]
    wn = np.zeros((T, KGn, 32, 8), np.uint32)
    for l in range(32):
        for st in range(4):
            for p in range(2):
                ws = 2 * int(FQ[l]) + (st >> 1)
                sh = 16 * (st & 1)
                lo = (w3[:, :, 16 * p + int(FM[l]), ws] >> sh) & 0xFFFF
                hi = (w3[:, :, 16 * p + int(FM[l]) + 8, ws] >> sh) & 0xFFFF
                wn[:, :, l, 2 * st + p] = lo | (hi << 16)
    return wn, s.reshape(T, 32, KGn).transpose(0, 2, 1).copy(), b.reshape(T, 32, KGn).transpose(0, 2, 1).copy()


def pack_gdn(w_all, s_all, b_all):
    """w_all [16480, 320] uint32 (qkv | z | a | b stacked, MLX 4-bit g64 layout), s_all / b_all [16480, 40]
    -> wn [16, 33, 40, 32, 8] uint32, sn / bn [16, 33, 40, 32]"""
    wn, sn, bn = [], [], []
    for hk in range(16):
        idx = head_rows(hk)
        pad = idx < 0
        ii = np.where(pad, 0, idx)
        w, s, b = w_all[ii].copy(), s_all[ii].copy(), b_all[ii].copy()
        w[pad] = 0; s[pad] = 0; b[pad] = 0
        a, c, d = repack_native(w, s, b)
        wn.append(a); sn.append(c); bn.append(d)
    return np.stack(wn), np.stack(sn), np.stack(bn)


def gdn_w(x16, P, conv_state, conv_w, scale_bf16, dt_bias, A_log, state, norm_w, eps, S):
    """the dispatch (needs MLX + a GPU; Leap runs it). x16 [16, 2560] bf16 padded; P = (wn, sn, bn) mx arrays from pack_gdn"""
    import mlx.core as mx
    k = mx.fast.metal_kernel(name="fb_gdn_w", input_names=IN, output_names=OUT, source=SRC, header=HDR)
    T = x16.dtype
    return k(inputs=[x16.reshape(-1), P[0].reshape(-1), P[1].reshape(-1), P[2].reshape(-1), conv_state.reshape(-1),
                     conv_w.reshape(-1), scale_bf16.reshape(-1), dt_bias, A_log, state.reshape(-1), norm_w.astype(T),
                     mx.array([eps], mx.float32)],
             template=[("T", T), ("S_C", S)], grid=(1024, 1, 16), threadgroup=(1024, 1, 1),
             output_shapes=[(16 * 6144,), (16 * 10240,), (16 * 6144,), (3 * 10240,), (48 * 128 * 128,), (16 * 2048,),
                            (16 * 2048,), (16 * 6144,), (16 * 48,), (16 * 48,)],
             output_dtypes=[T, T, T, T, mx.float32, T, T, T, T, T])
