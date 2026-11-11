"""hc_d.py (Fable B, DENSE_REBUILD, Fri 2026-11-06): DRAFT of kernel D, the head of an HC8 read. NOT RUN ON A GPU.
One dispatch, cut into 20 channel slabs (128 of the 2,560 channels each = one threadgroup of 1,024 threads):
  1  PEND 1: the slab's 128 columns of the previous block's output projection, bo = y @ W_out^T (4-bit g64, K 6144,
     split over K by 8 simdgroups per 32-column tile; partials reduced through a device scratch buffer).
     This is the GDN layer's out_proj, moved here: a slab needs only its own 128 output columns of it.
  2  the pending residual write h' = h + bo * inject for the slab's channels of all 4 streams (stored as `written`),
     A = bf16(h' * w_norm), and the slab's part of sum(h'^2) per (row, stream).
  3  the slab's K slice (4 streams x 128 channels = 8 quant groups) of the 8-bit down mix and of the 4 bf16 inject columns:
     fp32 partials [slab, stream, 16, 352].
Then Leap's HC8-1P K2 (gate) and K3 (up + tail) run unchanged, except that K2 sums 20 x 4 partials instead of NP.
The norm follows HC8-1P: norm weight before the GEMM, 1 / rms after it (per stream, in K2).
PEND 0: no pending write (the first read of a check). The MLP-side variant (pending = shared down + experts) is the same
skeleton with step 1 = the shared expert's down projection (K 640) plus the expert kernel's output; see DENSE_REBUILD.md.
Bits: FAST (HC8 is already a FAST read; out_proj's K order here is 8 partitions, not the reference's).
Knobs for Leap: slab width (128; 256 halves the partials K2 reads), KS of step 1 (8), the 8-bit decode (use hc8r K1's)."""
HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

SRC = r"""
    constexpr int G = 64, D = 2560, HC = 4, HCD = HC * D, LR = 320, NC = 352, SL = 128, KO = 6144, KGO = KO / G, KS = 8;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const int slab = int(threadgroup_position_in_grid.z);
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    threadgroup T bo[16 * SL];
    threadgroup T an[16 * HC * SL];
    threadgroup float ssp[1024];

    // ---------------- 1: the slab's out_proj columns (qmm16n's loop; tile = 4 slab + sg % 4, K part = sg / 4) ----------------
    if (PEND) {
        const int ti = (int)sg % 4, kp = (int)sg / 4;
        const size_t tb = ((size_t)(slab * 4 + ti)) * KGO;
        const device uint4* wq = (const device uint4*)(own + (tb * 32 + lane) * 8);
        const device T* sc_ = osn + tb * 32 + 4 * fq;
        const device T* bi_ = obn + tb * 32 + 4 * fq;
        const device T* yr0 = y + (size_t)fm * KO + 16 * fq;
        const device T* yr1 = y + (size_t)(fm + 8) * KO + 16 * fq;
        float acc[16];
        for (short i = 0; i < 16; i++) acc[i] = 0.0f;
        for (int g = kp * (KGO / KS); g < (kp + 1) * (KGO / KS); g++) {
            const uint4 q0 = wq[(size_t)g * 64];
            const uint4 q1 = wq[(size_t)g * 64 + 1];
            const uint wv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
            float xs0 = 0.0f, xs1 = 0.0f;
            const device T* x0 = yr0 + g * G;
            const device T* x1 = yr1 + g * G;
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
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
            scr[((((size_t)slab * KS + kp) * 4 + ti) * 16 + m) * 32 + nl] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_device);
        {
            const int m = (int)tid / 64, c0 = ((int)tid % 64) * 2;
            for (int cc = c0; cc < c0 + 2; cc++) {
                float sum = 0.0f;
                for (int p = 0; p < KS; p++) sum += scr[((((size_t)slab * KS + p) * 4 + cc / 32) * 16 + m) * 32 + cc % 32];
                bo[m * SL + cc] = static_cast<T>(sum);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------------- 2: pending write, A = bf16(h' w_norm), sum(h'^2) of the slab per (row, stream) ----------------
    {
        const int m = (int)tid / 64, e = (int)tid % 64, s = e / 16, c8 = (e % 16) * 8;
        float p = 0.0f;
        for (int i = 0; i < 8; i++) {
            const int c = c8 + i, d = slab * SL + c;
            const size_t idx = (size_t)m * HCD + s * D + d;
            T hv = h[idx];
            if (PEND) hv = hv + bo[m * SL + c] * inj[m * HC + s];
            written[idx] = hv;
            an[(m * HC + s) * SL + c] = static_cast<T>(hv * wnorm[s * D + d]);
            p += static_cast<float>(hv) * static_cast<float>(hv);
        }
        ssp[tid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) {
        const int m = (int)tid / 4, s = (int)tid % 4;
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) sum += ssp[m * 64 + s * 16 + i];
        ssq[((size_t)slab * 16 + m) * HC + s] = sum;
    }

    // ---------------- 3: the slab's K slice of the down mix: unit = (tile, stream), 2 quant groups each ----------------
    for (int u = (int)sg; u < 11 * HC; u += 32) {
        const int tile = u % 11, s = u / 11;
        float acc[16];
        for (short i = 0; i < 16; i++) acc[i] = 0.0f;
        for (int gl = 0; gl < SL / G; gl++) {
            const int kg0 = s * D + slab * SL + gl * G;         // K of this group's first column in the 10,240 layout
            for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
            float xs0 = 0.0f, xs1 = 0.0f;
            for (short st = 0; st < 4; st++) {
                const int kl = gl * G + 16 * fq + 4 * st;
                for (short j = 0; j < 4; j++) {
                    const T a0 = an[(fm * HC + s) * SL + kl + j], a1 = an[((fm + 8) * HC + s) * SL + kl + j];
                    ct_a[j] = a0; ct_a[4 + j] = a1;
                    xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
                }
                const int kk = kg0 + 16 * fq + 4 * st;
                for (short p = 0; p < 2; p++) {
                    const int n0 = tile * 32 + 16 * p + fm, n1 = n0 + 8;
                    if (tile < 10) {                           // 8-bit codes, MLX layout: 4 codes per uint32, low byte first
                        const uint w0 = dq[((size_t)n0 * HCD + kk) / 4], w1 = dq[((size_t)n1 * HCD + kk) / 4];
                        for (short j = 0; j < 4; j++) {
                            ct_b[p * 8 + j] = static_cast<T>(static_cast<float>((w0 >> (8 * j)) & 0xFFu));
                            ct_b[p * 8 + 4 + j] = static_cast<T>(static_cast<float>((w1 >> (8 * j)) & 0xFFu));
                        }
                    } else {                                   // the inject tile: 4 real columns, bf16 weights
                        for (short j = 0; j < 4; j++) {
                            ct_b[p * 8 + j] = (p == 0 && fm < HC) ? wi[(size_t)n0 % 32 * HCD + kk + j] : static_cast<T>(0);
                            ct_b[p * 8 + 4 + j] = static_cast<T>(0);
                        }
                    }
                }
                op.run(ct_a, ct_b, ct_c);
            }
            if (tile < 10) {
                xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
                xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
                for (short bk = 0; bk < 2; bk++)
                    for (short j = 0; j < 4; j++) {
                        const size_t o = (size_t)(tile * 32 + 16 * bk + 4 * fq + j) * (HCD / G) + kg0 / G;
                        const float s_ = static_cast<float>(dsn[o]), b_ = static_cast<float>(dbn[o]);
                        const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                        acc[i0] = acc[i0] + (s_ * ct_c[i0] + b_ * xs0);
                        acc[i1] = acc[i1] + (s_ * ct_c[i1] + b_ * xs1);
                    }
            } else {
                for (short i = 0; i < 16; i++) acc[i] = acc[i] + ct_c[i];
            }
        }
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int n = tile * 32 + (i >> 3) * 16 + 4 * fq + (i & 3);
            if (n < NC) part[((((size_t)slab * HC + s) * 16 + m) * NC) + n] = acc[i];
        }
    }
"""

IN = ["h", "y", "inj", "own", "osn", "obn", "wnorm", "dq", "dsn", "dbn", "wi"]
OUT = ["written", "ssq", "part", "scr"]
TYPES = {"own": "uint", "dq": "uint", "ssq": "float", "part": "float", "scr": "float"}


def hc_d(h16, y16, inj16, OW, wnorm, DQ, wi, pend=True):
    """the dispatch (needs MLX + a GPU; Leap runs it). OW = out_proj in the native layout (gdn_w.repack_native),
    DQ = (codes [320, 2560] uint32, scales, biases [320, 160]) of the 8-bit down mixer, wi [4, 10240] bf16"""
    import mlx.core as mx
    k = mx.fast.metal_kernel(name="fb_hc_d", input_names=IN, output_names=OUT, source=SRC, header=HDR)
    T = h16.dtype
    return k(inputs=[h16.reshape(-1), y16.reshape(-1), inj16.reshape(-1), OW[0].reshape(-1), OW[1].reshape(-1),
                     OW[2].reshape(-1), wnorm.astype(T).reshape(-1), DQ[0].reshape(-1), DQ[1].reshape(-1),
                     DQ[2].reshape(-1), wi.astype(T).reshape(-1)],
             template=[("T", T), ("PEND", int(pend))], grid=(1024, 1, 20), threadgroup=(1024, 1, 1),
             output_shapes=[(16 * 10240,), (20 * 16 * 4,), (20 * 4 * 16 * 352,), (20 * 8 * 4 * 16 * 32,)],
             output_dtypes=[T, mx.float32, mx.float32, mx.float32])
