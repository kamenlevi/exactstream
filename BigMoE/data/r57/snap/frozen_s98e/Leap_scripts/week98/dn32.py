"""dn32.py (Leap, Mon 16 Nov 2026 12:1x; CHECK25 step 1, data/week98/check25_plan.md): 32-row dense 4-bit GEMMs that
feed the 4-bit weights to the matrix unit as uint4 / int4 operands from threadgroup memory (NAX6's operand form for the
experts), so no ALU decode, and every weight is read once for all 32 rows. Two variants, both FAST (not bitwise to the
16-row qmm16n; checked against an fp32 reference):
  dn32  (bf16 x uint4(tg) -> fp32): per 64-group, two 16 x 32 x 32 ops per 16-row tile, y += s * P + b * sum(x)
  dn32q (int8 x int4(tg) -> int32): activations quantized per (row, 64-group) by quant64 (NAX-C's rule: scale = max|x| /
        127, round to nearest, clamp +-127; one extra dispatch per input, shared by every call on that input); codes
        flipped to int4 (code - 8, one XOR per word); y += sx * (s * (P + 8 qs) + b * qs). M27-C: 1.55-1.73x bf16's rate.
Layout ("NAX-dense", from MLX's 4-bit g64 arrays): wx [N/32, K/64, 32 columns, 8 words] = each column's 64 codes of a
group in MLX's nibble order, a tile-group's 1 KB contiguous (one coalesced uint4 x 2 load per lane); sx / bx [N/32, K/64,
32]. Grid: one simdgroup per (32-column tile, K split); KS > 1 reduces per tile through threadgroup memory.
  layout(w, s, b) -> (wx, sx, bx);  dn32(x [M, K], wx, sx, bx, N, ks, nt);  dn32q(xq, xs, xqs, wx, sx, bx, N, ks, nt)
  quant64(x) -> (xq int8, xs fp32 [M, K/64], xqs int32 [M, K/64]);  check() errors vs fp32 per FAST dense shape
usage: [BigMoE venv] dn32.py check"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98e/Leap_scripts/week98"))

HI = 32
_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_SRC = r"""
    constexpr int G = 64, KG = K_C / G;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint kg = sg / NT, nt = sg % NT;
    const int t = threadgroup_position_in_grid.x * NT + nt;          // 32-column tile
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    constexpr bool TWO = M_C > 16;                                 // 1-16 rows: one row tile (dn16 / dn16q)
    const bool ok0 = fm < M_C, ok1 = fm + 8 < M_C, ok2 = 16 + fm < M_C, ok3 = 24 + fm < M_C;
    const int r0 = ok0 ? fm : 0, r1 = ok1 ? fm + 8 : 0, r2 = ok2 ? 16 + fm : 0, r3 = ok3 ? 24 + fm : 0;
#if Q8
    const device int8_t* ar0 = xq + (size_t)r0 * K_C;
    const device int8_t* ar1 = xq + (size_t)r1 * K_C;
    const device int8_t* ar2 = xq + (size_t)r2 * K_C;
    const device int8_t* ar3 = xq + (size_t)r3 * K_C;
    using AT = int8_t;
    using BT = int4b_format;
    using CT = int32_t;
#else
    const device T* ar0 = x + (size_t)r0 * K_C;
    const device T* ar1 = x + (size_t)r1 * K_C;
    const device T* ar2 = x + (size_t)r2 * K_C;
    const device T* ar3 = x + (size_t)r3 * K_C;
    using AT = T;
    using BT = uint4b_format;
    using CT = float;
#endif
    constexpr int TBW = (KS * 256 > (KS - 1) * 512 ? KS * 256 : (KS - 1) * 512) * NT;   // staging, then split-K partials
    threadgroup uint32_t tb_all[TBW];
    constexpr uint FLIP = Q8 ? 0x88888888u : 0u;                    // int8: codes -> int4 (code - 8)
    threadgroup uint32_t* tb = tb_all + sg * (32 * 8);
    const device uint4* wq = (const device uint4*)(wx + (((size_t)t * KG) * 32 + lane) * 8);
    const device T* sc_ = sx + ((size_t)t * KG) * 32;
    const device T* bi_ = bx + ((size_t)t * KG) * 32;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto ca2 = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    using TB = tensor<threadgroup BT, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, CT>();
    auto cg2 = op.template get_destination_cooperative_tensor<decltype(ca), TB, CT>();
    float acc[16], acc2[16];
    for (short i = 0; i < 16; i++) { acc[i] = 0.0f; acc2[i] = 0.0f; }

    for (int g = kg * (KG / KS); g < (int)(kg + 1) * (KG / KS); g++) {
        const uint4 w0 = wq[(size_t)g * 64], w1 = wq[(size_t)g * 64 + 1];
#if NATIVE
        {   // DENSE-N-RAM's native words (qmm26.repack): word 2 st + p holds rows 16 p + fm (low half) and 16 p + fm + 8
            // (high half), each 4 codes of K = 16 fq + 4 st .. + 3 in MLX's nibble order -> each half to its column's run
            const uint wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            threadgroup ushort* tbs = (threadgroup ushort*)tb;
            const short fq = fn0 >> 2;
            for (short st = 0; st < 4; st++)
                for (short p = 0; p < 2; p++) {
                    const uint word = wv[st * 2 + p] ^ FLIP;
                    const int c0 = 16 * p + fm;
                    tbs[c0 * 16 + 4 * fq + st] = (ushort)(word & 0xFFFFu);
                    tbs[(c0 + 8) * 16 + 4 * fq + st] = (ushort)(word >> 16);
                }
        }
#else
        threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8);
        dst[0] = w0 ^ uint4(FLIP); dst[1] = w1 ^ uint4(FLIP);
#endif
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) { cg[i] = 0; cg2[i] = 0; }
#if !Q8
        float xs0 = 0.0f, xs1 = 0.0f, xs2 = 0.0f, xs3 = 0.0f;
#endif
        for (short hh = 0; hh < 2; hh++) {
            const int k0 = 64 * g + 32 * hh;
            vec<AT, 4> a00 = *(const device vec<AT, 4>*)(ar0 + k0 + fn0);
            vec<AT, 4> a10 = *(const device vec<AT, 4>*)(ar1 + k0 + fn0);
            vec<AT, 4> a01 = *(const device vec<AT, 4>*)(ar0 + k0 + 16 + fn0);
            vec<AT, 4> a11 = *(const device vec<AT, 4>*)(ar1 + k0 + 16 + fn0);
            if (!ok0) { a00 = vec<AT, 4>(0); a01 = vec<AT, 4>(0); }
            if (!ok1) { a10 = vec<AT, 4>(0); a11 = vec<AT, 4>(0); }
            vec<AT, 4> a20 = *(const device vec<AT, 4>*)(ar2 + k0 + fn0);
            vec<AT, 4> a30 = *(const device vec<AT, 4>*)(ar3 + k0 + fn0);
            vec<AT, 4> a21 = *(const device vec<AT, 4>*)(ar2 + k0 + 16 + fn0);
            vec<AT, 4> a31 = *(const device vec<AT, 4>*)(ar3 + k0 + 16 + fn0);
            if (!ok2) { a20 = vec<AT, 4>(0); a21 = vec<AT, 4>(0); }
            if (!ok3) { a30 = vec<AT, 4>(0); a31 = vec<AT, 4>(0); }
            for (short q = 0; q < 4; q++) {
                ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q];
                ca2[q] = a20[q]; ca2[4 + q] = a30[q]; ca2[8 + q] = a21[q]; ca2[12 + q] = a31[q];
#if !Q8
                xs0 += float(a00[q]) + float(a01[q]); xs1 += float(a10[q]) + float(a11[q]);
                xs2 += float(a20[q]) + float(a21[q]); xs3 += float(a30[q]) + float(a31[q]);
#endif
            }
            TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
            op.run(ca, tbt, cg);
            if (TWO) op.run(ca2, tbt, cg2);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
#if Q8
        const float sx0 = ok0 ? xs[(size_t)r0 * KG + g] : 0.0f, sx1 = ok1 ? xs[(size_t)r1 * KG + g] : 0.0f;
        const float sx2 = ok2 ? xs[(size_t)r2 * KG + g] : 0.0f, sx3 = ok3 ? xs[(size_t)r3 * KG + g] : 0.0f;
        const int q0 = ok0 ? xqs[(size_t)r0 * KG + g] : 0, q1 = ok1 ? xqs[(size_t)r1 * KG + g] : 0;
        const int q2 = ok2 ? xqs[(size_t)r2 * KG + g] : 0, q3 = ok3 ? xqs[(size_t)r3 * KG + g] : 0;
#else
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        xs2 += simd_shuffle_xor(xs2, 1); xs2 += simd_shuffle_xor(xs2, 8);
        xs3 += simd_shuffle_xor(xs3, 1); xs3 += simd_shuffle_xor(xs3, 8);
#endif
        const device T* scg = sc_ + (size_t)g * 32;
        const device T* big = bi_ + (size_t)g * 32;
        for (short i = 0; i < 16; i++) {
            const int col = 16 * (i >> 3) + fn0 + (i & 3);
            const float s_ = float(scg[col]), b_ = float(big[col]);
            const bool r1 = ((i >> 2) & 1) != 0;
#if Q8
            acc[i] += (r1 ? sx1 : sx0) * (s_ * float(cg[i] + 8 * (r1 ? q1 : q0)) + b_ * float(r1 ? q1 : q0));
            acc2[i] += (r1 ? sx3 : sx2) * (s_ * float(cg2[i] + 8 * (r1 ? q3 : q2)) + b_ * float(r1 ? q3 : q2));
#else
            acc[i] += s_ * cg[i] + b_ * (r1 ? xs1 : xs0);
            acc2[i] += s_ * cg2[i] + b_ * (r1 ? xs3 : xs2);
#endif
        }
    }

    if (KS > 1) {
        threadgroup float* part = (threadgroup float*)tb_all;     // reuse: (KS - 1) x NT x 16 x 32 floats per tile pass
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg == 0)
            for (int gg = 0; gg < KS - 1; gg++)
                for (short i = 0; i < 16; i++) acc[i] += part[((gg * NT + nt) * 16 + i) * 32 + lane];
        if (TWO) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc2[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kg == 0)
                for (int gg = 0; gg < KS - 1; gg++)
                    for (short i = 0; i < 16; i++) acc2[i] += part[((gg * NT + nt) * 16 + i) * 32 + lane];
        }
        if (kg != 0) return;
    }
    const int n0 = t * 32;
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i >> 2) & 1);
        const int n = n0 + 16 * (i >> 3) + fn0 + (i & 3);
        if (m < M_C) y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
        if (TWO && 16 + m < M_C) y[(size_t)(16 + m) * N_C + n] = static_cast<T>(acc2[i]);
    }
"""


@lru_cache(None)
def _k(q8, native=False):
    names = (["xq", "xs", "xqs"] if q8 else ["x"]) + ["wx", "sx", "bx"]
    src = (_SRC.replace("#if Q8", f"#if {int(q8)}").replace("#if !Q8", f"#if {int(not q8)}")
           .replace("#if NATIVE", f"#if {int(native)}").replace("Q8 ? 0x88888888u : 0u", "0x88888888u" if q8 else "0u"))
    return mx.fast.metal_kernel(name=f"leap_dn32{'q' if q8 else ''}{'n' if native else ''}", input_names=names,
                                output_names=["y"], source=src, header=_HDR)


def layout(w, s, b):
    """MLX 4-bit g64 [N, K/8] uint32 + [N, K/64] scales / biases -> (wx [N/32, K/64, 32, 8], sx, bx [N/32, K/64, 32])"""
    N, KW = w.shape
    KG = KW // 8
    wx = w.reshape(N // 32, 32, KG, 8).transpose(0, 2, 1, 3)
    sx = s.reshape(N // 32, 32, KG).transpose(0, 2, 1)
    bx = b.reshape(N // 32, 32, KG).transpose(0, 2, 1)
    return mx.contiguous(wx), mx.contiguous(sx), mx.contiguous(bx)


def _tg_ok(ks, nt):
    return max(ks * 256, (ks - 1) * 512) * nt * 4 <= 32768


def dn32(x, wx, sx, bx, N, ks=1, nt=1, native=False):
    """native=True: wx / sx / bx are DENSE-N-RAM's native copy (qmm26.repack layout), staged in the kernel"""
    M, K = x.shape
    assert 1 <= M <= HI and N % (32 * nt) == 0 and (K // 64) % ks == 0 and _tg_ok(ks, nt), (M, K, N, ks, nt)
    (y,) = _k(False, native)(inputs=[x, wx, sx, bx], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt),
                                                       ("M_C", M)],
                     grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(M, N)],
                     output_dtypes=[x.dtype])
    return y


_Q8 = r"""
    const uint g = thread_position_in_grid.x, r = thread_position_in_grid.y;
    if (g >= (uint)(L_C / 64)) return;
    const device T* p = a + (size_t)r * L_C + g * 64;
    float amax = 0.0f;
    for (int i = 0; i < 64; ++i) amax = max(amax, fabs((float)p[i]));
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    int qs = 0;
    device int8_t* qo = q + (size_t)r * L_C + g * 64;
    for (int i = 0; i < 64; ++i) {
        const int v = clamp((int)rint((float)p[i] * inv), -127, 127);
        qo[i] = (int8_t)v;
        qs += v;
    }
    scale[(size_t)r * (L_C / 64) + g] = sc;
    qsum[(size_t)r * (L_C / 64) + g] = qs;
"""


@lru_cache(None)
def _kq8():
    return mx.fast.metal_kernel(name="leap_dn32_q8", input_names=["a"], output_names=["q", "scale", "qsum"], source=_Q8)


def quant64(a):
    """NAX-C's activation quantizer: [rows, L] -> (int8 [rows, L], scale fp32 [rows, L/64], int32 sums [rows, L/64])"""
    R_, L = a.shape
    return _kq8()(inputs=[a], template=[("T", a.dtype), ("L_C", L)], grid=(L // 64, R_, 1), threadgroup=(min(L // 64, 64), 1, 1),
                  output_shapes=[(R_, L), (R_, L // 64), (R_, L // 64)], output_dtypes=[mx.int8, mx.float32, mx.int32])


def dn32q(xq, xs, xqs, wx, sx, bx, N, ks=1, nt=1, dtype=mx.bfloat16, native=False):
    M, K = xq.shape
    assert 1 <= M <= HI and N % (32 * nt) == 0 and (K // 64) % ks == 0 and _tg_ok(ks, nt), (M, K, N, ks, nt)
    (y,) = _k(True, native)(inputs=[xq, xs, xqs, wx, sx, bx], template=[("T", dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt),
                                                                ("M_C", M)],
                    grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(M, N)],
                    output_dtypes=[dtype])
    return y


def check(seeds=2, Ms=(17, 24, 32)):
    """errors vs an fp32 reference (max |err| / max |ref|) per FAST dense shape, next to qmm32n's / two 16-row calls'"""
    import numpy as np
    import qmm26 as Q26
    import qmm32n as Q
    rng = np.random.default_rng(1116)
    lines = []
    for name, N, K, _ in Q.SHAPES:
        ks = Q.ks_of(N, K)
        ksd = ks if _tg_ok(ks, 1) else 4
        w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        wd = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)
        wn, sn, bn = Q26.repack(wq, s, b)
        wx, sx, bx = layout(wq, s, b)
        mx.eval(wn, sn, bn, wx, sx, bx, wd)
        e = {"qmm32n": 0.0, "dn32": 0.0, "dn32q": 0.0}
        for _ in range(seeds):
            X = mx.array(rng.standard_normal((HI, K)).astype(np.float32)).astype(mx.bfloat16)
            for M in Ms:
                ref = X[:M].astype(mx.float32) @ wd.T
                xq, xs_, xqs = quant64(X[:M])
                outs = {"qmm32n": Q.qmm32n(X[:M], wn, sn, bn, N, ks), "dn32": dn32(X[:M], wx, sx, bx, N, ksd),
                        "dn32q": dn32q(xq, xs_, xqs, wx, sx, bx, N, ksd)}
                mx.eval(ref, *outs.values())
                r = float(mx.max(mx.abs(ref)))
                for k, v in outs.items():
                    e[k] = max(e[k], float(mx.max(mx.abs(v.astype(mx.float32) - ref))) / r)
        lines.append(f"  {name:36s} N {N:6d} K {K:5d} ks {ks:2d}/{ksd:2d}: max err / max |ref|: "
                     + ", ".join(f"{k} {v:.2e}" for k, v in e.items()))
        print(lines[-1], flush=True)
    return lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        lines = check()
        p = os.path.expanduser("~/Leap/data/week98/check25/dn32_check.txt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("DN32 check (errors vs fp32; bf16 output rounding included)\n" + "\n".join(lines) + "\n")
