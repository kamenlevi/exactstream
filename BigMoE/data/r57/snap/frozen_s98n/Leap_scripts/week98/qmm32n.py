"""qmm32n.py (Leap, Mon 16 Nov 2026 12:0x; CHECK25 step 1, data/week98/check25_plan.md): the FAST stack's native 4-bit
dense call (qmm26.qmm16n / qmm26m.qmm16n_m: DENSE-N, DENSE-N-RAM, SMALL-N, with DN-KS's K split) for 17-32 rows in ONE
dispatch and ONE weight pass. Today the FAST stack has no > 16-row form: _dnr_qmm falls to _dn_unpack above 16 rows, so a
32-row check would run two 16-row calls (two reads of every dense weight) or worse.

Each lane decodes its native weight fragment once per 16-value sub-step and runs the matrix-unit MMA for BOTH 16-row
tiles: tile 0 = rows (fm, fm + 8), tile 1 = rows (16 + fm, 24 + fm), each with its own A fragment, per-group
accumulator, row sums and output rows. Rows >= M load +0.0 in the kernel (qmm26m's rule) and are not written. The split-K
partials (KS > 1) are reduced per tile in qmm16n's order (kg 0, then 1 .. KS-1), through one threadgroup buffer used
twice (tile 0, then tile 1), so KS up to 16 fits in 32 KB. So each tile's arithmetic is qmm16n's at the same ks, nt:
qmm32n(x[:M]) == concat(qmm16n_m(x[:16]), qmm16n_m(x[16:M])) bit for bit (check() proves it per shape and M).
FAST-safe by construction (no new rounding); at ks 1 it is also qmm16c2a's arithmetic on the native layout.

  qmm32n(x [M, K] bf16, wn, sn, bn, N, ks=1, nt=1) -> y [M, N], 16 < M <= 32
  check()  bits vs two 16-row calls + error vs an fp32 reference, on the FAST dense shapes (DN-KS v1 ks)
usage: [BigMoE venv] qmm32n.py check"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98n/Leap_scripts"))
from qmm16 import _HDR

HI = 32

_SRC = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int t = threadgroup_position_in_grid.x * NT + nt;          // 32-column tile
    const int n0 = t * 32;
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_a2 = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                               metal::remove_addrspace_t<decltype(ct_b)>, float>();
    auto ct_c2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc[16], acc2[16];
    for (short i = 0; i < 16; i++) { acc[i] = 0.0f; acc2[i] = 0.0f; }

    constexpr int KG = K_C / G;
    const int kper = K_C / KS;
    // native weights: [t][g][lane][8 words]; scales / biases: [t][g][32]
    const device uint4* wq = (const device uint4*)(wn + (((size_t)t * KG) * 32 + lane) * 8);
    const device T* sc_ = sn + ((size_t)t * KG) * 32 + 4 * fq;
    const device T* bi_ = bn + ((size_t)t * KG) * 32 + 4 * fq;
    const bool ok2 = 16 + fm < M_C, ok3 = 24 + fm < M_C;             // rows 0..15 are always < M (M > 16)
    const device T* xr0 = x + (size_t)fm * K_C + 16 * fq;
    const device T* xr1 = x + (size_t)(fm + 8) * K_C + 16 * fq;
    const device T* xr2 = x + (size_t)(ok2 ? 16 + fm : 0) * K_C + 16 * fq;
    const device T* xr3 = x + (size_t)(ok3 ? 24 + fm : 0) * K_C + 16 * fq;

    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        const uint4 q0 = wq[(size_t)g * 64];
        const uint4 q1 = wq[(size_t)g * 64 + 1];
        const uint wv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        for (short i = 0; i < 16; i++) { ct_c[i] = 0.0f; ct_c2[i] = 0.0f; }
        float xs0 = 0.0f, xs1 = 0.0f, xs2 = 0.0f, xs3 = 0.0f;
        const device T* x0 = xr0 + k0;
        const device T* x1 = xr1 + k0;
        const device T* x2 = xr2 + k0;
        const device T* x3 = xr3 + k0;
        for (short st = 0; st < 4; st++) {
            for (short j = 0; j < 4; j++) {
                const T a0 = x0[4 * st + j], a1 = x1[4 * st + j];
                const T a2 = ok2 ? x2[4 * st + j] : static_cast<T>(0.0f);
                const T a3 = ok3 ? x3[4 * st + j] : static_cast<T>(0.0f);
                ct_a[j] = a0; ct_a[4 + j] = a1;
                ct_a2[j] = a2; ct_a2[4 + j] = a3;
                xs0 += static_cast<float>(a0); xs1 += static_cast<float>(a1);
                xs2 += static_cast<float>(a2); xs3 += static_cast<float>(a3);
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
            op.run(ct_a2, ct_b, ct_c2);
        }
        xs0 += simd_shuffle_xor(xs0, 1); xs0 += simd_shuffle_xor(xs0, 8);
        xs1 += simd_shuffle_xor(xs1, 1); xs1 += simd_shuffle_xor(xs1, 8);
        xs2 += simd_shuffle_xor(xs2, 1); xs2 += simd_shuffle_xor(xs2, 8);
        xs3 += simd_shuffle_xor(xs3, 1); xs3 += simd_shuffle_xor(xs3, 8);
        const device T* scg = sc_ + (size_t)g * 32;
        const device T* big = bi_ + (size_t)g * 32;
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const float s_ = static_cast<float>(scg[16 * bk + j]);
                const float b_ = static_cast<float>(big[16 * bk + j]) - 128.0f * s_;
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                acc[i0] = acc[i0] + (s_ * ct_c[i0] + b_ * xs0);
                acc[i1] = acc[i1] + (s_ * ct_c[i1] + b_ * xs1);
                acc2[i0] = acc2[i0] + (s_ * ct_c2[i0] + b_ * xs2);
                acc2[i1] = acc2[i1] + (s_ * ct_c2[i1] + b_ * xs3);
            }
    }

    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg == 0)
            for (int gg = 0; gg < KS - 1; gg++)
                for (short i = 0; i < 16; i++) acc[i] = acc[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg != 0) return;
        for (int gg = 0; gg < KS - 1; gg++)
            for (short i = 0; i < 16; i++) acc2[i] = acc2[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
    }
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(acc[i]);
        if (16 + m < M_C) y[(size_t)(16 + m) * N_C + n] = static_cast<T>(acc2[i]);
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="leap_qmm32n", input_names=["x", "wn", "sn", "bn"], output_names=["y"],
                                source=_SRC, header=_HDR)


def qmm32n(x, wn, sn, bn, N, ks=1, nt=1):
    """the native 4-bit dense call for 17-32 rows, one dispatch, one weight pass; == two 16-row qmm16n calls"""
    M, K = x.shape
    assert 16 < M <= HI and N % (32 * nt) == 0 and K % (64 * ks) == 0 and ks <= 16, (M, K, N, ks, nt)
    (y,) = _k()(inputs=[x, wn, sn, bn], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt), ("M_C", M)],
                grid=(N // 32 * 32 * ks, 1, 1), threadgroup=(32 * ks * nt, 1, 1), output_shapes=[(M, N)],
                output_dtypes=[x.dtype])
    return y


# the FAST dense shapes (N, K, calls per 32-row check) with DN-KS v1's ks (ref2._DNKS / its rule; lossy-45: REF2_DN_KS=1)
SHAPES = [("GDN qkv|z", 16384, 2560, 36), ("GDN b|a", 96, 2560, 36), ("out_proj / o_proj", 2560, 6144, 48),
          ("att q", 12288, 2560, 12), ("att k / v", 512, 2560, 24), ("indexer qk", 640, 2560, 12),
          ("shared gate|up|gate (SMALL-N pad)", 1312, 2560, 48), ("shared down", 2560, 640, 48),
          ("PLE key", 10240, 2560, 1), ("PLE value", 2560, 2560, 1), ("head (V1 16K list)", 16448, 2560, 1)]
DNKS1 = {(2560, 6144): 6, (96, 2560): 10, (512, 2560): 10, (1312, 2560): 8, (2560, 640): 5}


def ks_of(N, K):
    """DN-KS v1's ks (ref2._dn_ks with CFG dn_ks 1)"""
    ks = DNKS1.get((N, K))
    if ks is not None:
        return ks
    g, tiles, ks = K // 64, N // 32, 1
    for k in (1, 2, 4, 5, 8, 10, 16):
        if g % k == 0:
            ks = k
            if tiles * k >= 320:
                break
    return ks


def check(seeds=2, Ms=(17, 20, 24, 31, 32)):
    """per shape and M: qmm32n vs two qmm16n_m calls (bits), and both vs an fp32 reference (max |err| / max |ref|)"""
    import numpy as np
    import qmm26 as Q26
    from qmm26m import qmm16n_m
    rng = np.random.default_rng(1116)
    out, bad, n = [], 0, 0
    for name, N, K, _ in SHAPES:
        ks = ks_of(N, K)
        w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        wd = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)
        wn, sn, bn = Q26.repack(wq, s, b)
        mx.eval(wn, sn, bn, wd)
        e32 = e16 = 0.0
        sb = sn_ = 0
        for _ in range(seeds):
            X = mx.array(rng.standard_normal((HI, K)).astype(np.float32)).astype(mx.bfloat16)
            for M in Ms:
                a = qmm32n(X[:M], wn, sn, bn, N, ks)
                c = mx.concatenate([qmm16n_m(X[:16], wn, sn, bn, N, ks), qmm16n_m(X[16:M], wn, sn, bn, N, ks)], 0)
                ref = X[:M].astype(mx.float32) @ wd.T
                mx.eval(a, c, ref)
                n += 1
                ok = a.shape == c.shape and bool(mx.array_equal(a, c))
                bad += not ok
                sb += ok
                sn_ += 1
                r = float(mx.max(mx.abs(ref)))
                e32 = max(e32, float(mx.max(mx.abs(a.astype(mx.float32) - ref))) / r)
                e16 = max(e16, float(mx.max(mx.abs(c.astype(mx.float32) - ref))) / r)
        out.append(f"  {name:36s} N {N:6d} K {K:5d} ks {ks:2d}: bits {sb}/{sn_} == two qmm16n_m; max err / max |ref|: "
                   f"qmm32n {e32:.2e}, two 16-row {e16:.2e}")
    hdr = f"QMM32N check: {n - bad} of {n} cases bitwise == two 16-row qmm16n_m calls (M {list(Ms)} x {seeds} seeds)"
    print(hdr, flush=True)
    for s in out:
        print(s, flush=True)
    return bad, [hdr] + out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        b, lines = check()
        p = os.path.expanduser("~/Leap/data/week98/check25/qmm32n_check.txt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("\n".join(lines) + "\n")
        sys.exit(1 if b else 0)
