"""x2_mma_layout.py (Breakthrough, Tue 17 Nov 2026 09:5x; EXPERT REWRITE unit test): the facts the W4A8 kernel rests on,
checked on the GPU against numpy:
  (1) matmul2d(16, 32, 32) with a threadgroup int8 left [16 x 32] (extents (32, 16), strides {1, 64}) and a threadgroup
      int4b right in the expert kernels' code layout (row n = 64 nibbles; extents (32, 32), strides {1, 64}; transposed)
      -> int32 cooperative dest: C[m][n] = sum_k A[m][k] * B[n][k], exact;
  (2) int4b reads a nibble v as the signed value (v ^ 8) - 8, so XOR 0x8 of an unsigned code q gives q - 8;
  (3) the dest's capacity per lane, its mask, and get_multidimensional_index(i) = (column n, row m);
  (4) the second K half (pointer + 32 int8 / + 16 bytes of codes) accumulates into the same dest.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x2_mma_layout.py   (MEM < 0.2 GB)"""
import numpy as np
import mlx.core as mx

HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    threadgroup int8_t ta[16 * 64];
    threadgroup uint32_t tb[32 * 8];
    for (int i = lane; i < 16 * 64; i += 32) ta[i] = A[i];
    for (int i = lane; i < 32 * 8; i += 32) tb[i] = Bw[i] ^ (XOR_C ? 0x88888888u : 0u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<threadgroup int4b_format, dextents<int32_t, 2>, tensor_inline>;
    TA ta0(ta, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TA ta1(ta + 32, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB tb0((threadgroup uchar*)tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB tb1((threadgroup uchar*)tb + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, int32_t>();
    const uint cap = cg.get_capacity();
    for (uint i = 0; i < cap; ++i) cg[i] = 0;
    op.run(ta0, tb0, cg);
    if (HALVES_C == 2) op.run(ta1, tb1, cg);
    if (lane == 0) meta[0] = (int)cap;
    for (uint i = 0; i < 32; ++i) {
        const uint o = (lane * 32 + i) * 4;
        if (i < cap) {
            const auto ix = cg.get_multidimensional_index(i);
            res[o] = cg[i]; res[o + 1] = (int)ix[0]; res[o + 2] = (int)ix[1]; res[o + 3] = cg.is_valid_element(i) ? 1 : 0;
        } else { res[o] = 0; res[o + 1] = -1; res[o + 2] = -1; res[o + 3] = 0; }
    }
"""


def run(A, codes, xor, halves):
    k = mx.fast.metal_kernel(name="x2_layout", input_names=["A", "Bw"], output_names=["res", "meta"], source=SRC, header=HDR)
    # codes [32 rows n][64 nibbles k] -> 8 words per row; nibble j of word w = code 8w + j (bits 4j..4j+3)
    w = np.zeros((32, 8), np.uint32)
    for wi in range(8):
        acc = np.zeros(32, np.uint32)
        for j in range(8):
            acc |= (codes[:, 8 * wi + j].astype(np.uint32) & 0xF) << np.uint32(4 * j)
        w[:, wi] = acc
    res, meta = k(inputs=[mx.array(A.reshape(-1)), mx.array(w.reshape(-1))],
                  template=[("XOR_C", int(xor)), ("HALVES_C", halves)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(32 * 32 * 4,), (1,)], output_dtypes=[mx.int32, mx.int32])
    return np.array(res).reshape(32, 32, 4), int(np.array(meta)[0])


def main():
    rng = np.random.default_rng(1117)
    ok = True
    for halves in (1, 2):
        A = rng.integers(-127, 128, size=(16, 64)).astype(np.int8)
        q = rng.integers(0, 16, size=(32, 64)).astype(np.int64)      # unsigned codes as the decoder makes them
        for xor in (True, False):
            r, cap = run(A, q, xor, halves)
            Bv = (q - 8) if xor else np.where(q >= 8, q - 16, q)      # (2): the signed reading of the stored nibble
            kk = 32 * halves
            C = A[:, :kk].astype(np.int64) @ Bv[:, :kk].T              # [16 m, 32 n]
            got = np.full((16, 32), np.iinfo(np.int64).min)
            seen = np.zeros((16, 32), int)
            for lane in range(32):
                for i in range(cap):
                    v, c, m, mk = r[lane, i]
                    if not mk:
                        continue
                    got[m, c] = v
                    seen[m, c] += 1
            good = (seen == 1).all() and (got == C).all()
            ok &= bool(good)
            ex = r[0, :min(cap, 6), 1:3].tolist()
            print(f"halves {halves} xor {int(xor)}: capacity {cap}, every (m, n) once: {(seen == 1).all()}, "
                  f"exact vs numpy: {(got == C).all()} (max |diff| {np.abs(np.where(seen == 1, got - C, 0)).max()}); "
                  f"lane 0's first (n, m): {ex}")
    print("X2 LAYOUT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
