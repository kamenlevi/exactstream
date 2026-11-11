"""NAX probe 2 (Leap, Sun 2026-11-01): matmul2d 16 x 32 x 16 with transpose_right = FALSE (B given as [k][n], e.g. V [keys][dims]):
which B element does each lane hold? Uses MPP's get_multidimensional_index for A, B and C, fills from row-major arrays, compares
with numpy."""
import numpy as np
import mlx.core as mx
_HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
_SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < (short)ca.get_capacity(); i++) { auto ix = ca.get_multidimensional_index(i); ca[i] = a[ix[1] * 16 + ix[0]];
        if (lane == 0) idx[2 * i] = ix[0], idx[2 * i + 1] = ix[1]; }
    for (short i = 0; i < (short)cb.get_capacity(); i++) { auto ix = cb.get_multidimensional_index(i); cb[i] = b[ix[1] * 32 + ix[0]];
        if (lane == 0) idx[32 + 2 * i] = ix[0], idx[32 + 2 * i + 1] = ix[1]; }
    for (short i = 0; i < (short)cc.get_capacity(); i++) cc[i] = 0.0f;
    op.run(ca, cb, cc);
    for (short i = 0; i < (short)cc.get_capacity(); i++) { auto ix = cc.get_multidimensional_index(i); out[ix[1] * 32 + ix[0]] = cc[i]; }
"""
k = mx.fast.metal_kernel(name="nax_probe_nt", input_names=["a", "b"], output_names=["out", "idx"], source=_SRC, header=_HDR)
rng = np.random.default_rng(0)
A = rng.standard_normal((16, 16)).astype(np.float32); Bm = rng.standard_normal((16, 32)).astype(np.float32)   # B [k][n]
a = mx.array(A).astype(mx.bfloat16); b = mx.array(Bm).astype(mx.bfloat16)
c, ix = k(inputs=[a, b], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, 32), (96,)], output_dtypes=[mx.float32, mx.int32],
          init_value=-1)
ref = np.array(a.astype(mx.float32)) @ np.array(b.astype(mx.float32))
print("max |C - A @ B| (B as [k][n], index API: fast dim first):", float(np.abs(np.array(c) - ref).max()))
ixn = np.array(ix)
print("lane 0 A idx (fast, slow):", ixn[:16].reshape(-1, 2).tolist())
print("lane 0 B idx (fast, slow):", ixn[32:64].reshape(-1, 2).tolist())
