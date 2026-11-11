"""NAX probe (Leap, Sun 2026-11-01): can a custom mx.fast.metal_kernel use the M5's matrix units through Metal 4's
MetalPerformancePrimitives (mpp::tensor_ops::matmul2d, as MLX 0.32's steel/gemm/nax.h does)? One simdgroup computes
C [16 x 32] = A [16 x 16] @ B [16 x 32] from bf16 inputs with an fp32 accumulator, using MLX's BaseNAXFrag element map
(lane holds rows fm, fm + 8 x cols fn .. fn + 3 of each 16 x 16 frag); B is given transposed (W [32 x 16], rows = n),
like qmm_t. Checks against numpy; then row-invariance: C's row i must not depend on A's other rows."""
import numpy as np
import mlx.core as mx

_HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline short2 nax_coord(ushort lane, short idx) {
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (lane & 1)) * 4 + idx % 4;
    return short2{fn, fm};
}
"""
_SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                             metal::remove_addrspace_t<decltype(cb)>, float>();
    for (short i = 0; i < 8; i++) {
        const short2 c = nax_coord(lane, i);              // (col, row)
        ca[i] = a[c.y * 16 + c.x];                        // A [m][k]
        cb[i] = w[c.y * 16 + c.x];                        // frag 0: W rows n 0..15 (n = c.y), k = c.x
        cb[8 + i] = w[(16 + c.y) * 16 + c.x];             // frag 1: n 16..31
        cc[i] = 0.0f; cc[8 + i] = 0.0f;
    }
    op.run(ca, cb, cc);
    for (short i = 0; i < 8; i++) {
        const short2 c = nax_coord(lane, i);
        out[c.y * 32 + c.x] = cc[i];                      // C [m][n], n 0..15
        out[c.y * 32 + 16 + c.x] = cc[8 + i];             // n 16..31
    }
"""
k = mx.fast.metal_kernel(name="nax_probe", input_names=["a", "w"], output_names=["out"], source=_SRC, header=_HDR)
rng = np.random.default_rng(0)
A = rng.standard_normal((16, 16)).astype(np.float32)
W = rng.integers(0, 8, (32, 16)).astype(np.float32)
a = mx.array(A).astype(mx.bfloat16)
w = mx.array(W).astype(mx.bfloat16)
(c,) = k(inputs=[a, w], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, 32)], output_dtypes=[mx.float32])
C = np.array(c)
ref = np.array(a.astype(mx.float32)) @ np.array(w.astype(mx.float32)).T
print("max |C - ref|:", float(np.abs(C - ref).max()), " max |ref|:", float(np.abs(ref).max()))
# row invariance: change rows != 3 of A, row 3 of C must stay bitwise equal
outs = []
for s in range(6):
    A2 = rng.standard_normal((16, 16)).astype(np.float32) * (10 ** (s - 3))
    A2[3] = A[3]
    (c2,) = k(inputs=[mx.array(A2).astype(mx.bfloat16), w], grid=(32, 1, 1), threadgroup=(32, 1, 1),
              output_shapes=[(16, 32)], output_dtypes=[mx.float32])
    outs.append(np.array(c2)[3].view(np.uint32))
print("row 3 bitwise equal across 6 different other-row sets:", all((o == outs[0]).all() for o in outs),
      "| equals the base run:", bool((outs[0] == C[3].view(np.uint32)).all()))
