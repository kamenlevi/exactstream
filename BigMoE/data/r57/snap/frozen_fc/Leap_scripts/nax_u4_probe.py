"""NAX uint4 probe (Leap, Sun 2026-11-01): can mpp matmul2d take B as packed 4-bit unsigned codes from threadgroup
memory (bf16 cooperative A x uint4b_format threadgroup B -> fp32 cooperative C), and what is the nibble layout?
One simdgroup: A [16 x 16] bf16 (BaseNAXFrag map, verified by nax_probe.py), W [32 x 16] codes 0..15 written as nibbles
into threadgroup memory as rows n (16 nibbles = 8 bytes per row, low nibble = even k), B = W^T via transpose_right.
Tries a few stride / extent conventions and reports which one matches numpy."""
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
    threadgroup uchar wt[32 * 16];
    for (int i = lane; i < 32 * 16; i += 32) wt[i] = wpk[i];         // W rows n: 16 bytes = 32 nibbles (k)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, TR_C, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    // B tensor over threadgroup nibbles: extents (fast dim first)
    tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> tb(
        (threadgroup uchar*)wt, dextents<int32_t, 2>(E0_C, E1_C), array<int32_t, 2>{S0_C, S1_C});
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(tb), float>();
    for (short i = 0; i < (short)ca.get_capacity(); i++) {
        const auto ix = ca.get_multidimensional_index(i);         // (fast = k, slow = m)
        ca[i] = a[ix[1] * 32 + ix[0]];
        if (lane == 0 && DUMP_C) { idx[2 * i] = ix[0]; idx[2 * i + 1] = ix[1]; }
    }
    for (short i = 0; i < (short)cc.get_capacity(); i++) cc[i] = 0.0f;
    op.run(ca, tb, cc);
    for (short i = 0; i < (short)cc.get_capacity(); i++) {
        const auto ix = cc.get_multidimensional_index(i);         // (fast = n, slow = m)
        out[ix[1] * 32 + ix[0]] = cc[i];
        if (lane == 0 && DUMP_C) { idx[64 + 2 * i] = ix[0]; idx[64 + 2 * i + 1] = ix[1]; }
    }
"""
rng = np.random.default_rng(0)
A = rng.standard_normal((16, 32)).astype(np.float32)
W = rng.integers(0, 16, (32, 32))                                   # [n][k]
pk = (W[:, 0::2] | (W[:, 1::2] << 4)).astype(np.uint8).reshape(-1)  # low nibble = even k
a = mx.array(A).astype(mx.bfloat16)
ref = np.array(a.astype(mx.float32)) @ W.T.astype(np.float32)        # [16][32]
cands = [  # (transpose_right, extents fast-first, strides)
    (True, (32, 32), (1, 32)),     # B^T as [n][k], k fast
    (False, (32, 32), (1, 32)),    # B as [k][n] from the same bytes (a layout check)
    (True, (32, 32), (32, 1)),
]
for tr, (e0, e1), (s0, s1) in cands:
    try:
        k = mx.fast.metal_kernel(name=f"nax_u4_{int(tr)}_{e0}_{e1}_{s0}_{s1}", input_names=["a", "wpk"],
                                 output_names=["out", "idx"], source=_SRC, header=_HDR)
        c, ix = k(inputs=[a, mx.array(pk)], template=[("TR_C", tr), ("E0_C", e0), ("E1_C", e1), ("S0_C", s0), ("S1_C", s1),
                                                      ("DUMP_C", 1)],
                  grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, 32), (128,)],
                  output_dtypes=[mx.float32, mx.int32], init_value=-1)
        print("   lane 0 A idx (k, m):", np.array(ix)[:32].reshape(-1, 2).tolist())
        print("   lane 0 C idx (n, m):", np.array(ix)[64:96].reshape(-1, 2).tolist())
        C = np.array(c)
        best = min(np.abs(C - ref).max(), np.abs(C - (np.array(a.astype(mx.float32)) @ W.T.astype(np.float32))).max())
        print(f"tr {tr} ext {(e0, e1)} str {(s0, s1)}: max |C - ref| {np.abs(C - ref).max():.3g} (|ref| max {np.abs(ref).max():.3g})",
              flush=True)
    except Exception as ex:
        print(f"tr {tr} ext {(e0, e1)} str {(s0, s1)}: FAILED {str(ex)[:600]}", flush=True)
