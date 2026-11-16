"""WEEK98 MODULE 2 (MoonShots, Mon 16 Nov 2026 22:5x): can a segment of <= 8 pairs use an 8-row MMA tile and keep
ooo1cq's bits? The one-call's kernels run MPP's matmul2d(16, 32, 32, NT, relaxed) per K half, bf16 rows x uint4 nibbles ->
float, on segments of ~5.2 pairs (ablation: the MMA is ~30 % of kernel A). This test feeds the SAME operands (the
kernels' layouts: x rows as the left input, a 32 x 32 uint4b tile in threadgroup memory with strides {1, 64}, two K halves
accumulated in one float tile) to M = 16 and to M = 8 (rows 0-7, then 8-15), elements placed and read through each
cooperative tensor's own get_multidimensional_index, and compares every row's 32 outputs bit for bit, over NT random
tiles (x ~ N(0, 1) bf16 incl. cancellation columns; nibbles uniform 0..15). Also checks M = 16 here == the kernels'
hardcoded fragment map. usage: ~/BigMoE/.venv/bin/python -I -B m2_mma8.py [--tiles 256]   (MEM < 0.1 GB)"""
import argparse
import os
import sys

import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", type=int, default=256)
A_ = ap.parse_args()
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import tierglu_nax6 as N6                                # noqa: E402  (its header: MPP include + helpers)

SRC = r"""
    // one simdgroup per tile; x [NT, 16, 64] bf16 (rows, K), w [NT, 32, 32] u32 words = 32 rows x 64 nibbles
    const uint t = threadgroup_position_in_grid.x;
    const ushort lane = thread_index_in_simdgroup;
    threadgroup uint32_t tb[32 * 8];
    for (int i = lane; i < 32 * 8; i += 32) tb[i] = w[t * 256 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MR, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    for (int i = 0; i < (int)cg.get_capacity(); ++i) cg[i] = 0.0f;
    for (int hh = 0; hh < 2; ++hh) {
        for (int i = 0; i < (int)ca.get_capacity(); ++i) {
            if (!ca.is_valid_element(i)) continue;
            auto ix = ca.get_multidimensional_index(i);           // (k, m) for the left input: dim 0 = K, dim 1 = rows
            ca[i] = x[(size_t)t * 16 * 64 + (size_t)(R0 + ix[1]) * 64 + 32 * hh + ix[0]];
        }
        TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
        op.run(ca, tbt, cg);
    }
    for (int i = 0; i < (int)cg.get_capacity(); ++i) {
        if (!cg.is_valid_element(i)) continue;
        auto ix = cg.get_multidimensional_index(i);               // (n, m): dim 0 = output column, dim 1 = row
        out[(size_t)t * 16 * 32 + (size_t)(R0 + ix[1]) * 32 + ix[0]] = cg[i];
    }
"""
# the kernels' own hardcoded fragment map for M = 16 (tierglu_nax6._CORE4): lane -> rows fm0 / fm0 + 8, K cols fn0..+3 / +16
SRC_HARD = r"""
    const uint t = threadgroup_position_in_grid.x;
    const ushort lane = thread_index_in_simdgroup;
    threadgroup uint32_t tb[32 * 8];
    for (int i = lane; i < 32 * 8; i += 32) tb[i] = w[t * 256 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
    const device T* ar0 = x + (size_t)t * 16 * 64 + (size_t)fm0 * 64;
    const device T* ar1 = x + (size_t)t * 16 * 64 + (size_t)(fm0 + 8) * 64;
    for (int hh = 0; hh < 2; ++hh) {
        const int k0 = 32 * hh;
        for (int q = 0; q < 4; ++q) { ca[q] = ar0[k0 + fn0 + q]; ca[4 + q] = ar1[k0 + fn0 + q];
                                      ca[8 + q] = ar0[k0 + 16 + fn0 + q]; ca[12 + q] = ar1[k0 + 16 + fn0 + q]; }
        TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
        op.run(ca, tbt, cg);
    }
    for (int i = 0; i < 16; ++i) {
        const int row = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
        out[(size_t)t * 16 * 32 + row * 32 + col] = cg[i];
    }
"""


def kern(src, name):
    return mx.fast.metal_kernel(name=name, input_names=["x", "w"], output_names=["out"], source=src, header=N6._HDR4)


rng = np.random.default_rng(1116)
NT = A_.tiles
xf = rng.standard_normal((NT, 16, 64)).astype(np.float32)
xf[:, :, 1::7] = -xf[:, :, 0::7][:, :, :xf[:, :, 1::7].shape[2]]   # cancellation pairs: order-sensitive sums
x = mx.array(xf).astype(mx.bfloat16)
w = mx.array(rng.integers(0, 2 ** 32, (NT, 256), dtype=np.uint64).astype(np.uint32))
K16, K8, KH = kern(SRC, "m2mma_gen"), kern(SRC, "m2mma_gen8"), kern(SRC_HARD, "m2mma_hard")
o16 = K16(inputs=[x, w], template=[("T", mx.bfloat16), ("MR", 16), ("R0", 0)], grid=(32 * NT, 1, 1), threadgroup=(32, 1, 1),
          output_shapes=[(NT * 16 * 32,)], output_dtypes=[mx.float32], init_value=0)[0]
o8a = K8(inputs=[x, w], template=[("T", mx.bfloat16), ("MR", 8), ("R0", 0)], grid=(32 * NT, 1, 1), threadgroup=(32, 1, 1),
         output_shapes=[(NT * 16 * 32,)], output_dtypes=[mx.float32], init_value=0)[0]
o8b = K8(inputs=[x, w], template=[("T", mx.bfloat16), ("MR", 8), ("R0", 8)], grid=(32 * NT, 1, 1), threadgroup=(32, 1, 1),
         output_shapes=[(NT * 16 * 32,)], output_dtypes=[mx.float32], init_value=0)[0]
oh = KH(inputs=[x, w], template=[("T", mx.bfloat16)], grid=(32 * NT, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(NT * 16 * 32,)], output_dtypes=[mx.float32], init_value=0)[0]
mx.eval(o16, o8a, o8b, oh)
a16, ah = np.array(o16).reshape(NT, 16, 32), np.array(oh).reshape(NT, 16, 32)
a8 = np.concatenate([np.array(o8a).reshape(NT, 16, 32)[:, :8], np.array(o8b).reshape(NT, 16, 32)[:, 8:]], axis=1)
# a reference in fp64: nibble values from the words (row n: 64 nibbles = 8 words; K value j = nibble j)
wn = np.array(w).reshape(NT, 32, 8)
nib = np.stack([(wn >> (4 * s)) & 0xF for s in range(8)], -1).reshape(NT, 32, 64).astype(np.float64)
ref = np.einsum("tmk,tnk->tmn", np.array(x.astype(mx.float32)).astype(np.float64), nib)
print(f"M2 MMA8 {NT} tiles: generic M=16 == the kernels' hardcoded map bitwise: {bool((a16.view(np.uint32) == ah.view(np.uint32)).all())}")
print(f"  M=8 (rows 0-7, 8-15) == M=16 bitwise, per row: {bool((a8.view(np.uint32) == a16.view(np.uint32)).all())}; "
      f"rows differing {int((a8.view(np.uint32) != a16.view(np.uint32)).any(-1).sum())} of {NT * 16}")
print(f"  max |M=16 - fp64| {np.abs(a16 - ref).max():.3e}, max |M=8 - fp64| {np.abs(a8 - ref).max():.3e} "
      f"(scale {np.abs(ref).max():.1f})")
