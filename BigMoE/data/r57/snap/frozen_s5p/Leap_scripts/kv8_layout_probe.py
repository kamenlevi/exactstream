"""KV8 layout probe (Leap, Tue 2026-11-03 20:0x; TABLE_350 'KV8'): the lane -> (col, row) coordinates of every element of
MPP's cooperative tensors, from get_multidimensional_index, for fs4's bf16 matmul2d (16, 32, 16) and for the int8 form
(16, 32, 32) -> int32. Checks fs4's hard-coded layout (fatt16s.py: qa / kb / ss) against the API, then tells whether the
int8 destination has the same layout as fs4's float scores (so fs4's online softmax can run on it unchanged).
Usage: [BigMoE venv] kv8_layout_probe.py (GPU, seconds)"""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from fatt16 import _HDR

SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto d16 = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto d8 = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<d16, metal::execution_simdgroup> op16;
    mpp::tensor_ops::matmul2d<d8, metal::execution_simdgroup> op8;
    auto qa = op16.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb = op16.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ss = op16.template get_destination_cooperative_tensor<decltype(qa), decltype(kb), float>();
    auto qa8 = op8.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto kb8 = op8.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ss8 = op8.template get_destination_cooperative_tensor<decltype(qa8), decltype(kb8), int32_t>();
    // out[t][lane][i] = (col, row, mask) for tensor t = qa, kb, ss, qa8, kb8, ss8 (up to 64 elements each)
    #define DUMP(T_, CT) for (int i = 0; i < 64; ++i) { device int* o = out + (((T_) * 32 + lane) * 64 + i) * 3; \
        if (i < (int)CT.get_capacity()) { auto id = CT.get_multidimensional_index(i); o[0] = id[0]; o[1] = id[1]; o[2] = 1; } \
        else { o[0] = -1; o[1] = -1; o[2] = -1; } }
    DUMP(0, qa) DUMP(1, kb) DUMP(2, ss) DUMP(3, qa8) DUMP(4, kb8) DUMP(5, ss8)
"""
k = mx.fast.metal_kernel(name="kv8_layout_probe", input_names=["dummy"], output_names=["out"], source=SRC, header=_HDR)
(o,) = k(inputs=[mx.zeros((1,), mx.int32)], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(6, 32, 64, 3)],
         output_dtypes=[mx.int32])
o = np.array(o)
names = ["qa bf16 (16x16 left)", "kb bf16 (right, 32 keys x 16)", "ss float (16x32)", "qa8 int8 (16x32 left)",
         "kb8 int8 (right, 32 keys x 32)", "ss8 int32 (16x32)"]
cap = [(o[t, 0, :, 0] >= 0).sum() for t in range(6)]
print("capacity per lane:", dict(zip(names, cap)))
for t in range(6):
    print(names[t], "lane 0..3 (col,row):", [[tuple(o[t, l, i, :2]) for i in range(cap[t])] for l in range(4)])
# fs4's hard-coded layout (fatt16s.py)
ok = {"qa": True, "kb": True, "ss": True}
for l in range(32):
    fm0 = ((l >> 2) & 4) | ((l >> 1) & 3)
    fn0 = (((l >> 2) & 2) | (l & 1)) * 4
    for j in range(8):                                     # qa[j]: row fm0 + 8 (j >= 4), dim fn0 + (j & 3)
        exp = (fn0 + (j & 3), fm0 + 8 * (j >= 4))
        ok["qa"] &= tuple(o[0, l, j, :2]) in (exp, exp[::-1])
    for t_ in range(4):
        for j in range(4):                                 # kb[4 t + j]: key fm0 + 8 t, dim fn0 + j
            e_ = (fm0 + 8 * t_, fn0 + j)
            ok["kb"] &= tuple(o[1, l, 4 * t_ + j, :2]) in (e_, e_[::-1])
    for i in range(16):                                    # ss[i]: row fm0 + 8 ((i >> 2) & 1), key 16 (i >> 3) + fn0 + (i & 3)
        e_ = (16 * (i >> 3) + fn0 + (i & 3), fm0 + 8 * ((i >> 2) & 1))
        ok["ss"] &= tuple(o[2, l, i, :2]) in (e_, e_[::-1])
print("fs4's hard-coded layout matches the API (either coordinate order):", ok)
same = cap[2] == cap[5] and all(tuple(o[2, l, i, :2]) == tuple(o[5, l, i, :2]) for l in range(32) for i in range(cap[2]))
print("int32 destination (16, 32, 32) has fs4's float score layout:", same)
np.save(os.path.expanduser("~/Leap/data/kv8_layout.npy"), o)
