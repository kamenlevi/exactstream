"""x2b_dest_formula.py (Breakthrough, Tue 17 Nov 2026 10:2x; EXPERT REWRITE unit test): does the cooperative dest of
matmul2d(16, 32, 32) with THREADGROUP tensor operands (bf16 x uint4b -> float, int8 x int4b -> int32) place element i of
lane l at Leap's analytic position (the cooperative-left form's layout, tierglu_nax6._CORE4):
    col = 16 (i >> 3) + fn0 + (i & 3),  row = fm0 + 8 ((i >> 2) & 1),
    qid = l >> 2, fm0 = (qid & 4) | ((l >> 1) & 3), fn0 = ((qid & 2) | (l & 1)) * 4 ?
If yes, the epilogue can index scales by the formula (compile-time structure: 8 distinct columns, 2 rows per lane).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x2b_dest_formula.py"""
import numpy as np
import mlx.core as mx

HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    threadgroup LT_ ta[16 * 64];
    threadgroup uint32_t tb[32 * 8];
    for (int i = lane; i < 16 * 64; i += 32) ta[i] = (LT_)0;
    for (int i = lane; i < 32 * 8; i += 32) tb[i] = 0u;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TA = tensor<threadgroup LT_, dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<threadgroup RT_, dextents<int32_t, 2>, tensor_inline>;
    TA ta0(ta, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB tb0((threadgroup uchar*)tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, DT_>();
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    int bad = 0;
    for (int i = 0; i < 16; ++i) {
        const auto ix = cg.get_multidimensional_index(i);
        const int col = 16 * (i >> 3) + fn0 + (i & 3), row = fm0 + 8 * ((i >> 2) & 1);
        bad += (ix[0] != col) || (ix[1] != row);
    }
    out[lane] = bad + 1000 * (int)(cg.get_capacity() != 16);
"""


def run(lt, rt, dt):
    s = SRC.replace("LT_", lt).replace("RT_", rt).replace("DT_", dt)
    k = mx.fast.metal_kernel(name=f"x2b_{lt}_{dt}", input_names=["dummy"], output_names=["out"], source=s, header=HDR)
    o = k(inputs=[mx.zeros((1,), mx.int32)], template=[("T", mx.bfloat16)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
          output_shapes=[(32,)], output_dtypes=[mx.int32])[0]
    return np.array(o)


ok = True
for lt, rt, dt in (("bfloat", "uint4b_format", "float"), ("int8_t", "int4b_format", "int32_t")):
    r = run(lt, rt, dt)
    good = int(r.sum()) == 0
    ok &= good
    print(f"{lt} x {rt} -> {dt}: elements off the analytic layout per lane: {r.tolist()} -> {'MATCH' if good else 'DIFFERS'}")
print("X2B DEST FORMULA:", "PASS (use the formula)" if ok else "FAIL (keep get_multidimensional_index)")
