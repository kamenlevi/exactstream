"""x7_async.py (Breakthrough, Tue 17 Nov 2026 11:3x; MOE BLOCK / EXPERT REWRITE info screen): does matmul2d run
ASYNCHRONOUSLY within its simdgroup? One simdgroup per core (20 threadgroups x 32 threads), so no other simdgroup can
hide anything. ITER iterations of:
  A  MMA only: 2 x matmul2d(16, 32, 32) (bf16 threadgroup left x uint4b) accumulating
  B  ALU only: a decode-like chain (spread3 on 8 words, ~100 integer ops) folded into a register
  C  MMA issued, then B's ALU (independent), then the MMA result used (cg into acc)
  D  B's ALU, then the MMA (the dependent order of today's loop)
If C ~ max(A, B) the accelerator overlaps the simdgroup's own ALU work, and a decode-ahead loop can hide it. If
C ~ A + B it does not.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x7_async.py"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP

MP.load(2)
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402

HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline uint spread3(uint x) {
    uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
    a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
    return (a & 0x07070707u) | ((a & 0x38383838u) << 1);
}
"""
SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    threadgroup uint32_t tb[256];
    threadgroup bfloat ta[16 * 64];
    for (int i = lane; i < 256; i += 32) tb[i] = (0x76543210u ^ ((uint)i * 2654435761u)) & 0x77777777u;
    for (int i = lane; i < 1024; i += 32) ta[i] = (bfloat)(0.001f * (float)(i & 255));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TA = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    TA ta0(ta, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TA ta1(ta + 32, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB tb0((threadgroup uchar*)tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB tb1((threadgroup uchar*)tb + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, float>();
    float acc = 0.0f;
    uint fold = (uint)lane * 2654435761u;
    for (int it = 0; it < ITER_C; ++it) {
        if (MODE_C == 3) {                                   // D: ALU first
            for (int q = 0; q < 8 * ALU_C; ++q) fold = spread3(fold ^ (uint)(q * 97 + it)) + (fold >> 7);
        }
        if (MODE_C != 1) {
            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
            op.run(ta0, tb0, cg);
            op.run(ta1, tb1, cg);
        }
        if (MODE_C == 1 || MODE_C == 2) {                    // B / C: the ALU chain (after the issue in C)
            for (int q = 0; q < 8 * ALU_C; ++q) fold = spread3(fold ^ (uint)(q * 97 + it)) + (fold >> 7);
        }
        if (MODE_C != 1) {
            for (int i = 0; i < 16; ++i) acc += cg[i];
        }
    }
    out[threadgroup_position_in_grid.x * 32 + lane] = acc + (float)(fold & 1023u);
"""


def run(mode, alu, iters=4000, reps=5):
    k = mx.fast.metal_kernel(name=f"x7_m{mode}", input_names=["dummy"], output_names=["out"], source=SRC, header=HDR)
    d = mx.zeros((1,), mx.int32)
    mx.eval(d)
    tm = [("ITER_C", iters), ("MODE_C", mode), ("ALU_C", alu)]
    mx.eval(k(inputs=[d], template=tm, grid=(20 * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(20 * 32,)],
              output_dtypes=[mx.float32])[0])
    ts = []
    for _ in range(reps):
        mx.synchronize()
        c0 = MP.counts()["ncb"]
        mx.eval(k(inputs=[d], template=tm, grid=(20 * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(20 * 32,)],
                  output_dtypes=[mx.float32])[0])
        mx.synchronize()
        cb = MP.cbs(c0, MP.counts()["ncb"])
        g = [1e6 * (e - s) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > s]
        ts.append(max(g))
    return min(ts) / iters * 1e3                            # ns per iteration


def main():
    for alu in (1, 2):
        a, b, c, dd = (run(m, alu) for m in (0, 1, 2, 3))
        print(f"ALU x{alu}: ns / iteration: A (MMA) {a:.1f}, B (ALU) {b:.1f}, C (MMA, ALU, use) {c:.1f}, "
              f"D (ALU, MMA, use) {dd:.1f}; max(A, B) {max(a, b):.1f}, A + B {a + b:.1f}; C hides "
              f"{100 * (a + b - c) / min(a, b):.0f} % of the smaller", flush=True)


if __name__ == "__main__":
    main()
