"""NAX throughput probe (Leap, Sun 2026-11-01): MMAs per second for the 16 x 32 x 16 bf16 -> fp32 matmul2d op on
register (cooperative) operands, no memory traffic: (a) back-to-back op.run, (b) the same with the B operand rebuilt
from integer ops each step (like tierglu_nax's decode, ~2 ops per value), (c) with the A operand re-marshalled each step."""
import time
import numpy as np
import mlx.core as mx

_HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
_SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cg = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                             metal::remove_addrspace_t<decltype(cb)>, float>();
    for (int i = 0; i < 8; i++) ca[i] = bfloat(0.001f * (lane + i));
    for (int i = 0; i < 16; i++) { cb[i] = bfloat(0.5f + 0.01f * i); cg[i] = 0.0f; }
    uint w = lane * 2654435761u + threadgroup_position_in_grid.x;
    for (int it = 0; it < ITER_C; ++it) {
        if (MODE_C >= 1) {
            for (int t = 0; t < 4; ++t) {
                const uint f = (w >> (t * 3)) & 0xFFFu;
                const uint m = f | (f << 10);
                const uint r0 = (m & 0x00070007u) | 0x43004300u, r1 = ((m >> 3) & 0x00070007u) | 0x43004300u;
                const vec<bfloat, 2> p0 = as_type<vec<bfloat, 2>>(r0), p1 = as_type<vec<bfloat, 2>>(r1);
                cb[4 * t] = p0[0]; cb[4 * t + 1] = p0[1]; cb[4 * t + 2] = p1[0]; cb[4 * t + 3] = p1[1];
            }
            w = w * 1664525u + 1013904223u;
        }
        if (MODE_C >= 2) {
            for (int i = 0; i < 8; i++) ca[i] = bfloat(float(w >> i & 255) * 0.001f);
        }
        op.run(ca, cb, cg);
    }
    float s = 0.0f;
    for (int i = 0; i < 16; i++) s += cg[i];
    if (s == 12345.678f) out[0] = s;
"""
k = mx.fast.metal_kernel(name="nax_tput", input_names=["dummy"], output_names=["out"], source=_SRC, header=_HDR)
d = mx.zeros((1,), mx.float32)
ITER, NTG = 1024, 20 * 64
for mode in (0, 1, 2):
    f = lambda: k(inputs=[d], template=[("ITER_C", ITER), ("MODE_C", mode)], grid=(NTG * 32, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0]
    for _ in range(3): mx.eval(f())
    ts = []
    for _ in range(15):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    t = float(np.median(ts))
    n = ITER * NTG
    print(f"mode {mode}: {n / t / 1e9:.2f} G MMA/s = {n * 8192 / t / 1e12:.1f} T MAC/s; per core {n / t / 20 / 1.6e9 * 1e3:.1f} MMA "
          f"per 1000 cycles (1.6 GHz) -> {20 * 1.6e9 / (n / t):.1f} cycles per MMA per core", flush=True)
