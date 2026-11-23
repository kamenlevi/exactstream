#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

// probe: y[16 x 32] tile = sum over 64-value groups of x[16 x 64] * q[32 x 64]^T, q = packed uint4 weights
kernel void probe(device const bfloat* x [[buffer(0)]],
                  device const uint8_t* w [[buffer(1)]],
                  device float* y [[buffer(2)]],
                  constant int& K [[buffer(3)]],
                  constant int& N [[buffer(4)]],
                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr auto desc = tensor_ops::matmul2d_descriptor(16, 32, 64, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroup> op;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tX((device bfloat*)x, dextents<int32_t, 2>(K, 16));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tW((device uint8_t*)w, dextents<int32_t, 2>(K, N));
    auto ct = op.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    for (int i = 0; i < ct.get_capacity(); i++) ct[i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 64) {
        auto a = tX.slice<64, 16>(k0, 0);
        auto b = tW.slice<64, 32>(k0, 0);
        op.run(a, b, ct);
    }
    for (int i = 0; i < ct.get_capacity(); i++) y[lane * 16 + i] = ct[i];
}
