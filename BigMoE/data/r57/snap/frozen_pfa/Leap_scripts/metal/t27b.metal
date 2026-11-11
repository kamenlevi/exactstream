#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#define T bfloat
#define K_C 2560
#define N_C 1280
#define MROWS 16
#define P 160
kernel void probe2(device const bfloat* xg [[buffer(0)]], device const float* xs [[buffer(1)]],
                   device const uint32_t* w [[buffer(2)]], device const bfloat* scales [[buffer(3)]],
                   device const bfloat* biases [[buffer(4)]], device const int* sexp [[buffer(5)]],
                   device const int* seg_start [[buffer(6)]], device bfloat* y [[buffer(7)]],
                   uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    using namespace metal;
    constexpr int G = 64;
    const int t = threadgroup_position_in_grid.x;
    const int n0 = t * 32;
#if 1
    const int s = threadgroup_position_in_grid.y;
    const int p0 = seg_start[s];
    if (p0 >= P) return;
    const int cnt = seg_start[s + 1] - p0;
    const int e = sexp[p0];
#else
    const int p0 = 0, cnt = MROWS, e = 0;
#endif
    constexpr int KG = K_C / G;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, G, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tX((device bfloat*)(xg + (size_t)p0 * K_C),
                                                                dextents<int32_t, 2>(K_C, 16));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tW(
        (device uint8_t*)(w + (size_t)e * N_C * (K_C / 8)), dextents<int32_t, 2>(K_C, N_C));
    auto a0 = tX.slice<G, 16>(0, 0);
    auto b0 = tW.slice<G, 32>(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto part = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
    const device T* se = scales + ((size_t)e * N_C + n0) * KG;
    const device T* be = biases + ((size_t)e * N_C + n0) * KG;
    const device float* xsp = xs + (size_t)p0 * KG;
    for (int g = 0; g < KG; g++) {
        #pragma unroll
        for (uint16_t i = 0; i < part.get_capacity(); ++i) if (part.is_valid_element(i)) part[i] = 0.0f;
        auto a = tX.slice<G, 16>(g * G, 0);
        auto b = tW.slice<G, 32>(g * G, n0);
        op.run(a, b, part);
        #pragma unroll
        for (uint16_t i = 0; i < part.get_capacity(); ++i) {
            if (part.is_valid_element(i)) {
                auto idx = part.get_multidimensional_index(i);         // (column n, row m) within the 16 x 32 tile
                const int n = idx[0], m = idx[1];
                const float s_ = static_cast<float>(se[(size_t)n * KG + g]);
                const float b_ = static_cast<float>(be[(size_t)n * KG + g]);
                acc[i] = acc[i] + (s_ * part[i] + b_ * xsp[(size_t)m * KG + g]);
            }
        }
    }
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto idx = acc.get_multidimensional_index(i);
            const int n = idx[0], m = idx[1];
            if (m < cnt) y[(size_t)(p0 + m) * N_C + n0 + n] = static_cast<T>(acc[i]);
        }
    }

}
