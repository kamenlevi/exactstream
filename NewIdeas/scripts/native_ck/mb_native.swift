// NATIVE-CK M-B (NewIdeas Sun 8 Nov; pre-registered IDEAS.md 15:44, dd4c28f): native Metal encode cost per dispatch
// for a dependent chain of 2,000 dispatches. Three small kernels in rotation (pipeline switch every dispatch), each
// dispatch binds 2 buffers + 1 constant, one serial compute encoder in one command buffer. 20 timed reps after 3 warm-ups.
// Prints per-dispatch thread CPU for encoding (makeCommandBuffer .. endEncoding), commit + wait wall, and GPU time
// (gpuEndTime - gpuStartTime) per dispatch. usage: swiftc -O mb_native.swift -o mb_native && taskpolicy -b ./mb_native
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void k0(device float* x [[buffer(0)]], device const float* a [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint i [[thread_position_in_grid]]) { if (i < n) x[i] = x[i] * a[i] + 1.0f; }
kernel void k1(device float* x [[buffer(0)]], device const float* a [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint i [[thread_position_in_grid]]) { if (i < n) x[i] = x[i] - a[i] * 0.5f; }
kernel void k2(device float* x [[buffer(0)]], device const float* a [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint i [[thread_position_in_grid]]) { if (i < n) x[i] = fma(x[i], 0.999f, a[i]); }
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let psos = ["k0", "k1", "k2"].map { try! dev.makeComputePipelineState(function: lib.makeFunction(name: $0)!) }
let q = dev.makeCommandQueue()!
let n = 2560, N = 2000
let x = dev.makeBuffer(length: n * 4, options: .storageModeShared)!
let a = dev.makeBuffer(length: n * 4, options: .storageModeShared)!
var enc_us = [Double](), wall_us = [Double](), gpu_us = [Double]()
for rep in 0..<23 {
    let c0 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
    var nn = UInt32(n)
    for d in 0..<N {
        e.setComputePipelineState(psos[d % 3])
        e.setBuffer(x, offset: 0, index: 0)
        e.setBuffer(a, offset: 0, index: 1)
        e.setBytes(&nn, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    e.endEncoding()
    let c1 = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
    let w0 = DispatchTime.now().uptimeNanoseconds
    cb.commit(); cb.waitUntilCompleted()
    let w1 = DispatchTime.now().uptimeNanoseconds
    if rep >= 3 {
        enc_us.append(Double(c1 - c0) / 1e3 / Double(N))
        wall_us.append(Double(w1 - w0) / 1e3 / Double(N))
        gpu_us.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N))
    }
}
func med(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
print(String(format: "M-B native: per dispatch encode thread CPU %.3f us; commit+wait wall %.3f us; GPU %.3f us (%.2f ms GPU per %d-dispatch chain)",
             med(enc_us), med(wall_us), med(gpu_us), med(gpu_us) * Double(N) / 1e3, N))
