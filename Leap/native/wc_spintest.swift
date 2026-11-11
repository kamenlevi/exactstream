// wc_spintest.swift - WHOLE-CHIP-2 debug (Leap, Tue 2026-11-03 17:5x): does a running GPU kernel see a CPU store to a shared buffer? No.
// Build: swiftc -O -import-objc-header wc_bridge.h -o wc_spintest wc_spintest.swift; usage: wc_spintest [wc]   (wc: write-combined buffer)
// does a GPU spin kernel see a CPU store to a shared buffer? (WHOLE-CHIP-2 debug)
import Foundation
import Metal
setvbuf(stdout, nil, _IONBF, 0)
let SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void gspin(device const uint* f2 [[buffer(0)]], device uint* out [[buffer(1)]]) {
    const volatile device uint* f = (const volatile device uint*)f2;
    uint n = 0;
    while (f[0] == 0u && n < 4000000u) { n++; }
    out[0] = n;
}
kernel void gspin_atomic(device atomic_uint* f2 [[buffer(0)]], device uint* out [[buffer(1)]]) {
    uint n = 0;
    while (atomic_load_explicit(f2, memory_order_relaxed) == 0u && n < 4000000u) { n++; }
    out[0] = n;
}
kernel void gspin_rmw(device atomic_uint* f2 [[buffer(0)]], device uint* out [[buffer(1)]]) {
    uint n = 0;
    while (atomic_fetch_add_explicit(f2, 0u, memory_order_relaxed) == 0u && n < 4000000u) { n++; }
    out[0] = n;
}
"""
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: SRC, options: nil)
let wc = CommandLine.arguments.count > 1
let f2 = dev.makeBuffer(length: 64, options: wc ? [.storageModeShared, .cpuCacheModeWriteCombined] : .storageModeShared)!
print(wc ? "flag buffer: shared + write-combined" : "flag buffer: shared")
let out = dev.makeBuffer(length: 64, options: .storageModeShared)!
for name in ["gspin", "gspin_atomic", "gspin_rmw"] {
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    for delayMs in [0.0, 1.0] {
        wc_vstore(f2.contents(), 0)
        let cb = q.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso); e.setBuffer(f2, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1)
        e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        e.endEncoding()
        let t0 = now()
        cb.commit()
        if delayMs > 0 { let tw = now(); while now() - tw < delayMs / 1e3 {} }
        wc_vstore(f2.contents(), 1)
        let tw = now()
        cb.waitUntilCompleted()
        let n = out.contents().load(as: UInt32.self)
        print(String(format: "%@ delay %.0f ms: CB done %.2f ms after commit (%.3f ms after the CPU store), spin iterations %u%@", name, delayMs,
                     1e3 * (now() - t0), 1e3 * (now() - tw), n, n >= 4000000 ? " TIMED OUT" : ""))
    }
}
