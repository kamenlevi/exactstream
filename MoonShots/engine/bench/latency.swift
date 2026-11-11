// Engine latency microbench (MoonShots, 2026-10-21): what the loader ring's command-buffer design costs on the M5 Pro.
// Metal shows GPU writes to the CPU only at command-buffer completion (data/publish.txt), so a layer's routing must close a
// command buffer; the CPU learns the cold experts in the completion handler, reads them, and raises per-slot flags that a
// spinning kernel in an already-committed command buffer waits on. Measured here (all times in us; median [p10, p90]):
//   1. handler latency: GPU end of a command buffer -> its completed handler runs on the CPU (idle GPU / GPU busy on a 2nd queue)
//   2. flag latency: CPU store to a shared buffer -> a spinning GPU kernel's command buffer ends
//   3. cold reads: QD concurrent pread(F_NOCACHE) of whole MSB-3 expert parts (2.16 MB) from records3
//   4. end to end: routing command buffer's GPU end -> handler -> QD cold reads -> flag -> the waiting kernel's GPU end
// build: swiftc -O engine/bench/latency.swift -o scratch/latency   run: scratch/latency   (GPU seconds + SSD reads; quiet lock)
import Foundation
import Metal
import QuartzCore

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let bgQueue = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i] += 1u; }
kernel void spin(device atomic_uint* flag [[buffer(0)]], device uint* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint n = 0;
    while (atomic_load_explicit(flag, memory_order_relaxed) == 0u && n < 400000000u) n++;
    out[0] = n;
}
kernel void stream(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                   uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint k = i; k < (256u << 20) / 16u; k += n) acc += src[k];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xdeadbeefu) out[1] = 1u;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTiny = pso("tiny"), pSpin = pso("spin"), pStream = pso("stream")
let outBuf = dev.makeBuffer(length: 4096, options: .storageModeShared)!
let flagBuf = dev.makeBuffer(length: 64, options: .storageModeShared)!
let big = dev.makeBuffer(length: 256 << 20, options: .storageModePrivate)!
let flag = flagBuf.contents().bindMemory(to: UInt32.self, capacity: 16)

func enc(_ cb: MTLCommandBuffer, _ p: MTLComputePipelineState, _ bufs: [MTLBuffer], _ threads: Int, _ tg: Int) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    e.endEncoding()
}
func stats(_ name: String, _ v: [Double]) {
    let s = v.sorted()
    let q = { (f: Double) -> Double in s[min(s.count - 1, Int(f * Double(s.count)))] }
    print(String(format: "%@: median %.1f us [p10 %.1f, p90 %.1f] (n=%d)", name, q(0.5), q(0.1), q(0.9), s.count))
}

// 1. handler latency
func handlerLatency(busy: Bool, n: Int) -> [Double] {
    var lat = [Double]()
    for _ in 0..<n {
        if busy { let b = bgQueue.makeCommandBuffer()!; enc(b, pStream, [big, outBuf], 20 * 1024, 1024); b.commit() }
        let cb = queue.makeCommandBuffer()!
        enc(cb, pTiny, [outBuf], 256, 256)
        let sem = DispatchSemaphore(value: 0)
        var th = 0.0
        cb.addCompletedHandler { _ in th = CACurrentMediaTime(); sem.signal() }
        cb.commit()
        sem.wait()
        lat.append((th - cb.gpuEndTime) * 1e6)
    }
    return lat
}

// 2. CPU -> GPU flag latency
func flagLatency(n: Int) -> [Double] {
    var lat = [Double]()
    for _ in 0..<n {
        flag[0] = 0
        OSMemoryBarrier()
        let cb = queue.makeCommandBuffer()!
        enc(cb, pSpin, [flagBuf, outBuf], 32, 32)
        cb.commit()
        usleep(3000)                                   // the kernel is spinning by now
        let t0 = CACurrentMediaTime()
        flag[0] = 1
        OSMemoryBarrier()
        cb.waitUntilCompleted()
        lat.append((cb.gpuEndTime - t0) * 1e6)
    }
    return lat
}

// 3. cold reads of whole MSB-3 expert parts
let RECSZ = 2932736, MSBSZ = 2162688
let path = NSString(string: "~/BigMoE/models/bare/records3/L07.bin").expandingTildeInPath
let fd = open(path, O_RDONLY)
_ = fcntl(fd, F_NOCACHE, 1)
let bufs = (0..<16).map { _ in UnsafeMutableRawPointer.allocate(byteCount: MSBSZ, alignment: 16384) }
func readBatch(_ qd: Int) {
    let ids = (0..<qd).map { _ in Int.random(in: 0..<512) }
    DispatchQueue.concurrentPerform(iterations: qd) { j in _ = pread(fd, bufs[j], MSBSZ, off_t(ids[j] * RECSZ)) }
}
func coldRead(qd: Int, n: Int) -> [Double] {
    var lat = [Double]()
    for _ in 0..<n {
        let t0 = CACurrentMediaTime()
        readBatch(qd)
        lat.append((CACurrentMediaTime() - t0) * 1e6)
    }
    return lat
}

// 4. end to end: routing CB end -> handler -> reads -> flag -> waiting kernel's end
func endToEnd(qd: Int, n: Int) -> [Double] {
    var lat = [Double]()
    for _ in 0..<n {
        flag[0] = 0
        OSMemoryBarrier()
        let cbA = queue.makeCommandBuffer()!
        enc(cbA, pTiny, [outBuf], 256, 256)
        cbA.addCompletedHandler { _ in
            if qd > 0 { readBatch(qd) }
            flag[0] = 1
            OSMemoryBarrier()
        }
        let cbB = queue.makeCommandBuffer()!
        enc(cbB, pSpin, [flagBuf, outBuf], 32, 32)
        cbA.commit()
        cbB.commit()
        cbB.waitUntilCompleted()
        lat.append((cbB.gpuEndTime - cbA.gpuEndTime) * 1e6)
    }
    return lat
}

_ = handlerLatency(busy: false, n: 20)                    // warm up
stats("1a handler latency, idle GPU", handlerLatency(busy: false, n: 300))
stats("1b handler latency, GPU busy on a 2nd queue", handlerLatency(busy: true, n: 100))
stats("2  flag latency (CPU store -> spinning kernel's CB end)", flagLatency(n: 100))
for qd in [1, 4, 6, 10] { stats("3  cold read QD=\(qd) x 2.16 MB", coldRead(qd: qd, n: 40)) }
for qd in [0, 1, 6] { stats("4  end to end, routing CB end -> \(qd) cold reads -> flag -> GPU resumes", endToEnd(qd: qd, n: 60)) }
