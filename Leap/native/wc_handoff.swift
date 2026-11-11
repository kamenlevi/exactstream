// wc_handoff.swift - WHOLE-CHIP-2 H (Leap, Tue 2026-11-03 17:4x; TABLE_350 'WHOLE-CHIP-2'): the per-layer GPU -> Neural
// Engine -> GPU round trip. One command buffer per check of 48 layers. Per layer a kernel writes x (16 x 2560 fp16), the
// buffer signals an MTLSharedEvent, the GPU runs its own share (a bandwidth-bound read standing in for NAX6-N's part of
// the layer), waits for the event's next value, and a kernel combines the Neural Engine's output. The main thread spins
// on the event, runs the Core ML predict (x / scores / output as MLMultiArrays over the Metal buffers' memory, output
// backings) and signals back.
// Two ways to tell the CPU that x is ready: E1 a GPU-signalled MTLSharedEvent in the command buffer (BigMoE measured
// ~140 us before a dependent wait is released); E2 BigMoE's protocol: a one-thread post kernel stores a sequence number
// into shared memory (visible to a spinning CPU at kernel end) and the GPU waits on a CPU-signalled event value (free).
// Arms, interleaved over rounds: A GPU only (the full layer's read, no sync); E1 / E2 the round trip alone (no predict,
// no share); C<K> E2 + predict, no share; B<K> the split: E2 + predict + the share sized to the GPU's part; D<K> the
// predict alone, back to back. Inside C the predict calls are timed too, so overhead = C - that.
// No GPU spin arm: a running kernel never sees a CPU store to shared memory on this M5 Pro (native/wc_spintest.swift,
// Tue 17:5x: volatile, relaxed atomic and atomic RMW polls on a shared or write-combined buffer all time out; a store
// made before the kernel starts is seen at once). CPU -> GPU therefore has to be an event the CPU signals.
// Build: swiftc -O -import-objc-header wc_bridge.h -o wc_handoff wc_handoff.swift
// Usage: wc_handoff MODELS (comma-separated a4 .mlpackage paths, e.g. K 4,8,12,16) [checks 20] [layer_ms 0.70]
//        [window_experts 62] [used 0.71] [share_gbps 200]
// The share read is throttled (fewer threads in flight) to ~share_gbps, NAX6-N's rate at S 16 (190-210 GB/s), so the
// Neural Engine meets the bandwidth pressure of the real kernel rather than a full-speed stream.
import CoreML
import Foundation
import Metal

setvbuf(stdout, nil, _IOLBF, 0)
let a = CommandLine.arguments
let checks = a.count > 2 ? Int(a[2])! : 20
let layerMs = a.count > 3 ? Double(a[3])! : 0.70
let winExp = a.count > 4 ? Double(a[4])! : 62
let used = a.count > 5 ? Double(a[5])! : 0.71
let shareGBps = a.count > 6 ? Double(a[6])! : 200
var shareThreads = 65536
let NL = 48, S = 16, H = 2560

let SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void write_x(device half* x [[buffer(0)]], constant uint& layer [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    x[i] = half(float((i * 2654435761u + layer * 97u) & 1023u) / 1024.0f - 0.5f);
}
kernel void share(device const uint4* big [[buffer(0)]], device uint* sink [[buffer(1)]], constant uint& off [[buffer(2)]],
                  constant uint& n4 [[buffer(3)]], uint i [[thread_position_in_grid]], uint nt [[threads_per_grid]],
                  uint tg [[threadgroup_position_in_grid]]) {
    uint4 acc = 0;
    for (uint j = i; j < n4; j += nt) acc ^= big[off + j];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) sink[tg] = i;     // keeps the loads
}
kernel void post(device atomic_uint* flag [[buffer(0)]], constant uint& seq [[buffer(1)]]) {
    atomic_store_explicit(flag, seq, memory_order_relaxed);
}
kernel void combine(device const half* y [[buffer(0)]], device float* z [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    z[i] += float(y[i]);
}
"""

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: SRC, options: nil)
func pipe(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pWrite = pipe("write_x"), pShare = pipe("share"), pComb = pipe("combine"), pPost = pipe("post")
let flag = dev.makeBuffer(length: 64, options: .storageModeShared)!
let xbuf = dev.makeBuffer(length: S * H * 2, options: .storageModeShared)!
let ybuf = dev.makeBuffer(length: S * H * 2, options: .storageModeShared)!
let zbuf = dev.makeBuffer(length: S * H * 4, options: .storageModeShared)!
let sink = dev.makeBuffer(length: 1 << 20, options: .storageModeShared)!
let BIG = 1536 << 20
let big = dev.makeBuffer(length: BIG, options: .storageModePrivate)!
do {                                                               // touch every page of the read buffer once
    let cb = queue.makeCommandBuffer()!
    let b = cb.makeBlitCommandEncoder()!
    b.fill(buffer: big, range: 0..<BIG, value: 0x5A)
    b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
let ev = dev.makeSharedEvent()!

func encWrite(_ cb: MTLCommandBuffer, _ L: Int) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pWrite)
    e.setBuffer(xbuf, offset: 0, index: 0)
    var l = UInt32(L)
    e.setBytes(&l, length: 4, index: 1)
    e.dispatchThreads(MTLSize(width: S * H, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
}
func encShare(_ cb: MTLCommandBuffer, bytes: Int, layer: Int) {
    if bytes < 16 { return }
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pShare)
    e.setBuffer(big, offset: 0, index: 0)
    e.setBuffer(sink, offset: 0, index: 1)
    var off = UInt32(((layer * bytes) % (BIG - bytes)) / 16), n4 = UInt32(bytes / 16)
    e.setBytes(&off, length: 4, index: 2)
    e.setBytes(&n4, length: 4, index: 3)
    e.dispatchThreads(MTLSize(width: shareThreads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
}
func encPost(_ cb: MTLCommandBuffer, _ seq: UInt32) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pPost)
    e.setBuffer(flag, offset: 0, index: 0)
    var v = seq
    e.setBytes(&v, length: 4, index: 1)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding()
}
func encComb(_ cb: MTLCommandBuffer) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pComb)
    e.setBuffer(ybuf, offset: 0, index: 0)
    e.setBuffer(zbuf, offset: 0, index: 1)
    e.dispatchThreads(MTLSize(width: S * H, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
}

// ---- Core ML: inputs / outputs over the Metal buffers' memory
struct Ane {
    let model: MLModel
    let k: Int
    let input: MLDictionaryFeatureProvider
    let opts: MLPredictionOptions
    func predict() { _ = try! model.prediction(from: input, options: opts) }
}
func loadAne(_ path: String) -> Ane {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    let url = URL(fileURLWithPath: path)
    let compiled = url.pathExtension == "mlmodelc" ? url : try! MLModel.compileModel(at: url)
    let m = try! MLModel(contentsOf: compiled, configuration: cfg)
    let k = m.modelDescription.inputDescriptionsByName["sc"]!.multiArrayConstraint!.shape.last!.intValue
    let xa = try! MLMultiArray(dataPointer: xbuf.contents(), shape: [1, NSNumber(value: S), NSNumber(value: H)], dataType: .float16,
                               strides: [NSNumber(value: S * H), NSNumber(value: H), 1], deallocator: nil)
    let scp = UnsafeMutableRawPointer.allocate(byteCount: S * k * 2, alignment: 64)
    let sp = scp.assumingMemoryBound(to: UInt16.self)
    for i in 0..<(S * k) { sp[i] = Float16(1.0 / Float(k)).bitPattern }
    let sca = try! MLMultiArray(dataPointer: scp, shape: [1, NSNumber(value: S), NSNumber(value: k)], dataType: .float16,
                                strides: [NSNumber(value: S * k), NSNumber(value: k), 1], deallocator: nil)
    let ya = try! MLMultiArray(dataPointer: ybuf.contents(), shape: [1, NSNumber(value: S), NSNumber(value: H)], dataType: .float16,
                               strides: [NSNumber(value: S * H), NSNumber(value: H), 1], deallocator: nil)
    let opts = MLPredictionOptions()
    opts.outputBackings = ["out": ya]
    let input = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: xa), "sc": MLFeatureValue(multiArray: sca)])
    return Ane(model: m, k: k, input: input, opts: opts)
}
let anes = a[1].split(separator: ",").map { loadAne(String($0)) }
for m in anes { for _ in 0..<10 { m.predict() } }

// ---- one check (48 layers): wall time from commit to completion, the GPU's own time, the predict time inside
// mode 0: no sync; 1: GPU-signalled event (E1); 2: post kernel + CPU spin on shared memory + CPU-signalled wait (E2)
var evBase: UInt64 = 0
var seqBase: UInt32 = 0
func runCheck(ane: Ane?, shareBytes: Int, mode: Int) -> (Double, Double, Double) {
    let cb = queue.makeCommandBuffer()!
    let base = evBase, sbase = seqBase
    for L in 0..<NL {
        encWrite(cb, L)
        if mode == 1 { cb.encodeSignalEvent(ev, value: base + UInt64(2 * L + 1)) }
        if mode == 2 { encPost(cb, sbase + UInt32(L + 1)) }
        encShare(cb, bytes: shareBytes, layer: L)
        if mode != 0 { cb.encodeWaitForEvent(ev, value: base + UInt64(2 * L + 2)) }
        encComb(cb)
    }
    evBase += UInt64(2 * NL + 2)
    seqBase += UInt32(NL + 1)
    var tp = 0.0
    let t0 = now()
    cb.commit()
    if mode != 0 {
        let fp = flag.contents()
        for L in 0..<NL {
            let tw = now()
            if mode == 1 { let v1 = base + UInt64(2 * L + 1); while ev.signaledValue < v1 { if now() - tw > 2 { fatalError("E1 wait > 2 s") } } }
            else { let s1 = sbase + UInt32(L + 1); while wc_vload(fp) < s1 { if now() - tw > 2 { fatalError("E2 flag wait > 2 s") } } }
            if let m = ane { let t1 = now(); m.predict(); tp += now() - t1 }
            ev.signaledValue = base + UInt64(2 * L + 2)
        }
    }
    cb.waitUntilCompleted()
    return (now() - t0, cb.gpuEndTime - cb.gpuStartTime, tp)
}
func runAlone(_ m: Ane) -> Double {
    let t0 = now()
    for _ in 0..<NL { m.predict() }
    return now() - t0
}

// ---- calibrate the share read: bytes for layer_ms of GPU time
func steadyRate() -> Double {                                      // steady state: 48 reads per buffer, as in a check
    let probe = 128 << 20
    var ts: [Double] = []
    for i in 0..<6 {
        let cb = queue.makeCommandBuffer()!
        for L in 0..<NL { encShare(cb, bytes: probe, layer: L + i) }
        cb.commit(); cb.waitUntilCompleted()
        if i >= 2 { ts.append((cb.gpuEndTime - cb.gpuStartTime) / Double(NL)) }
    }
    ts.sort()
    return Double(probe) / ts[ts.count / 2]
}
var rate = 0.0
var best = (1e18, 65536, 0.0)
for th in [65536, 32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024] {   // the thread count whose rate is closest to the target
    shareThreads = th
    let r = steadyRate()
    if abs(r / 1e9 - shareGBps) < best.0 { best = (abs(r / 1e9 - shareGBps), th, r) }
}
shareThreads = best.1
rate = best.2
let fullBytes = Int(rate * layerMs / 1e3) / 16 * 16
func partBytes(_ k: Int) -> Int { Int(Double(fullBytes) * (winExp - used * Double(k)) / winExp) / 16 * 16 }
print(String(format: "share read %.0f GB/s (steady, %d threads; target %.0f); full layer %.1f MB (%.3f ms); models K %@", rate / 1e9,
             shareThreads, shareGBps, Double(fullBytes) / 1e6, layerMs, anes.map { String($0.k) }.joined(separator: " / ") as NSString))

// ---- arms, interleaved
typealias Arm = (name: String, run: () -> (Double, Double, Double))
var arms: [Arm] = [
    ("A gpu-only", { runCheck(ane: nil, shareBytes: fullBytes, mode: 0) }),
    ("E1 gpu-event", { runCheck(ane: nil, shareBytes: 0, mode: 1) }),
    ("E2 flag", { runCheck(ane: nil, shareBytes: 0, mode: 2) }),
]
for m in anes {
    arms.append(("C\(m.k) handoff", { runCheck(ane: m, shareBytes: 0, mode: 2) }))
    arms.append(("B\(m.k) split", { runCheck(ane: m, shareBytes: partBytes(m.k), mode: 2) }))
    arms.append(("D\(m.k) alone", { (runAlone(m), 0, 0) }))
}
for arm in arms { for _ in 0..<3 { _ = arm.run() } }                // warm-up
var wall = [String: [Double]](), gpu = [String: [Double]](), pin = [String: [Double]]()
for _ in 0..<3 {
    for arm in arms {
        for _ in 0..<checks {
            let (w, g, p) = arm.run()
            wall[arm.name, default: []].append(w / Double(NL))
            gpu[arm.name, default: []].append(g / Double(NL))
            pin[arm.name, default: []].append(p / Double(NL))
        }
    }
}
func med(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
func p90(_ v: [Double]) -> Double { let s = v.sorted(); return s[Int(Double(s.count) * 0.9)] }
var m = [String: Double]()
for arm in arms {
    m[arm.name] = med(wall[arm.name]!)
    print(String(format: "  %-12@ per layer: wall %7.1f us (p90 %7.1f), GPU %7.1f us, predict inside %6.1f us", arm.name as NSString,
                 1e6 * med(wall[arm.name]!), 1e6 * p90(wall[arm.name]!), 1e6 * med(gpu[arm.name]!), 1e6 * med(pin[arm.name]!)))
}
for k in anes.map({ $0.k }) {
    let c = m["C\(k) handoff"]!, d = m["D\(k) alone"]!, b = m["B\(k) split"]!, aa = m["A gpu-only"]!, pc = med(pin["C\(k) handoff"]!)
    print(String(format: "K %d: overhead per layer C - predict-inside %.1f us (C - D %.1f us) = %.2f ms per check; split B / A = %.3f (%.1f vs %.1f us)",
                 k, 1e6 * (c - pc), 1e6 * (c - d), 1e3 * 48 * (c - pc), b / aa, 1e6 * b, 1e6 * aa))
}
print(String(format: "round trip alone: E1 GPU-signalled event %.1f us, E2 post + CPU-signalled wait %.1f us per layer",
             1e6 * m["E1 gpu-event"]!, 1e6 * m["E2 flag"]!))
