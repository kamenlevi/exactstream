// Per-layer GPU<->CPU handshake cost with MTLSharedEvent (no CPU wait on the GPU).
// Each "layer": a bandwidth-bound kernel (~0.3 ms, reads LAYER_MB), then the GPU signals
// event=2L+1 ("router done"), and the next layer's command buffer encodeWait()s 2L+2,
// which a CPU listener sets on notification (optionally after a pread of a 2.77 MB expert).
// Baseline: same kernels in command buffers with no events.  -> stdout (caller saves)
import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void flag(device atomic_uint* f [[buffer(0)]], constant uint& v [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t == 0) atomic_store_explicit(f, v, memory_order_relaxed);
}
kernel void work(device const uint4* w [[buffer(0)]], device uint* o [[buffer(1)]],
                 constant uint& n [[buffer(2)]], uint t [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
  uint4 a = 0; for (uint i = t; i < n; i += g) a += w[i];
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0xdeadbeefu) o[0] = a.x;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "work")!)
let fpso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "flag")!)
let FLAG = dev.makeBuffer(length: 16384, options: .storageModeShared)!
let NL = 48
let layerMB = Int(ProcessInfo.processInfo.environment["LAYER_MB"] ?? "60")!
let doIO = ProcessInfo.processInfo.environment["IO"] == "1"
let bytes = layerMB << 20
let W = dev.makeBuffer(length: bytes, options: .storageModeShared)!
let O = dev.makeBuffer(length: 16, options: .storageModeShared)!
var n = UInt32(bytes / 16)
let slot = dev.makeBuffer(length: 2_764_800, options: .storageModeShared)!
let files = (try? FileManager.default.contentsOfDirectory(atPath: NSHomeDirectory() + "/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MLX-8bit"))?.filter { $0.hasSuffix(".safetensors") } ?? []
let fds: [Int32] = files.map { let fd = open(NSHomeDirectory() + "/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MLX-8bit/" + $0, O_RDONLY); _ = fcntl(fd, F_NOCACHE, 1); return fd }
func encodeWork(_ cb: MTLCommandBuffer) {
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pso); e.setBuffer(W, offset: 0, index: 0); e.setBuffer(O, offset: 0, index: 1)
  e.setBytes(&n, length: 4, index: 2)
  e.dispatchThreads(MTLSize(width: 20 * 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  e.endEncoding()
}
func baseline(splitPerLayer: Bool) -> Double {
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  if splitPerLayer {
    for _ in 0..<NL { let cb = q.makeCommandBuffer()!; encodeWork(cb); cb.commit(); last = cb }
  } else { let cb = q.makeCommandBuffer()!; for _ in 0..<NL { encodeWork(cb) }; cb.commit(); last = cb }
  last!.waitUntilCompleted()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
let listenerQ = DispatchQueue(label: "l", qos: .userInteractive)
let listener = MTLSharedEventListener(dispatchQueue: listenerQ)
var off: Int64 = 0
func handshake(missEvery: Int) -> Double {
  let ev = dev.makeSharedEvent()!
  ev.signaledValue = 0
  for L in 0..<NL {
    ev.notify(listener, atValue: UInt64(2 * L + 1)) { e, v in
      if missEvery > 0 && L % missEvery == 0 && !fds.isEmpty {
        off = (off + 7_340_032) % 4_000_000_000
        _ = pread(fds[L % fds.count], slot.contents(), 2_764_800, off)
      }
      e.signaledValue = v + 1
    }
  }
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  for L in 0..<NL {
    let cb = q.makeCommandBuffer()!
    if L > 0 { cb.encodeWaitForEvent(ev, value: UInt64(2 * L)) }
    encodeWork(cb)
    cb.encodeSignalEvent(ev, value: UInt64(2 * L + 1))
    cb.commit(); last = cb
  }
  last!.waitUntilCompleted()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
// Variant: a dedicated CPU thread busy-polls signaledValue (no listener/dispatch wake-up).
func handshakePoll(missEvery: Int) -> Double {
  let ev = dev.makeSharedEvent()!
  ev.signaledValue = 0
  let done = DispatchSemaphore(value: 0)
  let th = Thread {
    var L = 0
    while L < NL {
      let target = UInt64(2 * L + 1)
      while ev.signaledValue < target { }
      if missEvery > 0 && L % missEvery == 0 && !fds.isEmpty {
        off = (off + 7_340_032) % 4_000_000_000
        _ = pread(fds[L % fds.count], slot.contents(), 2_764_800, off)
      }
      ev.signaledValue = target + 1
      L += 1
    }
    done.signal()
  }
  th.qualityOfService = .userInteractive
  th.start()
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  for L in 0..<NL {
    let cb = q.makeCommandBuffer()!
    if L > 0 { cb.encodeWaitForEvent(ev, value: UInt64(2 * L)) }
    encodeWork(cb)
    cb.encodeSignalEvent(ev, value: UInt64(2 * L + 1))
    cb.commit(); last = cb
  }
  last!.waitUntilCompleted()
  done.wait()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
// Variant: GPU-only chain (signal then wait on a value the GPU itself signaled): event cost without the CPU.
func gpuOnlyEvents() -> Double {
  let ev = dev.makeSharedEvent()!
  ev.signaledValue = 0
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  for L in 0..<NL {
    let cb = q.makeCommandBuffer()!
    if L > 0 { cb.encodeWaitForEvent(ev, value: UInt64(L)) }
    encodeWork(cb)
    cb.encodeSignalEvent(ev, value: UInt64(L + 1))
    cb.commit(); last = cb
  }
  last!.waitUntilCompleted()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
// Waits whose value is already reached before commit (CPU pre-signals everything): is a satisfied wait free?
func presignaled(shared: Bool) -> Double {
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  if shared {
    let ev = dev.makeSharedEvent()!; ev.signaledValue = 1_000_000
    for L in 0..<NL { let cb = q.makeCommandBuffer()!; cb.encodeWaitForEvent(ev, value: UInt64(L + 1)); encodeWork(cb); cb.commit(); last = cb }
  } else {
    let ev = dev.makeEvent()!
    for L in 0..<NL { let cb = q.makeCommandBuffer()!; if L > 0 { cb.encodeWaitForEvent(ev, value: UInt64(L)) }; encodeWork(cb); cb.encodeSignalEvent(ev, value: UInt64(L + 1)); cb.commit(); last = cb }
  }
  last!.waitUntilCompleted()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
// GPU->CPU via a plain shared-memory store (tiny kernel at the end of layer L's buffer),
// CPU busy-reads it, then signals the shared event that layer L+1's buffer waits on.
let unretained = ProcessInfo.processInfo.environment["UNRETAINED"] == "1"
let smallWork = ProcessInfo.processInfo.environment["SMALL"] == "1"   // 20 small kernels per layer instead of 1 big one
func mkcb() -> MTLCommandBuffer { unretained ? q.makeCommandBufferWithUnretainedReferences()! : q.makeCommandBuffer()! }
func encodeLayer(_ cb: MTLCommandBuffer) {
  if smallWork { for _ in 0..<20 { encodeWorkN(cb, n / 20) } } else { encodeWork(cb) }
}
func encodeWorkN(_ cb: MTLCommandBuffer, _ nn: UInt32) {
  var m = nn
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pso); e.setBuffer(W, offset: 0, index: 0); e.setBuffer(O, offset: 0, index: 1)
  e.setBytes(&m, length: 4, index: 2)
  e.dispatchThreads(MTLSize(width: 2 * 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  e.endEncoding()
}
func plainLayers() -> Double {
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  for _ in 0..<NL { let cb = mkcb(); encodeLayer(cb); cb.commit(); last = cb }
  last!.waitUntilCompleted()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
func memflagHandshake(missEvery: Int) -> Double {
  let ev = dev.makeSharedEvent()!; ev.signaledValue = 0
  let fp = FLAG.contents().bindMemory(to: UInt32.self, capacity: 4)
  fp[0] = 0
  let done = DispatchSemaphore(value: 0)
  let th = Thread {
    for L in 0..<(NL - 1) {
      let want = UInt32(L + 1)
      while (fp[0] < want) { }                       // volatile-ish read of shared memory
      if missEvery > 0 && L % missEvery == 0 && !fds.isEmpty {
        off = (off + 7_340_032) % 4_000_000_000
        _ = pread(fds[L % fds.count], slot.contents(), 2_764_800, off)
      }
      ev.signaledValue = UInt64(L + 1)
    }
    done.signal()
  }
  th.qualityOfService = .userInteractive; th.start()
  let t0 = DispatchTime.now().uptimeNanoseconds
  var last: MTLCommandBuffer? = nil
  for L in 0..<NL {
    let cb = mkcb()
    if L > 0 { cb.encodeWaitForEvent(ev, value: UInt64(L)) }
    encodeLayer(cb)
    var v = UInt32(L + 1)
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(fpso); e.setBuffer(FLAG, offset: 0, index: 0); e.setBytes(&v, length: 4, index: 1)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding()
    cb.commit(); last = cb
  }
  last!.waitUntilCompleted(); done.wait()
  return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
func med(_ f: () -> Double) -> Double { var v = (0..<9).map { _ in f() }; v.sort(); return v[4] }
_ = baseline(splitPerLayer: false); _ = handshake(missEvery: 0)
let b1 = med { baseline(splitPerLayer: false) }, b2 = med { baseline(splitPerLayer: true) }
let h0 = med { handshake(missEvery: 0) }
print(String(format: "layer: %d MB read, %d layers", layerMB, NL))
print(String(format: "one command buffer, no events      %7.2f ms/token", b1))
print(String(format: "one command buffer per layer       %7.2f ms/token  (+%.1f us/layer)", b2, (b2 - b1) / Double(NL) * 1e3))
print(String(format: "per-layer GPU->CPU->GPU handshake  %7.2f ms/token  (+%.1f us/layer vs one buffer)", h0, (h0 - b1) / Double(NL) * 1e3))
let g0 = med { gpuOnlyEvents() }
print(String(format: "GPU-only signal->wait chain         %7.2f ms/token  (+%.1f us/layer vs one buffer)", g0, (g0 - b1) / Double(NL) * 1e3))
let s0 = med { presignaled(shared: true) }
print(String(format: "shared-event waits, pre-signaled    %7.2f ms/token  (+%.1f us/layer vs one buffer)", s0, (s0 - b1) / Double(NL) * 1e3))
let e0 = med { presignaled(shared: false) }
print(String(format: "MTLEvent (GPU-only) signal->wait    %7.2f ms/token  (+%.1f us/layer vs one buffer)", e0, (e0 - b1) / Double(NL) * 1e3))
let pl = med { plainLayers() }
print(String(format: "[unretained=%d small=%d] plain per-layer buffers %7.2f ms/token", unretained ? 1 : 0, smallWork ? 1 : 0, pl))
let m0 = med { memflagHandshake(missEvery: 0) }
print(String(format: "mem-flag GPU->CPU, CPU-signaled wait %7.2f ms/token  (+%.1f us/layer vs plain per-layer buffers)", m0, (m0 - pl) / Double(NL) * 1e3))
if doIO { for m in [12, 4] { let h = med { memflagHandshake(missEvery: m) }; let k = Double((NL + m - 1) / m)
  print(String(format: "  mem-flag + cold 2.77MB pread on %2.0f layers %7.2f ms/token (+%.0f us per stalled layer)", k, h, (h - m0) / k * 1e3)) } }
let p0 = med { handshakePoll(missEvery: 0) }
print(String(format: "handshake, CPU busy-polls value     %7.2f ms/token  (+%.1f us/layer vs one buffer)", p0, (p0 - b1) / Double(NL) * 1e3))
if doIO {
  for m in [12, 4] {
    let h = med { handshake(missEvery: m) }
    let k = Double((NL + m - 1) / m)
    print(String(format: "handshake + cold 2.77MB pread on %2.0f layers %7.2f ms/token  (+%.0f us per stalled layer)", k, h, (h - h0) / k * 1e3))
  }
}
