// CPU->GPU release without events: the GPU spins on FRESH cache lines.
// One command buffer per token, 48 "layers": [work kernel, row kernel (writes seq), spin kernel].
// The spin kernel polls line i, i+1, ... of a per-(layer,pass) ring region (never read before, so
// never in the GPU's caches); the CPU releases by writing the seq into every line of the region.
// Measures: per-layer overhead vs the same chain without row/spin kernels, CPU-observed visibility
// of the row write (kernel end vs buffer end), and spin timeouts (must be 0).
import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void work(device const uint4* w [[buffer(0)]], device uint* o [[buffer(1)]],
                 constant uint& n [[buffer(2)]], device uint* dep [[buffer(3)]], uint t [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
  uint4 a = 0; for (uint i = t; i < n; i += g) a += w[i];
  a.x += dep[0] & 0u;                                   // depends on the previous spin (STATUS)
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0xdeadbeefu) o[0] = a.x;
  if (t == 0) o[1] = a.y;                               // always writes o -> row depends on it
}
kernel void row(device atomic_int* r [[buffer(0)]], constant int& v [[buffer(1)]], device const uint* o [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  if (t == 0) atomic_store_explicit(r, v + (int)(o[1] & 0u), memory_order_relaxed);
}
// polls one int per 128-byte line, advancing a line per poll; wraps within the region
kernel void spin(device const int* region [[buffer(0)]], constant int& v [[buffer(1)]], constant uint& lines [[buffer(2)]],
                 device atomic_uint* status [[buffer(3)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  uint i = 0, polls = 0;
  while (true) {
    int x = ((volatile device const int*)region)[(i % lines) * 32];
    polls++;
    if (x == v) break;
    if (polls > 40000u) { atomic_fetch_add_explicit(status, 1u, memory_order_relaxed); break; }   // timeout
    i++;
  }
  atomic_fetch_max_explicit(status + 1, polls, memory_order_relaxed);
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pWork = pso("work"), pRow = pso("row"), pSpin = pso("spin")
let NL = 48
let layerMB = Int(ProcessInfo.processInfo.environment["LAYER_MB"] ?? "20")!
let LINES = 1024                                   // 128 KB region per (layer, pass)
let RING = 512                                     // regions in the ring (64 MB)
let untracked = ProcessInfo.processInfo.environment["UNTRACKED"] == "1"
let concurrent = ProcessInfo.processInfo.environment["CONCURRENT"] == "1"
let oneEncoder = ProcessInfo.processInfo.environment["ONE_ENCODER"] == "1"
let bopt: MTLResourceOptions = untracked ? [.storageModeShared, .hazardTrackingModeUntracked] : .storageModeShared
let W = dev.makeBuffer(length: layerMB << 20, options: bopt)!
let O = dev.makeBuffer(length: 16, options: bopt)!
let ROWS = dev.makeBuffer(length: NL * 128, options: bopt)!
let REG = dev.makeBuffer(length: RING * LINES * 128, options: bopt)!
let STATUS = dev.makeBuffer(length: 64, options: bopt)!
var nW = UInt32((layerMB << 20) / 16)
memset(REG.contents(), 0, REG.length)
var shared: MTLComputeCommandEncoder? = nil
func enc(_ cb: MTLCommandBuffer, _ p: MTLComputePipelineState, _ setup: (MTLComputeCommandEncoder) -> Void, _ threads: Int) {
  let e: MTLComputeCommandEncoder
  if oneEncoder { if shared == nil { shared = cb.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial)! }; e = shared! }
  else { e = cb.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial)! }
  e.setComputePipelineState(p); setup(e)
  e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(threads, 256), height: 1, depth: 1))
  if oneEncoder { e.memoryBarrier(scope: .buffers) } else { e.endEncoding() }
}
var pass = 0
let delayUS = Int(ProcessInfo.processInfo.environment["DELAY_US"] ?? "0")!
func token(gated: Bool) -> (Double, [Double]) {
  pass += 1
  let rows = ROWS.contents().bindMemory(to: Int32.self, capacity: NL * 32)
  var seenAt = [Double](repeating: 0, count: NL)
  let done = DispatchSemaphore(value: 0)
  let me = pass
  let cb = q.makeCommandBuffer()!
  for L in 0..<NL {
    enc(cb, pWork, { $0.setBuffer(W, offset: 0, index: 0); $0.setBuffer(O, offset: 0, index: 1); $0.setBytes(&nW, length: 4, index: 2)
      $0.setBuffer(STATUS, offset: 0, index: 3) }, 20 * 1024)
    if gated {
      var v = Int32(me)
      enc(cb, pRow, { $0.setBuffer(ROWS, offset: L * 128, index: 0); $0.setBytes(&v, length: 4, index: 1); $0.setBuffer(O, offset: 0, index: 2) }, 1)
      let regOff = ((me * NL + L) % RING) * LINES * 128
      var lines = UInt32(LINES)
      enc(cb, pSpin, { $0.setBuffer(REG, offset: regOff, index: 0); $0.setBytes(&v, length: 4, index: 1)
        $0.setBytes(&lines, length: 4, index: 2); $0.setBuffer(STATUS, offset: 0, index: 3) }, 1)
    }
  }
  var th: Thread? = nil
  if gated {
    th = Thread {
      let t0 = DispatchTime.now().uptimeNanoseconds
      for L in 0..<NL {
        while (rows + L * 32).pointee != Int32(me) { }
        seenAt[L] = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3
        if delayUS > 0 { let td = DispatchTime.now().uptimeNanoseconds; while DispatchTime.now().uptimeNanoseconds - td < UInt64(delayUS * 1000) {} }
        let reg = REG.contents().advanced(by: ((me * NL + L) % RING) * LINES * 128).bindMemory(to: Int32.self, capacity: LINES * 32)
        for i in 0..<LINES { reg[i * 32] = Int32(me) }      // release: every line of the region
      }
      done.signal()
    }
    th!.qualityOfService = .userInteractive; th!.start()
  }
  if oneEncoder { shared?.endEncoding(); shared = nil }
  let t0 = DispatchTime.now().uptimeNanoseconds
  cb.commit(); cb.waitUntilCompleted()
  let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
  if gated { done.wait() }
  return (dt, seenAt)
}
func med(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
_ = token(gated: false); _ = token(gated: true)
let st = STATUS.contents().bindMemory(to: UInt32.self, capacity: 2); st[0] = 0; st[1] = 0
var plain: [Double] = [], gated: [Double] = [], gaps: [Double] = []
for _ in 0..<(Int(ProcessInfo.processInfo.environment["REPS"] ?? "15")!) {
  plain.append(token(gated: false).0)
  let (g, seen) = token(gated: true); gated.append(g)
  for L in 1..<NL { gaps.append(seen[L] - seen[L - 1]) }
}
let mp = med(plain), mg = med(gated)
print(String(format: "[untracked=\(untracked ? 1 : 0) concurrent=\(concurrent ? 1 : 0) oneEncoder=\(oneEncoder ? 1 : 0)] delay %d us, layer %d MB: plain %.2f ms/token, fresh-line spin gate %.2f ms/token -> %+.1f us/layer; CPU sees rows every %.0f us (median); timeouts %d, max polls %d",
             delayUS, layerMB, mp, mg, (mg - mp) / Double(NL) * 1e3, med(gaps), st[0], st[1]))
