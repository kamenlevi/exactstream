// s7b_probe.swift - S7B-PROBE A and B (Leap, Tue 2026-11-03 22:3x; TABLE_350 'S7B-PROBE'): can the expert kernel start
// on each expert as it lands?
// A: streaming bandwidth of plain vs volatile coherent(system) vs coherent(device) loads over a 256 MB shared buffer.
// B: a 2 MB slot the GPU already read (P0) is rewritten by the CPU (P1) and released through a coherent flag; the GPU
//    re-reads it: B1 plain loads in the same kernel, B2 plain loads in the next dispatch, B3 = B2 + a buffer memory
//    barrier, B4 coherent loads in the same kernel, B0 a control that reads only after the CPU wrote P1, B5 the slot read
//    by an EARLIER command buffer, B6 = B2 with the re-read through a plain pointer cast to coherent(system). Counts P0.
// Build: swiftc -O -import-objc-header w1_bridge.h -o s7b_probe s7b_probe.swift
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
var tbi = mach_timebase_info_data_t()
mach_timebase_info(&tbi)
func us(_ t: UInt64) -> Double { Double(t) * Double(tbi.numer) / Double(tbi.denom) / 1000 }

let SYS = """
#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
using namespace metal;
"""
let SRC_A = SYS + """
kernel void rd_plain(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                     uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = g; i < n4; i += T) acc ^= src[i];
  out[g & 0xFFFFu] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
kernel void rd_coh(volatile coherent(system) device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                   constant uint& n4 [[buffer(2)]], uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = g; i < n4; i += T) acc ^= src[i];
  out[g & 0xFFFFu] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"""
let SRC_A_DEV = SYS + """
kernel void rd_cdev(volatile coherent(device) device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                    constant uint& n4 [[buffer(2)]], uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = g; i < n4; i += T) acc ^= src[i];
  out[g & 0xFFFFu] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"""
// B: 64 threadgroups x 256 threads cover the 2 MB slot, 32 words per thread. done[tg] / flag are system-scope words.
let SRC_B = SYS + """
constant uint NW = 524288;                       // 2 MB of uint
inline void tg_wait(volatile coherent(system) device uint* flag, uint seq, uint lid) {
  if (lid == 0) {
    for (uint it = 0; it < 50000000u; it++) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
      if (flag[0] >= seq) break;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
}
// phase 1 (plain reads of P0, kept live), done[tg] = seq, wait for the flag, phase 2 (plain or coherent re-read)
kernel void b_same(device const uint* x [[buffer(0)]], volatile coherent(system) device const uint* xc [[buffer(1)]],
                   volatile coherent(system) device uint* flag [[buffer(2)]], volatile coherent(system) device uint* done [[buffer(3)]],
                   device atomic_uint* stale [[buffer(4)]], constant uint4& a [[buffer(5)]],   // a = (seq, coherent phase 2, P1 base)
                   uint g [[thread_position_in_grid]], uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                   uint T [[threads_per_grid]]) {
  uint acc = 0;
  for (uint i = g; i < NW; i += T) acc ^= x[i];
  threadgroup_barrier(mem_flags::mem_device);
  if (lid == 0) { done[tg] = a.x; atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system); }
  tg_wait(flag, a.x, lid);
  uint n = 0;
  for (uint i = g; i < NW; i += T) { const uint w = a.y ? xc[i] : x[i]; n += (w != (a.z | i)) ? 1u : 0u; }
  if (acc == 0x12345678u) n += 1000000u;          // keeps phase 1's loads live
  atomic_fetch_add_explicit(stale, n, memory_order_relaxed);
}
kernel void b_read(device const uint* x [[buffer(0)]], volatile coherent(system) device uint* done [[buffer(3)]],
                   device uint* sink [[buffer(6)]], constant uint4& a [[buffer(5)]],
                   uint g [[thread_position_in_grid]], uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                   uint T [[threads_per_grid]]) {
  uint acc = 0;
  for (uint i = g; i < NW; i += T) acc ^= x[i];
  sink[g] = acc;
  threadgroup_barrier(mem_flags::mem_device);
  if (lid == 0) { done[tg] = a.x; atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system); }
}
kernel void b_spin(volatile coherent(system) device uint* flag [[buffer(2)]], volatile coherent(system) device uint* done [[buffer(3)]],
                   constant uint4& a [[buffer(5)]], uint lid [[thread_index_in_threadgroup]]) {
  if (lid == 0 && a.w) { done[64] = a.x; atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system); }
  if (lid == 0)
    for (uint it = 0; it < 50000000u; it++) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
      if (flag[0] >= a.x) break;
    }
}
kernel void b_count(device const uint* x [[buffer(0)]], device atomic_uint* stale [[buffer(4)]], constant uint4& a [[buffer(5)]],
                    uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  uint n = 0;
  for (uint i = g; i < NW; i += T) n += (x[i] != (a.z | i)) ? 1u : 0u;
  atomic_fetch_add_explicit(stale, n, memory_order_relaxed);
}
// B6: B2's re-read through a PLAIN parameter cast to coherent(system) inside (what an MLX metal_kernel can do)
kernel void b_count_cast(device const uint* x [[buffer(0)]], device atomic_uint* stale [[buffer(4)]], constant uint4& a [[buffer(5)]],
                         uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  volatile coherent(system) const device uint* xc = (volatile coherent(system) const device uint*)x;
  uint n = 0;
  for (uint i = g; i < NW; i += T) n += (xc[i] != (a.z | i)) ? 1u : 0u;
  atomic_fetch_add_explicit(stale, n, memory_order_relaxed);
}
"""
// B7: as B6 with a NON-volatile coherent(system) cast (COH's "cn" form)
let SRC_B7 = SRC_B + """
kernel void b_count_cn(device const uint* x [[buffer(0)]], device atomic_uint* stale [[buffer(4)]], constant uint4& a [[buffer(5)]],
                       uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  coherent(system) const device uint* xc = (coherent(system) const device uint*)x;
  uint n = 0;
  for (uint i = g; i < NW; i += T) n += (xc[i] != (a.z | i)) ? 1u : 0u;
  atomic_fetch_add_explicit(stale, n, memory_order_relaxed);
}
"""
func pso(_ src: String, _ fn: String) -> MTLComputePipelineState? {
  do { return try dev.makeComputePipelineState(function: try dev.makeLibrary(source: src, options: nil).makeFunction(name: fn)!) }
  catch { print("  (\(fn) does not build: \(String(describing: error).prefix(160)))"); return nil }
}

// ------------------------------------------------------------------------------------------------ A
let NA = 256 << 20
let bigA = dev.makeBuffer(length: NA, options: .storageModeShared)!
memset(bigA.contents(), 0x5A, NA)
let outA = dev.makeBuffer(length: 65536 * 4, options: .storageModeShared)!
print(String(format: "S7B-PROBE %@ %@", dev.name, Date().description))
var gbs: [String: Double] = [:]
for (name, p) in [("plain", pso(SRC_A, "rd_plain")), ("coherent(system)", pso(SRC_A, "rd_coh")), ("coherent(device)", pso(SRC_A_DEV, "rd_cdev"))] {
  guard let p = p else { continue }
  var ts: [Double] = []
  for r in 0..<12 {
    let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p); enc.setBuffer(bigA, offset: 0, index: 0); enc.setBuffer(outA, offset: 0, index: 1)
    var n4 = UInt32(NA / 16); enc.setBytes(&n4, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: 1 << 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if r >= 2 { ts.append(cb.gpuEndTime - cb.gpuStartTime) }
  }
  let med = ts.sorted()[ts.count / 2]
  gbs[name] = Double(NA) / med / 1e9
  print(String(format: "A %-17@: %.1f GB/s (median of 10, 256 MB)", name, gbs[name]!))
}
if let p = gbs["plain"], let c = gbs["coherent(system)"] {
  print(String(format: "A: coherent(system) / plain = %.2f -> %@", c / p, c / p >= 0.8 ? "the per-expert path can read weights coherently" : c / p < 0.5 ? "it cannot" : "between"))
}

// ------------------------------------------------------------------------------------------------ B
let NW = 524288, TG = 64, TPT = 256
let x = dev.makeBuffer(length: NW * 4, options: .storageModeShared)!
let flag = dev.makeBuffer(length: 256, options: .storageModeShared)!
let done = dev.makeBuffer(length: 4 * 128, options: .storageModeShared)!
let stale = dev.makeBuffer(length: 256, options: .storageModeShared)!
let sink = dev.makeBuffer(length: TG * TPT * 4, options: .storageModeShared)!
let xp = x.contents().bindMemory(to: UInt32.self, capacity: NW)
let donep = done.contents().bindMemory(to: UInt32.self, capacity: 128)
let stp = stale.contents().bindMemory(to: UInt32.self, capacity: 64)
guard let pSame = pso(SRC_B, "b_same"), let pRead = pso(SRC_B, "b_read"), let pSpin = pso(SRC_B, "b_spin"), let pCount = pso(SRC_B, "b_count"),
      let pCast = pso(SRC_B, "b_count_cast"), let pCn = pso(SRC_B7, "b_count_cn") else { exit(2) }
var seq: UInt32 = 0
func grid(_ enc: MTLComputeCommandEncoder) {
  enc.dispatchThreadgroups(MTLSize(width: TG, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: TPT, height: 1, depth: 1))
}
// one trial: the CPU waits for every threadgroup's done word, rewrites the slot to P1, releases the flag
func trial(_ mode: String) -> (UInt32, Double) {
  seq += 1
  let s = seq
  let tagv = (s & 0xFF) << 20                                // per-trial patterns: no aliasing with an earlier trial's lines
  for i in 0..<NW { xp[i] = 0xA0000000 | tagv | UInt32(i) }   // P0
  stp[0] = 0
  if mode == "B5" {                                           // B5: an EARLIER command buffer reads the slot and completes
    let c1 = queue.makeCommandBuffer()!, e1 = c1.makeComputeCommandEncoder()!
    var a1 = SIMD4<UInt32>(s, 0, 0xB0000000 | tagv, 0)
    e1.setComputePipelineState(pRead); e1.setBuffer(x, offset: 0, index: 0); e1.setBuffer(done, offset: 0, index: 3)
    e1.setBytes(&a1, length: 16, index: 5); e1.setBuffer(sink, offset: 0, index: 6); grid(e1)
    e1.endEncoding(); c1.commit(); c1.waitUntilCompleted()
  }
  let cb = queue.makeCommandBuffer()!
  var a = SIMD4<UInt32>(s, mode == "B4" ? 1 : 0, 0xB0000000 | tagv, mode == "B5" ? 1 : 0)
  let enc = cb.makeComputeCommandEncoder()!
  func bind(_ e: MTLComputeCommandEncoder) {
    e.setBuffer(x, offset: 0, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(flag, offset: 0, index: 2)
    e.setBuffer(done, offset: 0, index: 3); e.setBuffer(stale, offset: 0, index: 4); e.setBytes(&a, length: 16, index: 5)
    e.setBuffer(sink, offset: 0, index: 6)
  }
  switch mode {
  case "B1", "B4":
    enc.setComputePipelineState(pSame); bind(enc); grid(enc)
  case "B0", "B5":                                             // B0: the slot is first read after the release; B5: as B0,
                                                               // but an earlier command buffer read it (and the spin says "started")
    enc.setComputePipelineState(pSpin); bind(enc); enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.setComputePipelineState(pCount); bind(enc); grid(enc)
  default:                                                     // B2 / B3: read, spin, then count in the next dispatch
    enc.setComputePipelineState(pRead); bind(enc); grid(enc)
    enc.setComputePipelineState(pSpin); bind(enc); enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    if mode == "B3" { enc.memoryBarrier(scope: .buffers) }
    enc.setComputePipelineState(mode == "B6" ? pCast : mode == "B7" ? pCn : pCount); bind(enc); grid(enc)
  }
  enc.endEncoding()
  cb.commit()
  let t0 = w1_ticks()
  if mode == "B5" {                                            // the spin has started: the CPU now writes inside the running buffer
    while w1_load(donep + 64) != s { if us(w1_ticks() - t0) > 2e6 { print("  B5: spin start not seen in 2 s"); break } }
  } else if mode != "B0" {                                     // wait until every threadgroup has read P0
    while true {
      var all = true
      for t in 0..<TG where w1_load(donep + t) != s { all = false; break }
      if all { break }
      if us(w1_ticks() - t0) > 2e6 { print("  \(mode): phase-1 done words not all seen in 2 s"); break }
    }
  }
  for i in 0..<NW { xp[i] = 0xB0000000 | tagv | UInt32(i) }   // P1
  w1_fence_rel()
  w1_store_rel(flag.contents(), s)
  cb.waitUntilCompleted()
  return (stp[0], us(w1_ticks() - t0))
}
for mode in ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"] {
  var st: [UInt32] = [], ms: [Double] = []
  for _ in 0..<20 { let (n, t) = trial(mode); st.append(n); ms.append(t) }
  print(String(format: "%@: stale words per trial (of %d): min %u max %u, trials with any stale %d / 20; trial %.0f us median",
               mode, NW, st.min()!, st.max()!, st.filter { $0 > 0 }.count, ms.sorted()[10]))
}
