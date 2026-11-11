// w1_pickup.swift - W1-PICKUP (Leap, Tue 2026-11-03 19:3x; TABLE_350 'W1-PICKUP'): the loader's release -> the GPU's
// pickup. Today's spin (BigMoE's bigmoe_spin + its release, verbatim at a0a5fbe) vs a coherent(system) one-row spin
// (MLX's fast-synch pattern) vs MLX's own compiled fence_wait vs an MTLSharedEvent wait. Per trial, one command buffer:
// an m1 marker -> the wait -> an m2 marker. The CPU sees m1, waits w us, releases (timed), then times until it sees m2.
// Build: swiftc -O -import-objc-header w1_bridge.h -o w1_pickup w1_pickup.swift
// Usage: w1_pickup [trials 300] [arms A,B,B1,C,N] [w list -1,0,50,200,600,1000,2000]   (M = MLX's fence_wait: opt-in,
//        uncapped, so only after B shows no timeout). w -1 = released before the commit (a hidden burst).
import Metal
import Foundation

let args = CommandLine.arguments
let trials = args.count > 1 ? Int(args[1])! : 300
let arms = (args.count > 2 ? args[2] : "A,B,B1,C,N").split(separator: ",").map(String.init)
let wsList = (args.count > 3 ? args[3] : "-1,0,50,200,600,1000,2000").split(separator: ",").map { Int($0)! }
let RS = 192, PAIRS = 128, NREG = 128, N = 160
let BCAP: Int32 = 4_000_000
var tbi = mach_timebase_info_data_t()
mach_timebase_info(&tbi)
func us(_ t: UInt64) -> Double { Double(t) * Double(tbi.numer) / Double(tbi.denom) / 1000 }
func ticksOf(us x: Double) -> UInt64 { UInt64(x * 1000 * Double(tbi.denom) / Double(tbi.numer)) }
pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let SRC = """
#include <metal_stdlib>
using namespace metal;
#define RS \(RS)
#define SPIN_G1 1.1f
#define SPIN_G2 1.1f
#define SPIN_KNEE 0.0f
#define COUNT_FAST 1
kernel void w1_mark(device atomic_int* mk [[buffer(0)]], constant int2& a [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  atomic_store_explicit(mk + a.x, a.y, memory_order_relaxed);     // BigMoE's post pattern
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
// P: a stamped row published the way BigMoE's post does (device-scope atomic stores, a device fence, the tags last)
kernel void w1_post(device atomic_int* row [[buffer(0)]], constant int4& m [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const int seq = m.y, n = m.z;
  const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
  for (int i = 0; i < n; i++)
    atomic_store_explicit(row + (i < 31 ? 1 + i : i < 62 ? 32 + 1 + (i - 31) : i < 93 ? 64 + 1 + (i - 62) : 32 * (i / 31) + 1 + i % 31),
                          (int)(((uint)(i * 7 + 3) & 0x3FFFFFu) | STAMP), memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  for (int j = RS / 32 - 1; j >= 0; j--) atomic_store_explicit(row + 32 * j, seq, memory_order_relaxed);
}
// P1: 64 MB of reads after the post (the hit work's traffic)
kernel void w1_stream(device const uint4* src [[buffer(0)]], device uint* dst [[buffer(1)]], constant uint& total [[buffer(2)]],
                      uint g [[thread_position_in_grid]]) {
  uint4 acc = 0;
  for (uint k = 0; k < 16; k++) acc ^= src[g + k * total];
  dst[g & 0x3FFFFu] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
// bigmoe_spin verbatim (~/BigMoE/ext/bigmoe_ext.cpp at a0a5fbe, defaults G1 = G2 = 1.1, KNEE 0) + S0's planned
// histogram status[32 + got]
kernel void bigmoe_spin(device const int* dep [[buffer(0)]], device const int* reg [[buffer(1)]],
                        constant int4& m [[buffer(2)]], device atomic_int* status [[buffer(3)]],
                        device int* out [[buffer(4)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  int seq = m.y, n = m.z, pairs = m.w;
  if (dep[0] == 1) { for (int i = 0; i < n; i++) out[i] = dep[1 + i]; if (COUNT_FAST) atomic_fetch_add_explicit(status + 16, 1, memory_order_relaxed); return; }
  float x = 1.0f, d = 32.0f; int got = -1;
  for (int p = 0; p < pairs; p++) {
    const volatile device int* l0 = (const volatile device int*)(reg + p * RS);
    bool tags = l0[0] == seq && l0[32] == seq && (RS == 64 || l0[64] == seq);
    for (int j = 3; j < RS / 32 && tags; j++) tags = l0[32 * j] == seq;
    if (tags) {
      const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
      bool good = true;
      for (int i = 0; i < n && good; i++) {
        uint w = (uint)(i < 31 ? l0[1 + i] : i < 62 ? l0[32 + 1 + (i - 31)] : i < 93 ? l0[64 + 1 + (i - 62)] : l0[32 * (i / 31) + 1 + i % 31]);
        good = (w & 0xFFC00000u) == STAMP;
        out[i] = (w & 0x3FFFFFu) >= 0x3FFFFEu ? (int)(w & 0x3FFFFFu) - 0x400000 : (int)(w & 0x3FFFFFu);
      }
      if (good) { got = p; break; }
      atomic_fetch_add_explicit(status + 17, 1, memory_order_relaxed);
    }
    for (int j = 0; j < (int)d; j++) x = x * 1.0000001f + 1e-7f;
    d = d < SPIN_KNEE ? d * SPIN_G1 : min(d * SPIN_G2, 400000.0f);
  }
  if (got < 0) { atomic_fetch_add_explicit(status, 1, memory_order_relaxed); for (int i = 0; i < n; i++) out[i] = 0; return; }
  atomic_fetch_add_explicit(status + 32 + min(got, 127), 1, memory_order_relaxed);
  if (x == 12345.0f) out[0] = 0;
}
"""
// B: one simdgroup polls ONE row with coherent(system) loads and system-scope fences (MLX's fence_wait pattern); the
// lanes read the stamped words in parallel. Separate source: the internals pragma stays out of A's library.
let SRC_COH = """
#pragma METAL internals : enable
#include <metal_stdlib>
using namespace metal;
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
#define RS \(RS)
kernel void w1_mark_sys(volatile coherent(system) device int* mk [[buffer(0)]], constant int2& a [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  mk[a.x] = a.y;                                                    // MLX's fence_update pattern
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}
kernel void w1_post_sys(volatile coherent(system) device int* row [[buffer(0)]], constant int4& m [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;                                               // PS: the same row through a system-scope pointer
  const int seq = m.y, n = m.z;
  const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
  for (int i = 0; i < n; i++)
    row[i < 31 ? 1 + i : i < 62 ? 32 + 1 + (i - 31) : i < 93 ? 64 + 1 + (i - 62) : 32 * (i / 31) + 1 + i % 31] = (int)(((uint)(i * 7 + 3) & 0x3FFFFFu) | STAMP);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
  for (int j = RS / 32 - 1; j >= 0; j--) row[32 * j] = seq;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}
kernel void w1_coh_spin(volatile coherent(system) device int* row [[buffer(1)]], constant int4& m [[buffer(2)]],
                        device atomic_int* status [[buffer(3)]], device int* out [[buffer(4)]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const int seq = m.y, n = m.z, cap = m.w, nl = RS / 32;
  const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
  for (int it = 0; it < cap; it++) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    const int tg = row[32 * min((int)lane, nl - 1)];
    if (!simd_all(tg == seq)) continue;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    bool good = true;
    for (int i = (int)lane; i < n; i += 32) {
      uint w = (uint)(i < 31 ? row[1 + i] : i < 62 ? row[32 + 1 + (i - 31)] : i < 93 ? row[64 + 1 + (i - 62)] : row[32 * (i / 31) + 1 + i % 31]);
      good = good && (w & 0xFFC00000u) == STAMP;
      out[i] = (int)(w & 0x3FFFFFu);
    }
    if (simd_all(good)) {
      if (lane == 0) { atomic_fetch_add_explicit(status + 20, 1, memory_order_relaxed); atomic_store_explicit(status + 24, it, memory_order_relaxed); }
      return;
    }
    if (lane == 0) atomic_fetch_add_explicit(status + 17, 1, memory_order_relaxed);
  }
  if (lane == 0) atomic_fetch_add_explicit(status, 1, memory_order_relaxed);
}
"""
let lib = try! dev.makeLibrary(source: SRC, options: nil)
let psoMark = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "w1_mark")!)
let psoSpin = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "bigmoe_spin")!)
let psoPost = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "w1_post")!)
let psoStream = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "w1_stream")!)
var psoCoh: MTLComputePipelineState? = nil
let sysMark = ProcessInfo.processInfo.environment["W1_MARK"] == "sys"   // markers as MLX's fence_update (system scope)
var psoMarkUsed = psoMark
var psoPostSys: MTLComputePipelineState? = nil
if arms.contains(where: { $0.hasPrefix("B") || $0.hasPrefix("P") }) || sysMark {
  do {
    let l = try dev.makeLibrary(source: SRC_COH, options: nil)
    psoCoh = try dev.makeComputePipelineState(function: l.makeFunction(name: "w1_coh_spin")!)
    if sysMark { psoMarkUsed = try dev.makeComputePipelineState(function: l.makeFunction(name: "w1_mark_sys")!) }
    psoPostSys = try dev.makeComputePipelineState(function: l.makeFunction(name: "w1_post_sys")!)
  } catch { print("B: coherent(system) kernels do not compile: \(error)"); exit(2) }
}
var psoFence: MTLComputePipelineState? = nil
if arms.contains("M") {
  let p = NSHomeDirectory() + "/BigMoE/.venv/lib/python3.12/site-packages/mlx/lib/mlx.metallib"
  let l = try! dev.makeLibrary(URL: URL(fileURLWithPath: p))
  psoFence = try! dev.makeComputePipelineState(function: l.makeFunction(name: "fence_wait")!)
}

let mk = dev.makeBuffer(length: 4096, options: .storageModeShared)!
let reg = dev.makeBuffer(length: NREG * PAIRS * RS * 4, options: .storageModeShared)!   // 12.6 MB, the engine's ring
let rows = dev.makeBuffer(length: NREG * RS * 4, options: .storageModeShared)!
let status = dev.makeBuffer(length: 4096 * 4, options: .storageModeShared)!
let outb = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!
let dep = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!                 // dep[0] = 0: the miss path
let tsb = dev.makeBuffer(length: 256, options: .storageModeShared)!                     // M: MLX's timestamp word
let STREAM_T = 262_144
let prows = dev.makeBuffer(length: NREG * RS * 4, options: .storageModeShared)!          // P: the posted rows
let holdb = dev.makeBuffer(length: NREG * RS * 4, options: .storageModeShared)!          // P: the hold rows
let big = arms.contains("P1") || arms.contains("P2") ? dev.makeBuffer(length: 16 * STREAM_T * 16, options: .storageModeShared)! : tsb   // 64 MB
let bigOut = dev.makeBuffer(length: STREAM_T * 4, options: .storageModeShared)!
let ev = dev.makeSharedEvent()!
let mkp = mk.contents(), m2p = mk.contents() + 128
let regp = reg.contents().bindMemory(to: Int32.self, capacity: NREG * PAIRS * RS)
let rowsp = rows.contents().bindMemory(to: Int32.self, capacity: NREG * RS)
let st = status.contents().bindMemory(to: Int32.self, capacity: 4096)
let outp = outb.contents().bindMemory(to: Int32.self, capacity: 256)
let prowsp = prows.contents().bindMemory(to: Int32.self, capacity: NREG * RS)
let holdp = holdb.contents().bindMemory(to: Int32.self, capacity: NREG * RS)
func widx(_ i: Int) -> Int { i < 31 ? 1 + i : i < 62 ? 32 + 1 + (i - 31) : i < 93 ? 64 + 1 + (i - 62) : 32 * (i / 31) + 1 + i % 31 }

func one(_ enc: MTLComputeCommandEncoder, _ threads: Int = 1) {
  enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
}
func mark(_ enc: MTLComputeCommandEncoder, _ idx: Int32, _ s: Int32) {
  enc.setComputePipelineState(psoMarkUsed); enc.setBuffer(mk, offset: 0, index: 0)
  var a = SIMD2<Int32>(idx, s); enc.setBytes(&a, length: 8, index: 1); one(enc)
}
func rowIndex(_ arm: String, _ s: Int32) -> Int { arm == "B1" ? 0 : Int(s) % NREG }
func release(_ arm: String, _ s: Int32, _ sl: UnsafePointer<Int32>) {
  switch arm {
  case "A": w1_release_pairs(regp + (Int(s) % NREG) * PAIRS * RS, Int64(PAIRS), Int32(RS), sl, Int32(N), s)
  case "M": w1_release_pairs(rowsp + rowIndex(arm, s) * RS, 1, Int32(RS), sl, Int32(N), s); w1_store_rel(tsb.contents(), UInt32(bitPattern: s))
  case "C": w1_release_pairs(rowsp + rowIndex(arm, s) * RS, 1, Int32(RS), sl, Int32(N), s); ev.signaledValue = UInt64(s)
  default: w1_release_pairs(rowsp + rowIndex(arm, s) * RS, 1, Int32(RS), sl, Int32(N), s)   // B, B1
  }
}

struct Res { var rel: Double; var pick: Double; var start: Double; var ok: Bool; var got: Int; var it: Int; var first: Double = .nan }
var res: [String: [Res]] = [:]
var seq: Int32 = 0
var sl = [Int32](repeating: 0, count: N)

func trial(_ arm: String, _ w: Int) -> Res {
  seq += 1
  let s = seq
  for i in 0..<N { sl[i] = Int32(bitPattern: (UInt32(bitPattern: s) &* 2654435761 &+ UInt32(i) &* 40503) % 0x3FFFF0) }
  for i in 0..<160 { st[i] = 0 }
  for i in 0..<N { outp[i] = -7 }
  let cb = queue.makeCommandBuffer()!
  var enc = cb.makeComputeCommandEncoder()!
  mark(enc, 0, s)
  switch arm {
  case "A":
    enc.setComputePipelineState(psoSpin); enc.setBuffer(dep, offset: 0, index: 0)
    enc.setBuffer(reg, offset: (Int(s) % NREG) * PAIRS * RS * 4, index: 1)
    var m = SIMD4<Int32>(0, s, Int32(N), Int32(PAIRS)); enc.setBytes(&m, length: 16, index: 2)
    enc.setBuffer(status, offset: 0, index: 3); enc.setBuffer(outb, offset: 0, index: 4); one(enc)
  case "B", "B1":
    enc.setComputePipelineState(psoCoh!); enc.setBuffer(rows, offset: rowIndex(arm, s) * RS * 4, index: 1)
    var m = SIMD4<Int32>(0, s, Int32(N), BCAP); enc.setBytes(&m, length: 16, index: 2)
    enc.setBuffer(status, offset: 0, index: 3); enc.setBuffer(outb, offset: 0, index: 4); one(enc, 32)
  case "M":
    enc.setComputePipelineState(psoFence!); enc.setBuffer(tsb, offset: 0, index: 0)
    var v = UInt32(bitPattern: s); enc.setBytes(&v, length: 4, index: 1); one(enc)
  case "C":
    enc.endEncoding(); cb.encodeWaitForEvent(ev, value: UInt64(s)); enc = cb.makeComputeCommandEncoder()!
  default: break   // N
  }
  mark(enc, 32, s)
  enc.endEncoding()
  var r0: UInt64 = 0, r1: UInt64 = 0
  if w < 0 && arm != "N" { r0 = w1_ticks(); sl.withUnsafeBufferPointer { release(arm, s, $0.baseAddress!) }; r1 = w1_ticks() }
  let tc = w1_ticks()
  cb.commit()
  while w1_load(mkp) != UInt32(bitPattern: s) { if us(w1_ticks() - tc) > 2e6 { print("m1 not seen in 2 s (\(arm) w \(w)): stop"); exit(1) } }
  let t1 = w1_ticks()
  if w >= 0 && arm != "N" {
    let tw = t1 + ticksOf(us: Double(w))
    while w1_ticks() < tw {}
    r0 = w1_ticks(); sl.withUnsafeBufferPointer { release(arm, s, $0.baseAddress!) }; r1 = w1_ticks()
  }
  var t2: UInt64 = 0
  var seen = false
  while !seen {
    if w1_load(m2p) == UInt32(bitPattern: s) { t2 = w1_ticks(); seen = true }
    else if us(w1_ticks() - t1) > 500_000 { break }
  }
  cb.waitUntilCompleted()
  var ok = seen && st[0] == 0
  if arm == "A" || arm.hasPrefix("B") { for i in 0..<N where outp[i] != sl[i] { ok = false; break } }
  var got = -1
  if arm == "A" { for p in 0..<128 where st[32 + p] != 0 { got = p } }
  let from = (w < 0 || arm == "N") ? t1 : r1
  return Res(rel: us(r1 - r0), pick: seen ? us(t2 - from) : .nan, start: us(t1 - tc), ok: ok, got: got,
             it: arm.hasPrefix("B") ? Int(st[24]) : -1)
}

// P: m1 -> post (device or system scope) -> [P1: 64 MB of reads] -> hold (B's spin, released once the CPU saw the post or
// after 5 ms) -> m2; pick = m1 seen -> the post's tag seen; got = words not yet stamped when the tag was seen
func trialP(_ arm: String) -> Res {
  seq += 1
  let s = seq, pr = Int(s) % NREG
  for i in 0..<N { sl[i] = Int32(bitPattern: (UInt32(bitPattern: s) &* 2654435761 &+ UInt32(i) &* 40503) % 0x3FFFF0) }
  for i in 0..<160 { st[i] = 0 }
  for i in 0..<N { outp[i] = -7 }
  let cb = queue.makeCommandBuffer()!
  let enc = cb.makeComputeCommandEncoder()!
  mark(enc, 0, s)
  enc.setComputePipelineState(arm == "PS" ? psoPostSys! : psoPost)
  enc.setBuffer(prows, offset: pr * RS * 4, index: 0)
  var m = SIMD4<Int32>(0, s, Int32(N), 0); enc.setBytes(&m, length: 16, index: 1); one(enc)
  if arm == "P1" || arm == "P2" {   // P1: 64 MB of reads after the post, P2: 8 MB
    let nt = arm == "P1" ? STREAM_T : STREAM_T / 8
    enc.setComputePipelineState(psoStream); enc.setBuffer(big, offset: 0, index: 0); enc.setBuffer(bigOut, offset: 0, index: 1)
    var tot = UInt32(nt); enc.setBytes(&tot, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: nt, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  }
  enc.setComputePipelineState(psoCoh!); enc.setBuffer(holdb, offset: pr * RS * 4, index: 1)
  var mh = SIMD4<Int32>(0, s, Int32(N), BCAP); enc.setBytes(&mh, length: 16, index: 2)
  enc.setBuffer(status, offset: 0, index: 3); enc.setBuffer(outb, offset: 0, index: 4); one(enc, 32)
  mark(enc, 32, s)
  enc.endEncoding()
  let tc = w1_ticks()
  cb.commit()
  while w1_load(mkp) != UInt32(bitPattern: s) { if us(w1_ticks() - tc) > 2e6 { print("m1 not seen in 2 s (\(arm)): stop"); exit(1) } }
  let t1 = w1_ticks()
  let rowp = prowsp + pr * RS
  var tp: UInt64 = 0
  var seen = false
  while !seen {
    if w1_load(rowp) == UInt32(bitPattern: s) { tp = w1_ticks(); seen = true }
    else if us(w1_ticks() - t1) > 5000 { break }
  }
  var ta: UInt64 = 0                 // then every line's tag, the engine loader's criterion (loader_main 'all lines complete')
  if seen {
    while true {
      var all = true
      for j in 0..<(RS / 32) where w1_load(rowp + 32 * j) != UInt32(bitPattern: s) { all = false; break }
      if all { ta = w1_ticks(); break }
      if us(w1_ticks() - t1) > 5000 { seen = false; break }
    }
  }
  var unst = 0
  if seen {
    let stamp = (UInt32(bitPattern: s) & 0x3FF) << 22
    for i in 0..<N where (w1_load(rowp + widx(i)) & 0xFFC00000) != stamp { unst += 1 }
  }
  sl.withUnsafeBufferPointer { w1_release_pairs(holdp + pr * RS, 1, Int32(RS), $0.baseAddress!, Int32(N), s) }
  let th = w1_ticks()
  while w1_load(m2p) != UInt32(bitPattern: s) { if us(w1_ticks() - th) > 500_000 { break } }
  cb.waitUntilCompleted()
  var ok = seen && unst == 0 && st[0] == 0
  for i in 0..<N where outp[i] != sl[i] { ok = false; break }
  return Res(rel: 0, pick: seen ? us(ta - t1) : .nan, start: us(t1 - tc), ok: ok, got: unst, it: -1, first: tp > 0 ? us(tp - t1) : .nan)
}

var cfgs: [(String, Int)] = []
for a in arms { for w in (a == "N" || a.hasPrefix("P") ? [0] : wsList) { cfgs.append((a, w)) } }
func run(_ a: String, _ w: Int) -> Res { a.hasPrefix("P") ? trialP(a) : trial(a, w) }
var rng: UInt64 = 0x9E3779B97F4A7C15
func rnd() -> Int { rng = rng &* 6364136223846793005 &+ 1442695040888963407; return Int(rng >> 33) }
for (a, w) in cfgs { _ = run(a, w) }   // warm-up
let t0 = Date()
for _ in 0..<trials {
  var order = cfgs
  for i in stride(from: order.count - 1, to: 0, by: -1) { order.swapAt(i, rnd() % (i + 1)) }
  for (a, w) in order { res["\(a) \(w)", default: []].append(run(a, w)) }
}
func q(_ x: [Double], _ f: Double) -> Double { let s = x.filter { !$0.isNaN }.sorted(); return s.isEmpty ? .nan : s[min(s.count - 1, Int(Double(s.count) * f))] }
print(String(format: "W1-PICKUP %@: %d trials per (arm, w), %.1f s; RS %d, n %d, pairs %d; markers %@; times in us", dev.name, trials, Date().timeIntervalSince(t0), RS, N, PAIRS, sysMark ? "system scope (fence_update)" : "device scope (BigMoE post)"))
print("arm  w(us)   release med/p90    pickup med / p90 / max     start med   bad   (A: got p median; B: polls median)")
for (a, w) in cfgs {
  let r = res["\(a) \(w)"]!
  let bad = r.filter { !$0.ok }.count
  let extra = a == "A" ? String(format: "got p %d", Int(q(r.map { Double($0.got) }, 0.5))) : a.hasPrefix("B") ? String(format: "polls %d", Int(q(r.map { Double($0.it) }, 0.5)))
    : a.hasPrefix("P") ? String(format: "all tags (pickup col.); line 0's tag med %.1f; not all seen in 5 ms %d; unstamped words after all tags, max %d",
                                q(r.map { $0.first }, 0.5), r.filter { $0.pick.isNaN }.count, r.map { $0.got }.max() ?? 0) : ""
  print(String(format: "%-4@ %5d   %6.2f / %6.2f    %7.1f / %7.1f / %7.1f    %7.1f    %3d   %@", a, w,
               q(r.map { $0.rel }, 0.5), q(r.map { $0.rel }, 0.9), q(r.map { $0.pick }, 0.5), q(r.map { $0.pick }, 0.9),
               q(r.map { $0.pick }, 1.0), q(r.map { $0.start }, 0.5), bad, extra))
}
if arms.contains("A") {   // S0's hook: pair index p -> the spin's elapsed time at that poll (m1 seen -> release seen)
  var byp: [Int: [Double]] = [:]
  for (k, r) in res where k.hasPrefix("A ") {
    let w = Double(k.split(separator: " ")[1])!
    for x in r where x.ok && x.got >= 0 { byp[x.got, default: []].append(w >= 0 ? w + x.rel + x.pick : x.pick) }
  }
  print("A: pair p -> elapsed since the spin's start (us, median; n):")
  print(byp.keys.sorted().map { String(format: "p%d %.0f (%d)", $0, q(byp[$0]!, 0.5), byp[$0]!.count) }.joined(separator: ", "))
}
