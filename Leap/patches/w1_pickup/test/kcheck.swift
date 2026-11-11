// kcheck.swift - W1-PICKUP patch check (Leap, Tue 2026-11-03; TABLE_350 'W1-PICKUP'). The kernel texts extracted from
// the patched ext (test/*.metal, written by the extraction step from new/ext/bigmoe_ext.cpp) compiled and run on the GPU.
// 1. spin_coh, with the engine's region ring (128 regions x 128 pairs x RS 192) and the patched release (pair 0 only):
//    the pickup path (incl. the -1 / -2 slot codes), the fast path (dep[0] = 1), a stale word fixed late (must poll
//    again, not return it), the timeout (COH_CAP 1000, no release: status[0] = 1, out = 0); pickup timing at 0 / 1000 us.
// 2. post_ids: today's kernel (post_base), the patched default (post_default) and BIGMOE_POST_SYS (post_sys) on the
//    same inputs, without and with a residency table: route rows and dep bitwise identical.
// Build: swiftc -O -import-objc-header ../../../native/w1_bridge.h -o kcheck kcheck.swift
import Metal
import Foundation

let dir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
func pso(_ file: String, _ fn: String) -> MTLComputePipelineState {
  do {
    let l = try dev.makeLibrary(source: try String(contentsOfFile: dir + "/" + file + ".metal", encoding: .utf8), options: nil)
    return try dev.makeComputePipelineState(function: l.makeFunction(name: fn)!)
  } catch { print("COMPILE FAIL \(file): \(error)"); exit(2) }
}
var tbi = mach_timebase_info_data_t()
mach_timebase_info(&tbi)
func us(_ t: UInt64) -> Double { Double(t) * Double(tbi.numer) / Double(tbi.denom) / 1000 }
pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
var fails = 0
func check(_ ok: Bool, _ what: String) { print((ok ? "ok   " : "FAIL ") + what); if !ok { fails += 1 } }

// ---------------------------------------------------------------- 1. the coherent spin
let spin = pso("spin_coh", "bigmoe_spin"), spinCap = pso("spin_coh_cap", "bigmoe_spin")
let markSrc = """
#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
using namespace metal;
kernel void mark(volatile coherent(system) device int* mk [[buffer(0)]], constant int2& a [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  mk[a.x] = a.y;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}
"""
let markP = try! dev.makeComputePipelineState(function: try! dev.makeLibrary(source: markSrc, options: nil).makeFunction(name: "mark")!)
let RS = 192, PAIRS = 128, NREG = 128, NL = 96, N = 160
let reg = dev.makeBuffer(length: NREG * PAIRS * RS * 4, options: .storageModeShared)!
let regp = reg.contents().bindMemory(to: Int32.self, capacity: NREG * PAIRS * RS)
let dep = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!
let depp = dep.contents().bindMemory(to: Int32.self, capacity: 256)
let status = dev.makeBuffer(length: 4 * 4096, options: .storageModeShared)!
let st = status.contents().bindMemory(to: Int32.self, capacity: 4096)
let outb = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!
let outp = outb.contents().bindMemory(to: Int32.self, capacity: 256)
let mk = dev.makeBuffer(length: 4096, options: .storageModeShared)!
var sl = [Int32](repeating: 0, count: N)
func regOff(_ seq: Int32, _ L: Int) -> Int { ((Int(seq) * NL + L) % NREG) * PAIRS * RS }   // the engine's region index

// one spin; release(after: us) = nil: none; stale: word 10 carries the previous stamp first, fixed 200 us later
func spinTrial(_ p: MTLComputePipelineState, seq s: Int32, L: Int, fast: Bool, releaseAfter: Double?, stale: Bool = false) -> (Double, Bool) {
  for i in 0..<N { sl[i] = Int32(bitPattern: (UInt32(bitPattern: s) &* 2654435761 &+ UInt32(i) &* 40503) % 0x3FFFF0) }
  sl[5] = -1; sl[6] = -2                                            // LP1 codes ride through the stamp
  for i in 0..<64 { st[i] = 0 }
  for i in 0..<N { outp[i] = -7 }
  depp[0] = fast ? 1 : 0
  for i in 0..<N { depp[1 + i] = sl[i] ^ 0x55 }
  let cb = queue.makeCommandBuffer()!
  let enc = cb.makeComputeCommandEncoder()!
  func mark(_ idx: Int32) { enc.setComputePipelineState(markP); enc.setBuffer(mk, offset: 0, index: 0); var a = SIMD2<Int32>(idx, s); enc.setBytes(&a, length: 8, index: 1)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)) }
  mark(0)
  enc.setComputePipelineState(p); enc.setBuffer(dep, offset: 0, index: 0); enc.setBuffer(reg, offset: regOff(s, L) * 4, index: 1)
  var m = SIMD4<Int32>(Int32(L), s, Int32(N), Int32(PAIRS)); enc.setBytes(&m, length: 16, index: 2)
  enc.setBuffer(status, offset: 0, index: 3); enc.setBuffer(outb, offset: 0, index: 4)
  enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  mark(32)
  enc.endEncoding()
  cb.commit()
  let mkp = mk.contents(), m2p = mk.contents() + 128
  while w1_load(mkp) != UInt32(bitPattern: s) {}
  let t1 = w1_ticks()
  var tr: UInt64 = 0
  if let w = releaseAfter {
    while us(w1_ticks() - t1) < w {}
    let rp = regp + regOff(s, L)
    if stale {     // the engine's one-row release, but word 10 (line 0, word 11) keeps the old stamp until 200 us later
      for i in 0..<N { rp[i < 31 ? 1 + i : i < 62 ? 32 + 1 + (i - 31) : i < 93 ? 64 + 1 + (i - 62) : 32 * (i / 31) + 1 + i % 31] = w1_stamp_enc(sl[i], s) }
      rp[11] = w1_stamp_enc(sl[10], s &- 1)
      w1_fence_rel()
      for j in stride(from: RS / 32 - 1, through: 0, by: -1) { w1_store_rel(rp + 32 * j, UInt32(bitPattern: s)) }
      let t2 = w1_ticks(); while us(w1_ticks() - t2) < 200 {}
      w1_store_rel(rp + 11, UInt32(bitPattern: w1_stamp_enc(sl[10], s)))
    } else {
      sl.withUnsafeBufferPointer { w1_release_pairs(rp, 1, Int32(RS), $0.baseAddress!, Int32(N), s) }
    }
    tr = w1_ticks()
  }
  var t2: UInt64 = 0
  while w1_load(m2p) != UInt32(bitPattern: s) {}
  t2 = w1_ticks()
  cb.waitUntilCompleted()
  var ok = true
  if fast { for i in 0..<N where outp[i] != sl[i] ^ 0x55 { ok = false } ; ok = ok && st[16] == 1 && st[0] == 0 }
  else if releaseAfter == nil { for i in 0..<N where outp[i] != 0 { ok = false }; ok = ok && st[0] == 1 }
  else { for i in 0..<N where outp[i] != sl[i] { ok = false }; ok = ok && st[0] == 0 && (!stale || st[17] > 0) }
  return (tr > 0 ? us(t2 - tr) : .nan, ok)
}
var seq: Int32 = 100
var okAll = true
var t0: [Double] = [], t1000: [Double] = []
for r in 0..<200 {
  seq += 1; let a = spinTrial(spin, seq: seq, L: r % 96, fast: false, releaseAfter: 0); okAll = okAll && a.1; t0.append(a.0)
  seq += 1; let b = spinTrial(spin, seq: seq, L: r % 96, fast: false, releaseAfter: 1000); okAll = okAll && b.1; t1000.append(b.0)
}
let med = { (x: [Double]) -> Double in x.sorted()[x.count / 2] }
check(okAll, String(format: "spin_coh pickup path: 400 trials, slots (incl. -1 / -2) exact; pickup median %.1f us at w 0, %.1f us at w 1000", med(t0), med(t1000)))
seq += 1; check(spinTrial(spin, seq: seq, L: 5, fast: true, releaseAfter: nil).1, "spin_coh fast path (dep[0] = 1): out = dep[1..n], status[16] = 1")
seq += 1; let stl = spinTrial(spin, seq: seq, L: 9, fast: false, releaseAfter: 100, stale: true)
check(stl.1, String(format: "spin_coh stale word fixed 200 us late: polled again (status[17] = %d), final slots exact", st[17]))
seq += 1; check(spinTrial(spinCap, seq: seq, L: 11, fast: false, releaseAfter: nil).1, "spin_coh timeout (COH_CAP 1000, no release): status[0] = 1, out = 0")

// ---------------------------------------------------------------- 2. post_ids: three texts, same bytes
let posts = [("post_base", pso("post_base", "bigmoe_post_ids")), ("post_default", pso("post_default", "bigmoe_post_ids")), ("post_sys", pso("post_sys", "bigmoe_post_ids"))]
let ROW = 96 + 32 * 0 + 32 * 4
let ids = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!
let idsp = ids.contents().bindMemory(to: Int32.self, capacity: 256)
let preds = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!
let predsp = preds.contents().bindMemory(to: Int32.self, capacity: 256)
let tab = dev.makeBuffer(length: 4 * 17 * 32, options: .storageModeShared)!
let tabp = tab.contents().bindMemory(to: Int32.self, capacity: 17 * 32)
for i in 0..<N { idsp[i] = Int32((i * 37 + 11) % 512) }
for i in 0..<64 { predsp[i] = Int32((i * 91 + 5) % 512) }
for useTab in [false, true] {
  let pseq: Int32 = 345, L: Int32 = 3
  for li in 0..<17 {                                                // a table where some lines / entries are fresh
    tabp[li * 32] = li % 3 == 0 ? pseq : pseq - 1
    for j in 0..<31 { tabp[li * 32 + 1 + j] = (li + j) % 4 == 0 ? Int32(bitPattern: 0x3FFFFF | ((UInt32(pseq) & 0x3FF) << 22)) : w1_stamp_enc(Int32(li * 31 + j), (li + j) % 5 == 0 ? pseq - 1 : pseq) }
  }
  var rows: [[Int32]] = [], deps: [[Int32]] = []
  for (_, p) in posts {
    let route = dev.makeBuffer(length: 4 * 4096, options: .storageModeShared)!
    let dout = dev.makeBuffer(length: 4 * 256, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(ids, offset: 0, index: 0); enc.setBuffer(route, offset: 0, index: 1)
    var m = SIMD4<Int32>(L, pseq, Int32(N), useTab ? 1 : 0); enc.setBytes(&m, length: 16, index: 2)
    enc.setBuffer(dout, offset: 0, index: 3); enc.setBuffer(tab, offset: 0, index: 5); enc.setBuffer(preds, offset: 0, index: 6)
    var np: Int32 = 12 | (10 << 8); enc.setBytes(&np, length: 4, index: 7)
    enc.dispatchThreads(MTLSize(width: 192, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let rp = route.contents().bindMemory(to: Int32.self, capacity: 4096)
    rows.append((0..<ROW).map { rp[Int(L) * ROW + $0] })
    let dp = dout.contents().bindMemory(to: Int32.self, capacity: 256)
    deps.append((0...N).map { dp[$0] })
  }
  let nz = rows[0].filter { $0 != 0 }.count
  check(rows[1] == rows[0] && rows[2] == rows[0] && deps[1] == deps[0] && deps[2] == deps[0] && rows[0][0] == pseq && nz > N,
        "post_ids \(useTab ? "with" : "without") a residency table: patched default and POST_SYS == today's kernel, bitwise (row \(ROW) ints, \(nz) set; dep \(N + 1) ints, ok = \(deps[0][0]))")
}
print(fails == 0 ? "KCHECK PASS" : "KCHECK FAIL (\(fails))")
exit(fails == 0 ? 0 : 1)
