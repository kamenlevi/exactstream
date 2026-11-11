// PURGE-1 (b) (Leap, written Mon 2026-11-02 23:4x; TABLE_350 'PURGE-1 (b)'): under memory pressure, does macOS empty
// volatile Metal buffers BEFORE it compresses or swaps other memory? 64 shared buffers of 32 MB (2 GB) filled with
// pseudo-random bytes (incompressible, like expert records) and set volatile, plus a 65th buffer that only times the
// toggle (a toggle re-queues a buffer's purge token, so the 64 counted slabs are never toggled). First 5 s idle: the
// system's own compression / swap-out rate (the baseline). Then a balloon of anonymous memory, also pseudo-random,
// grows in 256 MB steps. After every step: slabs empty (setPurgeableState(.keepCurrent) == .empty), compressions and
// swap-outs since the start minus the baseline rate x time (host_statistics64, 16 KB pages), swap used (vm.swapusage),
// the kernel's pressure level, and one toggle of the 65th buffer (.nonVolatile then .volatile, timed).
// Stops at: every slab empty (then 2 more steps), swap used +SWAPCAP MB (default 1024), pressure level critical,
// 120 s, or the balloon at MAXGB. Frees the balloon at once, then reports swap 3 s later.
// Run ONLY with Kamen's apps closed and no big job running. Build: swiftc -O purge_balloon.swift -o purge_balloon
// Usage: ./purge_balloon [maxgb=48] [swapcap_mb=1024]
import Darwin
import Foundation
import Metal

let args = CommandLine.arguments
let maxGB = args.count > 1 ? Double(args[1])! : 48
let swapCapMB = args.count > 2 ? Double(args[2])! : 1024
let pageMB = Double(vm_page_size) / 1048576
func swapUsedMB() -> Double {
    var u = xsw_usage(); var sz = MemoryLayout<xsw_usage>.size
    sysctlbyname("vm.swapusage", &u, &sz, nil, 0)
    return Double(u.xsu_used) / 1048576
}
func pressureLevel() -> Int32 {                     // 1 normal, 2 warning, 4 critical
    var v: Int32 = 0; var sz = MemoryLayout<Int32>.size
    sysctlbyname("kern.memorystatus_vm_pressure_level", &v, &sz, nil, 0)
    return v
}
func vmStats() -> (swapouts: Double, compressions: Double, free: Double, purgeable: Double) {
    var s = vm_statistics64(); var n = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &s) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(n)) {
        host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &n) } }
    return (Double(s.swapouts), Double(s.compressions), Double(s.free_count), Double(s.purgeable_count))
}
var rng: UInt64 = 0x9E3779B97F4A7C15
func fillRandom(_ p: UnsafeMutableRawPointer, _ n: Int) {   // xorshift64: incompressible pages
    let q = p.bindMemory(to: UInt64.self, capacity: n / 8)
    var x = rng
    for i in 0..<(n / 8) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; q[i] = x }
    rng = x
}
func now() -> Double { var t = timespec(); clock_gettime(CLOCK_MONOTONIC, &t); return Double(t.tv_sec) + 1e-9 * Double(t.tv_nsec) }
let dev = MTLCreateSystemDefaultDevice()!
var slabs: [MTLBuffer] = []
for _ in 0..<64 {
    let b = dev.makeBuffer(length: 32 << 20, options: .storageModeShared)!
    fillRandom(b.contents(), b.length)
    _ = b.setPurgeableState(.volatile)
    slabs.append(b)
}
let probe = dev.makeBuffer(length: 32 << 20, options: .storageModeShared)!
fillRandom(probe.contents(), probe.length)
_ = probe.setPurgeableState(.volatile)
func toggleUs() -> (Double, String) {
    let t0 = now()
    let old = probe.setPurgeableState(.nonVolatile)
    _ = probe.setPurgeableState(.volatile)
    return (1e6 * (now() - t0), old == .empty ? "was empty" : "kept")
}
// baseline: 5 s idle
let b0 = vmStats(), tb0 = now()
while now() - tb0 < 5 { usleep(100_000) }
let b1 = vmStats(), tb1 = now()
let rateC = (b1.compressions - b0.compressions) / (tb1 - tb0), rateS = (b1.swapouts - b0.swapouts) / (tb1 - tb0)
let s0 = vmStats(), sw0 = swapUsedMB(), t0 = now()
print(String(format: "start %@: 64 volatile slabs (2 GB, random); swap used %.0f MB; free %.2f GB; purgeable %.2f GB; pressure %d; idle baseline: compressions %.0f MB/s, swap-outs %.0f MB/s",
             Date().description, sw0, s0.free * pageMB / 1024, s0.purgeable * pageMB / 1024, pressureLevel(), rateC * pageMB, rateS * pageMB))
var balloon: [UnsafeMutableRawPointer] = []
let step = 256 << 20
var firstEmptyAt = -1, allEmptyAt = -1
var togs: [Double] = []
var verdict = "no pressure reached (balloon at its max)"
var atFirst = "", atAll = ""
while Double(balloon.count) * 0.25 < maxGB {
    let p = mmap(nil, step, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
    if p == MAP_FAILED || p == nil { verdict = "STOPPED: mmap failed"; break }
    fillRandom(p!, step)
    balloon.append(p!)
    let empty = slabs.filter { $0.setPurgeableState(.keepCurrent) == .empty }.count
    let (tg, tgs) = toggleUs()
    togs.append(tg)
    let s = vmStats(), sw = swapUsedMB(), dt = now() - t0, lvl = pressureLevel()
    let exC = (s.compressions - s0.compressions - rateC * dt) * pageMB, exS = (s.swapouts - s0.swapouts - rateS * dt) * pageMB
    let line = String(format: "t %5.1f s balloon %5.2f GB: slabs empty %2d / 64; compressions %+.0f MB, swap-outs %+.0f MB over baseline; swap used %+.0f MB; free %.2f GB; purgeable %.2f GB; pressure %d; toggle %.1f us (%@)",
                      dt, Double(balloon.count) * 0.25, empty, exC, exS, sw - sw0, s.free * pageMB / 1024, s.purgeable * pageMB / 1024, lvl, tg, tgs)
    print(line); fflush(stdout)
    if empty > 0 && firstEmptyAt < 0 { firstEmptyAt = balloon.count; atFirst = line }
    if empty == 64 && allEmptyAt < 0 { allEmptyAt = balloon.count; atAll = line }
    if sw - sw0 > swapCapMB { verdict = "STOPPED: swap +\(Int(sw - sw0)) MB (cap)"; break }
    if lvl >= 4 { verdict = "STOPPED: pressure critical"; break }
    if dt > 120 { verdict = "STOPPED: 120 s"; break }
    if allEmptyAt >= 0 && balloon.count >= allEmptyAt + 2 { verdict = "all slabs emptied"; break }
}
for p in balloon { munmap(p, step) }
let tq = now(); while now() - tq < 3 { usleep(100_000) }
let s1 = vmStats()
togs.sort()
print("verdict: \(verdict)")
print("first slab empty at: \(firstEmptyAt < 0 ? "never" : atFirst)")
print("all slabs empty at:  \(allEmptyAt < 0 ? "never" : atAll)")
print(String(format: "toggle under the balloon: median %.1f us, max %.1f us over %d steps; 3 s after freeing: swap used %+.0f MB, compressions %+.0f MB, swap-outs %+.0f MB since the start",
             togs.isEmpty ? 0 : togs[togs.count / 2], togs.last ?? 0, togs.count, swapUsedMB() - sw0,
             (s1.compressions - s0.compressions) * pageMB, (s1.swapouts - s0.swapouts) * pageMB))
