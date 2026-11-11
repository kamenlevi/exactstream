// PURGE-1 (a) (Leap, Mon 2026-11-02 23:3x; TABLE_350 'PURGE-1'): the cost of toggling Metal buffers between volatile and
// nonVolatile, without memory pressure. 64 shared buffers of 32 MB (pages touched); per buffer and round: setPurgeableState
// (.volatile), then (.nonVolatile) (its return says whether the contents were discarded: .empty), then a read of the
// whole buffer (pages still resident?). 20 rounds; medians and p90 in us.
// Build: swiftc -O purge_probe.swift -o purge_probe ; run: ./purge_probe [nbuf=64] [mb=32] [rounds=20]
import Foundation
import Metal

let args = CommandLine.arguments
let nbuf = args.count > 1 ? Int(args[1])! : 64
let mb = args.count > 2 ? Int(args[2])! : 32
let rounds = args.count > 3 ? Int(args[3])! : 20
let dev = MTLCreateSystemDefaultDevice()!
func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
var bufs: [MTLBuffer] = []
for _ in 0..<nbuf {
    let b = dev.makeBuffer(length: mb << 20, options: .storageModeShared)!
    memset(b.contents(), 1, b.length)
    bufs.append(b)
}
var tv: [Double] = [], tn: [Double] = [], tr: [Double] = []
var empties = 0
var sink: UInt64 = 0
for _ in 0..<rounds {
    for b in bufs {
        let t0 = now()
        _ = b.setPurgeableState(.volatile)
        let t1 = now()
        let prev = b.setPurgeableState(.nonVolatile)
        let t2 = now()
        if prev == .empty { empties += 1 }
        let p = b.contents().assumingMemoryBound(to: UInt64.self)
        var s: UInt64 = 0
        var i = 0
        let n = b.length / 8
        while i < n { s &+= p[i]; i += 512 }              // one word per 4 KB page: touches every page
        sink &+= s
        let t3 = now()
        tv.append(Double(t1 - t0) / 1000); tn.append(Double(t2 - t1) / 1000); tr.append(Double(t3 - t2) / 1000)
    }
}
func q(_ v: [Double], _ f: Double) -> Double { let s = v.sorted(); return s[min(s.count - 1, Int(Double(s.count) * f))] }
print(String(format: "PURGE-1 (a): %d buffers x %d MB, %d rounds: ->volatile median %.1f us (p90 %.1f); ->nonVolatile median %.1f us (p90 %.1f); page walk after %.1f us (p90 %.1f); buffers found empty %d (sink %llu)",
             nbuf, mb, rounds, q(tv, 0.5), q(tv, 0.9), q(tn, 0.5), q(tn, 0.9), q(tr, 0.5), q(tr, 0.9), empties, sink % 7))
