// Round 24 microbenchmark: burst latency of cold expert-sized reads from the 125B's shard files, Metal fast resource
// loading (MTLIOCommandQueue) vs pread threads, the quantity that sets a stalled layer's wait (round 20: BigMoE's
// loader reads bursts at 6-7.6 GB/s against the SSD's 12-14.6). Each read is 2,768,896 bytes (169 x 16 KB, one
// expert's size) at a fresh random 16 KB-aligned offset of the 17 shards (~70 GB, never re-read), so the page cache
// doesn't help. Modes, per burst of n experts (n = 1, 2, 4, 8), 40 bursts each, times commit -> all data in memory:
//   io1   MTLIO, one load command per expert, all in one IO command buffer (concurrent queue)
//   io4   MTLIO, each expert as 4 chunk loads
//   io16  MTLIO, each expert as 16 chunk loads
//   pr1   pread, one per expert, n threads (F_NOCACHE)
//   pr4   pread, each expert as 4 chunks, 4n threads
// usage: mtlio_bench [BURSTS]     build: swiftc -O tools/mtlio_bench.swift -o tools/mtlio_bench -framework Metal
import Foundation
import Metal

let E = 2_768_896
let bursts = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 40
let dir = NSString(string: "~/BigMoE/models/bare").expandingTildeInPath
let paths = (1...17).map { String(format: "\(dir)/model-%05d-of-00017.safetensors", $0) }
let sizes = paths.map { (try! FileManager.default.attributesOfItem(atPath: $0)[.size] as! NSNumber).intValue }
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let qd = MTLIOCommandQueueDescriptor()
qd.type = .concurrent
qd.priority = .high
let queue = try! device.makeIOCommandQueue(descriptor: qd)
let handles = try! paths.map { try device.makeIOHandle(url: URL(fileURLWithPath: $0)) }
let fds: [Int32] = paths.map { p in let fd = open(p, O_RDONLY); _ = fcntl(fd, F_NOCACHE, 1); return fd }
let buf = device.makeBuffer(length: 8 * E, options: .storageModeShared)!
var used = Set<Int>()                                   // (shard << 32 | block) of every 16 KB block read so far
var rng = SystemRandomNumberGenerator()

func fresh() -> (Int, Int) {                            // a never-read expert-sized region
    while true {
        let s = Int.random(in: 0..<17, using: &rng)
        let blocks = (sizes[s] - E) / 16384
        let b = Int.random(in: 0..<blocks, using: &rng)
        let keys = (b..<(b + 169)).map { (s << 32) | $0 }
        if keys.allSatisfy({ !used.contains($0) }) { keys.forEach { used.insert($0) }; return (s, b * 16384) }
    }
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

func runIO(_ n: Int, chunks: Int) -> Double {
    let regions = (0..<n).map { _ in fresh() }
    let cb = queue.makeCommandBuffer()
    let c = E / chunks
    for (i, (s, off)) in regions.enumerated() {
        for k in 0..<chunks {
            let size = k == chunks - 1 ? E - c * k : c
            cb.load(buf, offset: i * E + k * c, size: size, sourceHandle: handles[s], sourceHandleOffset: off + k * c)
        }
    }
    let t0 = now()
    cb.commit()
    cb.waitUntilCompleted()
    let t = now() - t0
    if cb.status != .complete { fatalError("IO command buffer failed: \(cb.status.rawValue) \(String(describing: cb.error))") }
    return t
}

func runPread(_ n: Int, chunks: Int) -> Double {
    let regions = (0..<n).map { _ in fresh() }
    let c = E / chunks
    let base = buf.contents()
    let t0 = now()
    DispatchQueue.concurrentPerform(iterations: n * chunks) { j in
        let i = j / chunks, k = j % chunks
        let (s, off) = regions[i]
        let size = k == chunks - 1 ? E - c * k : c
        var done = 0
        while done < size {
            let r = pread(fds[s], base + i * E + k * c + done, size - done, off_t(off + k * c + done))
            if r <= 0 { fatalError("pread failed") }
            done += r
        }
    }
    return now() - t0
}

print("burst latency, ms (median / p90 of \(bursts) bursts), and GB/s at the median; expert = \(E) bytes, cold reads")
for n in [1, 2, 4, 8] {
    var line = String(format: "n=%d ", n)
    for (name, f) in [("io1", { runIO(n, chunks: 1) }), ("io4", { runIO(n, chunks: 4) }), ("io16", { runIO(n, chunks: 16) }),
                      ("pr1", { runPread(n, chunks: 1) }), ("pr4", { runPread(n, chunks: 4) })] {
        _ = f()                                             // warm the path (thread pool, queue)
        let t = (0..<bursts).map { _ in f() }.sorted()
        let med = t[t.count / 2], p90 = t[Int(Double(t.count) * 0.9)]
        line += String(format: " | %@ %.3f / %.3f (%.1f GB/s)", name, med, p90, Double(n * E) / med / 1e6)
    }
    print(line)
    fflush(stdout)
}
