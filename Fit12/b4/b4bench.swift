// Fit12 B4 (2026-10-25; NewIdeas' bet B4): per-call fixed cost of expert reads, today's loader path vs a page-aligned
// direct read vs Metal fast resource loading (MTLIOCommandQueue). READ-ONLY on ~/BigMoE/models/bare/records3 (B12's
// MSB-3 base part per expert: 2,162,688 B = 132 x 16 KB pages, 16 KB-aligned in the file; pieces 614,400 / 51,200 B).
//   A bounce  = v3k's default (load_jobs -> read_expert, rec_fmt 1): dispatch_apply over the misses, ONE pread (F_NOCACHE
//               fd) of the part into a 16 KB-aligned bounce, then 9 memcpys into 9 per-piece pools (Metal buffers).
//   B direct  = a kernel-native slot (the record image is the slot): one pread straight into a 16 KB-aligned Metal slot.
//   C io1     = MTLIOCommandQueue (concurrent, high priority): ONE IO command buffer with N loads, commit, wait.
//   D ioN     = N IO command buffers (one load each), all committed, then waited.
// One call = N distinct experts of one random layer (as one loader call per layer); every expert is read once in the run
// (none twice) and only experts with no page in the page cache (mincore) are used. Paths interleave per rep in random
// order; 1 ms gap between calls. Output: per path x N median / p10 / p90 ms, GB/s, CPU ms; fit median = a + b N.
// Build: xcrun swiftc -O -o b4bench b4bench.swift      Run: ./b4bench [reps]
import Foundation
import Metal
import Darwin

setvbuf(stdout, nil, _IOLBF, 0)
_ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
let reps = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 50
let Ns = [1, 2, 4, 8, 12, 16, 24]
let paths = ["A bounce", "B direct", "C io1", "D ioN"]
let root = NSHomeDirectory() + "/BigMoE/models/bare/records3"
let idx = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: root + "/index.json"))) as! [String: Any]
let lays = idx["layers"] as! [String: [String: Any]]
let NL = 48, NE = 512, PG = 16384
var files = [String](), recsz = [Int](), pOff = [Int](), pSz = [Int]()
var MSB = 0
for l in 0..<NL {
  let d = lays[String(l)]!
  files.append(root + "/" + (d["file"] as! String)); recsz.append(d["recsz"] as! Int)
  let m = d["msb_sz"] as! Int, po = d["msb_piece_off"] as! [Int], ps = d["msb_piece_size"] as! [Int]
  if l == 0 { MSB = m; pOff = po; pSz = ps } else { precondition(m == MSB && po == pOff && ps == pSz, "layer \(l) layout differs") }
}
precondition(MSB % PG == 0 && recsz.allSatisfy { $0 % PG == 0 })
let fds = files.map { f -> Int32 in let fd = open(f, O_RDONLY); precondition(fd >= 0); _ = fcntl(fd, F_NOCACHE, 1); return fd }
let fsize = recsz.map { $0 * NE }
let maps = (0..<NL).map { l -> UnsafeMutableRawPointer in
  let p = mmap(nil, fsize[l], PROT_READ, MAP_SHARED, fds[l], 0)
  precondition(p != nil && p != UnsafeMutableRawPointer(bitPattern: -1), "mmap failed"); return p! }
var vec = [CChar](repeating: 0, count: MSB / PG)
func residentPages(_ l: Int, _ e: Int) -> Int {
  _ = mincore(maps[l] + e * recsz[l], MSB, &vec)
  return vec.reduce(0) { $0 + Int($1 & 1) }
}
// cold experts per layer, shuffled
var rng = SystemRandomNumberGenerator()
var cold = (0..<NL).map { l in (0..<NE).filter { residentPages(l, $0) == 0 }.shuffled(using: &rng) }
let nCold = cold.reduce(0) { $0 + $1.count }
print("store: \(NL) layers x \(NE) experts, base part \(MSB) B; cold (no page resident): \(nCold) of \(NL * NE)")
let need = reps * Ns.reduce(0, +) * paths.count + 80
precondition(nCold >= need, "not enough cold experts: \(nCold) < \(need)")

let dev = MTLCreateSystemDefaultDevice()!
let NMAX = Ns.max()!
let poolsA = pSz.map { dev.makeBuffer(length: $0 * NMAX, options: .storageModeShared)! }
let poolB = dev.makeBuffer(length: MSB * NMAX, options: .storageModeShared)!
let poolC = dev.makeBuffer(length: MSB * NMAX, options: .storageModeShared)!
for b in poolsA + [poolB, poolC] { memset(b.contents(), 1, b.length) }
let bounce = (0..<NMAX).map { _ -> UnsafeMutableRawPointer in
  var p: UnsafeMutableRawPointer? = nil; precondition(posix_memalign(&p, PG, MSB) == 0); memset(p!, 1, MSB); return p! }
let qd = MTLIOCommandQueueDescriptor()
qd.type = .concurrent; qd.priority = .high; qd.maxCommandBufferCount = 64
let ioq = try! dev.makeIOCommandQueue(descriptor: qd)
let fh = files.map { try! dev.makeIOFileHandle(url: URL(fileURLWithPath: $0)) }

func preadFull(_ fd: Int32, _ dst: UnsafeMutableRawPointer, _ n: Int, _ off: Int) {
  var done = 0
  while done < n { let r = pread(fd, dst + done, n - done, off_t(off + done)); precondition(r > 0, "pread failed"); done += r }
}
func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
func cpuUs() -> Int64 { var ru = rusage(); getrusage(RUSAGE_SELF, &ru)
  return Int64(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1_000_000 + Int64(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) }
func take(_ n: Int) -> (Int, [Int]) {          // a random layer with n unused cold experts
  while true { let l = Int.random(in: 0..<NL); if cold[l].count >= n { let e = Array(cold[l].suffix(n)); cold[l].removeLast(n); return (l, e) } }
}
var usedBy = [String: [(Int, Int)]]()
func call(_ path: String, _ n: Int) -> (Double, Double) {
  let (l, es) = take(n)
  usedBy[path, default: []] += es.map { (l, $0) }
  let fd = fds[l], rs = recsz[l]
  let c0 = cpuUs(), t0 = now()
  switch path {
  case "A bounce":
    DispatchQueue.concurrentPerform(iterations: n) { j in
      let b = bounce[j]
      preadFull(fd, b, MSB, es[j] * rs)
      for i in 0..<pSz.count { memcpy(poolsA[i].contents() + j * pSz[i], b + pOff[i], pSz[i]) }
    }
  case "B direct":
    DispatchQueue.concurrentPerform(iterations: n) { j in preadFull(fd, poolB.contents() + j * MSB, MSB, es[j] * rs) }
  case "C io1":
    let cb = ioq.makeCommandBuffer()
    for j in 0..<n { cb.load(poolC, offset: j * MSB, size: MSB, sourceHandle: fh[l], sourceHandleOffset: es[j] * rs) }
    cb.commit(); cb.waitUntilCompleted()
    precondition(cb.status == .complete, "io1 status \(cb.status.rawValue)")
  default:
    let cbs = (0..<n).map { j -> MTLIOCommandBuffer in
      let cb = ioq.makeCommandBuffer()
      cb.load(poolC, offset: j * MSB, size: MSB, sourceHandle: fh[l], sourceHandleOffset: es[j] * rs); cb.commit(); return cb }
    for cb in cbs { cb.waitUntilCompleted(); precondition(cb.status == .complete, "ioN status \(cb.status.rawValue)") }
  }
  let t1 = now(), c1 = cpuUs()
  // content check: slot n-1 of this path == a fresh F_NOCACHE pread of the same bytes (after timing)
  let ref = bounce[NMAX - 1]
  preadFull(fd, ref, MSB, es[n - 1] * rs)
  let ok: Bool
  switch path {
  case "A bounce": ok = (0..<pSz.count).allSatisfy { memcmp(poolsA[$0].contents() + (n - 1) * pSz[$0], ref + pOff[$0], pSz[$0]) == 0 }
  case "B direct": ok = memcmp(poolB.contents() + (n - 1) * MSB, ref, MSB) == 0
  default: ok = memcmp(poolC.contents() + (n - 1) * MSB, ref, MSB) == 0
  }
  precondition(ok, "\(path): slot content differs from the file")
  return (Double(t1 - t0) / 1e6, Double(c1 - c0) / 1e3)
}
for _ in 0..<5 { for p in paths { _ = call(p, 4) } }      // warm-up (SSD power state, IO queue), not recorded
var res = [String: [Int: [(Double, Double)]]]()
let tStart = now()
for _ in 0..<reps {
  for n in Ns.shuffled() { for p in paths.shuffled() { res[p, default: [:]][n, default: []].append(call(p, n)); usleep(1000) } }
}
print(String(format: "run %.1f s, %d calls", Double(now() - tStart) / 1e9, reps * Ns.count * paths.count))
func q(_ a: [Double], _ f: Double) -> Double { let s = a.sorted(); return s[min(s.count - 1, Int(f * Double(s.count - 1) + 0.5))] }
print("path | N | median ms | p10 | p90 | GB/s at median | CPU ms / call")
var fit = [String: (Double, Double)]()
for p in paths {
  var xs = [Double](), ys = [Double]()
  for n in Ns {
    let r = res[p]![n]!, t = r.map { $0.0 }, c = r.map { $0.1 }
    let m = q(t, 0.5)
    xs.append(Double(n)); ys.append(m)
    print(String(format: "%@ | %d | %.3f | %.3f | %.3f | %.2f | %.3f", p, n, m, q(t, 0.1), q(t, 0.9), Double(n * MSB) / m / 1e6, q(c, 0.5)))
  }
  let mx: Double = xs.reduce(0, +) / Double(xs.count)
  let my: Double = ys.reduce(0, +) / Double(ys.count)
  var sxy = 0.0, sxx = 0.0
  for i in 0..<xs.count { let dx: Double = xs[i] - mx; sxy += dx * (ys[i] - my); sxx += dx * dx }
  let b: Double = sxy / sxx
  fit[p] = (my - b * mx, b)
}
print("path | fixed a (ms / call) | b (ms / expert) | marginal GB/s | a vs A")
for p in paths { let (a, b) = fit[p]!
  print(String(format: "%@ | %.3f | %.4f | %.2f | %.2fx", p, a, b, Double(MSB) / b / 1e6, a / fit["A bounce"]!.0)) }
// page-cache side effect: experts each path read that are now resident
for p in paths { let u = usedBy[p] ?? []
  print("\(p): \(u.filter { residentPages($0.0, $0.1) > 0 }.count) of \(u.count) experts it read now have pages in the page cache") }
