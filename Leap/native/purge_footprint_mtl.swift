// PURGE-1 accounting (Leap, Tue 2026-11-03 15:2x; NewIdeas' finding for vm_allocate purgeable memory, checked here for the
// mechanism the engine would use: a shared MTLBuffer with setPurgeableState). phys_footprint (TASK_VM_INFO) and the
// lifetime peak (RUSAGE_INFO_V4) across: allocate 256 MB, touch, volatile, non-volatile, a second buffer touched while
// volatile. Build: swiftc -O purge_footprint_mtl.swift -o purge_footprint_mtl
import Darwin
import Foundation
import Metal
func footprintMB() -> (Double, Double) {
    var info = task_vm_info_data_t(); var n = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(n)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &n) } }
    var ru = rusage_info_v4()
    _ = withUnsafeMutablePointer(to: &ru) { p in p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
        proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) } }
    return (Double(info.phys_footprint) / 1048576, Double(ru.ri_lifetime_max_phys_footprint) / 1048576)
}
func show(_ s: String) { let (f, l) = footprintMB(); print(String(format: "%-44@ footprint %7.1f MB, lifetime peak %7.1f MB", s as NSString, f, l)) }
let dev = MTLCreateSystemDefaultDevice()!
show("start")
let a = dev.makeBuffer(length: 256 << 20, options: .storageModeShared)!
show("buffer A allocated (untouched)")
memset(a.contents(), 1, a.length); show("A touched (non-volatile)")
_ = a.setPurgeableState(.volatile); show("A volatile")
_ = a.setPurgeableState(.nonVolatile); show("A non-volatile again")
_ = a.setPurgeableState(.volatile); show("A volatile again")
let b = dev.makeBuffer(length: 256 << 20, options: .storageModeShared)!
_ = b.setPurgeableState(.volatile)
memset(b.contents(), 2, b.length); show("B made volatile, then touched")
_ = b.setPurgeableState(.nonVolatile); show("B non-volatile")
