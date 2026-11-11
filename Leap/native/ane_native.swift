// ANE-2 native harness (Leap, Mon 2026-11-02; Fable mission 18 item 1): times one stateful CoreML call the way an engine
// would make it, without Python. Inputs are preallocated fp16 MLMultiArrays reused every call; outputs go into
// preallocated output backings, so the logits are not copied out. After each call the host commits block i's context
// rows (the model's kx* / vx* outputs) into the K / V ring states at the ring position with MLState.withMultiArray
// (16 rows x 8 heads x 128 per state), as the engine would after the check's cut.
// Build: swiftc -O ane_native.swift -o ane_native
// Usage: ane_native <model.mlpackage | model.mlmodelc> [iterations] [rowsCommitted]
import CoreML
import Foundation

func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b - a) / 1e6 }

let args = CommandLine.arguments
let src = URL(fileURLWithPath: args[1])
let iters = args.count > 2 ? Int(args[2])! : 60
let commit = args.count > 3 ? Int(args[3])! : 11
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndNeuralEngine
var t0 = DispatchTime.now().uptimeNanoseconds
let compiled = src.pathExtension == "mlmodelc" ? src : try MLModel.compileModel(at: src)
let model = try MLModel(contentsOf: compiled, configuration: cfg)
print(String(format: "compile + load %.1f s", ms(t0, DispatchTime.now().uptimeNanoseconds) / 1000))

let desc = model.modelDescription
var feats: [String: MLFeatureValue] = [:]
for (name, d) in desc.inputDescriptionsByName {
    let shape = d.multiArrayConstraint!.shape
    let arr = try MLMultiArray(shape: shape, dataType: .float16)
    memset(arr.dataPointer, 0, arr.count * 2)
    feats[name] = MLFeatureValue(multiArray: arr)
}
let provider = try MLDictionaryFeatureProvider(dictionary: feats)
let opts = MLPredictionOptions()
var backings: [String: Any] = [:]
for (name, d) in desc.outputDescriptionsByName {
    backings[name] = try MLMultiArray(shape: d.multiArrayConstraint!.shape, dataType: .float16)
}
opts.outputBackings = backings
let state = model.makeState()
let stateNames = Array(desc.stateDescriptionsByName.keys).sorted()
print("inputs \(desc.inputDescriptionsByName.keys.sorted()), outputs \(desc.outputDescriptionsByName.keys.sorted()), states \(stateNames)")

var ring = 0
func commitRows(_ out: MLFeatureProvider) {
    for sname in stateNames {                       // K0..K4 <- kx0..kx4, V0..V4 <- vx0..vx4
        let oname = (sname.hasPrefix("K") ? "kx" : "vx") + sname.dropFirst()
        guard let rows = out.featureValue(for: oname)?.multiArrayValue else { continue }
        state.withMultiArray(for: sname) { dst in
            let W = dst.shape[2].intValue, nh = dst.shape[1].intValue, hd = dst.shape[3].intValue
            let ds = dst.strides.map { $0.intValue }, ss = rows.strides.map { $0.intValue }
            let dp = dst.dataPointer.assumingMemoryBound(to: UInt16.self)
            let sp = rows.dataPointer.assumingMemoryBound(to: UInt16.self)
            for h in 0..<nh {
                for r in 0..<commit {
                    let slot = (ring + r) % W
                    memcpy(dp + h * ds[1] + slot * ds[2], sp + h * ss[1] + r * ss[2], hd * 2)
                }
            }
        }
    }
    ring = (ring + commit) % 2048
}

for _ in 0..<5 {
    let o = try model.prediction(from: provider, using: state, options: opts)
    commitRows(o)
}
var tp: [Double] = [], tc: [Double] = []
for _ in 0..<iters {
    let a = DispatchTime.now().uptimeNanoseconds
    let o = try model.prediction(from: provider, using: state, options: opts)
    let b = DispatchTime.now().uptimeNanoseconds
    commitRows(o)
    let c = DispatchTime.now().uptimeNanoseconds
    tp.append(ms(a, b))
    tc.append(ms(b, c))
}
tp.sort(); tc.sort()
let n = tp.count
print(String(format: "predict median %.2f ms (p10 %.2f, p90 %.2f); host commit of %d rows x %d states median %.3f ms; total median %.2f ms",
             tp[n / 2], tp[n / 10], tp[n * 9 / 10], commit, stateNames.count, tc[n / 2], tp[n / 2] + tc[n / 2]))
if compiled != src { try? FileManager.default.removeItem(at: compiled) }
