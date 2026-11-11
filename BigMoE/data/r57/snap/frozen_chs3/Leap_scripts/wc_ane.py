"""WHOLE-CHIP, the Neural Engine arm (Leap, Tue 2026-11-03 17:1x; TABLE_350 'WHOLE-CHIP'): one Core ML call computes
32 experts' gate / up / down (2560 -> 640 -> 2560, SiLU) for all 16 rows of a check and sums them with per-row scores.
Random weights of the real shapes (latency does not depend on the values). Two encodings:
  a8: int8 symmetric per channel (ANE-2's: ~79 G weights/s on the head);
  a4: 4-bit LUT per 16-channel group (constexpr_lut_to_dense; the ANE-native 4-bit form, if MLComputePlan places it).
Usage: [BigMoE venv] wc_ane.py build            -> data/wc/ane_e32_{a8,a4}.mlpackage, placement + time alone
       [BigMoE venv] wc_ane.py build4 K SEED     -> data/wc/ane_e{K}_a4_s{SEED}.mlpackage (WHOLE-CHIP-2: K experts, other weights)
       [BigMoE venv] wc_ane.py run PKG[,PKG...] SECONDS -> "ready", then one line a second: "<unix time> calls_per_s X
                     exp_per_ms Y"; several packages are called in turn (no weights reused within that many calls)
"""
import os, signal, sys, time
import numpy as np
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

HD, ID, S = 2560, 640, 16
NEXP = int(os.environ.get("WC_NEXP", "32"))
WD = os.path.expanduser("~/Leap/data/wc")


def build(kind, seed=1, name=None):
    rng = np.random.default_rng(seed)

    def weight(o, i):
        if kind == "a4":
            idx = rng.integers(0, 16, (o, i)).astype(types.np_uint4_dtype)
            lut = (rng.standard_normal((o // 16, 1, 16, 1)) * 0.03).astype(np.float16)
            return mb.constexpr_lut_to_dense(indices=idx, lut=lut)
        return (rng.standard_normal((o, i)) * 0.02).astype(np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, S, HD), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, S, NEXP), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def prog(x, sc):
        acc = None
        for e in range(NEXP):
            g = mb.linear(x=x, weight=weight(ID, HD))
            u = mb.linear(x=x, weight=weight(ID, HD))
            h = mb.mul(x=mb.silu(x=g), y=u)
            y = mb.linear(x=h, weight=weight(HD, ID))
            w = mb.slice_by_index(x=sc, begin=[0, 0, e], end=[1, S, e + 1])
            t = mb.mul(x=y, y=w)
            acc = t if acc is None else mb.add(x=acc, y=t)
        return mb.identity(x=acc, name="out")

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    if kind == "a8":
        m = cto.coreml.linear_quantize_weights(m, config=cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                             granularity="per_channel", weight_threshold=100000)))
    p = os.path.join(WD, name or f"ane_e32_{kind}.mlpackage")
    m.save(p)
    return p


def placement(m):
    from collections import Counter
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    c = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is not None:
            c[(op.operator_name.split(".")[-1], type(u.preferred_compute_device).__name__[2:-13])] += 1
    return dict(c)


def inputs(nexp=None):
    nexp = nexp or NEXP
    rng = np.random.default_rng(2)
    sc = rng.random((1, S, nexp)).astype(np.float32)
    return {"x": (rng.standard_normal((1, S, HD)) * 0.5).astype(np.float16), "sc": (sc / sc.sum(-1, keepdims=True)).astype(np.float16)}


if sys.argv[1] == "build":
    os.makedirs(WD, exist_ok=True)
    for kind in ("a8", "a4"):
        t0 = time.time()
        p = build(kind)
        gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e9
        m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        tb = time.time() - t0
        x = inputs()
        for _ in range(5):
            m.predict(x)
        ts = []
        for _ in range(40):
            t1 = time.perf_counter()
            m.predict(x)
            ts.append(time.perf_counter() - t1)
        med = float(np.median(ts))
        pl = {k: v for k, v in placement(m).items() if k[0] in ("linear", "constexpr_lut_to_dense", "constexpr_affine_dequantize",
                                                              "constexpr_blockwise_shift_scale", "silu", "mul", "add")}
        print(f"{kind}: {1e3 * med:.2f} ms per call (p90 {1e3 * np.percentile(ts, 90):.2f}) = {NEXP / med / 1e3:.1f} experts/ms, "
              f"{NEXP * 3 * HD * ID / med / 1e9:.0f} G weights/s; package {gb:.3f} GB; built + loaded in {tb:.0f} s; placement {pl}",
              flush=True)
        del m
elif sys.argv[1] == "build4":
    os.makedirs(WD, exist_ok=True)
    K, seed = int(sys.argv[2]), int(sys.argv[3])
    NEXP = K
    t0 = time.time()
    p = build("a4", seed, f"ane_e{K}_a4_s{seed}.mlpackage")
    m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = inputs(K)
    for _ in range(5):
        m.predict(x)
    ts = []
    for _ in range(30):
        t1 = time.perf_counter(); m.predict(x); ts.append(time.perf_counter() - t1)
    on_ane = all(v == "NeuralEngine" for (op, v) in placement(m).keys() if op == "linear")
    print(f"{os.path.basename(p)}: {1e3 * np.median(ts):.3f} ms per call (Python), linears on the Neural Engine: {on_ane}; "
          f"built + loaded in {time.time() - t0:.0f} s", flush=True)
else:
    ps, secs = sys.argv[2].split(","), float(sys.argv[3])
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    ms = [ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE) for p in ps]
    nexp = ms[0].get_spec().description.input[1].type.multiArrayType.shape[-1]
    x = inputs(nexp)
    for m in ms:
        for _ in range(3):
            m.predict(x)
    print("ready", flush=True)
    t_end = time.time() + secs
    last, n, i = time.time(), 0, 0
    while not stop and time.time() < t_end:
        ms[i].predict(x)
        i = (i + 1) % len(ms)
        n += 1
        t = time.time()
        if t - last >= 1.0:
            print(f"{t:.3f} calls_per_s {n / (t - last):.1f} exp_per_ms {nexp * n / (t - last) / 1e3:.3f}", flush=True)
            last, n = t, 0
