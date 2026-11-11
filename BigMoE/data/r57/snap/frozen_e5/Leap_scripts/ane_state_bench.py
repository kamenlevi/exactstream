"""ANE-2c (Leap, Mon 2026-11-02; Fable mission 18 item 1): can the drafter's context cache live inside the Neural Engine
model (CoreML states, macOS 15) instead of being copied in as 10 x 4 MB inputs per call? Ten (1, 8, W, 128) fp16 caches
(5 layers x K / V, W = 2064 slots), 16 new rows written per call at a host-chosen ring position, then an attention-
shaped read (64 query rows per kv head: scores + softmax + values), per variant:
  input:   the caches are inputs (ANE-1's body: copied per call by the Python binding)
  scatter: the caches are states; write = state * keep + onehot^T @ new (keep / onehot are small inputs)
  slice:   the caches are states; write = slice_update at a dynamic int32 position input
Line for mission 18: the cache read + write should cost <= ~2 ms per call on the Neural Engine.
Usage: [BigMoE venv] ane_state_bench.py -> ~/Leap/data/ane_state_bench.txt"""
import os, sys, time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D = os.path.expanduser("~/Leap/data")
AD = os.path.join(D, "ane")
OUT = os.path.join(D, "ane_state_bench.txt")
NL, NKV, HD, W, NN, NQ = 5, 8, 128, 2064, 16, 64
f16, fp = np.float16, types.fp16


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def build(kind):
    names = [f"c{i}" for i in range(2 * NL)]
    ins = [mb.TensorSpec(shape=(1, NKV, NN, HD), dtype=fp), mb.TensorSpec(shape=(1, NKV, NQ, HD), dtype=fp)]
    if kind == "input":
        ins += [mb.TensorSpec(shape=(1, NKV, W - NN, HD), dtype=fp) for _ in names]
        args = ["new", "q"] + names
    elif kind == "scatter":
        ins += [mb.TensorSpec(shape=(1, 1, W, NN), dtype=fp), mb.TensorSpec(shape=(1, 1, W, 1), dtype=fp)]
        ins += [mb.StateTensorSpec(shape=(1, NKV, W, HD), dtype=fp) for _ in names]
        args = ["new", "q", "onehot", "keep"] + names
    else:
        ins += [mb.TensorSpec(shape=(1,), dtype=types.int32)]
        ins += [mb.StateTensorSpec(shape=(1, NKV, W, HD), dtype=fp) for _ in names]
        args = ["new", "q", "pos"] + names

    def f(**a):
        outs = []
        for j, n in enumerate(names):
            if kind == "input":
                c = mb.concat(values=[a[n], a["new"]], axis=2)
            else:
                c = mb.read_state(input=a[n])
                if kind == "scatter":
                    c = mb.add(x=mb.mul(x=c, y=a["keep"]), y=mb.matmul(x=a["onehot"], y=a["new"]))
                else:
                    z = mb.const(val=np.zeros(1, np.int32))
                    b = mb.concat(values=[z, z, a["pos"], z], axis=0)
                    e = mb.concat(values=[mb.const(val=np.array([1, NKV], np.int32)), mb.add(x=a["pos"], y=NN),
                                          mb.const(val=np.array([HD], np.int32))], axis=0)
                    c = mb.slice_update(x=c, update=a["new"], begin=b, end=e)
                mb.coreml_update_state(state=a[n], value=c)
            s = mb.softmax(x=mb.matmul(x=a["q"], y=c, transpose_y=True), axis=-1)
            outs.append(mb.matmul(x=s, y=c))
        return mb.add(x=outs[0], y=outs[1], name="out") if len(outs) == 2 else mb.concat(values=outs, axis=1, name="out")
    src = f"def _p({', '.join(args)}):\n    return f({', '.join(f'{x}={x}' for x in args)})\n"
    ns = {"f": f}
    exec(src, ns)
    prog = mb.program(input_specs=ins, opset_version=ct.target.macOS15)(ns["_p"])
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    p = os.path.join(AD, f"st_{kind}.mlpackage")
    m.save(p)
    return p


rng = np.random.default_rng(0)
say(f"ANE-2c {time.strftime('%a %H:%M')}: W {W} slots, {2 * NL} caches of {NKV * W * HD * 2 / 1e6:.1f} MB")
for kind in ("input", "scatter", "slice"):
    try:
        p = build(kind)
        m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        inp = {"new": rng.standard_normal((1, NKV, NN, HD)).astype(f16), "q": rng.standard_normal((1, NKV, NQ, HD)).astype(f16)}
        st = None
        if kind == "input":
            for i in range(2 * NL):
                inp[f"c{i}"] = rng.standard_normal((1, NKV, W - NN, HD)).astype(f16)
        else:
            st = m.make_state()
            if kind == "scatter":
                oh = np.zeros((1, 1, W, NN), f16)
                oh[0, 0, 100 + np.arange(NN), np.arange(NN)] = 1
                kp = np.ones((1, 1, W, 1), f16)
                kp[0, 0, 100:100 + NN] = 0
                inp["onehot"], inp["keep"] = oh, kp
            else:
                inp["pos"] = np.array([100], np.int32)
        call = (lambda: m.predict(inp)) if st is None else (lambda: m.predict(inp, state=st))
        for _ in range(5):
            call()
        ts = []
        for _ in range(30):
            t1 = time.perf_counter()
            call()
            ts.append(time.perf_counter() - t1)
        from collections import Counter
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
        c = Counter()
        for op in plan.model_structure.program.functions["main"].block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is not None:
                c[(op.operator_name.split(".")[-1], type(u.preferred_compute_device).__name__[2:-13])] += 1
        say(f"  {kind}: {1e3 * np.median(ts):.2f} ms (p90 {1e3 * np.percentile(ts, 90):.2f}); placement {dict(c)}")
        del m
    except Exception as e:
        say(f"  {kind}: FAILED {type(e).__name__}: {str(e)[:300]}")
    import shutil
    shutil.rmtree(os.path.join(AD, f"st_{kind}.mlpackage"), ignore_errors=True)
