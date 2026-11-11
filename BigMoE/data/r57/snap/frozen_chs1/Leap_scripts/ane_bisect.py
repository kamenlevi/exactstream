"""ANE-2 bisect (Leap, Mon 2026-11-02): which construct of ane_v2.py's graph makes the Neural Engine fail at run time
(status 0x1d, Program Inference error) although CoreML places every op on it. One tiny CoreML model per construct, run
once on CPU_AND_NE. Usage: [BigMoE venv] ane_bisect.py -> stdout"""
import os, shutil
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

AD = os.path.expanduser("~/Leap/data/ane")
f16, fp = np.float16, types.fp16
NKV, HD, W, NC, S = 8, 128, 2064, 16, 48
TK = W + S
rng = np.random.default_rng(0)
R = lambda *s: rng.standard_normal(s).astype(f16)


def run(tag, specs, fn, inputs, stateful=False):
    args = [f"a{i}" for i in range(len(specs))]
    src = f"def _p({', '.join(args)}):\n    return fn({', '.join(args)})\n"
    ns = {"fn": fn}
    exec(src, ns)
    try:
        prog = mb.program(input_specs=specs, opset_version=ct.target.macOS15)(ns["_p"])
        m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        p = os.path.join(AD, f"bis_{tag}.mlpackage")
        m.save(p)
        ml = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        in_names = [i.name for i in ml.get_spec().description.input]
        feed = dict(zip(in_names, inputs.values()))
        if stateful:
            st = ml.make_state()
            ml.predict(feed, state=st)
            ml.predict(feed, state=st)
        else:
            ml.predict(feed)
        from collections import Counter
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(path=ml.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
        c = Counter()
        for op in plan.model_structure.program.functions["main"].block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is not None:
                c[type(u.preferred_compute_device).__name__[2:-13]] += 1
        print(f"{tag}: OK {dict(c)}", flush=True)
    except Exception as e:
        print(f"{tag}: FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
    shutil.rmtree(os.path.join(AD, f"bis_{tag}.mlpackage"), ignore_errors=True)


sp = lambda *s: mb.TensorSpec(shape=s, dtype=fp)
ss = lambda *s: mb.StateTensorSpec(shape=s, dtype=fp)
oh = np.zeros((1, 1, NC, W), f16)
oh[0, 0, np.arange(NC), np.arange(NC)] = 1
keep = np.ones((1, 1, 1, W), f16)
keep[..., :NC] = 0

# T1: V^T state written by v (1,8,128,16) @ onehot (1,1,16,W)
def t1(v, o, k, st):
    c = mb.add(x=mb.mul(x=mb.read_state(input=st), y=k), y=mb.matmul(x=v, y=o))
    mb.coreml_update_state(state=st, value=c)
    return mb.reduce_sum(x=c, axes=[3], keep_dims=False, name="out")
run("T1_vt_state_write", [sp(1, NKV, HD, NC), sp(1, 1, NC, W), sp(1, 1, 1, W), ss(1, NKV, HD, W)], t1,
    {"v": R(1, NKV, HD, NC), "o": oh, "k": keep}, stateful=True)

# T2: K state written by onehot^T (1,1,W,16) @ k^T (1,8,16,128)
def t2(k, o, kp, st):
    c = mb.add(x=mb.mul(x=mb.read_state(input=st), y=kp), y=mb.matmul(x=o, y=k))
    mb.coreml_update_state(state=st, value=c)
    return mb.reduce_sum(x=c, axes=[2], keep_dims=False, name="out")
run("T2_k_state_write", [sp(1, NKV, NC, HD), sp(1, 1, W, NC), sp(1, 1, W, 1), ss(1, NKV, W, HD)], t2,
    {"k": R(1, NKV, NC, HD), "o": oh.transpose(0, 1, 3, 2).copy(), "kp": keep.transpose(0, 1, 3, 2).copy()}, stateful=True)

# T3: key-major attention: scores (1,8,TK,192) = K (1,8,TK,128) @ q (1,8,128,192); softmax over axis 2; out = V^T @ P
def t3(K, q, VT, mask):
    p = mb.softmax(x=mb.add(x=mb.matmul(x=K, y=q), y=mask), axis=2)
    return mb.matmul(x=VT, y=p, name="out")
run("T3_keymajor_attn", [sp(1, NKV, TK, HD), sp(1, NKV, HD, 4 * S), sp(1, NKV, HD, TK), sp(1, 1, TK, 4 * S)], t3,
    {"K": R(1, NKV, TK, HD), "q": R(1, NKV, HD, 4 * S), "VT": R(1, NKV, HD, TK), "mask": np.zeros((1, 1, TK, 4 * S), f16)})

# T3b: the same with the softmax over the last axis (query-major scores)
def t3b(q, KT, V, mask):
    p = mb.softmax(x=mb.add(x=mb.matmul(x=q, y=KT), y=mask), axis=3)
    return mb.matmul(x=p, y=V, name="out")
run("T3b_querymajor_attn", [sp(1, NKV, 4 * S, HD), sp(1, NKV, HD, TK), sp(1, NKV, TK, HD), sp(1, 1, 4 * S, TK)], t3b,
    {"q": R(1, NKV, 4 * S, HD), "KT": R(1, NKV, HD, TK), "V": R(1, NKV, TK, HD), "mask": np.zeros((1, 1, 4 * S, TK), f16)})

# T4: layer norm over axis 2 of (1,8,256,192) and over axis 1 of (1,5120,1,48)
def t4(x, y):
    a = mb.layer_norm(x=mb.concat(values=[x, mb.mul(x=x, y=f16(-1))], axis=2), axes=[2], epsilon=f16(1e-6))
    b = mb.layer_norm(x=mb.concat(values=[y, mb.mul(x=y, y=f16(-1))], axis=1), axes=[1], epsilon=f16(1e-6))
    return mb.identity(x=a, name="a"), mb.identity(x=b, name="b")
run("T4_layernorm_axes", [sp(1, NKV, HD, 4 * S), sp(1, 2560, 1, S)], t4, {"x": R(1, NKV, HD, 4 * S), "y": R(1, 2560, 1, S)})

# T5: two-way broadcast add (1,160,1,48) + const (1,160,16,1) and the shift on the last axis
def t5(d, n):
    k = mb.add(x=d, y=R(1, 160, 16, 1))
    z = np.zeros((1, 160, 16, 1), f16)
    s = mb.concat(values=[z, mb.slice_by_index(x=n, begin=[0, 0, 0, 0], end=[1, 160, 16, S - 1])], axis=3)
    return mb.add(x=mb.mul(x=n, y=k), y=s, name="out")
run("T5_broadcast_conv", [sp(1, 160, 1, S), sp(1, 160, 16, S)], t5, {"d": R(1, 160, 1, S), "n": R(1, 160, 16, S)})

# T6: big-channel 1x1 conv (fc: 12800 -> 2560 on 16 rows)
def t6(x):
    return mb.conv(x=x, weight=(R(2560, 12800, 1, 1) * f16(0.01)).astype(f16), name="out")
run("T6_fc_conv", [sp(1, 12800, 1, NC)], t6, {"x": R(1, 12800, 1, NC)})
