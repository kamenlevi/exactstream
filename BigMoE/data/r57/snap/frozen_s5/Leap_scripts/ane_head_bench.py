"""ANE-2a (Leap, Mon 2026-11-02; Fable mission 18 item 1, ~/FableGod/ANE_DRAFTER.md): does the Neural Engine stream
4-bit weights natively, and what do 45 rows (three batched 15-row candidates) cost? The drafter's head (the target's
lm_head, 248,320 x 2560) as CoreML block-max heads (bmax only, as ANE-1's headm):
- h16: fp16 (1.27 GB); h8: int8 symmetric per channel (0.64 GB, ANE-1's); h4: the target's own 4-bit g64 head, exact
  (uint4 data + fp16 scale and offset per 64-block via constexpr_blockwise_shift_scale, offset = -bias / scale; 0.36 GB);
at 15 and 45 rows. If h4 runs at ~half h8's time, the Neural Engine reads 4-bit weights natively and the head drops to
~4 ms; if h4 ~ h8 ~ h16, the weights are expanded at load time and bytes are not the lever.
Usage: [BigMoE venv] ane_head_bench.py -> ~/Leap/data/ane_head_bench.txt"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D = os.path.expanduser("~/Leap/data")
AD = os.path.join(D, "ane")
TGT = os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors")
H, V, CH = 2560, 248320, 12416
OUT = os.path.join(D, "ane_head_bench.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


w = mx.load(TGT)
Wq, Sc, Bi = w["language_model.lm_head.weight"], w["language_model.lm_head.scales"], w["language_model.lm_head.biases"]
LM16 = np.array(mx.dequantize(Wq, Sc, Bi, group_size=64, bits=4).astype(mx.float16))
by = np.array(Wq).view(np.uint8)                                   # (V, H / 2): low nibble = even element
Q4 = np.empty((V, H), np.uint8)
Q4[:, 0::2] = by & 0xF
Q4[:, 1::2] = by >> 4
del by
sc = np.array(Sc.astype(mx.float32))
bi = np.array(Bi.astype(mx.float32))
del w, Wq, Sc, Bi
say(f"ANE-2a {time.strftime('%a %H:%M')}: head scales == 0: {int((sc == 0).sum())}; |scale| range {np.abs(sc).min():.2e}-{np.abs(sc).max():.2e}")
SC16 = sc.astype(np.float16)
OF16 = (-bi / np.where(sc == 0, 1, sc)).astype(np.float16)
chk = (SC16[:4].astype(np.float32).repeat(64, 1) * (Q4[:4].astype(np.float32) - OF16[:4].astype(np.float32).repeat(64, 1)))
say(f"  exact-4-bit rebuild vs MLX's dequantized head, first 4 rows: max |diff| {np.abs(chk - LM16[:4].astype(np.float32)).max():.2e} "
    f"(max |w| {np.abs(LM16[:4]).max():.3f})")
CFG8 = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=100000))


def build(kind, rows):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, rows, H), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def prog(hid):
        bm = []
        for c in range(V // CH):
            r = slice(c * CH, (c + 1) * CH)
            if kind == "h4":
                wt = mb.constexpr_blockwise_shift_scale(data=Q4[r].astype(types.np_uint4_dtype), scale=SC16[r], offset=OF16[r])
            else:
                wt = LM16[r]
            l_ = mb.linear(x=hid, weight=wt)
            bm.append(mb.reduce_max(x=mb.reshape(x=l_, shape=[1, rows, CH // 128, 128]), axes=[-1], keep_dims=False))
        return mb.concat(values=bm, axis=-1, name="bmax")
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    if kind == "h8":
        m = cto.coreml.linear_quantize_weights(m, config=CFG8)
    p = os.path.join(AD, f"hb_{kind}_{rows}.mlpackage")
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


rng = np.random.default_rng(0)
for kind in ("h8", "h4", "h16"):
    for rows in (15, 45):
        t0 = time.time()
        p = build(kind, rows)
        gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e9
        m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        tb = time.time() - t0
        x = {"hid": (rng.standard_normal((1, rows, H)) * 0.5).astype(np.float16)}
        for _ in range(5):
            m.predict(x)
        ts = []
        for _ in range(30):
            t1 = time.perf_counter()
            m.predict(x)
            ts.append(time.perf_counter() - t1)
        med = 1e3 * float(np.median(ts))
        lin = {k: v for k, v in placement(m).items() if k[0] in ("linear", "constexpr_blockwise_shift_scale")}
        say(f"  {kind} rows {rows}: {med:.2f} ms (p90 {1e3 * np.percentile(ts, 90):.2f}); package {gb:.3f} GB -> "
            f"{gb / med * 1e3:.0f} GB/s of stored bytes; built + loaded in {tb:.0f} s; placement {lin}")
        del m
