"""nax_peak_i8.py (Leap, Tue 17 Nov 2026 08:3x; Fable B's dense question via the orchestrator 08:2x): is the neural
accelerator's INT8 rate much higher than its bf16 rate? M27-P (nax_peak.py, Mon 11-02) measured Metal 4 MPP matmul2d
cooperative tensors (the neural-accelerator path) with bf16 x bf16 -> f32 operands in registers: best 13.7-15.0 T MAC/s.
Here, in ONE run, interleaved per shape: bf16 x bf16 -> f32, half x half -> f32 and int8 x int8 -> int32, same shapes, 1-2
independent accumulator chains per simdgroup, operands in registers (no memory traffic, no decode).
Fable B's lines (IDEAS_1117b_dense.md): int8 / bf16 >= 2.5x -> PASS (route (a) on the dense pass is possible);
< 1.3x -> KILL. Rough (shared GPU, ratio within one run).
usage: [BigMoE venv] nax_peak_i8.py -> ~/Leap/data/nax_peak_i8.txt"""
import os, time
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/nax_peak_i8.txt")
L = []
def say(s):
    print(s, flush=True); L.append(s)
HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
TYPES = {"bf16": ("bfloat", "float", "bfloat({v})", "0.0f"), "fp16": ("half", "float", "half({v})", "0.0f"),
         "int8": ("int8_t", "int32_t", "int8_t({iv})", "0")}
def src(mm, nn, kk, ch, ty):
    TA, TC, lit, z0 = TYPES[ty]
    val = lambda j, expr: lit.format(v=f"{0.5 / (j + 1)}f + 0.01f * i", iv=f"(({j + 1} + i) & 7) - 3")
    decl = "\n".join(f"    auto cb{j} = op.template get_right_input_cooperative_tensor<{TA}, {TA}, {TC}>();" for j in range(ch))
    dest = "\n".join(f"    auto c{j} = op.template get_destination_cooperative_tensor<CA, CB, {TC}>();" for j in range(ch))
    init = "\n".join(f"        cb{j}[i] = {val(j, None)};" for j in range(ch))
    zero = " ".join(f"c{j}[i] = {z0};" for j in range(ch))
    run = "\n".join(f"        op.run(ca, cb{j}, c{j});" for j in range(ch))
    tot = " + ".join(f"float(c{j}[i])" for j in range(ch))
    alit = lit.format(v="0.001f * (lane + i)", iv="((lane + i) & 7) - 3")
    return f"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor({mm}, {nn}, {kk}, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<{TA}, {TA}, {TC}>();
{decl}
    using CA = metal::remove_addrspace_t<decltype(ca)>;
    using CB = metal::remove_addrspace_t<decltype(cb0)>;
{dest}
    for (int i = 0; i < (int)ca.get_capacity(); i++) ca[i] = {alit};
    for (int i = 0; i < (int)cb0.get_capacity(); i++) {{
{init}
    }}
    for (int i = 0; i < (int)c0.get_capacity(); i++) {{ {zero} }}
    for (int it = 0; it < ITER_C; ++it) {{
{run}
    }}
    float s = 0.0f;
    for (int i = 0; i < (int)c0.get_capacity(); i++) s += {tot};
    if (s == 12345.678f) out[0] = s;
"""
d = mx.zeros((1,), mx.float32)
ITER, NSG = 1024, 20 * 64
def timed(f, reps=9):
    for _ in range(2):
        mx.eval(f())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
say(f"NAX int8 vs bf16 / fp16 peak {time.strftime('%a %d %b %H:%M')} (rough, shared GPU; ratios within one run); {NSG} simdgroups "
    f"x {ITER} ops per chain, operands in registers")
res = {}
for (mm, nn, kk) in ((16, 32, 16), (16, 32, 32), (32, 32, 16), (32, 32, 32)):
    for ch in (1, 2):
        for ty in ("bf16", "fp16", "int8"):
            try:
                k = mx.fast.metal_kernel(name=f"nxi8_{mm}_{nn}_{kk}_{ch}_{ty}", input_names=["dummy"], output_names=["out"],
                                         source=src(mm, nn, kk, ch, ty), header=HDR)
                f = lambda: k(inputs=[d], template=[("ITER_C", ITER)], grid=(NSG * 32, 1, 1), threadgroup=(32 * 4, 1, 1),
                              output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0]
                t = timed(f)
                r = ITER * NSG * ch * mm * nn * kk / t / 1e12
                res[(mm, nn, kk, ch, ty)] = r
                say(f"  {mm}x{nn}x{kk} ch{ch} {ty}: {r:5.1f} T MAC/s ({t * 1e3:.2f} ms per launch)")
            except Exception as e:
                say(f"  {mm}x{nn}x{kk} ch{ch} {ty}: FAILED {str(e).splitlines()[0][:150] if str(e) else repr(e)[:150]}")
best = {ty: max([v for k_, v in res.items() if k_[4] == ty], default=float("nan")) for ty in TYPES}
say(f"best: bf16 {best['bf16']:.1f}, fp16 {best['fp16']:.1f}, int8 {best['int8']:.1f} T MAC/s -> int8 / bf16 "
    f"{best['int8'] / best['bf16']:.2f}x; same-shape ratios: " + ", ".join(
        f"{k_[0]}x{k_[1]}x{k_[2]} ch{k_[3]} {res[k_] / res[k_[:4] + ('bf16',)]:.2f}x" for k_ in res if k_[4] == "int8" and k_[:4] + ("bf16",) in res))
r = best["int8"] / best["bf16"]
say(f"VERDICT (Fable B's lines): int8 / bf16 = {r:.2f}x -> {'PASS (>= 2.5x)' if r >= 2.5 else 'KILL (< 1.3x)' if r < 1.3 else 'BETWEEN'}")
open(OUT, "a").write("\n".join(L) + "\n")
