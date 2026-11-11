"""M27-C, the int8 x int8 matrix-unit probe (Leap, Mon 2026-11-02 22:4x; pre-registered in ~/Leap/TABLE_350.md 'M27-C';
EXPERT_KERNEL.md design C's gate; orchestrator 22:5x): does mpp::tensor_ops::matmul2d take int8 x int8 operands (with an
int32 or fp32 destination), and at what rate against bf16 x bf16 -> fp32, at 16 x 32 x 32 (NAX6's op)? M27-P's
register-operand loop (no memory traffic), 1 / 2 / 4 independent accumulators (each its own B and destination, A shared),
threadgroups of 4 simdgroups, 1,280 simdgroups x 1,024 ops per chain, median of 11; a calibration run of the bf16 form
first. A pair that does not build = no such mode on this SDK.
Lines: PASS an int8 mode at >= 1.6x bf16's best rate; KILL no int8 mode or < 1.2x; BETWEEN otherwise.
Usage: [BigMoE venv] nax_int8_probe.py -> ~/Leap/data/nax_int8_probe.txt"""
import os, time
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/nax_int8_probe.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
_HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
def src(ta, tb, tc, ch, init_a, init_b):
    decl = "\n".join(f"    auto cb{j} = op.template get_right_input_cooperative_tensor<{ta}, {tb}, {tc}>();" for j in range(ch))
    dest = "\n".join(f"    auto c{j} = op.template get_destination_cooperative_tensor<CA, CB, {tc}>();" for j in range(ch))
    init = "\n".join(f"        cb{j}[i] = {init_b.format(j=j)};" for j in range(ch))
    zero = " ".join(f"c{j}[i] = {tc}(0);" for j in range(ch))
    run = "\n".join(f"        op.run(ca, cb{j}, c{j});" for j in range(ch))
    tot = " + ".join(f"float(c{j}[i])" for j in range(ch))
    tc0 = tc
    return f"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<{ta}, {tb}, {tc}>();
{decl}
    using CA = metal::remove_addrspace_t<decltype(ca)>;
    using CB = metal::remove_addrspace_t<decltype(cb0)>;
{dest}
    for (int i = 0; i < (int)ca.get_capacity(); i++) ca[i] = {init_a};
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
TYPES = {
    "bf16 x bf16 -> fp32": ("bfloat", "bfloat", "float", "bfloat(0.001f * (lane + i))", "bfloat(0.5f / ({j} + 1) + 0.01f * i)"),
    "int8 x int8 -> int32": ("int8_t", "int8_t", "int32_t", "int8_t((lane + i) & 7)", "int8_t((i + {j}) & 7)"),
    "uint8 x uint8 -> int32": ("uint8_t", "uint8_t", "int32_t", "uint8_t((lane + i) & 7)", "uint8_t((i + {j}) & 7)"),
    "half x half -> fp32": ("half", "half", "float", "half(0.001h * (lane + i))", "half(0.5h / ({j} + 1) + 0.01h * i)"),
}
d = mx.zeros((1,), mx.float32)
ITER, NSG = 1024, 20 * 64
say(f"M27-C int8 x int8 probe {time.strftime('%a %H:%M')} ({os.environ.get('M27C_COND', 'shared GPU, rough')}); 16x32x32, "
    f"{NSG} simdgroups x {ITER} ops per chain, threadgroups of 4 simdgroups")
def timed(f, reps=11):
    for _ in range(3):
        mx.eval(f())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
best = {}
for name, (ta, tb, tc, ia, ib) in TYPES.items():
    for ch in (1, 2, 4):
        tag = f"{name} ch{ch}"
        try:
            k = mx.fast.metal_kernel(name=f"m27c_{ta}_{tb}_{tc}_{ch}", input_names=["dummy"], output_names=["out"],
                                     source=src(ta, tb, tc, ch, ia, ib), header=_HDR)
            t = timed(lambda: k(inputs=[d], template=[("ITER_C", ITER)], grid=(NSG * 32, 1, 1), threadgroup=(128, 1, 1),
                                output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0])
        except Exception as ex:
            msg = str(ex)
            err = [l for l in msg.splitlines() if "error" in l.lower()]
            say(f"  {tag}: does not build ({(err[0] if err else msg.splitlines()[0])[:220]})")
            break
        n = ITER * NSG * ch
        rate = n / t / 1e9
        best[name] = max(best.get(name, 0.0), rate)
        say(f"  {tag}: {rate:.3f} G op/s = {n * 16384 / t / 1e12:.1f} T MAC/s ({t * 1e3:.2f} ms)")
TGSRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint32_t tbuf[4 * 32 * 8];
    threadgroup uint32_t* tb = tbuf + sg * (32 * 8);
    for (int i = 0; i < 8; ++i) tb[lane * 8 + i] = (lane * 2654435761u + i * 40503u) & 0x77777777u;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TB = tensor<threadgroup BT_C, dextents<int32_t, 2>, tensor_inline>;
    auto ca = op.template get_left_input_cooperative_tensor<AT_C, BT_C, DT_C>();
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, DT_C>();
    for (int i = 0; i < (int)ca.get_capacity(); i++) ca[i] = AINIT_C;
    for (int i = 0; i < (int)cg.get_capacity(); i++) cg[i] = DT_C(0);
    for (int it = 0; it < ITER_C; ++it) {
        TB tbt((threadgroup uchar*)tb + 16 * (it & 1), dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
        op.run(ca, tbt, cg);
    }
    float s = 0.0f;
    for (int i = 0; i < (int)cg.get_capacity(); i++) s += float(cg[i]);
    if (s == 12345.678f) out[0] = s;
"""
TGTYPES = {
    "bf16 x uint4(tg) -> fp32 (NAX6's form)": ("bfloat", "uint4b_format", "float", "bfloat(0.001f * (lane + i))"),
    "int8 x int4(tg) -> int32": ("int8_t", "int4b_format", "int32_t", "int8_t((lane + i) & 7)"),
    "uint8 x uint4(tg) -> int32": ("uint8_t", "uint4b_format", "int32_t", "uint8_t((lane + i) & 7)"),
}
tgbest = {}
for name, (at, bt, dt, ai) in TGTYPES.items():
    hdr = _HDR + "using namespace metal;\n" + "".join(f"#define {k} {v}\n" for k, v in
                                                     (("AT_C", at), ("BT_C", bt), ("DT_C", dt), ("AINIT_C", ai)))
    try:
        k = mx.fast.metal_kernel(name=f"m27c_tg_{at}_{bt}_{dt}", input_names=["dummy"], output_names=["out"], source=TGSRC,
                                 header=hdr)
        t = timed(lambda: k(inputs=[d], template=[("ITER_C", ITER)], grid=(NSG * 32, 1, 1), threadgroup=(128, 1, 1),
                            output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0])
    except Exception as ex:
        msg = str(ex)
        err = [l for l in msg.splitlines() if "error" in l.lower()]
        say(f"  {name}: does not build ({(err[0] if err else msg.splitlines()[0])[:220]})")
        continue
    rate = ITER * NSG / t / 1e9
    tgbest[name] = rate
    say(f"  {name}: {rate:.3f} G op/s = {ITER * NSG * 16384 / t / 1e12:.1f} T MAC/s ({t * 1e3:.2f} ms)")
bf = best.get("bf16 x bf16 -> fp32")
i8 = {k: v for k, v in best.items() if k.startswith("int8") or k.startswith("uint8")}
nax6_form = tgbest.get("bf16 x uint4(tg) -> fp32 (NAX6's form)")
for k_, v_ in tgbest.items():
    if not k_.startswith("bf16") and nax6_form:
        say(f"  {k_}: {v_ / nax6_form:.2f}x NAX6's own form ({nax6_form:.3f} G op/s)")
        i8[k_] = v_ * bf / nax6_form          # compared like for like: the tg form's ratio, expressed against bf16
if not i8:
    say("M27-C verdict: KILL (no int8 x int8 mode builds on this SDK)")
else:
    kbest = max(i8, key=i8.get)
    r = i8[kbest] / bf
    v = "PASS" if r >= 1.6 else "KILL" if r < 1.2 else "BETWEEN"
    say(f"M27-C verdict (PASS >= 1.6x bf16, KILL no mode or < 1.2x): {v} (best int8 form {kbest}: {i8[kbest]:.3f} G op/s = "
        f"{r:.2f}x bf16's {bf:.3f})")
