"""M27-P, the matrix-unit peak (Leap, Mon 2026-11-02 21:1x; Fable mission 27 ~/FableGod/EXPERT_KERNEL.md section 0.3;
orchestrator 21:1x; pre-registered in ~/Leap/TABLE_350.md 'M27-P'): Sunday's probe (nax_tput.py / nax_shapes.py) ran ONE
dependent accumulator chain per simdgroup and reported only the 16 x 32 x 16 rate (1.3 G op/s, ~10.5 T MAC/s). Here, after
a calibration run of Sunday's kernel verbatim: every op shape that builds (16x32x16, 16x32x32, 32x32x16, 32x32x32;
16x32x64, 16x16x16 and 16x64x16 do not) x 1-4 independent accumulator chains per simdgroup (one kernel per count, each
chain its own B operand and destination, A shared as a GEMM shares it across column tiles) x threadgroups of 1 / 2 / 4 / 8
simdgroups. Operands held in registers: no memory traffic, no decode. MAC/s per configuration.
Decision (Fable): best rate >= 1.6 x Sunday's (16.8 T MAC/s) -> NAX7 alone can reach >= 85 % of streaming; otherwise
NAX7 + the hybrid.
Usage: [BigMoE venv] nax_peak.py -> ~/Leap/data/nax_peak.txt"""
import os, time
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/nax_peak.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
_HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
def src(mm, nn, kk, ch):
    """one kernel per configuration, declaring exactly ch accumulator chains (no unused cooperative tensors)"""
    decl = "\n".join(f"    auto cb{j} = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();" for j in range(ch))
    dest = "\n".join(f"    auto c{j} = op.template get_destination_cooperative_tensor<CA, CB, float>();" for j in range(ch))
    init = "\n".join(f"        cb{j}[i] = bfloat({0.5 / (j + 1)}f + 0.01f * i);" for j in range(ch))
    zero = " ".join(f"c{j}[i] = 0.0f;" for j in range(ch))
    run = "\n".join(f"        op.run(ca, cb{j}, c{j});" for j in range(ch))
    tot = " + ".join(f"c{j}[i]" for j in range(ch))
    return f"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor({mm}, {nn}, {kk}, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
{decl}
    using CA = metal::remove_addrspace_t<decltype(ca)>;
    using CB = metal::remove_addrspace_t<decltype(cb0)>;
{dest}
    for (int i = 0; i < (int)ca.get_capacity(); i++) ca[i] = bfloat(0.001f * (lane + i));
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
ITER, NSG = 1024, 20 * 64                      # ops per chain per simdgroup; simdgroups per launch (64 per core)
say(f"M27-P matrix-unit peak {time.strftime('%a %H:%M')} ({os.environ.get('M27P_COND', 'quiet lock')}); {NSG} simdgroups "
    f"x {ITER} ops per chain, operands in registers")
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
# calibration: Sunday's kernel verbatim (nax_tput.py mode 0)
txt = open(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts/nax_tput.py")).read()
SUN = txt[txt.index('_SRC = r"""') + len('_SRC = r"""'):txt.index('"""\nk = mx.fast')]
ks = mx.fast.metal_kernel(name="m27p_sunday", input_names=["dummy"], output_names=["out"], source=SUN, header=_HDR)
t = timed(lambda: ks(inputs=[d], template=[("ITER_C", ITER), ("MODE_C", 0)], grid=(NSG * 32, 1, 1), threadgroup=(32, 1, 1),
                     output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0])
cal = ITER * NSG * 8192 / t / 1e12
say(f"  calibration, Sunday's nax_tput mode 0 (16x32x16, 1 chain, tg1): {ITER * NSG / t / 1e9:.2f} G op/s = {cal:.1f} T MAC/s")
res = {}
for (mm, nn, kk) in ((16, 32, 16), (16, 32, 32), (32, 32, 16), (32, 32, 32)):
    for ch in (1, 2, 3, 4):
        k = mx.fast.metal_kernel(name=f"m27p_{mm}_{nn}_{kk}_{ch}", input_names=["dummy"], output_names=["out"],
                                 source=src(mm, nn, kk, ch), header=_HDR)
        for tgs in (1, 2, 4, 8):
            tag = f"{mm}x{nn}x{kk} ch{ch} tg{tgs}"
            try:
                t = timed(lambda: k(inputs=[d], template=[("ITER_C", ITER)], grid=(NSG * 32, 1, 1),
                                    threadgroup=(32 * tgs, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32],
                                    init_value=0)[0])
            except Exception as ex:
                say(f"  {tag}: FAILED {str(ex).splitlines()[0][:160]}")
                break
            n = ITER * NSG * ch
            res[tag] = n * mm * nn * kk / t / 1e12
            say(f"  {tag}: {n / t / 1e9:.2f} G op/s = {res[tag]:.1f} T MAC/s ({t * 1e3:.2f} ms per launch)")
if res:
    best = max(res, key=res.get)
    say(f"best {best}: {res[best]:.1f} T MAC/s = {res[best] / 10.5:.2f} x Sunday's 10.5 ({res[best] / cal:.2f} x this run's "
        f"calibration {cal:.1f}); per shape best: " + "; ".join(
            f"{s_} {max(v for t_, v in res.items() if t_.startswith(s_ + ' ')):.1f}"
            for s_ in dict.fromkeys(t_.split(' ')[0] for t_ in res)))
    v = "NAX7 world (>= 1.6x)" if res[best] >= 1.6 * 10.5 else "NAX7 + hybrid world (< 1.6x)"
    say(f"M27-P verdict (Fable's line: best >= 1.6 x 10.5 = 16.8 T MAC/s): {v}")
