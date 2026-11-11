"""N7-4, the matrix unit's rate in NAX6's own operand form (Leap, Mon 2026-11-02 21:5x; NAX7 groundwork): M27-P measured
16 x 32 x 32 at ~0.86 G op/s with both operands in registers (cooperative tensors). NAX6 runs the op with A in registers
and B as packed uint4 read straight from threadgroup memory (a tensor_inline view, 32 x 32 nibbles, row stride 64), and
N7-1-3 found its kernel A neither DRAM-, decode- nor occupancy-bound. Arms (no memory traffic beyond threadgroup memory,
1,280 simdgroups x 512 ops, median of 11; each simdgroup its own B region):
  reg    B in registers (bf16 cooperative tensor): M27-P's form
  tg4    B as uint4 from threadgroup memory (NAX6's form)
  tg16   B as bf16 from threadgroup memory (tensor_inline, 32 x 32, row stride 32 values)
x threadgroups of 1 / 4 / 8 simdgroups; plus 'tg4 +epi': NAX6's per-op-pair epilogue (16 FMAs with scale / bias from
threadgroup memory) after every second op.
Usage: [BigMoE venv] nax_tg_probe.py -> ~/Leap/data/nax_tg_probe.txt"""
import os, time
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/nax_tg_probe.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
import sys
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
import tierglu_nax6 as N6
_HDR = N6._HDR4                                                   # NAX6's own header (tierglu + MPP)
_REG = r"""
    const ushort lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    using CA = metal::remove_addrspace_t<decltype(ca)>;
    using CB = metal::remove_addrspace_t<decltype(cb)>;
    auto cg = op.template get_destination_cooperative_tensor<CA, CB, float>();
    for (int i = 0; i < (int)ca.get_capacity(); i++) ca[i] = bfloat(0.001f * (lane + i));
    for (int i = 0; i < (int)cb.get_capacity(); i++) cb[i] = bfloat(0.5f + 0.01f * i);
    for (int i = 0; i < (int)cg.get_capacity(); i++) cg[i] = 0.0f;
    for (int it = 0; it < ITER_C; ++it) op.run(ca, cb, cg);
    float s = 0.0f;
    for (int i = 0; i < (int)cg.get_capacity(); i++) s += cg[i];
    if (s == 12345.678f) out[0] = s;
"""
_TG = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint32_t tbuf[8 * 32 * 8 * BW_C];                    // per simdgroup: 32 rows x 8 words x BW_C
    threadgroup float sbuf[8 * 64];
    threadgroup uint32_t* tb = tbuf + sg * (32 * 8 * BW_C);
    for (int i = 0; i < 8 * BW_C; ++i) tb[lane * 8 * BW_C + i] = (lane * 2654435761u + i * 40503u) & 0x77777777u;
    sbuf[sg * 64 + lane] = 0.5f; sbuf[sg * 64 + 32 + lane] = 0.25f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
#if BF_C
    using TB = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
#else
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
#endif
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    for (int i = 0; i < (int)ca.get_capacity(); i++) ca[i] = bfloat(0.001f * (lane + i));
    for (int i = 0; i < (int)cg.get_capacity(); i++) cg[i] = 0.0f;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int it = 0; it < ITER_C; ++it) {
        const int hh = it & 1;
#if BF_C
        TB tbt((threadgroup bfloat*)tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 32});
#else
        TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
#endif
        op.run(ca, tbt, cg);
        if (EPI_C && hh == 1) {
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + (lane & 3) * 4 + (i & 3);
                acc[i] += sbuf[sg * 64 + col] * cg[i] + sbuf[sg * 64 + 32 + col] * 0.5f;
                cg[i] = 0.0f;
            }
        }
    }
    float s = 0.0f;
    for (int i = 0; i < (int)cg.get_capacity(); i++) s += cg[i];
    for (int i = 0; i < 16; ++i) s += acc[i];
    if (s == 12345.678f) out[0] = s;
"""
d = mx.zeros((1,), mx.float32)
ITER, NSG = 512, 20 * 64
say(f"N7-4 matrix unit in NAX6's operand form {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'quiet lock')}); "
    f"16x32x32, {NSG} simdgroups x {ITER} ops")
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
ARMS = {"reg": (_REG, {}), "tg4": (_TG, {"BF_C": 0, "EPI_C": 0, "BW_C": 1}), "tg16": (_TG, {"BF_C": 1, "EPI_C": 0, "BW_C": 2}),
        "tg4 +epi": (_TG, {"BF_C": 0, "EPI_C": 1, "BW_C": 1})}
for name, (src, defs) in ARMS.items():
    hdr = _HDR + "".join(f"#define {k} {v}\n" for k, v in defs.items())
    k = mx.fast.metal_kernel(name=f"n7tg_{name.replace(' ', '_').replace('+', 'p')}", input_names=["dummy"], output_names=["out"],
                             source=src, header=hdr)
    for tgs in (1, 4, 8):
        try:
            t = timed(lambda: k(inputs=[d], template=[("ITER_C", ITER)], grid=(NSG * 32, 1, 1), threadgroup=(32 * tgs, 1, 1),
                                output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0])
            n = ITER * NSG
            say(f"  {name:9s} tg{tgs}: {n / t / 1e9:.3f} G op/s = {n * 16384 / t / 1e12:.1f} T MAC/s ({t * 1e3:.2f} ms)")
        except Exception as ex:
            say(f"  {name:9s} tg{tgs}: FAILED {str(ex).splitlines()[0][:200]}")
            break
