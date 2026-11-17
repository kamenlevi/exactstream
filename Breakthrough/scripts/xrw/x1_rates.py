"""x1_rates.py (Breakthrough, Tue 17 Nov 2026 09:5x; EXPERT REWRITE, info screen beside X0): what does the MMA path of
kernel A cost by itself, and how fast are the accelerator's other operand types? All kernels run on kernel A's grid
(62 segments x 20 column tiles x 8 simdgroups, 10 quant groups x 2 matmul2d(16, 32, 32) per simdgroup = 198,400 ops per
launch, 3.25 G MAC) with threadgroup-resident static operands (no code loads, no decode):
  R1 today:  left bf16 cooperative from device x (Leap's fragment loads), right uint4b (threadgroup), float dest, reset
             per group + today's epilogue (scale x cg + bias x rowsum)
  R2         R1 without the epilogue: the dest accumulates across groups
  R3         left int8 threadgroup tensor, right int4b, int32 dest, accumulating (no epilogue)
  R4         left bf16 threadgroup tensor, right uint4b, float dest, accumulating
  R5         R3 + a per-group epilogue (int32 -> float, scale x sx, + bias x rowsum), reset per group
GPU us per launch (libmtlprof), ROUNDS interleaved rounds x REPS launches, median; T MAC/s = 3.25 G / time.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x1_rates.py [--rounds 5] [--reps 20] [--tag T] -> data/xrw/x1_rates*.txt"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP

MP.load(2)
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "..", "..", "data", "xrw")
NSEG, NT, KG, KS = 62, 20, 40, 4
GPC = KG // KS
K = 64 * KG
NOPS = NSEG * NT * 8 * GPC * 2
GMAC = NOPS * 16 * 32 * 32 / 1e9

HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

PRO = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint kc = sg >> 1;
    const uint sgi = threadgroup_position_in_grid.y, tile = threadgroup_position_in_grid.x;
    threadgroup uint32_t tb_all[8 * 256];                      // right: 32 rows x 64 nibbles per simdgroup
    threadgroup uint32_t ta_all[8 * 512];                      // left: 16 x 64 int8 (256 words) or bf16 (512 words)
    threadgroup float sb_all[8 * 64];
    threadgroup uint32_t* tb = tb_all + sg * 256;
    threadgroup uint32_t* ta = ta_all + sg * 512;
    threadgroup float* sb = sb_all + sg * 64;
    for (int i = lane; i < 256; i += 32) tb[i] = (0x76543210u ^ ((uint)i * 2654435761u)) & 0x77777777u;
    for (int i = lane; i < 512; i += 32) ta[i] = (0x3F803F80u ^ ((uint)i * 40503u)) & 0x3F7F3F7Fu;
    for (int i = lane; i < 64; i += 32) sb[i] = 0.001f * (float)(i & 63);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    using TB = tensor<threadgroup RT_, dextents<int32_t, 2>, tensor_inline>;
"""

R1 = PRO.replace("RT_", "uint4b_format") + r"""
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const uint ma0 = (sgi * 5u + (uint)fm0) % 32u, ma1 = (sgi * 5u + (uint)fm0 + 8u) % 32u;
    const device T* ar0 = x + (size_t)ma0 * K_C;
    const device T* ar1 = x + (size_t)ma1 * K_C;
    const device float* as0 = xs + (size_t)ma0 * KG_C;
    const device float* as1 = xs + (size_t)ma1 * KG_C;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        for (int hh = 0; hh < 2; ++hh) {
            const int k0 = 64 * g + 32 * hh;
            const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
            const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
            const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
            const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
            for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
            TB tbt((threadgroup uchar*)tb + 0, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
            op.run(ca, tbt, cg);
        }
        EPI_
    }
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += acc[i] + cg[i];
    out[((size_t)sgi * NT_C + tile) * 256 + thread_index_in_threadgroup] = s;
"""
EPI1 = r"""const float x0 = as0[g], x1 = as1[g];
        for (int i = 0; i < 16; ++i) {
            const int col = 16 * (i >> 3) + fn0 + (i & 3);
            acc[i] += sb[col] * cg[i] + sb[32 + col] * (((i >> 2) & 1) ? x1 : x0);
        }"""
SRC = {"R1": R1.replace("RT_", "uint4b_format").replace("EPI_", EPI1),
       "R2": R1.replace("RT_", "uint4b_format").replace("EPI_", "").replace(
           "        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;\n", "", 1)}
# R2: cg is zeroed once before the loop instead
SRC["R2"] = SRC["R2"].replace("    for (int g = kc * GPC_C;", "    for (int i = 0; i < 16; ++i) cg[i] = 0.0f;\n    for (int g = kc * GPC_C;", 1)


def tg_left(lt, rt, dt, epi):
    """left = a threadgroup tensor [16 x 32] of lt, right = threadgroup rt, dest cooperative dt"""
    lw = "int8_t" if lt == "int8_t" else "T"
    body = PRO.replace("RT_", rt) + r"""
    using TA = tensor<threadgroup LT_, dextents<int32_t, 2>, tensor_inline>;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    TA ta0((threadgroup LT_*)ta, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB tb0((threadgroup uchar*)tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, DT_>();
    const uint cap = cg.get_capacity();
    float acc[32];
    for (int i = 0; i < 32; ++i) acc[i] = 0.0f;
    for (uint i = 0; i < cap; ++i) cg[i] = (DT_)0;
    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {
        RESET_
        for (int hh = 0; hh < 2; ++hh) {
            TA tah((threadgroup LT_*)ta + 32 * hh, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
            op.run(tah, tb0, cg);
        }
        EPI_
    }
    float s = 0.0f;
    for (uint i = 0; i < cap; ++i) s += acc[i & 31] + (float)cg[i];
    out[((size_t)sgi * NT_C + tile) * 256 + thread_index_in_threadgroup] = s;
"""
    body = body.replace("LT_", lw).replace("RT_", rt).replace("DT_", dt)
    if epi:
        body = body.replace("RESET_", "for (uint i = 0; i < cap; ++i) cg[i] = (DT_)0;".replace("DT_", dt)).replace(
            "EPI_", r"""const float x0 = xs[(sgi * 5u) % 32u * KG_C + g];
        for (uint i = 0; i < cap; ++i) {
            const auto ix = cg.get_multidimensional_index(i);
            acc[i & 31] += sb[ix[0] & 31] * 0.01f * (float)cg[i] + sb[32 + (ix[0] & 31)] * x0;
        }""")
    else:
        body = body.replace("RESET_", "").replace("EPI_", "")
    return body


SRC["R3"] = tg_left("int8_t", "int4b_format", "int32_t", False)
SRC["R4"] = tg_left("T", "uint4b_format", "float", False)
SRC["R5"] = tg_left("int8_t", "int4b_format", "int32_t", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(1117)
    x = mx.array(rng.standard_normal((32, K)).astype(np.float32)).astype(mx.bfloat16)
    xs = mx.array(rng.standard_normal((32, KG)).astype(np.float32))
    mx.eval(x, xs)
    names = [n for n in SRC if not a.only or n in a.only.split(",")]
    tmpl = [("T", mx.bfloat16), ("NT_C", NT), ("KG_C", KG), ("GPC_C", GPC), ("K_C", K)]
    KK, ok, bad = {}, [], {}
    for n in names:
        try:
            KK[n] = mx.fast.metal_kernel(name=f"x1_{n.lower()}", input_names=["x", "xs"], output_names=["out"],
                                         source=SRC[n], header=HDR)
            mx.eval(KK[n](inputs=[x, xs], template=tmpl, grid=(NT * 256, NSEG, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(NSEG * NT * 256,)], output_dtypes=[mx.float32])[0])
            ok.append(n)
        except Exception as e:                              # a type combination that does not build is a finding
            bad[n] = repr(e)[:300]
    mx.synchronize()
    res = {n: [] for n in ok}
    for r in range(a.rounds):
        for n in (ok if r % 2 == 0 else ok[::-1]):
            mx.synchronize()
            c0 = MP.counts()["ncb"]
            for _ in range(a.reps):
                mx.eval(KK[n](inputs=[x, xs], template=tmpl, grid=(NT * 256, NSEG, 1), threadgroup=(256, 1, 1),
                              output_shapes=[(NSEG * NT * 256,)], output_dtypes=[mx.float32])[0])
            mx.synchronize()
            cb = MP.cbs(c0, MP.counts()["ncb"])
            g = [1e6 * (e - s) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > s]
            res[n].append(float(np.median(g)))
    lines = [f"X1 MMA RATES {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; {NOPS} matmul2d(16, 32, 32) per "
             f"launch = {GMAC:.2f} G MAC; {a.rounds} rounds x {a.reps}",
             "| kernel | GPU us per launch | spread | T MAC/s (padded) |", "|---|---|---|---|"]
    for n in ok:
        m = float(np.median(res[n]))
        lines.append(f"| {n} | {m:.1f} | {100 * (max(res[n]) - min(res[n])) / m:.1f} % | {GMAC / (m * 1e-6) / 1e3:.2f} |")
    for n, e in bad.items():
        lines.append(f"| {n} | DOES NOT BUILD / RUN: {e} | | |")
    print("\n".join(lines), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    open(os.path.join(OUTD, f"x1_rates{sfx}.txt"), "a").write("\n".join(lines) + "\n\n")
    json.dump(dict(res=res, bad=bad), open(os.path.join(OUTD, f"x1_rates{sfx}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
