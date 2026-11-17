"""x3_shapes.py (Breakthrough, Tue 17 Nov 2026 10:3x; EXPERT REWRITE info screen): does the accelerator charge for
padding in N (the tokens) when the WEIGHTS are the left operand? Kernel A's MMA today is matmul2d(16 tokens, 32 features,
32 K) with tokens on M; M = 8 runs padded (MoonShots). Here the weights go LEFT as int8 (the codes q - 8 are small
integers: exact) and the activations RIGHT as bf16 x^T, so the token count is N. The header lists int8 x bfloat -> float.
Per simdgroup and quant group (kernel A's grid, 10 groups per simdgroup, static threadgroup operands, no epilogue):
  W16  today's form: 2 x (16, 32, 32) bf16 x uint4b -> float              = 16 tokens x 32 features x 64 K
  F16  2 x (32, 16, 32) int8 (weights, left) x bf16 (x^T, right) -> float  = the same useful work, tokens on N
  F8   2 x (32, 8, 32)  int8 x bf16 -> float                               = 8 tokens: half the work if N pads to 8
  F16b 1 x (32, 16, 64) int8 x bf16 -> float                               = K 64 in one op (if it builds)
  F8b  1 x (32, 8, 64)
Timing: min over ROUNDS x REPS launches of the GPU interval (libmtlprof), interleaved; ratios vs W16. Info only.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x3_shapes.py [--rounds 5] [--reps 10] [--tag T]"""
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
OUTD = os.environ.get("XRW_OUTD") or os.path.join(HERE, "..", "..", "data", "xrw")
NSEG, NT, GPC = 62, 20, 10
HDR = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
DUMMY = mx.zeros((1,), mx.int32)
mx.eval(DUMMY)

BODY = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint32_t tl_all[8 * 512];                  // left: up to 32 x 64 int8 (512 words) or 16 x 64 bf16
    threadgroup uint32_t tr_all[8 * 512];                  // right: up to 32 x 64 nibbles / 16 x 64 bf16
    threadgroup uint32_t* tl = tl_all + sg * 512;
    threadgroup uint32_t* tr = tr_all + sg * 512;
    for (int i = lane; i < 512; i += 32) { tl[i] = (0x01020304u ^ ((uint)i * 2654435761u)) & 0x3F3F3F3Fu;
                                          tr[i] = (0x3F803F80u ^ ((uint)i * 40503u)) & 0x3F7F3F7Fu; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(M_C, N_C, K_C, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TL = tensor<threadgroup LT_, dextents<int32_t, 2>, tensor_inline>;
    using TR = tensor<threadgroup RT_, dextents<int32_t, 2>, tensor_inline>;
    TL l0(LPTR_, dextents<int32_t, 2>(K_C, M_C), array<int32_t, 2>{1, 64});
    TL l1(LPTR2_, dextents<int32_t, 2>(K_C, M_C), array<int32_t, 2>{1, 64});
    TR r0(RPTR_, dextents<int32_t, 2>(K_C, N_C), array<int32_t, 2>{1, 64});
    TR r1(RPTR2_, dextents<int32_t, 2>(K_C, N_C), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TL, TR, float>();
    for (uint i = 0; i < cg.get_capacity(); ++i) cg[i] = 0.0f;
    for (int g = 0; g < GPC_C; ++g) {
        op.run(l0, r0, cg);
        if (NOPS_C == 2) op.run(l1, r1, cg);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    float s = 0.0f;
    for (uint i = 0; i < cg.get_capacity(); ++i) s += cg[i];
    out[((size_t)threadgroup_position_in_grid.y * NT_C + threadgroup_position_in_grid.x) * 256 + thread_index_in_threadgroup] = s;
"""

# name: (M, N, K, nops, left type, right type, left ptr, left ptr 2, right ptr, right ptr 2)
SHAPES = {
    "W16": (16, 32, 32, 2, "bfloat", "uint4b_format", "(threadgroup bfloat*)tl", "(threadgroup bfloat*)tl + 32",
            "(threadgroup uchar*)tr", "(threadgroup uchar*)tr + 16"),
    "F16": (32, 16, 32, 2, "int8_t", "bfloat", "(threadgroup int8_t*)tl", "(threadgroup int8_t*)tl + 32",
            "(threadgroup bfloat*)tr", "(threadgroup bfloat*)tr + 32"),
    "F8": (32, 8, 32, 2, "int8_t", "bfloat", "(threadgroup int8_t*)tl", "(threadgroup int8_t*)tl + 32",
           "(threadgroup bfloat*)tr", "(threadgroup bfloat*)tr + 32"),
    "F16b": (32, 16, 64, 1, "int8_t", "bfloat", "(threadgroup int8_t*)tl", "(threadgroup int8_t*)tl",
             "(threadgroup bfloat*)tr", "(threadgroup bfloat*)tr"),
    "F8b": (32, 8, 64, 1, "int8_t", "bfloat", "(threadgroup int8_t*)tl", "(threadgroup int8_t*)tl",
            "(threadgroup bfloat*)tr", "(threadgroup bfloat*)tr"),
    "H16": (32, 16, 32, 2, "half", "bfloat", "(threadgroup half*)tl", "(threadgroup half*)tl + 32",
            "(threadgroup bfloat*)tr", "(threadgroup bfloat*)tr + 32"),
}


def build(name):
    M, N, K, nops, lt, rt, lp, lp2, rp, rp2 = SHAPES[name]
    src = (BODY.replace("LT_", lt).replace("RT_", rt).replace("LPTR2_", lp2).replace("LPTR_", lp)
           .replace("RPTR2_", rp2).replace("RPTR_", rp))
    k = mx.fast.metal_kernel(name=f"x3_{name.lower()}", input_names=["dummy"], output_names=["out"], source=src, header=HDR)
    tmpl = [("M_C", M), ("N_C", N), ("K_C", K), ("NOPS_C", nops), ("GPC_C", GPC), ("NT_C", NT)]
    return lambda: k(inputs=[DUMMY], template=tmpl, grid=(NT * 256, NSEG, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(NSEG * NT * 256,)], output_dtypes=[mx.float32])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fns, bad = {}, {}
    for n in SHAPES:
        try:
            f = build(n)
            mx.eval(f())
            fns[n] = f
        except Exception as e:
            bad[n] = repr(e)[:240]
    mx.synchronize()
    t = {n: [] for n in fns}
    names = list(fns)
    for r in range(a.rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            for _ in range(a.reps):
                c0 = MP.counts()["ncb"]
                mx.eval(fns[n]())
                mx.synchronize()
                cb = MP.cbs(c0, MP.counts()["ncb"])
                g = [1e6 * (e - s) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > s]
                if g:
                    t[n].append(max(g))                    # the kernel's command buffer (the only real work)
    L = [f"X3 SHAPES {time.strftime('%a %d %b %H:%M:%S')}: kernel A's grid ({NSEG} x {NT} x 8 simdgroups x {GPC} groups); "
         f"min GPU us over {a.rounds} x {a.reps} launches (interleaved); static threadgroup operands, no epilogue",
         "| form | shape (M, N, K) x ops | types | min us | / W16 | useful tokens per op |", "|---|---|---|---|---|---|"]
    w = min(t["W16"]) if "W16" in t else float("nan")
    for n in fns:
        M, N, K, nops, lt, rt = SHAPES[n][:6]
        tok = 16 if n == "W16" else N
        L.append(f"| {n} | ({M}, {N}, {K}) x {nops} | {lt} x {rt} | {min(t[n]):.1f} | {min(t[n]) / w:.3f} | {tok} |")
    for n, e in bad.items():
        L.append(f"| {n} | does not build / run | {e} | | | |")
    print("\n".join(L), flush=True)
    sfx = f"_{a.tag}" if a.tag else ""
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, f"x3_shapes{sfx}.txt"), "a").write("\n".join(L) + "\n\n")
    json.dump(dict(t=t, bad=bad), open(os.path.join(OUTD, f"x3_shapes{sfx}.json"), "w"))


if __name__ == "__main__":
    main()
