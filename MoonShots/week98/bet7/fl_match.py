"""WEEK98 BET 7: which stock kernel does MLX 0.32.2 run for each of the level's matmul shapes? (MoonShots, Mon 16 Nov)

MLX's host-side dispatch (matmul.cpp / quantized.cpp) isn't shipped, only the kernel headers. So each candidate
(kernel family x template parameters) is built from the shipped header code and its output compared bit for bit with
the stock op on random inputs; the candidates that match on every input are the stock order (or equivalent to it).
Families:
  bf16 x @ W.T, M = 1: GEMVKernel (gemv.h) over (BN, SN, TN) (the parameters that fix one output's reduction order),
                       GemvWide (k_lanes in 8 / 16 / 32)
  bf16 x @ W.T, M = 4: GemvWide (k_lanes), and the M = 1 GEMV per row
usage: ~/BigMoE/.venv/bin/python -I -B fl_match.py [--only gemv] [--n 16]"""
import argparse
import itertools
import os
import re

import numpy as np
import mlx.core as mx

KDIR = os.path.expanduser("~/BigMoE/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels")


def header_slice(fname, start_pat, end_pat):
    """the text of a shipped header between two regex anchors (start inclusive, end exclusive)"""
    s = open(os.path.join(KDIR, fname)).read()
    a = re.search(start_pat, s, re.M).start()
    b = re.search(end_pat, s[a + 1:], re.M).start() + a + 1
    return s[a:b]


# GEMVKernel::run, from gemv.h (struct GEMVKernel ... up to the GEMVTKernel comment block)
_GEMV_STRUCT = header_slice("gemv.h", r"^template <\n    typename T,\n    const int BM,", r"^/{70,}\n/// Vector matrix")
# by-value scalars: MLX's `const constant int&` parameters can't bind a kernel's local values (same arithmetic)
_GEMV_STRUCT = _GEMV_STRUCT.replace("const constant int&", "const int").replace("const constant float&", "const float")
_GEMV_STRUCT = re.sub(r"\s*\[\[buffer\(\d+\)\]\]", "", _GEMV_STRUCT)     # attributes need reference types
# GemvWide::run, from gemv.h; the function constant gemv_wide_do_axpby is replaced by false (no bias here)
_GEMV_WIDE = header_slice("gemv.h", r"^template <\n    typename T,\n    const int vecs_per_tg,", r"^template <typename T, const int vecs_per_tg, const int k_lanes>\n\[\[kernel\]\] void gemv_wide\(")
_GEMV_WIDE = _GEMV_WIDE.replace("gemv_wide_do_axpby", "false")

GEMV_HEADER = """
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
#define MLX_MTL_CONST static constant constexpr const
#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif
template <typename U> struct DefaultAccT { using type = float; };
""" + _GEMV_STRUCT + "\n" + _GEMV_WIDE

_GEMV_SRC = """
  threadgroup float tgp_memory[GK::tgp_mem_size > 0 ? GK::tgp_mem_size : 1];
  int insz = IN, outsz = OUT, ld = IN;
  float alpha = 1.0f, beta = 0.0f;
  int bstride = 0;
  GK::run(mat, vec + tid3.z * IN, mat, out + tid3.z * OUT, insz, outsz, ld, alpha, beta, bstride, tgp_memory,
          tid3, lid3, simd_gid, simd_lid);
"""
_GEMV_WIDE_SRC = """
  int insz = IN, outsz = OUT, m = M, ld = IN, vld = IN;
  GW::run(mat, vec, mat, out, insz, outsz, m, ld, vld, 1.0f, 0.0f, 0, 0, tid3, tgpg3, simd_gid, simd_lid);
"""


def _gemv_kernel(BM, BN, SM, SN, TM, TN):
    hdr = GEMV_HEADER + f"\ntypedef GEMVKernel<bfloat16_t, {BM}, {BN}, {SM}, {SN}, {TM}, {TN}, false> GK;\n"
    src = ("uint3 tid3 = threadgroup_position_in_grid; uint3 lid3 = thread_position_in_threadgroup;\n"
           "uint simd_gid = simdgroup_index_in_threadgroup; uint simd_lid = thread_index_in_simdgroup;\n" + _GEMV_SRC)
    return mx.fast.metal_kernel(name=f"flm_gemv_{BM}_{BN}_{SM}_{SN}_{TM}_{TN}", input_names=["mat", "vec"],
                                output_names=["out"], header=hdr, source=src)


def _gemv_wide_kernel(V, KL):
    hdr = GEMV_HEADER + f"\ntypedef GemvWide<bfloat16_t, {V}, {KL}> GW;\n"
    src = ("uint3 tid3 = threadgroup_position_in_grid; uint3 tgpg3 = threadgroups_per_grid;\n"
           "uint simd_gid = simdgroup_index_in_threadgroup; uint simd_lid = thread_index_in_simdgroup;\n" + _GEMV_WIDE_SRC)
    return mx.fast.metal_kernel(name=f"flm_gemvw_{V}_{KL}", input_names=["mat", "vec"], output_names=["out"],
                                header=hdr, source=src)


def gemv_candidate(x, W, BM, BN, SM, SN, TM, TN):
    """out[m] = W @ x[m] with GEMVKernel<BM, BN, SM, SN, TM, TN>, one grid z per row of x"""
    M, K = x.shape
    N = W.shape[0]
    blockM = BM * SM * TM
    k = _gemv_kernel(BM, BN, SM, SN, TM, TN)
    return k(inputs=[W, x], template=[("IN", K), ("OUT", N)],
             grid=((N + blockM - 1) // blockM * BN * SN, BM * SM, M), threadgroup=(BN * SN, BM * SM, 1),
             output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def gemv_wide_candidate(x, W, V, KL):
    M, K = x.shape
    N = W.shape[0]
    rows_per_tg = (32 // KL) * (KL // 8)
    k = _gemv_wide_kernel(V, KL)
    return k(inputs=[W, x], template=[("IN", K), ("OUT", N), ("M", M)],
             grid=(32 * (KL // 8), (N + rows_per_tg - 1) // rows_per_tg, 1), threadgroup=(32 * (KL // 8), 1, 1),
             output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def bits_equal(a, b):
    mx.eval(a, b)
    return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def cancel_inputs(M, N, K, n, seed=0):
    """inputs whose partial sums are huge but whose exact total is small: in every group of 4 columns, column 0
    carries +H * a and column 1 carries -H * a (W[:, 4j + 1] = -W[:, 4j], x[4j + 1] = x[4j]), columns 2-3 are O(1).
    Any change in the order of summation then changes the fp32 rounding by far more than a bf16 ulp of the result,
    so wrong candidates fail on most outputs"""
    import numpy as np
    rng = np.random.default_rng(seed + N + K)
    W = rng.standard_normal((N, K)).astype(np.float32)
    # random column pairs (p, q): the cancellations land in every part of the reduction tree (inside a thread,
    # across lanes, across simdgroups), not only inside one thread's contiguous run
    perm = rng.permutation(K)
    npair = K // 4
    p, q = perm[:npair], perm[npair:2 * npair]
    H = (2.0 ** 12) * (1.0 + rng.random((N, npair)))
    W[:, p] = H
    W[:, q] = -H
    xs = []
    for i in range(n):
        x = rng.standard_normal((M, K)).astype(np.float32)
        x[:, q] = x[:, p]
        xs.append(mx.array(x).astype(mx.bfloat16))
    return mx.array(W).astype(mx.bfloat16), xs


def match_bf16(name, M, N, K, n_inputs, say, cancel=False):
    """which candidates reproduce x @ W.T (stock) bit for bit on every input"""
    W = (mx.random.normal((N, K), key=mx.random.key(N + K)) * 0.05).astype(mx.bfloat16)
    # heavy-tailed inputs (log-normal magnitudes): large cancellations make any change of reduction order flip bf16
    # output bits often, so few candidates survive by chance
    xs = [(mx.random.normal((M, K), key=mx.random.key(1000 + i))
           * mx.exp(2.5 * mx.random.normal((M, K), key=mx.random.key(5000 + i)))).astype(mx.bfloat16)
          for i in range(n_inputs)]
    if cancel:
        W, xs = cancel_inputs(M, N, K, n_inputs)
    refs = [x @ W.T for x in xs]
    mx.eval(W, xs, refs)
    hits = []
    if M == 1 or True:
        for BN, SN, TN in itertools.product((1, 2, 4, 8, 16), (4, 8, 16, 32), (1, 4, 8)):
            SM = 32 // SN
            for BM, TM in ((1, 4),):
                if BN * SN * BM * SM > 1024:
                    continue
                try:
                    ok = all(bits_equal(gemv_candidate(x, W, BM, BN, SM, SN, TM, TN), r) for x, r in zip(xs, refs))
                except Exception as e:                               # noqa: BLE001
                    say(f"    gemv BN{BN} SN{SN} TN{TN}: error {str(e)[:120]}")
                    continue
                if ok:
                    hits.append(f"gemv BM{BM} BN{BN} SM{SM} SN{SN} TM{TM} TN{TN}")
    for KL in (8, 16, 32):
        for V in (1, 2, 4, 8):
            try:
                ok = all(bits_equal(gemv_wide_candidate(x, W, V, KL), r) for x, r in zip(xs, refs))
            except Exception as e:                                   # noqa: BLE001
                say(f"    gemv_wide V{V} K{KL}: error {str(e)[:120]}")
                continue
            if ok:
                hits.append(f"gemv_wide V{V} k_lanes{KL}")
    say(f"  {name} (M {M}, N {N}, K {K}): {len(hits)} matching: {hits}")
    return hits


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_match.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--cancel", action="store_true", help="cancellation-structured inputs (sharper discrimination)")
    a = ap.parse_args()

    def say(s):
        print(s, flush=True)
    res = {}
    SHAPES = [("fc_embedding", 1, 2560, 2560), ("hc down", 1, 320, 10240), ("hc up", 1, 10240, 320),
              ("hc inject", 1, 4, 10240), ("fc_hidden (4 substreams)", 4, 2560, 2560)]
    for name, M, N, K in SHAPES:
        if a.only and not any(o in name for o in a.only.split(",")):
            continue
        res[name] = match_bf16(name, M, N, K, a.n, say, cancel=a.cancel)
    json.dump(res, open(a.out, "w"), indent=1)
