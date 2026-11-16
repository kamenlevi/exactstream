"""WEEK98 BET 7: which stock quantized GEMV does MLX 0.32.2 run for each of the level's quantized shapes? (MoonShots)

As fl_match.py, for mx.quantized_matmul(x, w, scales, biases, transpose=True) at M = 1 (and 4): candidates qmv_fast,
qmv and qmv_wide (k_lanes 8 / 16 / 32), built from quantized.h's own impl functions (only the scalar parameters
passed by value), compared bit for bit with the stock op on heavy-tailed random inputs.
usage: ~/BigMoE/.venv/bin/python -I -B fl_qmatch.py [--n 16] [--only q_proj,router]"""
import argparse
import json
import os
import re

import mlx.core as mx

KDIR = os.path.expanduser("~/BigMoE/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels")
_Q = open(os.path.join(KDIR, "quantized.h")).read()


def _between(s, start_pat, end_pat):
    a = re.search(start_pat, s, re.M).start()
    b = re.search(end_pat, s[a + 1:], re.M).start() + a + 1
    return s[a:b]


# helpers (pack factors, load_vector*, qdot*, dequantize) and the qmv impls, from quantized.h as shipped
_HELPERS = _between(_Q, r"^#define MLX_MTL_CONST", r"^template <\n    typename T,\n    short BROWS,")
_IMPLS = _between(_Q, r"^template <typename T, int group_size, int bits, int D>\nMETAL_FUNC void qmv_quad_impl",
                  r"^template <typename T, const int group_size, const int bits>\nMETAL_FUNC void qvm_impl")
QHEADER = ("#include <metal_simdgroup>\n#include <metal_stdlib>\nusing namespace metal;\n" + _HELPERS + "\n" + _IMPLS)
QHEADER = QHEADER.replace("const constant int&", "const int").replace("constant int& in_vec_size", "const int in_vec_size")
QHEADER = re.sub(r"\s*\[\[buffer\(\d+\)\]\]", "", QHEADER)

_PRE = ("uint3 tid3 = threadgroup_position_in_grid; uint simd_gid = simdgroup_index_in_threadgroup;\n"
        "uint simd_lid = thread_index_in_simdgroup;\n")
_KERNELS = {}


def _kernel(kind, gs, bits, extra=()):
    key = (kind, gs, bits) + tuple(extra)
    if key in _KERNELS:
        return _KERNELS[key]
    if kind == "fast":
        call = f"qmv_fast_impl<T, {gs}, {bits}>(w, scales, biases, x, y, IN, OUT, tid3, simd_gid, simd_lid);"
    elif kind == "qmv":
        call = f"qmv_impl<T, {gs}, {bits}>(w, scales, biases, x, y, IN, OUT, tid3, simd_gid, simd_lid);"
    else:
        V, KL = extra
        call = (f"qmv_wide_impl<T, {gs}, {bits}, {V}, {KL}>(w, scales, biases, x, y, IN, OUT, MROWS, tid3, simd_gid, "
                f"simd_lid);")
    k = mx.fast.metal_kernel(name=f"flq_{kind}_{gs}_{bits}_{'_'.join(map(str, extra))}",
                             input_names=["w", "scales", "biases", "x"], output_names=["y"], header=QHEADER,
                             source=_PRE + call)
    _KERNELS[key] = k
    return k


def qmv_candidate(kind, x, w, s, b, gs, bits, V=1, KL=32):
    M, K = x.shape
    N = w.shape[0]
    if kind in ("fast", "qmv"):
        k = _kernel(kind, gs, bits)
        grid, tg = (M * 32, (N + 7) // 8 * 2, 1), (32, 2, 1)
        tmpl = [("T", x.dtype), ("IN", K), ("OUT", N)]
    else:
        k = _kernel("wide", gs, bits, (V, KL))
        rows_tg = (32 // KL) * 2
        grid, tg = ((M + V - 1) // V * 32, (N + rows_tg - 1) // rows_tg * 2, 1), (32, 2, 1)
        tmpl = [("T", x.dtype), ("IN", K), ("OUT", N), ("MROWS", M)]
    return k(inputs=[w, s, b, x], template=tmpl, grid=grid, threadgroup=tg, output_shapes=[(M, N)],
             output_dtypes=[x.dtype], init_value=0)[0]


def heavy(shape, seed):
    return (mx.random.normal(shape, key=mx.random.key(seed))
            * mx.exp(2.5 * mx.random.normal(shape, key=mx.random.key(seed + 77)))).astype(mx.bfloat16)


def match_q(name, M, N, K, gs, bits, n, say):
    W = (mx.random.normal((N, K), key=mx.random.key(N + K + bits)) * 0.05).astype(mx.bfloat16)
    w, s, b = mx.quantize(W, group_size=gs, bits=bits)
    xs = [heavy((M, K), 1000 + i) for i in range(n)]
    refs = [mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits) for x in xs]
    mx.eval(w, s, b, xs, refs)
    cands = [("fast", {}), ("qmv", {})] + [("wide", dict(V=1, KL=kl)) for kl in (8, 16, 32)]
    hits = []
    for kind, kw in cands:
        if kind == "fast" and (K % 512 or N % 8):
            continue
        try:
            ok = True
            for x, r in zip(xs, refs):
                y = qmv_candidate(kind, x, w, s, b, gs, bits, **kw)
                mx.eval(y)
                if not bool(mx.array_equal(y.view(mx.uint16), r.view(mx.uint16)).item()):
                    ok = False
                    break
        except Exception as e:                                       # noqa: BLE001
            say(f"    {kind} {kw}: error {str(e)[:300]}")
            continue
        if ok:
            hits.append(f"{kind}{'' if not kw else ' k_lanes' + str(kw['KL'])}")
    say(f"  {name} (M {M}, N {N}, K {K}, {bits}-bit g{gs}): {len(hits)} matching: {hits}")
    return hits


SHAPES = [("q_proj", 1, 12288, 2560, 64, 4), ("k/v_proj", 1, 512, 2560, 64, 4), ("indexer qk", 1, 640, 2560, 64, 8),
          ("o_proj", 1, 2560, 6144, 64, 4), ("router", 1, 512, 2560, 64, 8), ("shared gate/up", 1, 640, 2560, 64, 4),
          ("shared down", 1, 2560, 640, 64, 4), ("shared_expert_gate", 1, 1, 2560, 64, 4),
          ("head 32K screen", 1, 32768, 2560, 64, 4)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_qmatch.json"))
    a = ap.parse_args()

    def say(s):
        print(s, flush=True)
    res = {}
    for name, M, N, K, gs, bits in SHAPES:
        if a.only and not any(o in name for o in a.only.split(",")):
            continue
        res[name] = match_q(name, M, N, K, gs, bits, a.n, say)
    json.dump(res, open(a.out, "w"), indent=1)
