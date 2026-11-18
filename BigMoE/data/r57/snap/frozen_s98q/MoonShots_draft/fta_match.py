"""FUSED TREE LEVEL, step 1 (MoonShots, Tue 17 Nov 2026): which stock MLX 0.32.2 kernels run dd_tree.tree_attn's two
fp32 matmuls at B = 1 (the r57 draft's pop), so a fused attention kernel can reproduce their per-output order bitwise.
As BET 7's fl_match.py: candidates built from the shipped gemv.h, compared bit for bit with the stock op on the REAL
captured inputs (draft/attn_cap_*.npz) and on cancellation inputs (partial sums huge, totals small: any order change
shows).
  A  st = qg @ K1^T   [1,KV,r,1,hd] @ [1,KV,1,hd,S1] -> GEMVKernel<float, BM, BN, SM, SN, TM, TN> (out = K1 @ q)
  B  o = et @ V1      [1,KV,r,1,S1] @ [1,KV,1,S1,hd] -> GEMVTKernel<float, ...>             (out = et @ V1)
usage: ~/BigMoE/.venv/bin/python -I -B fta_match.py   (GPU, seconds, < 1 GB)"""
import itertools
import os
import re

import numpy as np
import mlx.core as mx

KDIR = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")  # [bundle] the running mlx's headers (was ~/BigMoE/.venv/.../kernels)
_S = open(os.path.join(KDIR, "gemv.h")).read()


def _slice(start_pat, end_pat):
    a = re.search(start_pat, _S, re.M).start()
    b = re.search(end_pat, _S[a + 1:], re.M).start() + a + 1
    return _S[a:b]


def _fix(t):
    t = t.replace("const constant int&", "const int").replace("const constant float&", "const float")
    return re.sub(r"\s*\[\[buffer\(\d+\)\]\]", "", t)


GEMV = _fix(_slice(r"^template <\n    typename T,\n    const int BM,", r"^/{70,}\n/// Vector matrix"))
GEMVT = _fix(_slice(r"^/{70,}\n/// Vector matrix multiplication\n", r"^/{70,}\n/// Matrix vector multiplication kernel"))
HDR = """
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
#define MLX_MTL_CONST static constant constexpr const
#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif
template <typename U> struct DefaultAccT { using type = float; };
""" + GEMV + "\n" + GEMVT
_K = {}


def gemv_cand(W, X, p, transposed):
    """transposed False: out[m] = W @ X[m] (W [N, K]); True: out[m] = X[m] @ W (W [K, N]); one grid z per row of X"""
    BM, BN, SM, SN, TM, TN = p
    name = f"fta_{'gt' if transposed else 'g'}_{BM}_{BN}_{SM}_{SN}_{TM}_{TN}"
    if name not in _K:
        kt = "GEMVTKernel" if transposed else "GEMVKernel"
        hdr = HDR + f"\ntypedef {kt}<float, {BM}, {BN}, {SM}, {SN}, {TM}, {TN}, false> GK;\n"
        src = """
  uint3 tid3 = threadgroup_position_in_grid; uint3 lid3 = thread_position_in_threadgroup;
  uint simd_gid = simdgroup_index_in_threadgroup; uint simd_lid = thread_index_in_simdgroup;
  threadgroup float tgp_memory[GK::tgp_mem_size > 0 ? GK::tgp_mem_size : 1];
  int insz = IN, outsz = OUT, ld = LD;
  GK::run(mat, vec + tid3.z * IN, mat, out + tid3.z * OUT, insz, outsz, ld, 1.0f, 0.0f, 0, tgp_memory,
          tid3, lid3, simd_gid, simd_lid);
"""
        _K[name] = mx.fast.metal_kernel(name=name, input_names=["mat", "vec"], output_names=["out"], header=hdr,
                                        source=src)
    M = X.shape[0]
    if transposed:
        K_, N = W.shape
        blockN = BN * SN * TN
        grid = ((N + blockN - 1) // blockN * BN * SN, BM * SM, M)
        ld = N
    else:
        N, K_ = W.shape
        blockM = BM * SM * TM
        grid = ((N + blockM - 1) // blockM * BN * SN, BM * SM, M)
        ld = K_
    return _K[name](inputs=[W, X], template=[("IN", K_), ("OUT", N), ("LD", ld)], grid=grid,
                    threadgroup=(BN * SN, BM * SM, 1), output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]


def cands():
    out = []
    for BM, BN, TM, TN in itertools.product((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)):
        for SM, SN in ((1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)):
            if BM * BN * 32 <= 1024:
                out.append((BM, BN, SM, SN, TM, TN))
    return out


def eq(a, b):
    mx.eval(a, b)
    return bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())


def cancel_like(x, seed):
    """x with pairs of huge opposite entries along its last axis (any reduction-order change shows)"""
    rng = np.random.default_rng(seed)
    a = np.array(x).copy()
    K_ = a.shape[-1]
    perm = rng.permutation(K_)
    p, q = perm[:K_ // 4], perm[K_ // 4:2 * (K_ // 4)]
    big = (2.0 ** 14) * (1.0 + rng.random(a.shape[:-1] + (len(p),)))
    a[..., p] = big
    a[..., q] = -big
    return mx.array(a.astype(np.float32))


def main():
    D = os.path.dirname(os.path.abspath(__file__))
    caps = [np.load(os.path.join(D, f)) for f in ("attn_cap_he32.npz", "attn_cap_mb16.npz")]
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for which in (sys.argv[1:] or ["A", "B"]):
        alive = cands()
        print(f"== {which}: {len(alive)} candidates", flush=True)
        for ci, z in enumerate(caps):
            K1, V1 = mx.array(z["K1"]), mx.array(z["V1"])
            KV, S1, hd = K1.shape[1], K1.shape[2], K1.shape[3]
            for i in (0, int(z["n"]) // 2, int(z["n"]) - 1):
                q = mx.array(z[f"q_{i}"])
                r = q.shape[1] // KV
                qg = q.reshape(1, KV, r, 1, hd)
                tests = []
                for variant in ("real", "cancel"):
                    if which == "A":
                        K1X = K1 if variant == "real" else cancel_like(K1, 7)
                        qgX = qg if variant == "real" else cancel_like(qg, 9)
                        stock = qgX @ K1X[:, :, None].swapaxes(-1, -2)            # the stock op at the real shapes
                        for kv in range(KV):
                            tests.append((mx.contiguous(K1X[0, kv]), mx.contiguous(qgX[0, kv, :, 0, :]), stock[0, kv, :, 0, :]))
                    else:
                        st = qg @ K1[:, :, None].swapaxes(-1, -2)
                        et = mx.exp(st - st.max(-1, keepdims=True))
                        V1X = V1 if variant == "real" else mx.contiguous(cancel_like(V1.swapaxes(-1, -2), 11).swapaxes(-1, -2))
                        etX = et if variant == "real" else cancel_like(et, 13)
                        stock = etX @ V1X[:, :, None]
                        for kv in range(KV):
                            tests.append((mx.contiguous(V1X[0, kv]), mx.contiguous(etX[0, kv, :, 0, :]), stock[0, kv, :, 0, :]))
                keep = []
                for p in alive:
                    ok = True
                    for Wm, Xm, stock in tests:
                        try:
                            got = gemv_cand(Wm, Xm, p, transposed=(which == "B"))
                            if not eq(got, stock):
                                ok = False
                                break
                        except Exception:
                            ok = False
                            break
                    if ok:
                        keep.append(p)
                alive = keep
                print(f"   cap {ci} call {i} (S1 {S1}): {len(alive)} left", flush=True)
                if not alive:
                    break
            if not alive:
                break
        print(f"== {which}: survivors (BM, BN, SM, SN, TM, TN): {alive[:12]}{' ...' if len(alive) > 12 else ''}", flush=True)


if __name__ == "__main__":
    main()
