"""FUSED TREE LEVEL, step 1b (MoonShots, Tue 17 Nov 2026): which stock MLX 0.32.2 row reduction runs each of
dd_tree.tree_attn's fp32 sums at B = 1, so the fused attention kernel can reproduce them bitwise:
  C  (qg * AK_j).sum(-1)   24 rows of hd = 256 (the tree columns' scores)
  D  et.sum(-1)            24 rows of S1 (the prefix softmax denominators)
  E  ec.sum(-1)            24 rows of J (the tree part of the denominators)
Candidates (the shipped reduce_row.h logic, one row per threadgroup, N_READS = 4):
  seq      one thread, left to right (row_reduce_small's thread_reduce)
  tg<L>    row_reduce_simple with L threads: thread t sums its 4-element runs at 4t + 4Lb, then the tail, in order;
           simd_sum per simdgroup; then lanes < #simdgroups take the partials, the rest 0, and simd_sum again
Compared bit for bit with the stock op on the REAL captured inputs (draft/attn_cap_*.npz) and on cancellation rows.
usage: ~/BigMoE/.venv/bin/python -I -B fta_rmatch.py   (GPU, seconds)"""
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fta_match import cancel_like, eq                     # noqa: E402

SRC_TG = """
  // one threadgroup of L threads per row (row_reduce_simple with N_WRITES = 1 per row)
  uint row = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint L = threads_per_threadgroup.x;
  uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  uint nsg = (L + 31) / 32;
  threadgroup float shared_vals[32];
  const device float* in = x + row * N;
  int blocks = N / (L * 4);
  int extra = N - blocks * (L * 4);
  float total = 0.0f;
  const device float* p = in + lid * 4;
  for (int b = 0; b < blocks; b++) {
    for (int i = 0; i < 4; i++) { total = p[i] + total; }
    p += L * 4;
  }
  int index = lid * 4;
  if (index + 4 <= extra) {
    for (int i = 0; i < 4; i++) { total = p[i] + total; }
  } else {
    for (int i = 0; index + i < extra; i++) { total = p[i] + total; }
  }
  total = simd_sum(total);
  if (nsg > 1) {
    if (lane == 0) { shared_vals[sg] = total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = (lid < nsg) ? shared_vals[lid] : 0.0f;
    total = simd_sum(v);
  }
  if (lid == 0) { out[row] = total; }
"""
SRC_SEQ = """
  uint row = thread_position_in_grid.x;
  const device float* r = x + row * N;
  float total = 0.0f;
  int blocks = N / 4, extra = N % 4;
  for (int b = 0; b < blocks; b++) {
    float v[4];
    for (int j = 0; j < 4; j++) { v[j] = r[j]; }
    for (int j = 0; j < 4; j++) { total = v[j] + total; }
    r += 4;
  }
  for (int i = 0; i < extra; i++) { total = r[i] + total; }
  out[row] = total;
"""
_K = {}


def cand(x2, mode):
    """x2 [R, N] f32 contiguous -> [R] with the candidate's order"""
    R, N = x2.shape
    if mode == "seq":
        k = _K.setdefault("seq", mx.fast.metal_kernel(name="fta_rseq", input_names=["x"], output_names=["out"], source=SRC_SEQ))
        return k(inputs=[x2], template=[("N", N)], grid=(R, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(R,)],
                 output_dtypes=[mx.float32])[0]
    L = int(mode[2:])
    k = _K.setdefault("tg", mx.fast.metal_kernel(name="fta_rtg", input_names=["x"], output_names=["out"], source=SRC_TG))
    return k(inputs=[x2], template=[("N", N)], grid=(R * L, 1, 1), threadgroup=(L, 1, 1), output_shapes=[(R,)],
             output_dtypes=[mx.float32])[0]


MODES = ["seq"] + [f"tg{L}" for L in (32, 64, 96, 128, 160, 192, 224, 256, 512, 1024)]


def main():
    D = os.path.dirname(os.path.abspath(__file__))
    caps = [np.load(os.path.join(D, f)) for f in ("attn_cap_he32.npz", "attn_cap_mb16.npz")]
    NEG = -1e9
    for which in ("C", "D", "E"):
        alive = list(MODES)
        for ci, z in enumerate(caps):
            K1 = mx.array(z["K1"])
            KV, S1, hd = K1.shape[1], K1.shape[2], K1.shape[3]
            scale = float(z["scale"])
            for i in range(int(z["n"])):
                q = mx.array(z[f"q_{i}"]); AK = mx.array(z[f"AK_{i}"]); am = mx.array(z[f"am_{i}"])
                anc = mx.array(z[f"anc_{i}"])
                r = q.shape[1] // KV
                qg = q.reshape(1, KV, r, 1, hd)
                J = AK.shape[3]
                rows = []
                if which == "C":
                    for jj in range(J):
                        prod = qg * AK[:, :, None, :, jj]
                        rows.append((prod, prod.sum(-1)))
                    pc = cancel_like(qg * AK[:, :, None, :, 0], 3 + i)
                    rows.append((pc, pc.sum(-1)))
                else:
                    st = (qg @ K1[:, :, None].swapaxes(-1, -2)) * scale
                    st = mx.where(mx.arange(S1)[None, :] <= anc[:, None], st, NEG)
                    sc = mx.stack([(qg * AK[:, :, None, :, jj]).sum(-1) for jj in range(J)], -1) * scale
                    sc = mx.where(am, sc, NEG)
                    m = mx.maximum(st.max(-1, keepdims=True), sc.max(-1, keepdims=True))
                    et, ec = mx.exp(st - m), mx.exp(sc - m)
                    x = et if which == "D" else ec
                    rows.append((x, x.sum(-1, keepdims=True)))
                    xc = cancel_like(x, 5 + i)
                    rows.append((xc, xc.sum(-1, keepdims=True)))
                keep = []
                for mode in alive:
                    ok = True
                    for x, stock in rows:
                        x2 = mx.contiguous(x).reshape(-1, x.shape[-1])
                        if mode.startswith("tg") and int(mode[2:]) > 1024:
                            ok = False; break
                        if not eq(cand(x2, mode), stock.reshape(-1)):
                            ok = False; break
                    if ok:
                        keep.append(mode)
                alive = keep
                if not alive:
                    break
            print(f"   {which} after cap {ci}: {alive}", flush=True)
            if not alive:
                break
        print(f"== {which}: survivors {alive}", flush=True)


if __name__ == "__main__":
    main()
