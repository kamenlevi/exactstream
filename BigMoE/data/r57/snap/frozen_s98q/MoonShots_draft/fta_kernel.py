"""FUSED TREE LEVEL, step 1c (MoonShots, Tue 17 Nov 2026): dd_tree.tree_attn for one pop (n = 1 row) as ONE Metal
kernel, bitwise. One threadgroup (32 x 32 threads) per query head h (kv = h / r), phases in order, each reproducing
the stock op's per-output order as matched on real captured inputs (fta_match.py / fta_rmatch.py):
  1 st = q_h . K1[kv][s] for s < S1   GEMVKernel<float, 32, 1, 1, 32, 1, 4> run per 32-row block (stock: SN 32, TN 4)
  2 st * scale; where(s <= anc, st, NEG)
  3 sc_j = (q_h * AK_j) row sum       products rounded alone (no FMA), lane l sums its 4-runs at 4l, 4l + 128; simd_sum
    sc_j * scale; where(am_j, sc_j, NEG)
  4 m = max(max st, max sc)           (exact in any order)
  5 et = exp(st - m), ec = exp(sc - m) (precise exp, the subtract rounded alone)
  6 o = et @ V1[kv]                   GEMVTKernel<float, 1, 8, 8, 4, 4, 8> (stock: BM 1, SM 8, SN 4, TM 4)
  7 o = o + ec_j * AV_j for j in order (multiply and add rounded alone)
  8 o / (et.sum + ec.sum)             et.sum: one simdgroup (lane 4-runs, blocks of 128, the stock tail rule), simd_sum;
                                      ec.sum: sequential; precise divide
Scratch: per-head scores (S1 floats) and the prefix output (256) in output buffers (any S1).
Bit test: against dd_tree.tree_attn on every captured call (draft/attn_cap_*.npz) + timing of both.
usage: ~/BigMoE/.venv/bin/python -I -B fta_kernel.py [--reps 200]   (GPU, seconds)"""
import argparse
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fta_match import HDR as GEMV_HDR                      # noqa: E402  (GEMVKernel / GEMVTKernel from gemv.h)

HDR = GEMV_HDR + """
typedef GEMVKernel<float, 32, 1, 1, 32, 1, 4, false> FTA_GK;      // scores: 32 rows per block, SN 32, TN 4
typedef GEMVTKernel<float, 1, 8, 8, 4, 4, 8, false> FTA_GT;       // et @ V1: 256 outputs in one block
METAL_FUNC float fta_nofma_mul(float a, float b) { return as_type<float>(as_type<uint>(a * b)); }
"""

SRC = """
  #pragma clang fp contract(off)
  const uint h = threadgroup_position_in_grid.x;
  const uint kv = h / R;
  const uint lx = thread_position_in_threadgroup.x, ly = thread_position_in_threadgroup.y;
  const uint lid = ly * 32 + lx;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const float scale = scl[0];
  const float NEGV = -1e9f;
  const device float* qh = q + h * HD;
  const device float* K1kv = K1 + kv * S1 * HD;
  const device float* V1kv = V1 + kv * S1 * HD;
  device float* sth = st + h * S1;
  device float* opre = op + h * HD;
  threadgroup float tgp_memory[1];
  threadgroup float scj[32];
  threadgroup float ecj[32];
  threadgroup float red[32];
  threadgroup float mshared[2];
  // 1. scores, GEMVKernel's own order (per 32-row block)
  for (int blk = 0; blk < (S1 + 31) / 32; ++blk) {
    FTA_GK::run(K1kv, qh, K1kv, sth, HD, S1, HD, 1.0f, 0.0f, 0, tgp_memory, uint3(blk, 0, 0), uint3(lx, ly, 0), sg, lane);
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  // 2. scale and the prefix mask
  for (int s = lid; s < S1; s += 1024) {
    float v = sth[s] * scale;
    sth[s] = (s <= anc[0]) ? v : NEGV;
  }
  // 3. the tree columns (one simdgroup per column)
  for (int j = sg; j < J; j += 32) {
    const device float* ak = AK + (kv * J + j) * HD;
    float total = 0.0f;
    for (int b = 0; b < HD / 128; ++b) {
      for (int i = 0; i < 4; ++i) {
        int k = b * 128 + lane * 4 + i;
        float pr = fta_nofma_mul(qh[k], ak[k]);
        total = pr + total;
      }
    }
    total = simd_sum(total);
    if (lane == 0) { float v = total * scale; scj[j] = am[j] ? v : NEGV; }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  // 4. the max (exact in any order)
  float mx_ = -INFINITY;
  for (int s = lid; s < S1; s += 1024) { mx_ = max(mx_, sth[s]); }
  mx_ = simd_max(mx_);
  if (lane == 0) { red[sg] = mx_; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float a = -INFINITY;
    for (int i = 0; i < 32; ++i) { a = max(a, red[i]); }
    float c = -INFINITY;
    for (int j = 0; j < J; ++j) { c = max(c, scj[j]); }
    mshared[0] = max(a, c);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float m = mshared[0];
  // 5. exp
  for (int s = lid; s < S1; s += 1024) { float d = sth[s] - m; sth[s] = metal::precise::exp(d); }
  if (lid < (uint)J) { float d = scj[lid] - m; ecj[lid] = metal::precise::exp(d); }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  // 6. the prefix output, GEMVTKernel's own order (threads y < 8)
  if (ly < 8) {
    FTA_GT::run(V1kv, sth, V1kv, opre, S1, HD, HD, 1.0f, 0.0f, 0, tgp_memory, uint3(0, 0, 0), uint3(lx, ly, 0), sg, lane);
  }
  // 8a. the denominators: et.sum (one simdgroup, the stock row_reduce_simple order with 32 threads)
  if (sg == 0) {
    int blocks = S1 / 128;
    int extra = S1 - blocks * 128;
    float total = 0.0f;
    const device float* p = sth + lane * 4;
    for (int b = 0; b < blocks; ++b) {
      for (int i = 0; i < 4; ++i) { total = p[i] + total; }
      p += 128;
    }
    int index = lane * 4;
    if (index + 4 <= extra) {
      for (int i = 0; i < 4; ++i) { total = p[i] + total; }
    } else {
      for (int i = 0; index + i < extra; ++i) { total = p[i] + total; }
    }
    total = simd_sum(total);
    if (lane == 0) {
      float ce = 0.0f;
      for (int j = 0; j < J; ++j) { ce = ecj[j] + ce; }
      mshared[1] = total + ce;
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  // 7 + 8b. the tree values in order, then the divide
  if (lid < HD) {
    float o = opre[lid];
    for (int j = 0; j < J; ++j) {
      float t = fta_nofma_mul(ecj[j], AV[(kv * J + j) * HD + lid]);
      o = o + t;
    }
    out[h * HD + lid] = metal::precise::divide(o, mshared[1]);
  }
"""
_K = {}


def tree_attn_fused(scale, q, K1, V1, AK, AV, amask, anc):
    """q [1,H,1,hd] f32, K1 / V1 [1,KV,S1,hd], AK / AV [1,KV,1,J,hd], amask [1,J] bool, anc [1] int32 -> [1,H,1,hd]"""
    _, Hh, n, hd = q.shape
    assert n == 1 and hd == 256
    KV, S1 = K1.shape[1], K1.shape[2]
    J = AK.shape[3]
    assert J <= 32
    k = _K.setdefault("fta", mx.fast.metal_kernel(
        name="fta_tree_attn", input_names=["q", "K1", "V1", "AK", "AV", "am", "anc", "scl"],
        output_names=["out", "st", "op"], header=HDR, source=SRC))
    o = k(inputs=[q.reshape(-1), K1.reshape(-1), V1.reshape(-1), AK.reshape(-1), AV.reshape(-1),
                  amask.reshape(-1), anc.reshape(-1).astype(mx.int32), mx.array([scale], mx.float32)],
          template=[("S1", S1), ("J", J), ("HD", hd), ("R", Hh // KV)],
          grid=(Hh * 32, 32, 1), threadgroup=(32, 32, 1),
          output_shapes=[(Hh * hd,), (Hh * S1,), (Hh * hd,)], output_dtypes=[mx.float32] * 3)
    return o[0].reshape(1, Hh, 1, hd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dd_tree as DT
    D = os.path.dirname(os.path.abspath(__file__))
    tot = eqn = 0
    worst = 0.0
    times = {"stock": [], "fused": []}
    for f in ("attn_cap_he32.npz", "attn_cap_mb16.npz"):
        z = np.load(os.path.join(D, f))
        K1, V1 = mx.array(z["K1"]), mx.array(z["V1"])
        scale = float(z["scale"])
        for i in range(int(z["n"])):
            q = mx.array(z[f"q_{i}"]); AK = mx.array(z[f"AK_{i}"]); AV = mx.array(z[f"AV_{i}"])
            am = mx.array(z[f"am_{i}"]); anc = mx.array(z[f"anc_{i}"])
            ref = DT.tree_attn(mx, scale, q, K1, V1, AK, AV, am, anc)
            got = tree_attn_fused(scale, q, K1, V1, AK, AV, am, anc)
            mx.eval(ref, got)
            same = bool(mx.array_equal(ref.view(mx.uint32), got.view(mx.uint32)).item())
            tot += 1
            eqn += int(same)
            if not same:
                worst = max(worst, float(mx.max(mx.abs(ref - got)).item()))
                if tot - eqn <= 3:
                    nd = int((ref.view(mx.uint32) != got.view(mx.uint32)).sum().item())
                    print(f"  {f} call {i} (J {AK.shape[3]}): {nd} / {ref.size} values differ, max abs {worst:.3g}", flush=True)
            if i in (0, int(z["n"]) // 2, int(z["n"]) - 1):
                for name, fn in (("stock", lambda: DT.tree_attn(mx, scale, q, K1, V1, AK, AV, am, anc)),
                                 ("fused", lambda: tree_attn_fused(scale, q, K1, V1, AK, AV, am, anc))):
                    for _ in range(5):
                        mx.eval(fn())
                    mx.synchronize(); t0 = time.perf_counter()
                    outs = [fn() for _ in range(a.reps)]
                    mx.eval(outs)
                    mx.synchronize()
                    times[name].append((time.perf_counter() - t0) * 1e6 / a.reps)
    print(f"bitwise: {eqn} / {tot} calls equal{'' if eqn == tot else f' (max abs diff {worst:.3g})'}", flush=True)
    print("us per call (independent calls, pipelined): stock " + ", ".join(f"{x:.1f}" for x in times["stock"])
          + " | fused " + ", ".join(f"{x:.1f}" for x in times["fused"]), flush=True)


if __name__ == "__main__":
    main()
