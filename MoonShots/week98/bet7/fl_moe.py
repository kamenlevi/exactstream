"""WEEK98 BET 7: the exact MoE block (steps S9, S9b, S10, S11 of the plan; F = 1).

Stock (mlx_lm Qwen3NextSparseMoeBlock.__call__, eager), every op rounding at its dtype:
  gates = softmax(gate(x), precise)            qmv_fast 8-bit g64 -> softmax_single_row (fast exp, AccT float)
  inds  = argpartition(gates, kth=-10)[-10:]   stable block merge sort -> the 10 largest by (value, index), ascending
  sc    = gates[inds]; sc = sc / sc.sum()      row_reduce_small: bf16 running sum in slot order; bf16 Divide
  y     = SwitchGLU(x, inds)                   gather_qmv_fast 4-bit g32 (gate, up) -> swiglu (compiled) ->
                                               gather_qmv 4-bit g32 (down, K = 640)
  y     = (y * sc[:, None]).sum(axis=-2)       bf16 Multiply; col_reduce_small: bf16 running sum in slot order
  sy    = sigmoid(seg(x)) * shared_expert(x)   qmv (N = 1); swiglu(qmv_fast gate, up) -> qmv (K = 640) down
  out   = y + sy
Fused, four dependent kernels:
  S9   router (64 TGs) | shared gate + up + swiglu (80 TGs) | shared-expert gate (1 TG)
  S9b  route: softmax, top-10 by rank (the sort's total order without sorting), scores, sum, weights (1 TG)
  S10  routed gate + up + swiglu for the 10 experts (800 TGs) | shared down (320 TGs)
  S11  routed down for the 10 experts + the weighted sum + the shared branch (320 TGs, 8 columns each)
All GEMVs call quantized.h's own qmv_fast_impl / qmv_impl (fl_qmatch.QHEADER)."""
import mlx.core as mx

from fl_exact import EW_HEADER
from fl_qmatch import QHEADER

HDR = QHEADER + EW_HEADER

_S9 = """
  uint tgx = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint tid = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
  if (tgx < 64) {
    qmv_fast_impl<bfloat16_t, 64, 8>(rw, rs, rb, x, rl, 2560, 512, uint3(0, tgx, 0), sg, lane);
  } else if (tgx < 144) {
    uint t = tgx - 64;
    qmv_fast_impl<bfloat16_t, 64, 4>(gw, gsc, gb, x, gout, 2560, 640, uint3(0, t, 0), sg, lane);
    qmv_fast_impl<bfloat16_t, 64, 4>(uw, usc, ub, x, uout, 2560, 640, uint3(0, t, 0), sg, lane);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < 8) { uint j = t * 8 + tid; sh[j] = fl_silu_jit(gout[j]) * uout[j]; }   // swiglu (compiled)
  } else {
    qmv_impl<bfloat16_t, 64, 4>(segw, segs, segb, x, segl, 2560, 1, uint3(0, 0, 0), sg, lane);
  }
"""

_S9B = """
  // one threadgroup of 128 threads = the stock softmax_single_row layout for 512 values (4 per lane)
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
  threadgroup float local_max[32];
  threadgroup float local_norm[32];
  threadgroup bfloat16_t P[512];
  float ld[4];
  for (int i = 0; i < 4; i++) { ld[i] = float(rl[lid * 4 + i]); }
  if (sgi == 0) { local_max[lane] = Limits<float>::min; local_norm[lane] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  for (int i = 0; i < 4; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
  maxval = simd_max(maxval);
  if (lane == 0) { local_max[sgi] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0) { maxval = simd_max(local_max[lane]); if (lane == 0) { local_max[0] = maxval; } }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  for (int i = 0; i < 4; i++) { float e = fast::exp(ld[i] - maxval); ld[i] = e; normalizer += e; }
  normalizer = simd_sum(normalizer);
  if (lane == 0) { local_norm[sgi] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0) { normalizer = simd_sum(local_norm[lane]); if (lane == 0) { local_norm[0] = normalizer; } }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = metal::precise::divide(1.0f, local_norm[0]);
  for (int i = 0; i < 4; i++) { P[lid * 4 + i] = static_cast<bfloat16_t>(ld[i] * normalizer); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // top-10 = the last 10 of the stable ascending sort: rank = elements ahead in (value, index) descending order
  threadgroup uint ti[10];
  threadgroup bfloat16_t tv[10];
  for (int k = 0; k < 4; k++) {
    uint i = lid * 4 + k;
    bfloat16_t v = P[i];
    uint rank = 0;
    for (uint j = 0; j < 512; j++) {
      bfloat16_t u = P[j];
      rank += (u > v || (u == v && j > i)) ? 1 : 0;
    }
    if (rank < 10) { ti[9 - rank] = i; tv[9 - rank] = v; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    bfloat16_t s = static_cast<bfloat16_t>(0.0f);
    for (int e = 0; e < 10; e++) { s = tv[e] + s; }                          // row_reduce_small, slot order
    for (int e = 0; e < 10; e++) {
      inds[e] = ti[e];
      wts[e] = static_cast<bfloat16_t>(metal::precise::divide(float(tv[e]), float(s)));   // bf16 Divide
      probs_top[e] = tv[e];
    }
  }
"""

_S10 = """
  uint tgx = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint tid = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
  if (tgx < 800) {
    uint e = tgx / 80, t = tgx % 80;
    size_t ex = inds[e];
    qmv_fast_impl<bfloat16_t, 32, 4>(egw + ex * (640 * 320), egs + ex * (640 * 80), egb + ex * (640 * 80), x,
                                    gout + e * 640, 2560, 640, uint3(0, t, 0), sg, lane);
    qmv_fast_impl<bfloat16_t, 32, 4>(euw + ex * (640 * 320), eus + ex * (640 * 80), eub + ex * (640 * 80), x,
                                    uout + e * 640, 2560, 640, uint3(0, t, 0), sg, lane);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < 8) { uint j = e * 640 + t * 8 + tid; eh[j] = fl_silu_jit(gout[j]) * uout[j]; }
  } else {
    uint t = tgx - 800;
    qmv_impl<bfloat16_t, 64, 4>(sdw, sds, sdb, sh, sout, 640, 2560, uint3(0, t, 0), sg, lane);
  }
"""

_S11 = """
  uint t = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  uint tid = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
  for (int e = 0; e < 10; e++) {
    size_t ex = inds[e];
    qmv_impl<bfloat16_t, 32, 4>(edw + ex * (2560 * 80), eds + ex * (2560 * 20), edb + ex * (2560 * 20), eh + e * 640,
                               yout + e * 2560, 640, 2560, uint3(0, t, 0), sg, lane);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid < 8) {
    uint c = t * 8 + tid;
    // col_reduce_small with 8 threads along the 10 rows: thread j sums rows j, j + 8 from Sum's init (bf16),
    // then thread 0 folds partials 1..7 into its own, in order (found by bit-matching the stock sum)
    bfloat16_t part[8];
    for (int j = 0; j < 8; j++) {
      bfloat16_t t8 = static_cast<bfloat16_t>(0.0f);
      for (int e = j; e < 10; e += 8) {
        bfloat16_t p = yout[e * 2560 + c] * wts[e];                          // y * scores (bf16 Multiply)
        t8 = p + t8;
      }
      part[j] = t8;
    }
    bfloat16_t acc = part[0];
    for (int j = 1; j < 8; j++) { acc = part[j] + acc; }
    bfloat16_t sy = fl_sigmoid_pre(segl[0]) * sout[c];                       // sigmoid(seg) * shared (bf16)
    out[c] = acc + sy;                                                       // y + shared_y (bf16)
  }
"""

_K = {}


def _k(name, inputs, outputs, src):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, header=HDR, source=src)
    return _K[name]


def moe(blk, x):
    """the fused MoE on x [2560] bf16 (blk = the stock SparseMoeBlock, its weights read in place).
    Returns (out [2560], intermediates dict)"""
    x = x.reshape(-1)
    g, se, sw, seg = blk.gate, blk.shared_expert, blk.switch_mlp, blk.shared_expert_gate
    BF = mx.bfloat16
    s9 = _k("fl_moe_s9", ["x", "rw", "rs", "rb", "gw", "gsc", "gb", "uw", "usc", "ub", "segw", "segs", "segb"],
            ["rl", "gout", "uout", "sh", "segl"], _S9)
    rl, _, _, sh, segl = s9(
        inputs=[x, g["weight"], g["scales"], g["biases"], se.gate_proj["weight"], se.gate_proj["scales"],
                se.gate_proj["biases"], se.up_proj["weight"], se.up_proj["scales"], se.up_proj["biases"],
                seg["weight"], seg["scales"], seg["biases"]],
        grid=(145 * 32, 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(512,), (640,), (640,), (640,), (1,)], output_dtypes=[BF] * 5)
    s9b = _k("fl_moe_s9b", ["rl"], ["inds", "wts", "probs_top"], _S9B)
    inds, wts, ptop = s9b(inputs=[rl], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(10,), (10,), (10,)], output_dtypes=[mx.uint32, BF, BF])
    s10 = _k("fl_moe_s10", ["x", "inds", "egw", "egs", "egb", "euw", "eus", "eub", "sh", "sdw", "sds", "sdb"],
             ["gout", "uout", "eh", "sout"], _S10)
    _, _, eh, sout = s10(
        inputs=[x, inds, sw.gate_proj["weight"], sw.gate_proj["scales"], sw.gate_proj["biases"],
                sw.up_proj["weight"], sw.up_proj["scales"], sw.up_proj["biases"], sh, se.down_proj["weight"],
                se.down_proj["scales"], se.down_proj["biases"]],
        grid=((800 + 320) * 32, 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(6400,), (6400,), (6400,), (2560,)], output_dtypes=[BF] * 4)
    s11 = _k("fl_moe_s11", ["inds", "wts", "eh", "edw", "eds", "edb", "segl", "sout"], ["yout", "out"], _S11)
    yout, out = s11(
        inputs=[inds, wts, eh, sw.down_proj["weight"], sw.down_proj["scales"], sw.down_proj["biases"], segl, sout],
        grid=(320 * 32, 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(25600,), (2560,)], output_dtypes=[BF, BF])
    return out, dict(rl=rl, inds=inds, wts=wts, sh=sh, segl=segl, eh=eh, sout=sout, yout=yout)
