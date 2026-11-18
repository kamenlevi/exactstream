"""WEEK98 BET 7: the exact hyper-connection read as two kernels (steps S2 / S3, S7 / S8, S12 / S13 of the plan).

Stock (qwen4_exp.GatedResidual.__call__, eager, F = 1), every op rounding to bf16:
  [pending write]  hyper' = hyper + (block_out[None, :] * inject[:, None])          (Multiply, then Add)
  normed = rms_norm(hyper' as 4 rows of 2560, no weight) * hc_norm.weight           (RMSNorm, Multiply)
  mix    = nn.silu(normed @ W_down.T / 4)            (gemv; Divide; silu is mx.compile'd -> JIT, fast exp)
  mix    = mx.sigmoid(mix @ W_up.T)                   (gemv; standalone Sigmoid -> precompiled, precise exp)
  mixed  = mean(mix * normed, over the 4 streams)     (Multiply; Sum over axis -2 in bf16; x 0.25)
  inject = 2 * mx.sigmoid(normed @ W_inj.T / 4)       (gemv; Divide; Sigmoid; Multiply)   [combine variant only]
Fused:
  K_down: every threadgroup recomputes hyper' and the grouped RMS statistic in the stock lane layout (fl_exact's
          virtual lanes), keeps normed in threadgroup memory, and runs MLX's own GEMVKernel arithmetic over it
          (the shipped gemv.h text, with the input vector read from threadgroup memory); the epilogue applies
          / 4 + silu (down rows) or / 4 + sigmoid + x 2 (inject rows). Threadgroups also write their slice of normed
          (and of hyper') to device memory for K_up and the final residual write.
  K_up:   one simdgroup per column c computes the 4 stream rows c, c + 2560, c + 5120, c + 7680 (GEMVKernel with a
          row stride of 2560 rows), then sigmoid, x normed and the 4-stream mean in the stock order.
The GEMV parameters (BN, SN, TN) are the stock ones found by fl_match.py; the mean's order by fl_bittest.py."""
import os
import re

import mlx.core as mx

from fl_exact import RMS_HEADER, EW_HEADER

KDIR = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")  # [bundle] the running mlx's headers (was ~/BigMoE/.venv/.../kernels)


def _gemv_tg_struct():
    """GEMVKernel from gemv.h, renamed FLGEMV: the input vector and the outputs in threadgroup memory, the scalars
    by value; the arithmetic text is untouched"""
    s = open(os.path.join(KDIR, "gemv.h")).read()
    a = re.search(r"^template <\n    typename T,\n    const int BM,", s, re.M).start()
    b = re.search(r"^/{70,}\n/// Vector matrix", s[a + 1:], re.M).start() + a + 1
    t = s[a:b]
    t = t.replace("struct GEMVKernel", "struct FLGEMV")
    t = t.replace("const constant int&", "const int").replace("const constant float&", "const float")
    t = re.sub(r"\s*\[\[buffer\(\d+\)\]\]", "", t)
    t = re.sub(r"\s*\[\[threadgroup\(0\)\]\]", "", t)
    # the input vector lives in threadgroup memory: run() and the vector loads take threadgroup pointers
    t = t.replace("const device T* in_vec,", "const threadgroup T* in_vec,")
    t = t.replace("device T* out_vec,", "threadgroup T* out_vec,")
    # load_unsafe / load_safe: add threadgroup overloads (same bodies)
    for fn in ("load_unsafe", "load_safe"):
        m = re.search(r"  template <typename U = T>\n  static METAL_FUNC void\s*\n?\s*" + fn + r"\(.*?\n  }\n", t, re.S)
        body = m.group(0)
        t = t.replace(body, body + body.replace("const device T* src", "const threadgroup T* src"))
    return t


GEMV_TG = """
#define MLX_MTL_CONST static constant constexpr const
#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif
template <typename U> struct DefaultAccT { using type = float; };
""" + _gemv_tg_struct()

HC_HEADER = RMS_HEADER + EW_HEADER + GEMV_TG + """
// grouped stock RMS statistic of hyper' (4 groups of 2560) with the residual write folded in; part: >= 4 * 20 + 4
template <bool PENDING, typename PI>      // PI: small inputs arrive in constant memory
METAL_FUNC float fl_hc_val(const device bfloat16_t* hyper, const device bfloat16_t* bout, PI pinj, uint i) {
  if (PENDING) {
    bfloat16_t prod = bout[i % 2560] * pinj[i / 2560];          // Multiply (bf16)
    return float(static_cast<bfloat16_t>(hyper[i] + prod));      // Add (bf16)
  }
  return float(hyper[i]);
}

template <bool PENDING, typename PI>
METAL_FUNC void fl_hc_stats(const device bfloat16_t* hyper, const device bfloat16_t* bout, PI pinj, float eps,
                            threadgroup float* part, threadgroup float* inv, uint lane, uint sg, uint nsg) {
  // stock rms_single_row for axis 2560: 640 lanes = 20 simdgroups per group; 4 groups -> 80 virtual simdgroups
  for (uint v = sg; v < 80; v += nsg) {
    uint g = v / 20, vs = v % 20;
    uint vl = vs * 32 + lane;
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
      float xi = fl_hc_val<PENDING>(hyper, bout, pinj, g * 2560 + vl * 4 + i);
      acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (lane == 0) { part[g * 20 + vs] = acc; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 4) {
    float v = (lane < 20) ? part[sg * 20 + lane] : 0.0f;
    float tot = simd_sum(v);
    if (lane == 0) { inv[sg] = metal::precise::rsqrt(tot / 2560.0f + eps); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

# ---- K_down: grid (NDOWN + COMBINE) threadgroups of BN * 32 threads; rows per threadgroup = TM (= 4)
_DOWN_SRC = """
  constexpr int TM_ = 4;
  uint tg = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  uint nsg = threads_per_threadgroup.x / 32;
  threadgroup float part[84];
  threadgroup float inv[4];
  threadgroup bfloat16_t tn[10240];
  threadgroup float tgp_mem[GK::tgp_mem_size > 0 ? GK::tgp_mem_size : 1];
  threadgroup bfloat16_t tout[TM_];
  fl_hc_stats<PENDING>(hyper, bout, pinj, eps[0], part, inv, lane, sg, nsg);
  for (uint i = lid; i < 10240; i += threads_per_threadgroup.x) {
    float hv = fl_hc_val<PENDING>(hyper, bout, pinj, i);
    bfloat16_t nr = static_cast<bfloat16_t>(hv * inv[i / 2560]);        // rms_norm output (x 1, the no-weight form)
    bfloat16_t nv = nr * nw[i];                                          // x hc_norm.weight (bf16 Multiply)
    tn[i] = nv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // this threadgroup's slice of normed (and of the written stream) for the later kernels
  uint per = (10240 + NTG - 1) / NTG;
  for (uint i = tg * per + lid; i < min(10240u, (tg + 1) * per); i += threads_per_threadgroup.x) {
    normed[i] = tn[i];
    if (PENDING) { hw[i] = static_cast<bfloat16_t>(fl_hc_val<PENDING>(hyper, bout, pinj, i)); }
  }
  bool is_inj = COMBINE && (tg == NDOWN);
  const device bfloat16_t* mat = is_inj ? wi : (wd + size_t(tg) * TM_ * 10240);
  uint3 tid3 = uint3(0, 0, 0);
  uint3 lid3 = uint3(lid, 0, 0);
  GK::run(mat, tn, mat, tout, 10240, TM_, 10240, 1.0f, 0.0f, 0, tgp_mem, tid3, lid3, sg, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < TM_) {
    bfloat16_t d = tout[lid];
    bfloat16_t q = static_cast<bfloat16_t>(metal::precise::divide(float(d), 4.0f));   // / hc_count (bf16 Divide)
    if (is_inj) {
      bfloat16_t sgm = fl_sigmoid_pre(q);                                // standalone mx.sigmoid
      inj[lid] = static_cast<bfloat16_t>(2.0f * float(sgm));             // 2.0 * (bf16 Multiply, exact)
    } else {
      mix1[tg * TM_ + lid] = fl_silu_jit(q);                             // nn.silu (compiled)
    }
  }
"""

# ---- K_up: grid 2560 / CPT threadgroups of CPT simdgroups; simdgroup = one column, rows c + s * 2560
_UP_SRC = """
  constexpr int TM_ = 4;
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  uint c = threadgroup_position_in_grid.x * CPT + sg;
  threadgroup bfloat16_t tm1[320];
  threadgroup float tgp_mem[1];
  threadgroup bfloat16_t tout[CPT * TM_];
  for (uint i = thread_position_in_threadgroup.x; i < 320; i += threads_per_threadgroup.x) { tm1[i] = mix1[i]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device bfloat16_t* mat = wu + size_t(c) * 320;
  uint3 tid3 = uint3(0, 0, 0);
  uint3 lid3 = uint3(lane, 0, 0);
  GK::run(mat, tm1, mat, tout + sg * TM_, 320, TM_, 2560 * 320, 1.0f, 0.0f, 0, tgp_mem, tid3, lid3, 0, lane);
  if (lane == 0) {
    bfloat16_t p[4];
    for (int s = 0; s < 4; s++) {
      bfloat16_t z = fl_sigmoid_pre(tout[sg * TM_ + s]);                 // standalone mx.sigmoid
      p[s] = z * normed[s * 2560 + c];                                   // mix * grouped (bf16 Multiply)
    }
    bfloat16_t acc;
    if (MEAN_MODE == 0) {                    // bf16 running sum from Sum's init 0, in row order
      acc = static_cast<bfloat16_t>(0.0f);
      for (int s = 0; s < 4; s++) { acc = p[s] + acc; }
    } else if (MEAN_MODE == 1) {             // float running sum, one rounding
      float f = 0.0f;
      for (int s = 0; s < 4; s++) { f = float(p[s]) + f; }
      acc = static_cast<bfloat16_t>(f);
    } else {                                 // pairwise in bf16
      acc = static_cast<bfloat16_t>(float(static_cast<bfloat16_t>(p[0] + p[1])) +
                                    float(static_cast<bfloat16_t>(p[2] + p[3])));
    }
    mixed[c] = acc * static_cast<bfloat16_t>(0.25f);                     // x 1 / 4 (bf16 Multiply)
  }
"""


def _typedef(BN, SN, TN, TM=4):
    return f"\ntypedef FLGEMV<bfloat16_t, 1, {BN}, {32 // SN}, {SN}, {TM}, {TN}, false> GK;\n"


_KS = {}


def k_down(BN, SN, TN, combine, pending):
    key = ("down", BN, SN, TN, combine, pending)
    if key not in _KS:
        _KS[key] = mx.fast.metal_kernel(
            name=f"fl_hc_down_{BN}_{SN}_{TN}_{int(combine)}_{int(pending)}",
            input_names=["hyper", "bout", "pinj", "nw", "wd", "wi", "eps"],
            output_names=["mix1", "inj", "normed", "hw"],
            header=HC_HEADER + _typedef(BN, SN, TN), source=_DOWN_SRC)
    return _KS[key]


def k_up(BN, SN, TN, mean_mode, cpt=8):
    key = ("up", BN, SN, TN, mean_mode, cpt)
    if key not in _KS:
        _KS[key] = mx.fast.metal_kernel(
            name=f"fl_hc_up_{BN}_{SN}_{TN}_{mean_mode}_{cpt}", input_names=["mix1", "wu", "normed"],
            output_names=["mixed"], header=HC_HEADER + _typedef(BN, SN, TN), source=_UP_SRC)
    return _KS[key]


_CONST = {}


def _const(key, make):
    """small constant inputs made once (a fresh mx.zeros / mx.array per call would add work to every level)"""
    if key not in _CONST:
        _CONST[key] = make()
        mx.eval(_CONST[key])
    return _CONST[key]


DEFAULT_MEAN_MODE = 0      # 0 = the stock order; a test sets 1 to run a deliberately wrong level (negative control)


def hc_read(gr, hyper, pending=None, down=(8, 32, 4), up=(1, 32, 4), mean_mode=None, cpt=8):
    """the fused read of GatedResidual `gr` on hyper [10240] (F = 1). Returns (mixed [2560], inject [4] or None,
    the written stream [10240] (hyper itself without a pending write), normed [10240])"""
    combine = "block_inject_weight" in gr
    ndown = 320 // 4
    ntg = ndown + (1 if combine else 0)
    if pending is not None:
        bout, pinj = pending[0].reshape(-1), pending[1].reshape(-1)
    else:
        bout = _const("z2560", lambda: mx.zeros((2560,), mx.bfloat16))
        pinj = _const("z4", lambda: mx.zeros((4,), mx.bfloat16))
    wi = gr.block_inject_weight.weight if combine else gr.input_mix_weight_down.weight
    BN, SN, TN = down
    kd = k_down(BN, SN, TN, combine, pending is not None)
    mix1, inj, normed, hw = kd(
        inputs=[hyper.reshape(-1), bout, pinj, gr.hc_norm.weight, gr.input_mix_weight_down.weight, wi,
                _const(("eps", gr.hc_norm.eps), lambda: mx.array([gr.hc_norm.eps], mx.float32))],
        template=[("PENDING", pending is not None), ("COMBINE", combine), ("NDOWN", ndown), ("NTG", ntg)],
        grid=(ntg * BN * 32, 1, 1), threadgroup=(BN * 32, 1, 1),
        output_shapes=[(320,), (4,), (10240,), (10240,)], output_dtypes=[mx.bfloat16] * 4)
    mean_mode = DEFAULT_MEAN_MODE if mean_mode is None else mean_mode
    BNu, SNu, TNu = up
    ku = k_up(BNu, SNu, TNu, mean_mode, cpt)
    mixed = ku(inputs=[mix1, gr.input_mix_weight_up.weight, normed], template=[("MEAN_MODE", mean_mode), ("CPT", cpt)],
               grid=(2560 // cpt * 32 * cpt, 1, 1), threadgroup=(32 * cpt, 1, 1), output_shapes=[(2560,)],
               output_dtypes=[mx.bfloat16])[0]
    written = hw if pending is not None else hyper.reshape(-1)
    return mixed, (inj if combine else None), written, normed
