"""WEEK98 BET 7: the remaining fused steps of the level (S1 prep, S4 projections, S6 o_proj, S14 head, S15 tail)
and the level driver. Each step reproduces the stock eager arithmetic (see fl_hc.py / fl_moe.py for the others):

S1  prep: tok_emb = dequantize(embed rows[tok]) (affine_dequantize: scale * d + bias); en = fc_embedding(rms(tok_emb))
    (gemv BN1 SN32 TN4); hn = rms(widened) (looped, 1024 lanes) as 4 substreams; fh = fc_hidden(hn) (gemv_wide,
    k_lanes 32, 4 vectors); h = fh + en (bf16 Add). Every threadgroup recomputes the norms (and the embedding row).
S4  q (12,288), k, v (512 each) at 4-bit g64 and the indexer qk (640) at 8-bit g64: qmv_fast over mixed, one dispatch.
S6  o_proj: qmv_fast 4-bit g64 (K = 6,144) over the gated attention output.
S14 head: qmv_fast 4-bit g64 over the mixer's x, the screen's rows (bf16 logits; the stock casts them to fp32).
S15 tail (one threadgroup of 1024): logsumexp_looped (the stock order; fast exp, precise log), lp = lg - lse (fp32),
    top-K of lp by (value, index) in ascending order (= argpartition's last K), klp, cand = sc + klp, the frontier
    (F = 1: the largest cand by (value, position)), the token id and the path score."""
import os
import re

import mlx.core as mx

from fl_exact import RMS_HEADER
from fl_hc import GEMV_TG, _const
from fl_qmatch import QHEADER

KDIR = os.path.expanduser("~/BigMoE/.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels")


def _gemv_wide_tg():
    """GemvWide from gemv.h as FLGEMVW: the input vectors and the outputs in threadgroup memory, no bias"""
    s = open(os.path.join(KDIR, "gemv.h")).read()
    a = re.search(r"^template <\n    typename T,\n    const int vecs_per_tg,", s, re.M).start()
    b = re.search(r"^template <typename T, const int vecs_per_tg, const int k_lanes>\n\[\[kernel\]\] void gemv_wide\(",
                  s[a + 1:], re.M).start() + a + 1
    t = s[a:b].replace("struct GemvWide", "struct FLGEMVW").replace("gemv_wide_do_axpby", "false")
    t = t.replace("const device T* in_vec,", "const threadgroup T* in_vec,")
    t = t.replace("device T* out_vec,", "threadgroup T* out_vec,")
    t = t.replace("const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;",
                  "const threadgroup vec<T, 4>* x4 = (const threadgroup vec<T, 4>*)in_vec;")
    assert "threadgroup vec<T, 4>* x4" in t and "threadgroup T* out_vec" in t
    return t


S1_HEADER = RMS_HEADER + GEMV_TG + _gemv_wide_tg() + """
typedef FLGEMV<bfloat16_t, 1, 1, 1, 32, 4, 4, false> GKE;     // fc_embedding: gemv BN1 SN32 TN4
typedef FLGEMVW<bfloat16_t, 4, 32> GKW;                         // fc_hidden: gemv_wide k_lanes 32 (4 vectors)
"""

_S1 = """
  // threadgroup: 4 simdgroups (128 threads); 4 output columns c0..c0+3 per threadgroup
  uint tg = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  uint nsg = threads_per_threadgroup.x / 32;
  threadgroup float part[33];
  threadgroup bfloat16_t tn[10240];
  threadgroup bfloat16_t te[2560];
  threadgroup float tgp[1];
  threadgroup bfloat16_t ofh[16];
  threadgroup bfloat16_t oen[4];
  // the embedding row, as affine_dequantize writes it (2 values per byte, out = scale * d + bias)
  const device uint8_t* wrow = (const device uint8_t*)(ew + size_t(tok[0]) * 320);
  for (uint o = lid; o < 1280; o += threads_per_threadgroup.x) {
    uint oindex = o * 2;
    bfloat16_t scale = es[size_t(tok[0]) * 40 + oindex / 64];
    bfloat16_t bias = eb[size_t(tok[0]) * 40 + oindex / 64];
    uint val = wrow[o];
    for (int i = 0; i < 2; i++) {
      uint8_t d = (val >> (4 * i)) & 0x0f;
      te[oindex + i] = scale * d + bias;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_e = fl_rms_inv_p<2560>((const threadgroup bfloat16_t*)te, eps[0], part, lane, sg, nsg);
  for (uint i = lid; i < 2560; i += threads_per_threadgroup.x) {
    te[i] = nwe[i] * static_cast<bfloat16_t>(float(te[i]) * inv_e);           // pre_fc_norm_embedding
  }
  float inv_h = fl_rms_inv_p<10240>(widened, eps[0], part, lane, sg, nsg);
  for (uint i = lid; i < 10240; i += threads_per_threadgroup.x) {
    tn[i] = nwh[i] * static_cast<bfloat16_t>(float(widened[i]) * inv_h);      // pre_fc_norm_hidden
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint c0 = tg * 4;
  // fc_hidden: 4 rows x 4 substream vectors (gemv_wide k_lanes 32: one row per simdgroup)
  GKW::run(wh + size_t(c0) * 2560, tn, wh, ofh, 2560, 4, 4, 2560, 2560, 1.0f, 0.0f, 0, 0,
           uint3(0, 0, 0), uint3(1, 1, 1), sg, lane);
  // fc_embedding: 4 rows (gemv BN1 SN32 TN4 TM4, one simdgroup)
  if (sg == 0) {
    GKE::run(we + size_t(c0) * 2560, te, we, oen, 2560, 4, 2560, 1.0f, 0.0f, 0, tgp, uint3(0, 0, 0),
             uint3(lane, 0, 0), 0, lane);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 16) {
    uint v = lid / 4, r = lid % 4;
    h[v * 2560 + c0 + r] = ofh[v * 4 + r] + oen[r];                          // fc_hidden(hn) + en (bf16 Add)
  }
"""

_S4 = """
  uint tgx = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  if (tgx < 1536) {
    qmv_fast_impl<bfloat16_t, 64, 4>(qw, qs, qb, x, qo, 2560, 12288, uint3(0, tgx, 0), sg, lane);
  } else if (tgx < 1600) {
    qmv_fast_impl<bfloat16_t, 64, 4>(kw, ks, kb, x, ko, 2560, 512, uint3(0, tgx - 1536, 0), sg, lane);
  } else if (tgx < 1664) {
    qmv_fast_impl<bfloat16_t, 64, 4>(vw, vs, vb, x, vo, 2560, 512, uint3(0, tgx - 1600, 0), sg, lane);
  } else {
    qmv_fast_impl<bfloat16_t, 64, 8>(iw, is, ib, x, io, 2560, 640, uint3(0, tgx - 1664, 0), sg, lane);
  }
"""

_S6 = """
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  qmv_fast_impl<bfloat16_t, 64, 4>(ow, os, ob, x, y, 6144, 2560, uint3(0, threadgroup_position_in_grid.x, 0), sg, lane);
"""

_S14 = """
  uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  qmv_fast_impl<bfloat16_t, 64, 4>(hw, hs, hb, x, lg, 2560, NH, uint3(0, threadgroup_position_in_grid.x, 0), sg, lane);
"""

S15_HEADER = """
// (value, index) order of argpartition's stable ascending sort: a before b iff a.v < b.v or (a.v == b.v and a.i < b.i)
METAL_FUNC bool fl_after(float av, uint ai, float bv, uint bi) { return av > bv || (av == bv && ai > bi); }
"""

_S15 = """
  // one threadgroup of 1024 threads: the stock logsumexp_looped layout (4 values per lane per round)
  uint lid = thread_position_in_threadgroup.x;
  uint lsize = threads_per_threadgroup.x;
  uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
  threadgroup float local_max[32];
  threadgroup float local_norm[32];
  threadgroup float lse_s[1];
  float prevmax;
  float maxval = Limits<float>::finite_min;
  float normalizer = 0;
  for (int r = 0; r < int((NH + 4 * lsize - 1) / (4 * lsize)); r++) {
    int offset = r * lsize * 4 + lid * 4;
    float vals[4];
    if (offset + 4 <= NH) {
      for (int i = 0; i < 4; i++) { vals[i] = float(lg[offset + i]); }
    } else {
      for (int i = 0; i < 4; i++) { vals[i] = (offset + i < NH) ? float(lg[offset + i]) : Limits<float>::min; }
    }
    prevmax = maxval;
    for (int i = 0; i < 4; i++) { maxval = (maxval < vals[i]) ? vals[i] : maxval; }
    normalizer *= fast::exp(prevmax - maxval);
    for (int i = 0; i < 4; i++) { normalizer += fast::exp(vals[i] - maxval); }
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);
  prevmax = maxval;
  if (lane == 0) { local_max[sgi] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[lane]);
  normalizer *= fast::exp(prevmax - maxval);
  if (lane == 0) { local_norm[sgi] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_norm[lane]);
  if (lid == 0) { lse_s[0] = isinf(maxval) ? maxval : (metal::precise::log(normalizer) + maxval); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float lse = lse_s[0];
  // each lane: its top-K of lp = lg - lse (fp32 Subtract) over its elements, kept in descending (v, i) order
  float bv[KTOP]; uint bi[KTOP];
  for (int k = 0; k < KTOP; k++) { bv[k] = -INFINITY; bi[k] = 0xffffffffu; }
  int nfill = 0;
  for (uint j = lid; j < uint(NH); j += lsize) {
    float v = float(lg[j]) - lse;
    if (nfill < KTOP || fl_after(v, j, bv[KTOP - 1], bi[KTOP - 1])) {
      int p = min(nfill, KTOP - 1);
      while (p > 0 && fl_after(v, j, bv[p - 1], bi[p - 1])) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--; }
      bv[p] = v; bi[p] = j;
      nfill = min(nfill + 1, KTOP);
    }
  }
  // merge in two levels, 2 barriers: (1) each simdgroup merges its 32 lanes' lists in KTOP rounds of an all-lane
  // butterfly argmax (the order is total, so every lane converges on the same winner; its owner advances);
  // (2) simdgroup 0 merges the 32 simdgroup lists the same way
  threadgroup float sgv[32 * KTOP]; threadgroup uint sgx[32 * KTOP];
  threadgroup float topv[KTOP]; threadgroup uint topi[KTOP];
  int head = 0;
  for (int k = 0; k < KTOP; k++) {
    bool live = head < nfill;
    float mv = live ? bv[head] : -INFINITY;
    uint mi = live ? bi[head] : 0u;
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ov = simd_shuffle_xor(mv, off); uint oi = simd_shuffle_xor(mi, off);
      if (fl_after(ov, oi, mv, mi)) { mv = ov; mi = oi; }
    }
    if (live && bi[head] == mi && bv[head] == mv) { head++; }
    if (lane == 0) { sgv[sgi * KTOP + k] = mv; sgx[sgi * KTOP + k] = mi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0) {
    int hd = 0;
    for (int k = 0; k < KTOP; k++) {
      bool live = hd < KTOP;
      float mv = live ? sgv[lane * KTOP + hd] : -INFINITY;
      uint mi = live ? sgx[lane * KTOP + hd] : 0u;
      float lv = mv; uint li = mi;
      for (ushort off = 16; off >= 1; off >>= 1) {
        float ov = simd_shuffle_xor(mv, off); uint oi = simd_shuffle_xor(mi, off);
        if (fl_after(ov, oi, mv, mi)) { mv = ov; mi = oi; }
      }
      if (live && li == mi && lv == mv) { hd++; }
      if (lane == 0) { topv[k] = mv; topi[k] = mi; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    // argpartition's last K, ascending: position p holds the (K - 1 - p)-th largest
    for (int p = 0; p < KTOP; p++) { kidx[p] = topi[KTOP - 1 - p]; klp[p] = topv[KTOP - 1 - p]; }
    float best = 0; uint bp = 0;
    for (int p = 0; p < KTOP; p++) {
      float cd = sc[0] + topv[KTOP - 1 - p];                                 // sc + klp (fp32 Add)
      cand[p] = cd;
      if (p == 0 || !fl_after(best, bp, cd, uint(p))) { best = cd; bp = p; }
    }
    sel[0] = bp;
    nsc[0] = best;
    ntok[0] = ids[kidx[bp]];
  }
"""

_K = {}


def _k(name, header, src, inputs, outputs):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, header=header, source=src)
    return _K[name]


BF = mx.bfloat16


def s1_prep(mtp, emb, widened, tok):
    """h [10240] = the stock _prepare_inputs_eager(widened [1, 1, 10240], emb(tok)) for one row"""
    k = _k("fl_s1", S1_HEADER, _S1, ["widened", "tok", "ew", "es", "eb", "nwe", "nwh", "we", "wh", "eps"], ["h"])
    return k(inputs=[widened.reshape(-1), tok.reshape(-1), emb["weight"], emb["scales"],
                     emb["biases"], mtp.pre_fc_norm_embedding.weight, mtp.pre_fc_norm_hidden.weight,
                     mtp.fc_embedding.weight, mtp.fc_hidden.weight,
                     _const(("eps", mtp.pre_fc_norm_hidden.eps), lambda: mx.array([mtp.pre_fc_norm_hidden.eps], mx.float32))],
             grid=(640 * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(10240,)], output_dtypes=[BF])[0]


def s4_proj(attn, x):
    k = _k("fl_s4", QHEADER, _S4, ["x", "qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "iw", "is", "ib"],
           ["qo", "ko", "vo", "io"])
    ix = attn.indexer.index_qk_proj
    return k(inputs=[x.reshape(-1)] + [m[n] for m in (attn.q_proj, attn.k_proj, attn.v_proj, ix)
                                       for n in ("weight", "scales", "biases")],
             grid=(1744 * 32, 2, 1), threadgroup=(32, 2, 1),
             output_shapes=[(12288,), (512,), (512,), (640,)], output_dtypes=[BF] * 4)


def s6_oproj(attn, gated):
    k = _k("fl_s6", QHEADER, _S6, ["x", "ow", "os", "ob"], ["y"])
    o = attn.o_proj
    return k(inputs=[gated.reshape(-1), o["weight"], o["scales"], o["biases"]], grid=(320 * 32, 2, 1),
             threadgroup=(32, 2, 1), output_shapes=[(2560,)], output_dtypes=[BF])[0]


def s14_head(sh, x):
    nh = int(sh.w.shape[0])
    k = _k("fl_s14", QHEADER, _S14, ["x", "hw", "hs", "hb"], ["lg"])
    return k(inputs=[x.reshape(-1), sh.w, sh.s, sh.b], template=[("NH", nh)], grid=(nh // 8 * 32, 2, 1),
             threadgroup=(32, 2, 1), output_shapes=[(nh,)], output_dtypes=[BF])[0]


def s15_tail(lg, sc, ids, K=8):
    nh = int(lg.shape[0])
    k = _k("fl_s15", S15_HEADER, _S15, ["lg", "sc", "ids"], ["kidx", "klp", "cand", "sel", "nsc", "ntok"])
    return k(inputs=[lg, sc.reshape(-1).astype(mx.float32), ids], template=[("NH", nh), ("KTOP", K)],
             grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(K,), (K,), (K,), (1,), (1,), (1,)],
             output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.uint32, mx.float32, mx.int32])


# ---------------------------------------------------------------------------------------------------------------------
# Next-level prep folded into S15 (S15P) and S1 without a prologue (S1G). S15's 1,024 threads are exactly the stock
# rms_looped layout for the 10,240-wide pre_fc_norm_hidden, so the norm of hh is computed there once (not in each of
# S1's 640 threadgroups), and so are the next token's embedding row (affine_dequantize) and pre_fc_norm_embedding
# (640 lanes of 4). S1G is then fc_hidden (gemv_wide k32) + fc_embedding (gemv BN1 SN32 TN4) + the add, reading the
# prepared vectors. S0 does the same prep for a block's first level (no S15 before it).
_PREP = """
  {
    // pre_fc_norm_hidden(hh): rms_looped, 1024 lanes; lane lid walks r = 0, 4096, 8192
    float acc = 0.0f;
    for (uint r = 0; r < 10240; r += 4096) {
      for (int i = 0; i < 4; i++) {
        uint idx = r + lid * 4 + i;
        if (idx < 10240) { float xi = hh[idx]; acc += xi * xi; }
      }
    }
    acc = simd_sum(acc);
    if (lane == 0) { pp[sgi] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
      float tot = simd_sum(pp[lane]);
      if (lane == 0) { pp[32] = metal::precise::rsqrt(tot / 10240.0f + eps[0]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invh = pp[32];
    for (uint i = lid; i < 10240; i += 1024) {
      hn[i] = nwh[i] * static_cast<bfloat16_t>(float(hh[i]) * invh);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the token's embedding row (affine_dequantize: 2 values per byte, out = scale * d + bias)
    size_t tk = size_t(TOKEXPR);
    const device uint8_t* wrow = (const device uint8_t*)(ew + tk * 320);
    for (uint o = lid; o < 1280; o += 1024) {
      uint oindex = o * 2;
      bfloat16_t scale = es[tk * 40 + oindex / 64];
      bfloat16_t bias = eb[tk * 40 + oindex / 64];
      uint val = wrow[o];
      for (int i = 0; i < 2; i++) {
        uint8_t d = (val >> (4 * i)) & 0x0f;
        te[oindex + i] = scale * d + bias;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // pre_fc_norm_embedding: rms_single_row, 640 lanes of 4 (20 simdgroups; slots 20-31 stay 0)
    if (sgi == 0) { pp[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 640) {
      float a2 = 0.0f;
      for (int i = 0; i < 4; i++) { float xi = te[lid * 4 + i]; a2 += xi * xi; }
      a2 = simd_sum(a2);
      if (lane == 0) { pp[sgi] = a2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
      float tot = simd_sum(pp[lane]);
      if (lane == 0) { pp[33] = metal::precise::rsqrt(tot / 2560.0f + eps[0]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inve = pp[33];
    for (uint i = lid; i < 2560; i += 1024) {
      en_in[i] = nwe[i] * static_cast<bfloat16_t>(float(te[i]) * inve);
    }
  }
"""

_S15P = _S15.replace("  uint lid = thread_position_in_threadgroup.x;",
                     "  uint lid = thread_position_in_threadgroup.x;\n  threadgroup float pp[34];\n"
                     "  threadgroup bfloat16_t te[2560];\n  threadgroup int tok_s[1];") + """
  if (lid == 0) { tok_s[0] = ntok[0]; }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
""" + _PREP.replace("TOKEXPR", "tok_s[0]")

_S0 = """
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
  threadgroup float pp[34];
  threadgroup bfloat16_t te[2560];
""" + _PREP.replace("TOKEXPR", "tok[0]")


def _gemv_dev_variants():
    """FLGEMV / FLGEMVW with the input vector in DEVICE memory (outputs still in threadgroup memory)"""
    t = GEMV_TG.replace("struct FLGEMV", "struct FLGEMVD").replace("const threadgroup T* in_vec,", "const device T* in_vec,")
    w = _gemv_wide_tg().replace("struct FLGEMVW", "struct FLGEMVWD").replace(
        "const threadgroup T* in_vec,", "const device T* in_vec,").replace(
        "const threadgroup vec<T, 4>* x4 = (const threadgroup vec<T, 4>*)in_vec;",
        "const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;")
    t = re.sub(r"#define MLX_MTL_CONST.*?\ntemplate <typename U> struct DefaultAccT \{ using type = float; \};\n", "", t,
               flags=re.S)
    return t + w


S1G_HEADER = RMS_HEADER + GEMV_TG + _gemv_wide_tg() + _gemv_dev_variants() + """
typedef FLGEMVD<bfloat16_t, 1, 1, 1, 32, 4, 4, false> GKE;
typedef FLGEMVWD<bfloat16_t, 4, 32> GKW;
"""

_S1G = """
  uint tg = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float tgp[1];
  threadgroup bfloat16_t ofh[16];
  threadgroup bfloat16_t oen[4];
  uint c0 = tg * 4;
  GKW::run(wh + size_t(c0) * 2560, hn, wh, ofh, 2560, 4, 4, 2560, 2560, 1.0f, 0.0f, 0, 0,
           uint3(0, 0, 0), uint3(1, 1, 1), sg, lane);
  if (sg == 0) {
    GKE::run(we + size_t(c0) * 2560, en_in, we, oen, 2560, 4, 2560, 1.0f, 0.0f, 0, tgp, uint3(0, 0, 0),
             uint3(lane, 0, 0), 0, lane);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 16) {
    uint v = lid / 4, r = lid % 4;
    h[v * 2560 + c0 + r] = ofh[v * 4 + r] + oen[r];
  }
"""


def _prep_inputs(mtp, emb):
    return [emb["weight"], emb["scales"], emb["biases"], mtp.pre_fc_norm_embedding.weight,
            mtp.pre_fc_norm_hidden.weight,
            _const(("eps", mtp.pre_fc_norm_hidden.eps), lambda: mx.array([mtp.pre_fc_norm_hidden.eps], mx.float32))]


_PREP_NAMES = ["ew", "es", "eb", "nwe", "nwh", "eps"]


def s0_prep(mtp, emb, widened, tok):
    """a block's first level: (hn [10240], en_in [2560]) from (widened, tok)"""
    k = _k("fl_s0", RMS_HEADER, _S0, ["hh", "tok"] + _PREP_NAMES, ["hn", "en_in"])
    return k(inputs=[widened.reshape(-1), tok.reshape(-1)] + _prep_inputs(mtp, emb), grid=(1024, 1, 1),
             threadgroup=(1024, 1, 1), output_shapes=[(10240,), (2560,)], output_dtypes=[BF, BF])


def s1g(mtp, hn, en_in):
    k = _k("fl_s1g", S1G_HEADER, _S1G, ["hn", "en_in", "we", "wh"], ["h"])
    return k(inputs=[hn, en_in, mtp.fc_embedding.weight, mtp.fc_hidden.weight], grid=(640 * 128, 1, 1),
             threadgroup=(128, 1, 1), output_shapes=[(10240,)], output_dtypes=[BF])[0]


def s15p_tail(lg, sc, ids, hh, mtp, emb, K=8):
    """S15 + the next level's prep. Returns (kidx, klp, cand, sel, nsc, ntok, hn_next, en_next)"""
    nh = int(lg.shape[0])
    k = _k("fl_s15p", S15_HEADER, _S15P, ["lg", "sc", "ids", "hh"] + _PREP_NAMES,
           ["kidx", "klp", "cand", "sel", "nsc", "ntok", "hn", "en_in"])
    return k(inputs=[lg, sc.reshape(-1).astype(mx.float32), ids, hh.reshape(-1)] + _prep_inputs(mtp, emb),
             template=[("NH", nh), ("KTOP", K)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(K,), (K,), (K,), (1,), (1,), (1,), (10240,), (2560,)],
             output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.uint32, mx.float32, mx.int32, BF, BF])
