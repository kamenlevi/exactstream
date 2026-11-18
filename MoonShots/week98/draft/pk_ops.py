"""PK v2 bitwise building blocks (MoonShots, Wed 18 Nov 2026): Metal device functions for the persistent pop that
reproduce MLX 0.32.2's stock kernels per output, whatever simdgroup / threadgroup computes them.
GEMV:
  qmv_pf<T, gs, bits, R, UB, FAST, TOP>: rows row0 .. row0 + R - 1, one simdgroup. Per row and lane the stock order:
    FAST (qmv_fast_impl, K % (32 * vpt) == 0): every block load_vector + qdot, packs_per_thread 2;
    !FAST (qmv_impl): packs_per_thread 1, blocks while k < K - block (strict), then ONE last block through
    load_vector_safe / qdot_safe (even when full), as qmv_impl does;
    then simd_sum. The weights of UB blocks are staged in registers first (qdot_r = qdot's arithmetic on registers).
    TOP: each finished row (rounded to T) enters the simdgroup's running top-8 (value desc, index asc).
  qmv_wide_impl (shipped text): fc_hidden's 4 streams (affine_qmv_wide nv 4, k_lanes 8).
pk_qbit.py bit-tests them against mx.quantized_matmul / mx.gather_qmm on every Q8H32 shape."""
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/bet7"))
from fl_qmatch import QHEADER, KDIR  # noqa: E402

_Q = open(os.path.join(KDIR, "quantized.h")).read()
_a = _Q.index("template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>\nMETAL_FUNC void qmv_wide_impl")
_b = _Q.index("\ntemplate", _a + 10)
WIDE = _Q[_a:_b].replace("const constant int&", "const int")
WIDE = re.sub(r"\s*\[\[\w+\]\]", "", WIDE)

PF = r"""
template <int vpt, int bits>
inline float qdot_r(const thread uint32_t* wp, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  if (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)wp;
    for (int i = 0; i < (vpt / 4); i++) {
      accum += (x_thread[4 * i] * (ws[i] & 0x000f) + x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                x_thread[4 * i + 2] * (ws[i] & 0x0f00) + x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else {
    const thread uint8_t* w = (const thread uint8_t*)wp;
    for (int i = 0; i < vpt; i++) accum += x_thread[i] * w[i];
  }
  return scale * accum + sum * bias;
}

METAL_FUNC void top8_insert(thread float& tv, thread int& ti, float v, int i, uint lane) {
  bool bet = (lane < 8u) && (tv > v || (tv == v && ti < i));
  uint p = uint(simd_sum(bet ? 1 : 0));
  if (p >= 8u) return;
  float uv = simd_shuffle_up(tv, ushort(1));
  int ui = simd_shuffle_up(ti, ushort(1));
  if (lane == p) { tv = v; ti = i; }
  else if (lane > p && lane < 8u) { tv = uv; ti = ui; }
}

template <typename T, int gs, int bits, int R, int UB, bool FAST, bool TOP>
METAL_FUNC void qmv_pf(const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
                       device T* y, const int K, const int row0, const uint lane, thread float& tv, thread int& ti) {
  constexpr int PPT = FAST ? 2 : 1;
  constexpr int pf = 32 / bits;
  constexpr int vpt = pf * PPT;
  constexpr int bsz = vpt * 32;
  constexpr int sstep = gs / vpt;
  const int Kw = K / pf, Kg = K / gs;
  const device uint32_t* wr = w + row0 * Kw + int(lane) * PPT;
  const device T* sr = scales + row0 * Kg + int(lane) / sstep;
  const device T* br = biases + row0 * Kg + int(lane) / sstep;
  const device T* xr = x + int(lane) * vpt;
  float res[R];
  for (int r = 0; r < R; ++r) res[r] = 0.0f;
  float xt[vpt];
  const int nmain = FAST ? K / bsz : (K - 1) / bsz;          // qmv: k < K - block -> the last block is the safe one
  int b = 0;
  for (; b + UB <= nmain; b += UB) {
    uint32_t wb[UB][R][PPT];
    float sb[UB][R], bb[UB][R];
    for (int u = 0; u < UB; ++u)
      for (int r = 0; r < R; ++r) {
        for (int p = 0; p < PPT; ++p) wb[u][r][p] = wr[r * Kw + u * (bsz / pf) + p];
        sb[u][r] = sr[r * Kg + u * (bsz / gs)];
        bb[u][r] = br[r * Kg + u * (bsz / gs)];
      }
    for (int u = 0; u < UB; ++u) {
      float sum = load_vector<T, float, vpt, bits>(xr + u * bsz, xt);
      for (int r = 0; r < R; ++r) res[r] += qdot_r<vpt, bits>(wb[u][r], xt, sb[u][r], bb[u][r], sum);
    }
    wr += UB * (bsz / pf); sr += UB * (bsz / gs); br += UB * (bsz / gs); xr += UB * bsz;
  }
  for (; b < nmain; ++b) {
    uint32_t wb[R][PPT];
    float sb[R], bb[R];
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PPT; ++p) wb[r][p] = wr[r * Kw + p];
      sb[r] = sr[r * Kg]; bb[r] = br[r * Kg];
    }
    float sum = load_vector<T, float, vpt, bits>(xr, xt);
    for (int r = 0; r < R; ++r) res[r] += qdot_r<vpt, bits>(wb[r], xt, sb[r], bb[r], sum);
    wr += bsz / pf; sr += bsz / gs; br += bsz / gs; xr += bsz;
  }
  if (!FAST) {
    const int remaining = clamp(K - nmain * bsz - int(lane) * vpt, 0, vpt);
    if (remaining > 0) {
      float sum = load_vector_safe<T, float, vpt, bits>(xr, xt, remaining);
      for (int r = 0; r < R; ++r) {
        float s = sr[r * Kg], bi = br[r * Kg];
        res[r] += qdot_safe<float, vpt, bits>((const device uint8_t*)(wr + r * Kw), xt, s, bi, sum, remaining);
      }
    }
  }
  for (int r = 0; r < R; ++r) {
    float s = simd_sum(res[r]);
    if (lane == 0u) y[row0 + r] = static_cast<T>(s);
    if (TOP) top8_insert(tv, ti, float(static_cast<T>(s)), row0 + r, lane);
  }
}
"""

HEADER = QHEADER + "\n" + PF                      # QHEADER already carries qmv_wide_impl

# ---------------------------------------------------------------------------------------------------- the HC read
sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/bet7"))
import fl_hc  # noqa: E402
from fl_exact import EW_HEADER  # noqa: E402

_HCS = fl_hc.HC_HEADER[fl_hc.HC_HEADER.index("// grouped stock RMS statistic"):]   # fl_hc_val / fl_hc_stats (stock order)

HC = EW_HEADER + _HCS + r"""
// one row of a quantized matrix, returned (as T) to every lane: qmv_pf's R = 1 arithmetic
template <typename T, int gs, int bits, bool FAST>
METAL_FUNC T qmv_row(const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
                     const int K, const int row, const uint lane) {
  constexpr int PPT = FAST ? 2 : 1;
  constexpr int pf = 32 / bits;
  constexpr int vpt = pf * PPT;
  constexpr int bsz = vpt * 32;
  constexpr int sstep = gs / vpt;
  const int Kw = K / pf, Kg = K / gs;
  const device uint32_t* wr = w + row * Kw + int(lane) * PPT;
  const device T* sr = scales + row * Kg + int(lane) / sstep;
  const device T* br = biases + row * Kg + int(lane) / sstep;
  const device T* xr = x + int(lane) * vpt;
  float res = 0.0f;
  float xt[vpt];
  const int nmain = FAST ? K / bsz : (K - 1) / bsz;
  for (int b = 0; b < nmain; ++b) {
    uint32_t wb[PPT];
    for (int p = 0; p < PPT; ++p) wb[p] = wr[p];
    float s = sr[0], bi = br[0];
    float sum = load_vector<T, float, vpt, bits>(xr, xt);
    res += qdot_r<vpt, bits>(wb, xt, s, bi, sum);
    wr += bsz / pf; sr += bsz / gs; br += bsz / gs; xr += bsz;
  }
  if (!FAST) {
    const int remaining = clamp(K - nmain * bsz - int(lane) * vpt, 0, vpt);
    if (remaining > 0) {
      float sum = load_vector_safe<T, float, vpt, bits>(xr, xt, remaining);
      float s = sr[0], bi = br[0];
      res += qdot_safe<float, vpt, bits>((const device uint8_t*)wr, xt, s, bi, sum, remaining);
    }
  }
  return static_cast<T>(simd_sum(res));
}

// R rows of a quantized matrix, returned (as T) to every lane in outv[R]; qmv_pf's arithmetic (4-bit needs R >= 2)
template <typename T, int gs, int bits, int R, bool FAST>
METAL_FUNC void qmv_rows(const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
                         const int K, const int row0, const uint lane, thread T* outv) {
  constexpr int PPT = FAST ? 2 : 1;
  constexpr int pf = 32 / bits;
  constexpr int vpt = pf * PPT;
  constexpr int bsz = vpt * 32;
  constexpr int sstep = gs / vpt;
  const int Kw = K / pf, Kg = K / gs;
  const device uint32_t* wr = w + row0 * Kw + int(lane) * PPT;
  const device T* sr = scales + row0 * Kg + int(lane) / sstep;
  const device T* br = biases + row0 * Kg + int(lane) / sstep;
  const device T* xr = x + int(lane) * vpt;
  float res[R];
  for (int r = 0; r < R; ++r) res[r] = 0.0f;
  float xt[vpt];
  const int nmain = FAST ? K / bsz : (K - 1) / bsz;
  for (int b = 0; b < nmain; ++b) {
    uint32_t wb[R][PPT];
    float sb[R], bb[R];
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PPT; ++p) wb[r][p] = wr[r * Kw + p];
      sb[r] = sr[r * Kg]; bb[r] = br[r * Kg];
    }
    float sum = load_vector<T, float, vpt, bits>(xr, xt);
    for (int r = 0; r < R; ++r) res[r] += qdot_r<vpt, bits>(wb[r], xt, sb[r], bb[r], sum);
    wr += bsz / pf; sr += bsz / gs; br += bsz / gs; xr += bsz;
  }
  if (!FAST) {
    const int remaining = clamp(K - nmain * bsz - int(lane) * vpt, 0, vpt);
    if (remaining > 0) {
      float sum = load_vector_safe<T, float, vpt, bits>(xr, xt, remaining);
      for (int r = 0; r < R; ++r) {
        float s = sr[r * Kg], bi = br[r * Kg];
        res[r] += qdot_safe<float, vpt, bits>((const device uint8_t*)(wr + r * Kw), xt, s, bi, sum, remaining);
      }
    }
  }
  for (int r = 0; r < R; ++r) outv[r] = static_cast<T>(simd_sum(res[r]));
}

// HC phase A (call from EVERY threadgroup; the PK's grid barrier follows): the stock eager GatedResidual read of hyper'
// (= hyper + bout * pinj when PENDING): grouped stats, normed (this TG's device copy nrm; TG 0 also writes hw = hyper'),
// the 320 down rows (8-bit qmv_fast, one simdgroup per row) -> mix1 = silu_jit(bf16(d / 4)); with COMBINE, threadgroup
// ITG's simdgroups 0..7 write the inject gemv's partials (stock BN 8 / SN 32 / TM 4 / TN 4 order) to ipart[32].
template <bool PENDING, bool COMBINE, int BITS = 8, int RR = 1>
METAL_FUNC void pk_hc_a(const device bfloat16_t* hyper, const device bfloat16_t* bout, const device bfloat16_t* pinj,
                        const device bfloat16_t* nw, float eps, const device uint32_t* wd, const device bfloat16_t* sd,
                        const device bfloat16_t* bd, const device bfloat16_t* wi, device bfloat16_t* nrm,
                        device bfloat16_t* hw, device bfloat16_t* mix1, device float* ipart, threadgroup float* part,
                        threadgroup float* inv, uint tg, uint ntg, uint itg, uint lid, uint nt, uint sg, uint nsg,
                        uint lane) {
  fl_hc_stats<PENDING>(hyper, bout, pinj, eps, part, inv, lane, sg, nsg);
  for (uint i = lid; i < 10240u; i += nt) {
    float hv = fl_hc_val<PENDING>(hyper, bout, pinj, i);
    bfloat16_t nr = static_cast<bfloat16_t>(hv * inv[i / 2560u]);
    nrm[i] = nr * nw[i];
    if (PENDING && tg == 0u) hw[i] = static_cast<bfloat16_t>(hv);
  }
  threadgroup_barrier(mem_flags::mem_device);
  const bool inj = COMBINE && tg == itg && sg < 8u;
  if (inj) {
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = int(32u * sg + lane) * 4;
    for (int it = 0; it < 10; ++it) {
      float vc[4];
      for (int tn = 0; tn < 4; ++tn) vc[tn] = static_cast<float>(nrm[bn + tn]);
      for (int tm = 0; tm < 4; ++tm) {
        bfloat16_t inter[4];
        for (int tn = 0; tn < 4; ++tn) inter[tn] = wi[tm * 10240 + bn + tn];
        for (int tn = 0; tn < 4; ++tn) result[tm] += inter[tn] * vc[tn];
      }
      bn += 1024;
    }
    for (int tm = 0; tm < 4; ++tm)
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[tm] += simd_shuffle_down(result[tm], sn);
    if (lane == 0u) for (int tm = 0; tm < 4; ++tm) ipart[sg * 4u + tm] = result[tm];
  }
  const uint skip = COMBINE ? 8u : 0u;                       // the inject's 8 simdgroups take no down rows
  const bool worker = !inj;
  const uint wid = (COMBINE && tg > itg) ? tg * nsg + sg - 8u : ((COMBINE && tg == itg) ? itg * nsg + sg - 8u : tg * nsg + sg);
  for (uint u = worker ? wid : 320u / uint(RR); u < 320u / uint(RR); u += ntg * nsg - skip) {
    bfloat16_t dv[RR];
    qmv_rows<bfloat16_t, 64, BITS, RR, true>(wd, sd, bd, nrm, 10240, int(u) * RR, lane, dv);
    for (int r = 0; r < RR; ++r) {
      bfloat16_t q = static_cast<bfloat16_t>(metal::precise::divide(float(dv[r]), 4.0f));
      if (lane == 0u) mix1[u * uint(RR) + uint(r)] = fl_silu_jit(q);
    }
  }
}

// HC phase B (every threadgroup): per column c (one simdgroup): the 4 up rows c + s * 2560 (8-bit qmv, K 320) ->
// sigmoid_pre -> x normed (bf16) -> the stock mean (bf16 running sum from 0 in row order) x 0.25 -> mixed[c];
// with COMBINE the inject: simdgroup 0's partial + 1..7 in order -> bf16 -> / 4 -> sigmoid_pre -> x 2 (one thread).
template <bool COMBINE, int BITS = 8, int RR = 1>
METAL_FUNC void pk_hc_b(const device bfloat16_t* mix1, const device uint32_t* wu, const device bfloat16_t* su,
                        const device bfloat16_t* bu, const device bfloat16_t* nrm, device bfloat16_t* mixed,
                        const device float* ipart, device bfloat16_t* inj, uint tg, uint ntg, uint sg, uint nsg,
                        uint lane) {
  for (uint cg = tg * nsg + sg; cg < 2560u / uint(RR); cg += ntg * nsg) {
    const uint c0 = cg * uint(RR);
    bfloat16_t acc[RR];
    for (int r = 0; r < RR; ++r) acc[r] = static_cast<bfloat16_t>(0.0f);
    for (uint s = 0; s < 4u; ++s) {
      bfloat16_t up[RR];
      qmv_rows<bfloat16_t, 64, BITS, RR, false>(wu, su, bu, mix1, 320, int(s * 2560u + c0), lane, up);
      for (int r = 0; r < RR; ++r) {
        bfloat16_t z = fl_sigmoid_pre(up[r]);
        bfloat16_t p = z * nrm[s * 2560u + c0 + uint(r)];
        acc[r] = p + acc[r];
      }
    }
    if (lane == 0u) for (int r = 0; r < RR; ++r) mixed[c0 + uint(r)] = acc[r] * static_cast<bfloat16_t>(0.25f);
  }
  if (COMBINE && tg == 0u && sg == 0u && lane < 4u) {
    float r = ipart[lane];
    for (uint g = 1; g < 8u; ++g) r += ipart[g * 4u + lane];
    bfloat16_t d = static_cast<bfloat16_t>(r);
    bfloat16_t q = static_cast<bfloat16_t>(metal::precise::divide(float(d), 4.0f));
    bfloat16_t sgm = fl_sigmoid_pre(q);
    inj[lane] = static_cast<bfloat16_t>(2.0f * float(sgm));
  }
}
"""

# ---------------------------------------------------------------------------------------------------- attention
import fl_attn  # noqa: E402

ATTN = r"""
#ifndef PK_ATTN_PRAGMA
#define PK_ATTN_PRAGMA
#endif
// no FMA contraction: stock RoPE is three separate fp32 kernels (Multiply, Multiply, Add)
METAL_FUNC float pk_nofma_mul(float a, float b) { return as_type<float>(as_type<uint>(a * b)); }

// one head h, by ONE whole threadgroup (1024 threads = sdpa_vector's 32 simdgroups): fl_attn's S5 (BET 7, bitwise vs
// the stock level) with the tree's key set: keys 0 .. Lc - 1 = the MTP cache backing (kc / vc + g * khs, row stride 256),
// Lc .. Lc + t - 1 = the drafted rows (kd / vd: row r, group g at (r * 2 + g) * 256), key N - 1 = this node (rotated k,
// v = vf); drafted row r is used iff bit r of anc is set (the stock bool mask SKIPS a masked key). RoPE at POS.
// q_norm / k_norm, partial RoPE (64 dims), sdpa_vector one-pass (N < 1024), attention_gate (standalone sigmoid).
METAL_FUNC void pk_attn_head(uint h, const device bfloat16_t* qf, const device bfloat16_t* kf, const device bfloat16_t* vf,
                             const device bfloat16_t* kc, const device bfloat16_t* vc, uint khs, const device bfloat16_t* kd,
                             const device bfloat16_t* vd, uint Lc, uint t, uint anc, uint POS,
                             const device bfloat16_t* qnw, const device bfloat16_t* knw, const device float* inv_freq,
                             float eps, device bfloat16_t* gated, device bfloat16_t* krot,
                             threadgroup float* part, threadgroup bfloat16_t* qr, threadgroup bfloat16_t* kr,
                             threadgroup float* cs, threadgroup float* sn, threadgroup bfloat16_t* qn,
                             threadgroup bfloat16_t* kn, threadgroup float* outputs, threadgroup float* max_scores,
                             threadgroup float* sum_exp_scores, uint lid, uint simd_gid, uint simd_lid, uint nsg) {
  PK_ATTN_PRAGMA
  constexpr int BN = 32, BD = 32, qk_per_thread = 8, v_per_thread = 8;
  const uint g = h / 12u;
  typedef float U;
  if (lid < 64u) {
    float a = float(POS) * inv_freq[lid % 32u];
    cs[lid] = metal::precise::cos(a);
    sn[lid] = metal::precise::sin(a);
  }
  const device bfloat16_t* xq = qf + h * 512u;
  const device bfloat16_t* xk = kf + g * 256u;
  float invq = fl_rms_inv_p<256>(xq, eps, part, simd_lid, simd_gid, nsg);
  float invk = fl_rms_inv_p<256>(xk, eps, part, simd_lid, simd_gid, nsg);
  if (lid < 256u) {
    qn[lid] = qnw[lid] * static_cast<bfloat16_t>(float(xq[lid]) * invq);
    kn[lid] = knw[lid] * static_cast<bfloat16_t>(float(xk[lid]) * invk);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 256u) {
    uint i = lid;
    if (i < 64u) {
      bfloat16_t rq = (i < 32u) ? static_cast<bfloat16_t>(-qn[i + 32u]) : qn[i - 32u];
      bfloat16_t rk = (i < 32u) ? static_cast<bfloat16_t>(-kn[i + 32u]) : kn[i - 32u];
      float q1 = pk_nofma_mul(float(qn[i]), cs[i]);
      float q2 = pk_nofma_mul(float(rq), sn[i]);
      qr[i] = static_cast<bfloat16_t>(q1 + q2);
      float k1 = pk_nofma_mul(float(kn[i]), cs[i]);
      float k2 = pk_nofma_mul(float(rk), sn[i]);
      kr[i] = static_cast<bfloat16_t>(k1 + k2);
    } else {
      qr[i] = qn[i];
      kr[i] = kn[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (h % 12u == 0u && lid < 256u) { krot[g * 256u + lid] = kr[lid]; }
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];
  const float scale = 0.0625f;
  for (int i = 0; i < qk_per_thread; i++) { q[i] = static_cast<U>(scale) * qr[simd_lid * qk_per_thread + i]; }
  for (int i = 0; i < v_per_thread; i++) { o[i] = 0; }
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  const int N = int(Lc + t + 1u);
  for (int i = int(simd_gid); i < N; i += BN) {
    const bool is_new = (i == N - 1);
    const bool drafted = !is_new && i >= int(Lc);
    if (drafted && ((anc >> uint(i - int(Lc))) & 1u) == 0u) continue;        // the bool mask: skip the key
    const device bfloat16_t* kp = drafted ? (kd + (size_t(i - int(Lc)) * 2u + g) * 256u) : (kc + size_t(g) * khs + size_t(i) * 256u);
    const device bfloat16_t* vp = drafted ? (vd + (size_t(i - int(Lc)) * 2u + g) * 256u) : (vc + size_t(g) * khs + size_t(i) * 256u);
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = is_new ? float(kr[simd_lid * qk_per_thread + j]) : float(kp[simd_lid * qk_per_thread + j]);
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) { score += q[j] * k[j]; }
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) {
      float vv = is_new ? float(vf[g * 256u + simd_lid * v_per_thread + j]) : float(vp[simd_lid * v_per_thread + j]);
      o[j] = o[j] * factor + exp_score * vv;
    }
  }
  if (simd_lid == 0u) { max_scores[simd_gid] = max_score; sum_exp_scores[simd_gid] = sum_exp_score; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : metal::precise::divide(o[i], sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0u) {
    for (int i = 0; i < v_per_thread; i++) {
      uint d = simd_gid * v_per_thread + i;
      bfloat16_t ov = static_cast<bfloat16_t>(o[i]);
      bfloat16_t gt = fl_sigmoid_pre(qf[h * 512u + 256u + d]);
      gated[h * 256u + d] = ov * gt;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);                         // the TG's next head reuses the scratch
}
"""

# ---------------------------------------------------------------------------------------------------- fuse + MoE
MOE = r"""
// P1 prologue (every TG, into its private scratch): tok_emb = the embedding row (affine_dequantize 4-bit g64:
// scale * d + bias), en_in = rms_norm(tok_emb) * w_e (2560, single row), hn = rms_norm(widened) * w_h (10240, looped)
METAL_FUNC void pk_fuse_prep(const device uint8_t* ew, const device bfloat16_t* es, const device bfloat16_t* eb, uint tok,
                             const device bfloat16_t* widened, const device bfloat16_t* wne, const device bfloat16_t* wnh,
                             float eps_e, float eps_h, device bfloat16_t* emb, device bfloat16_t* en_in,
                             device bfloat16_t* hn, threadgroup float* part, uint lid, uint nt, uint sg, uint nsg,
                             uint lane) {
  for (uint o = lid; o < 1280u; o += nt) {                         // 2560 values = 1280 bytes, 2 per byte
    uint val = ew[size_t(tok) * 1280u + o];
    bfloat16_t scale = es[size_t(tok) * 40u + (o * 2u) / 64u];
    bfloat16_t bias = eb[size_t(tok) * 40u + (o * 2u) / 64u];
    for (int i = 0; i < 2; i++) {
      uint8_t d = (val >> (4 * i)) & 0x0f;
      emb[o * 2u + i] = scale * d + bias;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  float inve = fl_rms_inv_p<2560>(emb, eps_e, part, lane, sg, nsg);
  float invh = fl_rms_inv_p<10240>(widened, eps_h, part, lane, sg, nsg);
  for (uint i = lid; i < 2560u; i += nt) en_in[i] = wne[i] * static_cast<bfloat16_t>(float(emb[i]) * inve);
  for (uint i = lid; i < 10240u; i += nt) hn[i] = wnh[i] * static_cast<bfloat16_t>(float(widened[i]) * invh);
  threadgroup_barrier(mem_flags::mem_device);
}

// the shared-expert gate (CastLinear: x.astype(f32) @ w.T): MLX's dot_product<float, 32, 512, 16> partials, by the
// 16 simdgroups sg < 16 of one threadgroup: per thread float4 c over its items, ((c0 + c1) + c2) + c3, simd_sum
METAL_FUNC void pk_seg_partial(const device bfloat16_t* x, const device float* w, device float* spart, uint sg, uint lane) {
  const int n = 2560;
  const int start = int(sg * 32u) * 32 + int(lane) * 4;
  float4 c = 0.0f;
  for (int i = 0; i < 32; i += 4) {
    int idx = start + i * 32;
    if (idx + 4 <= n) {
      c += float4(float(x[idx]), float(x[idx + 1]), float(x[idx + 2]), float(x[idx + 3])) *
           float4(w[idx], w[idx + 1], w[idx + 2], w[idx + 3]);
    } else {
      for (int j = 0; j < 4; ++j) { int nidx = idx + j; if (nidx < n) c[j & 3] += float(x[nidx]) * w[nidx]; }
    }
  }
  float sum = c[0] + c[1] + c[2] + c[3];
  sum = simd_sum(sum);
  if (lane == 0u) spart[sg] = sum;
}

// the route (one whole TG; threads < 128 = the stock softmax_single_row layout for 512): fl_moe's S9b
METAL_FUNC void pk_route(const device bfloat16_t* rl, device uint* inds, device bfloat16_t* wts, threadgroup float* local_max,
                         threadgroup float* local_norm, threadgroup bfloat16_t* P, threadgroup uint* ti,
                         threadgroup bfloat16_t* tv, threadgroup uint* sgk, uint lid, uint lane, uint sgi) {
  const bool act = lid < 128u;
  float ld[4];
  for (int i = 0; i < 4; i++) { ld[i] = act ? float(rl[lid * 4 + i]) : 0.0f; }
  if (sgi == 0u) { local_max[lane] = Limits<float>::min; local_norm[lane] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  for (int i = 0; i < 4; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
  maxval = simd_max(maxval);
  if (act && lane == 0u) { local_max[sgi] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0u) { maxval = simd_max(local_max[lane]); if (lane == 0u) { local_max[0] = maxval; } }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  for (int i = 0; i < 4; i++) { float e = fast::exp(ld[i] - maxval); ld[i] = e; normalizer += e; }
  normalizer = simd_sum(normalizer);
  if (act && lane == 0u) { local_norm[sgi] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0u) { normalizer = simd_sum(local_norm[lane]); if (lane == 0u) { local_norm[0] = normalizer; } }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = metal::precise::divide(1.0f, local_norm[0]);
  if (act) for (int i = 0; i < 4; i++) { P[lid * 4 + i] = static_cast<bfloat16_t>(ld[i] * normalizer); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint key[4];
  for (int k = 0; k < 4; k++) { uint i = lid * 4 + k; key[k] = act ? ((uint(as_type<ushort>(P[i])) << 16) | i) : 0u; }
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3 - a; b++)
      if (key[b] < key[b + 1]) { uint t = key[b]; key[b] = key[b + 1]; key[b + 1] = t; }
  int hd = 0;
  for (int r = 0; r < 10; r++) {
    uint mine = hd < 4 ? key[hd] : 0u;
    uint best = simd_max(mine);
    if (hd < 4 && mine == best) { hd++; }
    if (act && lane == 0u) { sgk[sgi * 10 + r] = best; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0u) {
    int h2 = 0;
    for (int r = 0; r < 10; r++) {
      uint mine = (lane < 4u && h2 < 10) ? sgk[lane * 10 + h2] : 0u;
      uint best = simd_max(mine);
      if (lane < 4u && h2 < 10 && mine == best) { h2++; }
      if (lane == 0u) { uint i = best & 0xffffu; ti[9 - r] = i; tv[9 - r] = P[i]; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    bfloat16_t s = static_cast<bfloat16_t>(0.0f);
    for (int e = 0; e < 10; e++) { s = tv[e] + s; }
    for (int e = 0; e < 10; e++) {
      inds[e] = ti[e];
      wts[e] = static_cast<bfloat16_t>(metal::precise::divide(float(tv[e]), float(s)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// the shared-expert gate's value: simd_sum over the 16 partials in lanes 0..15 (dot_product's tid < 16 step), as bf16
METAL_FUNC bfloat16_t pk_seg_value(const device float* spart, uint lane) {
  float s = lane < 16u ? spart[lane] : 0.0f;
  return static_cast<bfloat16_t>(simd_sum(s));
}

// the MoE block output for column c (one thread): col_reduce_small's order over the 10 expert rows (8 threads: j, j + 8;
// thread 0 folds 1..7 in order), + sigmoid(seg) * shared (bf16)
METAL_FUNC bfloat16_t pk_moe_out(const device bfloat16_t* yout, const device bfloat16_t* wts, bfloat16_t seg,
                                 const device bfloat16_t* sout, uint c) {
  bfloat16_t part[8];
  for (int j = 0; j < 8; j++) {
    bfloat16_t t8 = static_cast<bfloat16_t>(0.0f);
    for (int e = j; e < 10; e += 8) { bfloat16_t p = yout[e * 2560 + c] * wts[e]; t8 = p + t8; }
    part[j] = t8;
  }
  bfloat16_t acc = part[0];
  for (int j = 1; j < 8; j++) { acc = part[j] + acc; }
  bfloat16_t sy = fl_sigmoid_pre(seg) * sout[c];
  return acc + sy;
}
"""

# ---------------------------------------------------------------------------------------------------- head + tail
TAIL = r"""
// logsumexp of the f32 row lg[0 .. NH) by one whole TG of 1024: the stock logsumexp_looped order (fl_level S15)
METAL_FUNC float pk_lse(const device float* lg, int NH, threadgroup float* local_max, threadgroup float* local_norm,
                        threadgroup float* lse_s, uint lid, uint lsize, uint lane, uint sgi) {
  float prevmax;
  float maxval = Limits<float>::finite_min;
  float normalizer = 0;
  for (int r = 0; r < int((NH + 4 * int(lsize) - 1) / (4 * int(lsize))); r++) {
    int offset = r * int(lsize) * 4 + int(lid) * 4;
    float vals[4];
    if (offset + 4 <= NH) {
      for (int i = 0; i < 4; i++) { vals[i] = lg[offset + i]; }
    } else {
      for (int i = 0; i < 4; i++) { vals[i] = (offset + i < NH) ? lg[offset + i] : Limits<float>::min; }
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
  if (lane == 0u) { local_max[sgi] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[lane]);
  normalizer *= fast::exp(prevmax - maxval);
  if (lane == 0u) { local_norm[sgi] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_norm[lane]);
  if (lid == 0u) { lse_s[0] = isinf(maxval) ? maxval : (metal::precise::log(normalizer) + maxval); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float lse = lse_s[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return lse;
}

// the frontier step s in place (sf_tree.STEP_SRC's arithmetic and order; one thread): push node s - 1's KK children
// (tis token ids, tp probabilities; the roots at s = 0), pop node s: the largest float64 path_p, the earliest pushed on
// ties. Writes pops[s] = (par, tok, depth - 1, bad).
METAL_FUNC void pk_step(device ulong* fk, device int* fpar, device int* ftok, device int* fdep, device int* nf,
                        device ulong* npp, device int* ndep, device int* nanc, device int* pops, const thread int* tis,
                        const thread float* tp, int s, int KK, int CAP, int NMAX) {
  int n = nf[0];
  int bad = 0;
  ulong pp = (s > 0) ? npp[s - 1] : 0x3FF0000000000000ul;
  int pd = (s > 0) ? ndep[s - 1] + 1 : 0;
  for (int c = 0; c < KK; ++c) {
    if (n >= CAP) { bad = 2; break; }
    fk[n] = f64_mul_f32(pp, tp[c], bad); fpar[n] = s - 1; ftok[n] = tis[c]; fdep[n] = pd; n += 1;
  }
  int best = -1; ulong bk = 0ul;
  for (int i = 0; i < n; ++i) { if (fk[i] > bk) { bk = fk[i]; best = i; } }
  if (best < 0) { bad = 3; best = 0; }
  fk[best] = 0ul;
  int par = fpar[best], d = fdep[best];
  npp[s] = bk; ndep[s] = d;
  for (int j = 0; j < NMAX; ++j) { nanc[s * NMAX + j] = -1; }
  if (par >= 0) {
    for (int j = 0; j < d - 1; ++j) { nanc[s * NMAX + j] = nanc[par * NMAX + j]; }
    nanc[s * NMAX + d - 1] = par;
  }
  nf[0] = n;
  pops[s * 4 + 0] = par; pops[s * 4 + 1] = ftok[best]; pops[s * 4 + 2] = d; pops[s * 4 + 3] = bad;
}
"""
