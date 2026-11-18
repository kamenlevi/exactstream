"""PK SKELETON v3 (MoonShots, Wed 18 Nov 2026; ALL_LEVERS item 1, M-46 resumed): can a persistent pop beat today's
live sync-free pop (ra_sf.draft_sf on r57's live Q8H32 head: 0.93 ms per pop, 178.8 MB, pk_census_q8h32.json)?
Same bytes and phase chain as pk_skel2 (M-46: 1.053 ms at S1 1024, 1.006 at S1 192), with the changes the M-46
ablation pointed at:
  - GEMV: qmv_pf = qmv_fast / qmv's per-lane arithmetic (load_vector, qdot, simd_sum: the stock order), with the
    weights / scales / biases of UB blocks staged in registers first (more loads in flight), R rows per simdgroup,
    one simdgroup per task (no 2-simdgroup vTGs), so latency-bound phases (HC downs, router, shared expert) fill the GPU;
  - attention: one TG per head, 32 key-strided virtual simdgroups with an online softmax (sdpa_vector's shape), keys
    unrolled by 2, combine through scratch (M-46: scores + softmax + PV as two phases);
  - head + tail: each simdgroup keeps a running top-8 of the logits it computes, each TG merges its 16 lists inside
    the head phase; after the last barrier TG 0 does the full-row logsumexp (one TG, the stock looped shape) and a
    parallel rank merge of 40 x 8 candidates (M-46: two more phases, a 1-thread merge from device memory: 90 us).
Phases: P1 fuse | P2-P3 attention HC | P4 q/k/v | P5 attention | P6 o_proj | P7-P8 MLP HC | P9 router + shared
gate/up | P10 route + shared down | P11 experts gate/up | P12 experts down | P13-P14 mixer HC | P15 head + top-8 |
P16 tail (TG 0)  (15 grid barriers). Non-bitwise skeleton math outside the GEMVs (norms, RoPE skipped, routing).
usage: ~/BigMoE/.venv/bin/python -I -B pk_skel3.py [--s1 192] [--rounds 8] [--trees 3] [--ablate] [--out <json>]
       [--r8 R,UB] [--rs R,UB] [--rh R,UB] [--re R,UB]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/bet7"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/draft"))
import mlx.core as mx                                      # noqa: E402
from fl_qmatch import QHEADER                              # noqa: E402
import pk_skel2 as S2                                      # noqa: E402

DENSE, NE, EW_PER, ES_PER = S2.DENSE, S2.NE, S2.EW_PER, S2.ES_PER
NBAR = 15
HELP = S2.HDR[len(QHEADER) + len(S2.FAST4):]               # gbar, prep4, mix4, silu320 (M-46)

PF = r"""
// qdot with the packed weights in registers: the shipped qdot's arithmetic (bits 4 / 8)
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

// the simdgroup's running top-8 (lanes 0..7, value desc, index asc); v, i uniform over the simdgroup
METAL_FUNC void top8_insert(thread float& tv, thread int& ti, float v, int i, uint lane) {
  bool bet = (lane < 8u) && (tv > v || (tv == v && ti < i));
  uint p = uint(simd_sum(bet ? 1 : 0));
  if (p >= 8u) return;
  float uv = simd_shuffle_up(tv, ushort(1));
  int ui = simd_shuffle_up(ti, ushort(1));
  if (lane == p) { tv = v; ti = i; }
  else if (lane > p && lane < 8u) { tv = uv; ti = ui; }
}

// one simdgroup, rows row0 .. row0 + R - 1 of a quantized [N, K] matrix times NV vectors: per row and lane the
// stock per-block order (load_vector, then qdot per row; simd_sum at the end); UB blocks of weights staged first
template <typename T, int gs, int bits, int R, int UB, int PPT, int NV, bool TOP>
METAL_FUNC void qmv_pf(const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
                       device T* y, const int K, const int N, const int row0, const uint lane, thread float& tv,
                       thread int& ti) {
  constexpr int pf = 32 / bits;
  constexpr int vpt = pf * PPT;
  constexpr int bsz = vpt * 32;
  constexpr int sstep = gs / vpt;
  const int Kw = K / pf, Kg = K / gs;
  const device uint32_t* wr = w + row0 * Kw + int(lane) * PPT;
  const device T* sr = scales + row0 * Kg + int(lane) / sstep;
  const device T* br = biases + row0 * Kg + int(lane) / sstep;
  const device T* xr = x + int(lane) * vpt;
  float res[NV][R];
  for (int v = 0; v < NV; ++v) for (int r = 0; r < R; ++r) res[v][r] = 0.0f;
  float xt[vpt];
  const int kfull = (K / bsz) * bsz;
  int k = 0;
  for (; k + UB * bsz <= kfull; k += UB * bsz) {
    uint32_t wb[UB][R][PPT];
    float sb[UB][R], bb[UB][R];
    for (int u = 0; u < UB; ++u)
      for (int r = 0; r < R; ++r) {
        for (int p = 0; p < PPT; ++p) wb[u][r][p] = wr[r * Kw + u * (bsz / pf) + p];
        sb[u][r] = float(sr[r * Kg + u * (bsz / gs)]);
        bb[u][r] = float(br[r * Kg + u * (bsz / gs)]);
      }
    for (int u = 0; u < UB; ++u)
      for (int v = 0; v < NV; ++v) {
        float sum = load_vector<T, float, vpt, bits>(xr + v * K + u * bsz, xt);
        for (int r = 0; r < R; ++r) res[v][r] += qdot_r<vpt, bits>(wb[u][r], xt, sb[u][r], bb[u][r], sum);
      }
    wr += UB * (bsz / pf); sr += UB * (bsz / gs); br += UB * (bsz / gs); xr += UB * bsz;
  }
  for (; k < kfull; k += bsz) {
    uint32_t wb[R][PPT];
    float sb[R], bb[R];
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PPT; ++p) wb[r][p] = wr[r * Kw + p];
      sb[r] = float(sr[r * Kg]); bb[r] = float(br[r * Kg]);
    }
    for (int v = 0; v < NV; ++v) {
      float sum = load_vector<T, float, vpt, bits>(xr + v * K, xt);
      for (int r = 0; r < R; ++r) res[v][r] += qdot_r<vpt, bits>(wb[r], xt, sb[r], bb[r], sum);
    }
    wr += bsz / pf; sr += bsz / gs; br += bsz / gs; xr += bsz;
  }
  if (k < K && int(lane) * vpt < K - k) {                  // qmv's last partial block (skeleton form)
    uint32_t wb[R][PPT];
    float sb[R], bb[R];
    for (int r = 0; r < R; ++r) {
      for (int p = 0; p < PPT; ++p) wb[r][p] = wr[r * Kw + p];
      sb[r] = float(sr[r * Kg]); bb[r] = float(br[r * Kg]);
    }
    for (int v = 0; v < NV; ++v) {
      float sum = load_vector<T, float, vpt, bits>(xr + v * K, xt);
      for (int r = 0; r < R; ++r) res[v][r] += qdot_r<vpt, bits>(wb[r], xt, sb[r], bb[r], sum);
    }
  }
  for (int v = 0; v < NV; ++v)
    for (int r = 0; r < R; ++r) {
      float s = simd_sum(res[v][r]);
      if (lane == 0u) y[v * N + row0 + r] = static_cast<T>(s);
      if (TOP) top8_insert(tv, ti, float(static_cast<T>(s)), row0 + r, lane);
    }
}
"""

HDR = QHEADER + PF + HELP


def sg_dispatch(segs, indent="    "):
    """segs: [(call using {u} = the task index inside its segment, n_tasks)]: every simdgroup takes tasks in turn"""
    total = sum(n for _, n in segs)
    out, lo = [f"{indent}for (uint ut = tg * NSG + sg; ut < {total}u; ut += NTG * NSG) {{"], 0
    for i, (call, n) in enumerate(segs):
        if i < len(segs) - 1:
            out.append(f"{indent}  {'if' if i == 0 else 'else if'} (ut < {lo + n}u) {{ uint u = ut - {lo}u; {call} }}")
        else:
            out.append(f"{indent}  {'else ' if i else ''}{{ uint u = ut - {lo}u; {call} }}")
        lo += n
    out.append(f"{indent}}}")
    return "\n".join(out)


def qc(name, x, y, K, N, cfg, bits=8, gs=64, ppt=2, nv=1):
    R, UB = cfg
    return (f"qmv_pf<bfloat16_t, {gs}, {bits}, {R}, {UB}, {ppt}, {nv}, false>(dw + W_{name}, ds + S_{name}, "
            f"db + S_{name}, {x}, {y}, {K}, {N}, int(u) * {R}, lane, tv, ti);", N // R)


def build_src(c):
    r8, rs, rh, re_ = c["r8"], c["rs"], c["rh"], c["re"]
    G = {}
    G["P1"] = sg_dispatch([qc("fc_emb", "mine", "scr + O_EN", 2560, 2560, r8),
                           qc("fc_hid", "mine + 2560u", "scr + O_FH", 2560, 2560, (2, 1), nv=4)])
    G["P2"] = sg_dispatch([qc("ahc_dn", "mine", "scr + O_HDN0", 10240, 320, rs)])
    G["P3"] = sg_dispatch([qc("ahc_up", "mine + 10240u", "scr + O_HUP0", 320, 10240, (4, 1), ppt=1)])
    G["P4"] = sg_dispatch([qc("q", "mine + 12288u", "scr + O_Q", 2560, 12288, r8),
                           qc("k", "mine + 12288u", "scr + O_K", 2560, 512, rs),
                           qc("v", "mine + 12288u", "scr + O_V", 2560, 512, rs)])
    G["P6"] = sg_dispatch([qc("o", "scr + O_O", "scr + O_BLK", 6144, 2560, r8)])
    G["P7"] = sg_dispatch([qc("mhc_dn", "mine", "scr + O_HDN1", 10240, 320, rs)])
    G["P8"] = sg_dispatch([qc("mhc_up", "mine + 10240u", "scr + O_HUP1", 320, 10240, (4, 1), ppt=1)])
    G["P9"] = sg_dispatch([qc("router", "mine + 12288u", "scr + O_RL", 2560, 512, rs),
                           qc("se_gate", "mine + 12288u", "scr + O_SEG", 2560, 640, rs),
                           qc("se_up", "mine + 12288u", "scr + O_SEU", 2560, 640, rs)])
    G["P10"] = sg_dispatch([qc("se_down", "mine + 13000u", "scr + O_SY", 640, 2560, (4, 1), ppt=1)])
    R, UB = re_
    ng = 640 // R
    G["P11"] = sg_dispatch([(
        f"uint e = u / {2 * ng}u, rr = u % {2 * ng}u, gu = rr / {ng}u, rg = rr % {ng}u; "
        f"uint ex = (uint(((device int*)(scf + O_TOPf))[e]) + e * 13u) % uint(NE_C); "
        f"qmv_pf<bfloat16_t, 32, 4, {R}, {UB}, 2, 1, false>(ew + size_t(ex) * EW_PER + gu * 204800u, "
        f"es + size_t(ex) * ES_PER + gu * 51200u, eb + size_t(ex) * ES_PER + gu * 51200u, scr + O_MX2, "
        f"scr + O_EGU + e * 1280u + gu * 640u, 2560, 640, int(rg) * {R}, lane, tv, ti);", 10 * 2 * ng)])
    G["P12"] = sg_dispatch([(
        "uint e = u % 10u, cb = u / 10u; uint ex = (uint(((device int*)(scf + O_TOPf))[e]) + e * 13u) % uint(NE_C); "
        "qmv_pf<bfloat16_t, 32, 4, 4, 1, 1, 1, false>(ew + size_t(ex) * EW_PER + 409600u, es + size_t(ex) * ES_PER + "
        "102400u, eb + size_t(ex) * ES_PER + 102400u, mine + e * 640u, scr + O_EY + e * 2560u, 640, 2560, int(cb) * 4, "
        "lane, tv, ti);", 10 * 2560 // 4)])
    G["P13"] = sg_dispatch([qc("xhc_dn", "mine", "scr + O_HDN2", 10240, 320, rs)])
    G["P14"] = sg_dispatch([qc("xhc_up", "mine + 10240u", "scr + O_HUP2", 320, 10240, (4, 1), ppt=1)])
    R, UB = rh
    G["P15"] = sg_dispatch([(f"qmv_pf<bfloat16_t, 64, 4, {R}, {UB}, 2, 1, true>(hw, hs, hb, mine + 12288u, scr + O_LG, "
                             f"2560, NH_C, int(u) * {R}, lane, tv, ti);", c["nh"] // R)])
    src = SRC_T
    for k_, v_ in G.items():
        src = src.replace("{" + k_ + "}", v_)
    return src


SRC_T = r"""
  const uint tg = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint t = uint(meta[0]), base = uint(meta[1]), S1 = uint(meta[2]);
  device atomic_uint* ctr = (device atomic_uint*)syncb;
  device bfloat16_t* scr = (device bfloat16_t*)scratch;
  device float* scf = (device float*)scratch;
  device bfloat16_t* mine = scr + O_PRIV + tg * 16384u;
  device bfloat16_t* hsw = (device bfloat16_t*)hsbuf;
  device bfloat16_t* tkw = (device bfloat16_t*)tkv;
  threadgroup uint tflag;
  threadgroup float red[96];
  threadgroup float stv[128];
  threadgroup int sti[128];
  threadgroup float cvv[NTG * 8u];
  threadgroup int cii[NTG * 8u];
  float tv = -INFINITY;
  int ti = 0x7FFFFFFF;
  uint nb = 0;
  bool ok = true;
  const uint J = min(t + 1u, 16u), L = S1 + J;
#define PH(k) (WORK && ((PHMASK >> (k)) & 1u))
#define BAR() { nb += 1u; if (ok) ok = gbar(ctr, (base + nb) * NTG, tid, tflag, SPIN); }
  const uint par = (t == 0u) ? 0u : uint(fst[0]), tok = (t == 0u) ? 1u : uint(fst[1]) % 1024u;

  // P1 fuse: normed emb row + the parent's 4 normed streams (every TG, its own slot); fc_embedding + fc_hidden (x4)
  if (PH(1)) {
    prep4(hsbuf + par * 10240u, scr, scr, 1u, (device bfloat16_t*)0, mine + 2560u, tid, NT, red);
    if (tid < 640u) { float4 e = float4(*(const device vec<bfloat16_t, 4>*)(embt + tok * 2560u + tid * 4u)); *(device vec<bfloat16_t, 4>*)(mine + tid * 4u) = vec<bfloat16_t, 4>(e); }
    threadgroup_barrier(mem_flags::mem_device);
{P1}
  }
  BAR();
  // P2 attention HC down
  if (ok && PH(2)) {
    prep4(scr + O_FH, scr + O_EN, scr, 2u, (device bfloat16_t*)0, mine, tid, NT, red);
{P2}
  }
  BAR();
  // P3 attention HC up
  if (ok && PH(3)) {
    silu320(scr + O_HDN0, mine + 10240u, tid, NT);
{P3}
  }
  BAR();
  // P4 mixed -> q / k / v
  if (ok && PH(4)) {
    mix4(scr + O_HUP0, mine, mine + 12288u, tid, NT);
{P4}
  }
  BAR();
  // P5 attention: TG h = head h (24); 32 key-strided virtual simdgroups (2 per simdgroup), online softmax, keys x2
  if (ok && PH(5)) {
    if (tg == NTG - 1u) for (uint i = tid; i < 1024u; i += NT) tkw[t * 1024u + i] = (i < 512u) ? scr[O_K + i] : scr[O_V + i - 512u];
    if (tg < 24u) {
      const uint h = tg, g = h / 12u;
      float qv[8];
      for (uint e = 0; e < 8u; ++e) qv[e] = float(scr[O_Q + h * 512u + lane * 8u + e]) * 0.0625f;
      for (uint hv = 0; hv < 2u; ++hv) {
        const uint vs = sg + hv * NSG;
        float m = -INFINITY, l = 0.0f, o[8];
        for (uint e = 0; e < 8u; ++e) o[e] = 0.0f;
        for (uint j = vs; j < L; j += 64u) {
          const uint j2 = j + 32u;
          const bool two = j2 < L;
          const device bfloat16_t* ka = (j < S1) ? (k1 + (g * S1 + j) * 256u) : ((j + 1u == L) ? (scr + O_K + g * 256u) : (tkv + (j - S1) * 1024u + g * 256u));
          const device bfloat16_t* va = (j < S1) ? (v1 + (g * S1 + j) * 256u) : ((j + 1u == L) ? (scr + O_V + g * 256u) : (tkv + (j - S1) * 1024u + 512u + g * 256u));
          const uint jb = two ? j2 : j;
          const device bfloat16_t* kb = (jb < S1) ? (k1 + (g * S1 + jb) * 256u) : ((jb + 1u == L) ? (scr + O_K + g * 256u) : (tkv + (jb - S1) * 1024u + g * 256u));
          const device bfloat16_t* vb = (jb < S1) ? (v1 + (g * S1 + jb) * 256u) : ((jb + 1u == L) ? (scr + O_V + g * 256u) : (tkv + (jb - S1) * 1024u + 512u + g * 256u));
          vec<bfloat16_t, 8> k8a = *(const device vec<bfloat16_t, 8>*)(ka + lane * 8u);
          vec<bfloat16_t, 8> k8b = *(const device vec<bfloat16_t, 8>*)(kb + lane * 8u);
          vec<bfloat16_t, 8> v8a = *(const device vec<bfloat16_t, 8>*)(va + lane * 8u);
          vec<bfloat16_t, 8> v8b = *(const device vec<bfloat16_t, 8>*)(vb + lane * 8u);
          float sa = 0.0f, sb = 0.0f;
          for (uint e = 0; e < 8u; ++e) { sa += qv[e] * float(k8a[e]); sb += qv[e] * float(k8b[e]); }
          sa = simd_sum(sa);
          sb = simd_sum(sb);
          float mn = max(m, sa), fa = metal::exp(m - mn), ea = metal::exp(sa - mn);
          l = l * fa + ea;
          for (uint e = 0; e < 8u; ++e) o[e] = o[e] * fa + ea * float(v8a[e]);
          m = mn;
          if (two) {
            mn = max(m, sb); fa = metal::exp(m - mn); float eb2 = metal::exp(sb - mn);
            l = l * fa + eb2;
            for (uint e = 0; e < 8u; ++e) o[e] = o[e] * fa + eb2 * float(v8b[e]);
            m = mn;
          }
        }
        device float* ao = scf + O_AOf + (h * 32u + vs) * 264u;
        *(device float4*)(ao + lane * 8u) = float4(o[0], o[1], o[2], o[3]);
        *(device float4*)(ao + lane * 8u + 4u) = float4(o[4], o[5], o[6], o[7]);
        if (lane == 0u) { ao[256] = m; ao[257] = l; }
      }
      threadgroup_barrier(mem_flags::mem_device);
      if (tid < 256u) {
        const device float* a0 = scf + O_AOf + h * 32u * 264u;
        float M = -INFINITY;
        for (uint q = 0; q < 32u; ++q) M = max(M, a0[q * 264u + 256u]);
        float S = 0.0f, O = 0.0f;
        for (uint q = 0; q < 32u; ++q) { float f = metal::exp(a0[q * 264u + 256u] - M); S += a0[q * 264u + 257u] * f; O += a0[q * 264u + tid] * f; }
        float gt = float(scr[O_Q + h * 512u + 256u + tid]);
        scr[O_O + h * 256u + tid] = bfloat16_t(O / S / (1.0f + metal::exp(-gt)));
      }
    }
  }
  BAR();
  // P6 o_proj
  if (ok && PH(6)) {
{P6}
  }
  BAR();
  // P7 MLP HC down: hyper' = x4 + blk (TG 0 keeps it)
  if (ok && PH(7)) {
    prep4(scr + O_FH, scr + O_EN, scr + O_BLK, 3u, tg == 0u ? scr + O_HYP : (device bfloat16_t*)0, mine, tid, NT, red);
{P7}
  }
  BAR();
  // P8 MLP HC up
  if (ok && PH(8)) {
    silu320(scr + O_HDN1, mine + 10240u, tid, NT);
{P8}
  }
  BAR();
  // P9 mixed2 -> router | shared gate / up
  if (ok && PH(9)) {
    mix4(scr + O_HUP1, mine, mine + 12288u, tid, NT);
    if (tg == 0u) for (uint c = tid; c < 2560u; c += NT) scr[O_MX2 + c] = mine[12288u + c];
{P9}
  }
  BAR();
  // P10 route (TG 0, simdgroup 0: fp32 softmax over 512, top-10) | swiglu + shared down
  if (ok && PH(10)) {
    if (tg == 0u && sg == 0u) {
      float v[16];
      for (uint e = 0; e < 16u; ++e) v[e] = float(scr[O_RL + lane * 16u + e]);
      float m = -INFINITY;
      for (uint e = 0; e < 16u; ++e) m = max(m, v[e]);
      m = simd_max(m);
      float s = 0.0f;
      for (uint e = 0; e < 16u; ++e) { v[e] = metal::precise::exp(v[e] - m); s += v[e]; }
      s = simd_sum(s);
      for (uint e = 0; e < 16u; ++e) v[e] /= s;
      for (uint r = 0; r < 10u; ++r) {
        float bv = -1.0f; uint bi = 0;
        for (uint e = 0; e < 16u; ++e) if (v[e] > bv) { bv = v[e]; bi = lane * 16u + e; }
        float mv = simd_max(bv);
        uint mi = simd_min(bv == mv ? bi : 0xFFFFFFFFu);
        if (lane == 0u) { ((device int*)(scf + O_TOPf))[r] = int(mi); scf[O_TOPf + 16u + r] = mv; }
        if (mi / 16u == lane) v[mi % 16u] = -2.0f;
      }
    }
    for (uint i = tid; i < 640u; i += NT) {
      float gg = float(scr[O_SEG + i]), uu = float(scr[O_SEU + i]);
      mine[13000u + i] = bfloat16_t(gg / (1.0f + metal::exp(-gg)) * uu);
    }
    threadgroup_barrier(mem_flags::mem_device);
{P10}
  }
  BAR();
  // P11 experts gate / up (10 x 4-bit g32)
  if (ok && PH(11)) {
{P11}
  }
  BAR();
  // P12 experts down (10 x 4-bit g32, qmv)
  if (ok && PH(12)) {
    for (uint i = tid; i < 6400u; i += NT) {
      uint e = i / 640u, c = i % 640u;
      float gg = float(scr[O_EGU + e * 1280u + c]), uu = float(scr[O_EGU + e * 1280u + 640u + c]);
      mine[i] = bfloat16_t(gg / (1.0f + metal::exp(-gg)) * uu);
    }
    threadgroup_barrier(mem_flags::mem_device);
{P12}
  }
  BAR();
  // P13 h = sy + sum_e sc_e y_e; hyper'' = hyper' + h (TG 0 writes the node state); mixer HC down
  if (ok && PH(13)) {
    for (uint q = tid; q < 640u; q += NT) {
      uint c0 = q * 4u;
      float4 hh = float4(*(const device vec<bfloat16_t, 4>*)(scr + O_SY + c0));
      for (uint e = 0; e < 10u; ++e) hh += scf[O_TOPf + 16u + e] * float4(*(const device vec<bfloat16_t, 4>*)(scr + O_EY + e * 2560u + c0));
      *(device vec<bfloat16_t, 4>*)(mine + 12288u + c0) = vec<bfloat16_t, 4>(hh);
    }
    threadgroup_barrier(mem_flags::mem_device);
    prep4(scr + O_HYP, mine + 12288u, scr, 2u, tg == 0u ? hsw + (t + 1u) * 10240u : (device bfloat16_t*)0, mine, tid, NT, red);
{P13}
  }
  BAR();
  // P14 mixer HC up
  if (ok && PH(14)) {
    silu320(scr + O_HDN2, mine + 10240u, tid, NT);
{P14}
  }
  BAR();
  // P15 x_head -> screened head (4-bit g64) + each simdgroup's running top-8; the TG merges its 16 lists
  if (ok && PH(15)) {
    mix4(scr + O_HUP2, mine, mine + 12288u, tid, NT);
{P15}
    if (lane < 8u) { stv[sg * 8u + lane] = tv; sti[sg * 8u + lane] = ti; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float cv[NSG / 8u]; int ci[NSG / 8u];
      for (uint q = 0; q < NSG / 8u; ++q) { cv[q] = stv[lane * (NSG / 8u) + q]; ci[q] = sti[lane * (NSG / 8u) + q]; }
      for (uint r = 0; r < 8u; ++r) {
        float bv = -INFINITY; int bi = 0x7FFFFFFF; uint bq = 99u;
        for (uint q = 0; q < NSG / 8u; ++q) if (cv[q] > bv || (cv[q] == bv && ci[q] < bi)) { bv = cv[q]; bi = ci[q]; bq = q; }
        float mv = simd_max(bv);
        int mi = simd_min(bv == mv ? bi : 0x7FFFFFFF);
        if (bv == mv && bi == mi && bq < 99u) { cv[bq] = -INFINITY; ci[bq] = 0x7FFFFFFF; }
        if (lane == 0u) { scf[O_TK8f + tg * 16u + r] = mv; ((device int*)(scf + O_TK8f))[tg * 16u + 8u + r] = mi; }
      }
    }
  }
  BAR();
  // P16 tail (TG 0): logsumexp over the full row (one TG, 4 per read, online max / sum), rank merge of 40 x 8
  if (tg == 0u) {
    float lse = 0.0f;
    if (ok && PH(16)) {
      float m = -INFINITY, s = 0.0f;
      for (uint i = tid * 4u; i < NH_C; i += NT * 4u) {
        float4 x = float4(*(const device vec<bfloat16_t, 4>*)(scr + O_LG + i));
        float mn = max(m, max(max(x[0], x[1]), max(x[2], x[3])));
        s = s * metal::exp(m - mn) + metal::exp(x[0] - mn) + metal::exp(x[1] - mn) + metal::exp(x[2] - mn) + metal::exp(x[3] - mn);
        m = mn;
      }
      float M = simd_max(m);
      s = simd_sum(s * metal::exp(m - M));
      if (lane == 0u) { red[sg] = M; red[32u + sg] = s; }
      if (tid < NTG * 8u) {
        cvv[tid] = scf[O_TK8f + (tid / 8u) * 16u + tid % 8u];
        cii[tid] = ((device int*)(scf + O_TK8f))[(tid / 8u) * 16u + 8u + tid % 8u];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < NTG * 8u) {
        float v = cvv[tid]; int i = cii[tid]; uint rk = 0;
        for (uint q = 0; q < NTG * 8u; ++q) rk += (cvv[q] > v || (cvv[q] == v && cii[q] < i)) ? 1u : 0u;
        if (rk < 8u) { stv[rk] = v; sti[rk] = i; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float MM = red[0];
      for (uint q = 1; q < NSG; ++q) MM = max(MM, red[q]);
      float SS = 0.0f;
      for (uint q = 0; q < NSG; ++q) SS += red[32u + q] * metal::exp(red[q] - MM);
      lse = MM + metal::precise::log(SS);
    } else {
      if (tid < 8u) { stv[tid] = 0.0f; sti[tid] = int(tid); }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      for (uint r = 0; r < 8u; ++r) { out_tis[r] = sti[r]; out_tp[r] = metal::precise::exp(stv[r] - lse); }
      out_pop[0] = int(t); out_pop[1] = sti[0]; out_pop[2] = int(J); out_pop[3] = ok ? 0 : 1; out_pop[4] = int(nb);
      fst_out[0] = int(t + 1u); fst_out[1] = sti[0] & 0x3FF; fst_out[2] = ok ? 0 : 1;
      for (uint r = 3; r < 8u; ++r) fst_out[r] = 0;
    }
  }
"""


def scratch_layout(ntg):
    o, cur = {}, 0
    for name, n in (("EN", 2560), ("FH", 10240), ("HDN0", 512), ("HUP0", 10240), ("Q", 12288), ("K", 512), ("V", 512),
                    ("O", 6144), ("BLK", 2560), ("HYP", 10240), ("HDN1", 512), ("HUP1", 10240), ("MX2", 2560),
                    ("RL", 512), ("SEG", 640), ("SEU", 640), ("SY", 2560), ("EGU", 12800), ("EY", 25600),
                    ("HDN2", 512), ("HUP2", 10240), ("LG", 37120)):
        o[name] = cur
        cur += (n + 255) // 256 * 256
    o["AOf"] = (cur + 1) // 2                                       # float offsets from here
    cur = (o["AOf"] + 24 * 32 * 264) * 2
    o["TOPf"] = cur // 2
    cur = (o["TOPf"] + 64) * 2
    o["TK8f"] = cur // 2
    cur = (o["TK8f"] + ntg * 16 + 64) * 2
    o["PRIV"] = (cur + 255) // 256 * 256
    return o, o["PRIV"] + ntg * 16384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntg", type=int, default=40)
    ap.add_argument("--nt", type=int, default=512)
    ap.add_argument("--s1", type=int, default=192)
    ap.add_argument("--nh", type=int, default=36864)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--trees", type=int, default=3)
    ap.add_argument("--spin", type=int, default=400000)
    ap.add_argument("--r8", default="4,2", help="R,UB of the big 8-bit GEMVs (fc_emb, q, o)")
    ap.add_argument("--rs", default="1,2", help="R,UB of the few-row 8-bit GEMVs (HC downs, k, v, router, shared)")
    ap.add_argument("--rh", default="2,2", help="R,UB of the head (4-bit)")
    ap.add_argument("--re", default="2,1", help="R,UB of the experts gate / up (4-bit)")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    NTG, NT, S1, NH = a.ntg, a.nt, a.s1, a.nh
    cfg = {k: tuple(int(x) for x in getattr(a, k).split(",")) for k in ("r8", "rs", "rh", "re")}
    cfg["nh"] = NH
    W, S, nw32, ns16 = S2.offsets()
    sc, scr_n = scratch_layout(NTG)
    mx.random.seed(0)
    dw = mx.random.randint(0, 2 ** 31, (nw32,), dtype=mx.uint32)
    ds = (mx.random.normal((ns16,)) * 0.002).astype(mx.bfloat16)
    db = (mx.random.normal((ns16,)) * 0.002).astype(mx.bfloat16)
    ew = mx.random.randint(0, 2 ** 31, (NE * EW_PER,), dtype=mx.uint32)
    es = (mx.random.normal((NE * ES_PER,)) * 0.002).astype(mx.bfloat16)
    eb = (mx.random.normal((NE * ES_PER,)) * 0.002).astype(mx.bfloat16)
    hw = mx.random.randint(0, 2 ** 31, (NH * 320,), dtype=mx.uint32)
    hs = (mx.random.normal((NH * 40,)) * 0.002).astype(mx.bfloat16)
    hb = (mx.random.normal((NH * 40,)) * 0.002).astype(mx.bfloat16)
    embt = mx.random.normal((1024 * 2560,)).astype(mx.bfloat16)
    k1 = mx.random.normal((2 * S1 * 256,)).astype(mx.bfloat16)
    v1 = mx.random.normal((2 * S1 * 256,)).astype(mx.bfloat16)
    hsbuf = mx.random.normal((40 * 10240,)).astype(mx.bfloat16)
    tkv = mx.zeros((40 * 1024,), dtype=mx.bfloat16)
    scratch = mx.zeros((scr_n,), dtype=mx.bfloat16)
    mx.eval(dw, ds, db, ew, es, eb, hw, hs, hb, embt, k1, v1, hsbuf, tkv, scratch)
    bytes_pop = dw.nbytes + ds.nbytes + db.nbytes + 10 * (EW_PER * 4 + 2 * ES_PER * 2) + hw.nbytes + hs.nbytes + hb.nbytes
    floor = bytes_pop / 292.7e6
    print(f"[pk_skel3] NTG {NTG} x {NT}, S1 {S1}, NH {NH}, cfg {cfg}; bytes per pop {bytes_pop / 1e6:.1f} MB; floor at "
          f"292.7 GB/s {floor:.3f} ms, at 260 GB/s {bytes_pop / 260e6:.3f} ms", flush=True)
    defs = "".join(f"#define W_{n} {W[n]}u\n#define S_{n} {S[n]}u\n" for n, _, _ in DENSE)
    defs += "".join(f"#define O_{k} {v}u\n" for k, v in sc.items())
    defs += f"#define EW_PER {EW_PER}u\n#define ES_PER {ES_PER}u\n#define NSG {NT // 32}u\n#define VPP {NT // 64}u\n"
    defs += f"#define NHEAD_C 24u\n#define NE_C {NE}\n#define NH_C {NH}u\n"
    SRC = build_src(cfg)
    tagc = "_".join(f"{k}{v[0]}{v[1]}" for k, v in cfg.items() if k != "nh")

    def kern(work, mask=(1 << 17) - 2):
        src = defs + f"#define PHMASK {mask}u\n#define WORK {int(work)}\n#define NTG {NTG}u\n#define NT {NT}u\n#define SPIN {a.spin}u\n" + SRC
        return mx.fast.metal_kernel(
            name=f"pk_skel3_w{int(work)}_m{mask}_{NTG}_{NT}_{S1}_{NH}_{tagc}",
            input_names=["dw", "ds", "db", "ew", "es", "eb", "hw", "hs", "hb", "embt", "k1", "v1", "hsbuf", "tkv",
                         "scratch", "syncb", "fst", "meta"],
            output_names=["out_pop", "out_tis", "out_tp", "fst_out"], source=src, header=HDR)

    KW, KB = kern(1), kern(0)
    ks = mx.fast.metal_kernel(name="pk_stream2", input_names=["dw", "ds", "db", "ew", "es", "eb", "hw", "hs", "hb"],
                              output_names=["out"], header="#define NDW4 %du\n#define NDS4 %du\n#define EW_PER %du\n"
                              "#define ES_PER %du\n#define NHW4 %du\n#define NHS4 %du\n" % (
                                  dw.size // 4, ds.size // 8, EW_PER, ES_PER, hw.size // 4, hs.size // 8), source=S2.STREAM)
    state = {"base": 0, "sync": mx.zeros((64,), dtype=mx.uint32), "fst": mx.zeros((8,), dtype=mx.int32)}

    def tree(K):
        fst, outs = state["fst"], []
        for t in range(15):
            meta = mx.array([t, state["base"], S1, 0], dtype=mx.int32)
            o_pop, o_tis, o_tp, fst = K(inputs=[dw, ds, db, ew, es, eb, hw, hs, hb, embt, k1, v1, hsbuf, tkv, scratch,
                                               state["sync"], fst, meta], grid=(NTG * NT, 1, 1), threadgroup=(NT, 1, 1),
                                       output_shapes=[(8,), (8,), (8,), (8,)],
                                       output_dtypes=[mx.int32, mx.int32, mx.float32, mx.int32])
            state["base"] += NBAR
            outs += [o_pop, o_tis, o_tp]
        state["fst"] = fst
        return outs

    def stream():
        return [ks(inputs=[dw, ds, db, ew, es, eb, hw, hs, hb], grid=(65536, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(256,)], output_dtypes=[mx.int32])[0]]

    def timed(f, n):
        mx.synchronize()
        t0 = time.perf_counter()
        outs = []
        for _ in range(n):
            outs += f()
        mx.eval(outs)
        mx.synchronize()
        return time.perf_counter() - t0, outs

    _, outs = timed(lambda: tree(KW), 1)
    pops = [o.tolist() for o in outs[0::3]]
    fails = sum(p[3] for p in pops)
    print(f"[pk_skel3] smoke: barrier fails {fails} of {len(pops)} pops; barriers counted {pops[-1][4]}; "
          f"top-8 ids {outs[1].tolist()[:4]} p {[round(x, 4) for x in outs[2].tolist()[:3]]}", flush=True)
    if a.ablate:
        tb = sorted(timed(lambda: tree(KB), 2)[0] / 30 * 1e3 for _ in range(5))[2]
        prev, rows = tb, []
        for k in range(1, 17):
            Kk = kern(1, sum(1 << j for j in range(1, k + 1)))
            timed(lambda: tree(Kk), 1)
            tk = sorted(timed(lambda: tree(Kk), 2)[0] / 30 * 1e3 for _ in range(5))[2]
            rows.append(dict(phase=k, us=1e3 * (tk - prev), cum_ms=tk))
            print(f"[pk_skel3] + P{k:2d}: {1e3 * (tk - prev):7.1f} us (phases 1..{k}: {tk:.3f} ms)", flush=True)
            prev = tk
        print(f"[pk_skel3] barriers only {tb:.3f} ms", flush=True)
        if a.out:
            json.dump(dict(cfg={k: v for k, v in cfg.items()}, barriers_ms=tb, rows=rows), open(a.out, "w"), indent=1)
        return
    res = dict(ntg=NTG, nt=NT, s1=S1, nh=NH, cfg={k: v for k, v in cfg.items()}, bytes_pop=bytes_pop, floor_ms=floor,
               fails_smoke=fails, skel=[], bar=[], stream=[])
    for r in range(a.rounds):
        for arm in (["skel", "bar", "stream"] if r % 2 == 0 else ["stream", "bar", "skel"]):
            if arm == "skel":
                dt, outs = timed(lambda: tree(KW), a.trees)
                fails += sum(int(o.tolist()[3]) for o in outs[0::3])
                res["skel"].append(dt / (15 * a.trees) * 1e3)
            elif arm == "bar":
                res["bar"].append(timed(lambda: tree(KB), a.trees)[0] / (15 * a.trees) * 1e3)
            else:
                res["stream"].append(timed(stream, 15 * a.trees)[0] / (15 * a.trees) * 1e3)
    res["fails_total"] = fails
    for arm in ("skel", "bar", "stream"):
        v = sorted(res[arm][2:])
        res[arm + "_med"] = v[len(v) // 2]
    res["f_line"] = res["skel_med"] - floor
    print(f"[pk_skel3] ms per pop (median of {a.rounds - 2} rounds, {a.trees} trees x 15 pops): skel {res['skel_med']:.3f}, "
          f"barriers only {res['bar_med']:.3f}, stream {res['stream_med']:.3f} ({bytes_pop / res['stream_med'] / 1e6:.1f} GB/s); "
          f"f = {res['f_line']:.3f}; barrier fails {fails}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
