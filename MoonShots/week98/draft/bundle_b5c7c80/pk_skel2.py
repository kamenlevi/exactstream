"""PK SKELETON v2 (MoonShots, Tue 17 Nov 2026; PK_PLAN.md, QUEUE M-46): the pre-registered "tune once" of the skeleton.
One persistent kernel per pop of the Q8H32 head on synthetic weights of the exact real shapes, the real phase chain and
grid barriers (Leap's fenced counter). Line (M-46): skeleton f >= 0.25 ms -> KILL the line; <= 0.15 -> GO.
Every quantized GEMV is MLX's shipped qmv_fast_impl / qmv_impl (bitwise == mx.quantized_matmul on every PK shape,
pk_qpf.py), run as stock "virtual threadgroups" (2 simdgroups, 8 rows) by independent simdgroup pairs; fc_hidden's 4
streams share one weight read (qmv_fast4: the same per-output arithmetic). Norms / attention / routing / tail move the
real bytes with simplified math. Changes from v1: the PV loop keeps 8 V loads in flight; the tail is distributed (all
threadgroups: partial logsumexp + top-8 of a slice; one more barrier; threadgroup 0 merges); one-pass vector preps.
Phases: P1 fuse | P2-P3 attention HC | P4 q/k/v | P5 scores | P6 softmax + PV | P7 o_proj | P8-P9 MLP HC |
P10 router + shared gate/up | P11 route + shared down | P12 experts gate/up | P13 experts down | P14-P15 mixer HC |
P16 head | P17 tail slices | P18 merge + frontier (17 grid barriers).
usage: ~/BigMoE/.venv/bin/python -I -B pk_skel2.py [--ntg 40] [--nt 512] [--s1 1024] [--rounds 12] [--trees 4]
       [--ablate] [--out <json>]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx                                      # noqa: E402
from fl_qmatch import QHEADER                              # noqa: E402

DENSE = [("fc_emb", 2560, 2560), ("fc_hid", 2560, 2560), ("ahc_dn", 320, 10240), ("ahc_up", 10240, 320),
         ("q", 12288, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 6144),
         ("mhc_dn", 320, 10240), ("mhc_up", 10240, 320), ("router", 512, 2560), ("se_gate", 640, 2560),
         ("se_up", 640, 2560), ("se_down", 2560, 640), ("xhc_dn", 320, 10240), ("xhc_up", 10240, 320)]
NE = 128
NBAR = 17
EW_PER, ES_PER = 614400, 153600

FAST4 = r"""
// qmv_fast_impl for 4 input vectors sharing one weight read: per vector, the shipped per-output arithmetic
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast4_impl(const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x0, device T* y0, const int in_vec_size, const int out_vec_size, uint3 tid, uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U xt[4][values_per_thread];
  thread U result[4][results_per_simdgroup] = {{0}};
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x0 + simd_lid * values_per_thread;
  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum[4];
    for (int v = 0; v < 4; ++v) sum[v] = load_vector<T, U, values_per_thread, bits>(xs + v * in_vec_size, xt[v]);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      U s = scales[row * in_vec_size_g];
      U b = biases[row * in_vec_size_g];
      for (int v = 0; v < 4; ++v) result[v][row] += qdot<U, values_per_thread, bits>(wl, xt[v], s, b, sum[v]);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    xs += block_size;
  }
  for (int v = 0; v < 4; ++v)
    for (int row = 0; row < results_per_simdgroup; row++) {
      U r = simd_sum(result[v][row]);
      if (simd_lid == 0) y0[v * out_vec_size + out_row + row] = static_cast<T>(r);
    }
}
"""

HDR = QHEADER + FAST4 + r"""
#define VLOOP(NVT) for (uint vt = tg * VPP + (sg >> 1); vt < (uint)(NVT); vt += NTG * VPP)

METAL_FUNC bool gbar(device atomic_uint* c, uint target, uint tid, threadgroup uint& tflag, uint spin) {
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
    uint it = 0, f = 0;
    while (atomic_load_explicit(c, memory_order_relaxed) < target) { if (++it > spin) { f = 1; break; } }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    tflag = f;
  }
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return tflag == 0u;
}

// 4 x 2560: x = a + b (+ c) (bf16 per add); optional raw copy; per-stream rms-normed -> out (device, this TG's slot)
METAL_FUNC void prep4(const device bfloat16_t* a, const device bfloat16_t* b, const device bfloat16_t* c, uint nadd,
                      device bfloat16_t* raw, device bfloat16_t* out, uint tid, uint nt, threadgroup float* red) {
  float4 acc = 0.0f;
  for (uint q = tid; q < 2560u; q += nt) {
    uint i = q * 4u, c0 = i % 2560u, r = i / 2560u;
    float4 x = float4(*(const device vec<bfloat16_t, 4>*)(a + i));
    if (nadd > 1u) x = float4(vec<bfloat16_t, 4>(x + float4(*(const device vec<bfloat16_t, 4>*)(b + c0))));
    if (nadd > 2u) x = float4(vec<bfloat16_t, 4>(x + float4(*(const device vec<bfloat16_t, 4>*)(c + c0))));
    if (raw) *(device vec<bfloat16_t, 4>*)(raw + i) = vec<bfloat16_t, 4>(x);
    *(device vec<bfloat16_t, 4>*)(out + i) = vec<bfloat16_t, 4>(x);
    acc[r] += dot(x, x);
  }
  for (uint r = 0; r < 4u; ++r) acc[r] = simd_sum(acc[r]);
  if ((tid & 31u) == 0u) for (uint r = 0; r < 4u; ++r) red[r * 16u + (tid >> 5)] = acc[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4u) { float s = 0.0f; for (uint q = 0; q < nt / 32u; ++q) s += red[tid * 16u + q]; red[64u + tid] = metal::precise::rsqrt(s / 2560.0f + 1e-6f); }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint q = tid; q < 2560u; q += nt) {
    uint i = q * 4u, r = i / 2560u;
    *(device vec<bfloat16_t, 4>*)(out + i) = vec<bfloat16_t, 4>(float4(*(const device vec<bfloat16_t, 4>*)(out + i)) * red[64u + r]);
  }
  threadgroup_barrier(mem_flags::mem_device);
}

// a 2560-wide mixed input: out[c] = 0.25 * sum_s sigmoid(up[s][c]) * normed[s][c]  (4 columns per thread)
METAL_FUNC void mix4(const device bfloat16_t* up, const device bfloat16_t* nrm, device bfloat16_t* out, uint tid, uint nt) {
  for (uint q = tid; q < 640u; q += nt) {
    uint c0 = q * 4u;
    float4 a = 0.0f;
    for (uint s = 0; s < 4u; ++s) {
      float4 u = float4(*(const device vec<bfloat16_t, 4>*)(up + s * 2560u + c0));
      a += float4(*(const device vec<bfloat16_t, 4>*)(nrm + s * 2560u + c0)) / (1.0f + metal::exp(-u));
    }
    *(device vec<bfloat16_t, 4>*)(out + c0) = vec<bfloat16_t, 4>(a * 0.25f);
  }
  threadgroup_barrier(mem_flags::mem_device);
}

METAL_FUNC void silu320(const device bfloat16_t* d, device bfloat16_t* out, uint tid, uint nt) {
  for (uint i = tid; i < 320u; i += nt) { float v = float(d[i]) * 0.25f; out[i] = bfloat16_t(v / (1.0f + metal::exp(-v))); }
  threadgroup_barrier(mem_flags::mem_device);
}
"""


def gemv_dispatch(segs, indent="    "):
    """segs: [(call_template, n_vtg)]: one VLOOP over all vTGs, if / else by range; call uses {v} for the local vTG"""
    total = sum(n for _, n in segs)
    out, lo = [f"{indent}VLOOP({total}) {{"], 0
    for i, (call, n) in enumerate(segs):
        cond = "if" if i == 0 else "else if"
        lim = f"(vt < {lo + n}u)" if i < len(segs) - 1 else ""
        if i < len(segs) - 1:
            out.append(f"{indent}  {cond} {lim} {{ uint v = vt - {lo}u; {call} }}")
        else:
            out.append(f"{indent}  {'else ' if i else ''}{{ uint v = vt - {lo}u; {call} }}")
        lo += n
    out.append(f"{indent}}}")
    return "\n".join(out)


def q8(name, x, y, K, N, fn="QMVF"):
    impl = "qmv_impl" if fn == "qmv" else "qmv_fast_impl"
    return (f"{impl}<bfloat16_t, 64, 8>(dw + W_{name}, ds + S_{name}, db + S_{name}, {x}, {y}, {K}, {N}, "
            f"uint3(0, v, 0), sg & 1u, lane);", N // 8)


def build_src():
    G = {}
    G["P1"] = gemv_dispatch([q8("fc_emb", "mine", "scr + O_EN", 2560, 2560),
                             ("qmv_fast4_impl<bfloat16_t, 64, 8>(dw + W_fc_hid, ds + S_fc_hid, db + S_fc_hid, mine + 2560u, "
                              "scr + O_FH, 2560, 2560, uint3(0, v, 0), sg & 1u, lane);", 320)])
    G["P2"] = gemv_dispatch([q8("ahc_dn", "mine", "scr + O_HDN0", 10240, 320)])
    G["P3"] = gemv_dispatch([q8("ahc_up", "mine + 10240u", "scr + O_HUP0", 320, 10240, "qmv")])
    G["P4"] = gemv_dispatch([q8("q", "mine + 12288u", "scr + O_Q", 2560, 12288), q8("k", "mine + 12288u", "scr + O_K", 2560, 512),
                             q8("v", "mine + 12288u", "scr + O_V", 2560, 512)])
    G["P7"] = gemv_dispatch([q8("o", "scr + O_O", "scr + O_BLK", 6144, 2560)])
    G["P8"] = gemv_dispatch([q8("mhc_dn", "mine", "scr + O_HDN1", 10240, 320)])
    G["P9"] = gemv_dispatch([q8("mhc_up", "mine + 10240u", "scr + O_HUP1", 320, 10240, "qmv")])
    G["P10"] = gemv_dispatch([q8("router", "mine + 12288u", "scr + O_RL", 2560, 512), q8("se_gate", "mine + 12288u", "scr + O_SEG", 2560, 640),
                              q8("se_up", "mine + 12288u", "scr + O_SEU", 2560, 640)])
    G["P11"] = gemv_dispatch([q8("se_down", "mine + 13000u", "scr + O_SY", 640, 2560, "qmv")])
    G["P14"] = gemv_dispatch([q8("xhc_dn", "mine", "scr + O_HDN2", 10240, 320)])
    G["P15"] = gemv_dispatch([q8("xhc_up", "mine + 10240u", "scr + O_HUP2", 320, 10240, "qmv")])
    return SRC_T.replace("{P1}", G["P1"]).replace("{P2}", G["P2"]).replace("{P3}", G["P3"]).replace("{P4}", G["P4"]) \
        .replace("{P7}", G["P7"]).replace("{P8}", G["P8"]).replace("{P9}", G["P9"]).replace("{P10}", G["P10"]) \
        .replace("{P11}", G["P11"]).replace("{P14}", G["P14"]).replace("{P15}", G["P15"])


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
  threadgroup float qn[NHEAD_C * 256];
  threadgroup float stv[128];
  threadgroup int sti[128];
  uint nb = 0;
  bool ok = true;
  const uint J = min(t + 1u, 16u), L = S1 + J;
#define PH(k) (WORK && ((PHMASK >> (k)) & 1u))
#define BAR() { nb += 1u; if (ok) ok = gbar(ctr, (base + nb) * NTG, tid, tflag, SPIN); }
  const uint par = (t == 0u) ? 0u : uint(fst[0]), tok = (t == 0u) ? 1u : uint(fst[1]) % 1024u;

  // P1 fuse: normed emb row + the parent's 4 normed streams (every TG, its own slot); fc_embedding + fc_hidden (Q8)
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
  // P5 scores: one simdgroup per (kv group, key), 12 heads; the node's own k from scratch
  if (ok && PH(5)) {
    if (tg == 0u) for (uint i = tid; i < 1024u; i += NT) tkw[t * 1024u + i] = (i < 512u) ? scr[O_K + i] : scr[O_V + i - 512u];
    for (uint i = tid; i < NHEAD_C * 256u; i += NT) qn[i] = float(scr[O_Q + (i / 256u) * 512u + (i % 256u)]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = tg * NSG + sg; w < 2u * L; w += NTG * NSG) {
      uint g = w / L, j = w % L;
      const device bfloat16_t* kr = (j < S1) ? (k1 + (g * S1 + j) * 256u)
                                    : ((j + 1u == L) ? (scr + O_K + g * 256u) : (tkv + (j - S1) * 1024u + g * 256u));
      float kv[8];
      for (uint e = 0; e < 8u; ++e) kv[e] = float(kr[lane * 8u + e]);
      for (uint hh = 0; hh < 12u; ++hh) {
        uint h = g * 12u + hh;
        float a = 0.0f;
        for (uint e = 0; e < 8u; ++e) a += qn[h * 256u + lane * 8u + e] * kv[e];
        a = simd_sum(a);
        if (lane == 0u) scf[O_SCf + h * 2048u + j] = a * 0.0625f;
      }
    }
  }
  BAR();
  // P6 softmax + PV: (head, 32-dim block) tasks; keys over simdgroups; 8 V loads in flight per lane
  if (ok && PH(6)) {
    threadgroup float* part = qn;
    for (uint task = tg; task < 192u; task += NTG) {
      uint h = task / 8u, d0 = (task % 8u) * 32u, g = h / 12u;
      const device float* sc = scf + O_SCf + h * 2048u;
      float m = -INFINITY;
      for (uint j = tid * 4u; j < L; j += NT * 4u) { float4 v4 = *(const device float4*)(sc + j); for (uint u = 0; u < 4u; ++u) if (j + u < L) m = max(m, v4[u]); }
      m = simd_max(m);
      if (lane == 0u) red[sg] = m;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      m = red[0];
      for (uint q = 1; q < NSG; ++q) m = max(m, red[q]);
      float acc = 0.0f, ssum = 0.0f;
      for (uint c = sg; c * 32u < L; c += NSG) {
        uint j = c * 32u + lane;
        float pj = (j < L) ? metal::exp(sc[j] - m) : 0.0f;
        ssum += pj;
        uint nk = min(32u, L - c * 32u);
        for (uint k0 = 0; k0 < nk; k0 += 8u) {
          float vv[8];
          for (uint u = 0; u < 8u; ++u) {
            uint jj = c * 32u + k0 + u;
            vv[u] = 0.0f;
            if (k0 + u < nk) {
              const device bfloat16_t* vr = (jj < S1) ? (v1 + (g * S1 + jj) * 256u)
                                            : ((jj + 1u == L) ? (scr + O_V + g * 256u) : (tkv + (jj - S1) * 1024u + 512u + g * 256u));
              vv[u] = float(vr[d0 + lane]);
            }
          }
          for (uint u = 0; u < 8u; ++u) acc += simd_shuffle(pj, k0 + u) * vv[u];
        }
      }
      ssum = simd_sum(ssum);
      part[sg * 32u + lane] = acc;
      if (lane == 0u) red[32u + sg] = ssum;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {
        float o = 0.0f, st = 0.0f;
        for (uint q = 0; q < NSG; ++q) { o += part[q * 32u + lane]; st += red[32u + q]; }
        float gt = float(scr[O_Q + h * 512u + 256u + d0 + lane]);
        scr[O_O + h * 256u + d0 + lane] = bfloat16_t(o / st / (1.0f + metal::exp(-gt)));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  BAR();
  // P7 o_proj
  if (ok && PH(7)) {
{P7}
  }
  BAR();
  // P8 MLP HC down: hyper' = x4 + blk (TG 0 keeps it)
  if (ok && PH(8)) {
    prep4(scr + O_FH, scr + O_EN, scr + O_BLK, 3u, tg == 0u ? scr + O_HYP : (device bfloat16_t*)0, mine, tid, NT, red);
{P8}
  }
  BAR();
  // P9 MLP HC up
  if (ok && PH(9)) {
    silu320(scr + O_HDN1, mine + 10240u, tid, NT);
{P9}
  }
  BAR();
  // P10 mixed2 -> router | shared gate / up
  if (ok && PH(10)) {
    mix4(scr + O_HUP1, mine, mine + 12288u, tid, NT);
    if (tg == 0u) for (uint c = tid; c < 2560u; c += NT) scr[O_MX2 + c] = mine[12288u + c];
{P10}
  }
  BAR();
  // P11 route (TG 0, simdgroup 0: fp32 softmax over 512, top-10) | swiglu + shared down
  if (ok && PH(11)) {
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
{P11}
  }
  BAR();
  // P12 experts gate / up (10 x 4-bit g32)
  if (ok && PH(12)) {
    VLOOP(1600) {
      uint e = vt / 160u, r = vt % 160u;
      uint ex = (uint(((device int*)(scf + O_TOPf))[e]) + e * 13u) % uint(NE_C);
      uint gu = r / 80u;
      qmv_fast_impl<bfloat16_t, 32, 4>(ew + size_t(ex) * EW_PER + gu * 204800u, es + size_t(ex) * ES_PER + gu * 51200u,
                                       eb + size_t(ex) * ES_PER + gu * 51200u, scr + O_MX2, scr + O_EGU + e * 1280u + gu * 640u,
                                       2560, 640, uint3(0, r % 80u, 0), sg & 1u, lane);
    }
  }
  BAR();
  // P13 experts down (10 x 4-bit g32, qmv)
  if (ok && PH(13)) {
    for (uint i = tid; i < 6400u; i += NT) {
      uint e = i / 640u, c = i % 640u;
      float gg = float(scr[O_EGU + e * 1280u + c]), uu = float(scr[O_EGU + e * 1280u + 640u + c]);
      mine[i] = bfloat16_t(gg / (1.0f + metal::exp(-gg)) * uu);
    }
    threadgroup_barrier(mem_flags::mem_device);
    VLOOP(3200) {
      uint e = vt % 10u, cb = vt / 10u;
      uint ex = (uint(((device int*)(scf + O_TOPf))[e]) + e * 13u) % uint(NE_C);
      qmv_impl<bfloat16_t, 32, 4>(ew + size_t(ex) * EW_PER + 409600u, es + size_t(ex) * ES_PER + 102400u,
                                  eb + size_t(ex) * ES_PER + 102400u, mine + e * 640u, scr + O_EY + e * 2560u, 640, 2560,
                                  uint3(0, cb, 0), sg & 1u, lane);
    }
  }
  BAR();
  // P14 h = sy + sum_e sc_e y_e; hyper'' = hyper' + h (TG 0 writes the node state); mixer HC down
  if (ok && PH(14)) {
    for (uint q = tid; q < 640u; q += NT) {
      uint c0 = q * 4u;
      float4 h = float4(*(const device vec<bfloat16_t, 4>*)(scr + O_SY + c0));
      for (uint e = 0; e < 10u; ++e) h += scf[O_TOPf + 16u + e] * float4(*(const device vec<bfloat16_t, 4>*)(scr + O_EY + e * 2560u + c0));
      *(device vec<bfloat16_t, 4>*)(mine + 12288u + c0) = vec<bfloat16_t, 4>(h);
    }
    threadgroup_barrier(mem_flags::mem_device);
    prep4(scr + O_HYP, mine + 12288u, scr, 2u, tg == 0u ? hsw + (t + 1u) * 10240u : (device bfloat16_t*)0, mine, tid, NT, red);
{P14}
  }
  BAR();
  // P15 mixer HC up
  if (ok && PH(15)) {
    silu320(scr + O_HDN2, mine + 10240u, tid, NT);
{P15}
  }
  BAR();
  // P16 x_head -> screened head (4-bit g64)
  if (ok && PH(16)) {
    mix4(scr + O_HUP2, mine, mine + 12288u, tid, NT);
    VLOOP(NH_C / 8) qmv_fast_impl<bfloat16_t, 64, 4>(hw, hs, hb, mine + 12288u, scr + O_LG, 2560, NH_C, uint3(0, vt, 0), sg & 1u, lane);
  }
  BAR();
  // P17 tail slices (every TG): partial (max, sum exp) and top-8 of its slice of the logits
  if (ok && PH(17)) {
    const uint per = (NH_C + NTG - 1u) / NTG, lo = tg * per, hi = min(NH_C, lo + per);
    float m = -INFINITY;
    float bv[8]; int bi[8];
    for (uint r = 0; r < 8u; ++r) { bv[r] = -INFINITY; bi[r] = 0x7FFFFFFF; }
    float xv[4]; uint nx = 0;
    for (uint i = lo + tid; i < hi; i += NT) {
      float x = float(scr[O_LG + i]);
      if (nx < 4u) xv[nx++] = x;
      m = max(m, x);
      if (x > bv[7]) { uint p = 7; while (p > 0u && x > bv[p - 1u]) { bv[p] = bv[p - 1u]; bi[p] = bi[p - 1u]; --p; } bv[p] = x; bi[p] = int(i); }
    }
    float mm = simd_max(m);
    if (lane == 0u) red[sg] = mm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = red[0];
    for (uint q = 1; q < NSG; ++q) M = max(M, red[q]);
    float s = 0.0f;
    for (uint u = 0; u < nx; ++u) s += metal::exp(xv[u] - M);
    s = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) red[32u + sg] = s;
    uint ptr = 0;
    for (uint r = 0; r < 8u; ++r) {
      float v = -INFINITY; int ii = 0x7FFFFFFF;
      for (uint q = 0; q < 8u; ++q) if (q == ptr) { v = bv[q]; ii = bi[q]; }
      float mv = simd_max(v);
      int mi = simd_min(v == mv ? ii : 0x7FFFFFFF);
      if (v == mv && ii == mi) ptr += 1u;
      if (lane == 0u) { stv[sg * 8u + r] = mv; sti[sg * 8u + r] = mi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float S = 0.0f;
      for (uint q = 0; q < NSG; ++q) S += red[32u + q];
      float fv[8]; int fi[8];
      for (uint r = 0; r < 8u; ++r) { fv[r] = -INFINITY; fi[r] = 0x7FFFFFFF; }
      for (uint q = 0; q < NSG * 8u; ++q) {
        float x = stv[q]; int xi = sti[q];
        if (x > fv[7] || (x == fv[7] && xi < fi[7])) { uint p = 7; while (p > 0u && (x > fv[p - 1u] || (x == fv[p - 1u] && xi < fi[p - 1u]))) { fv[p] = fv[p - 1u]; fi[p] = fi[p - 1u]; --p; } fv[p] = x; fi[p] = xi; }
      }
      scf[O_TLf + tg * 20u] = M; scf[O_TLf + tg * 20u + 1u] = S;
      for (uint r = 0; r < 8u; ++r) { scf[O_TLf + tg * 20u + 2u + r] = fv[r]; ((device int*)(scf + O_TLf))[tg * 20u + 10u + r] = fi[r]; }
    }
  }
  BAR();
  // P18 merge (TG 0): lse, top-8, probabilities; a skeleton frontier
  if (tg == 0u && tid == 0u) {
    float M = -INFINITY, S = 0.0f;
    float fv[8]; int fi[8];
    for (uint r = 0; r < 8u; ++r) { fv[r] = -INFINITY; fi[r] = 0x7FFFFFFF; }
    if (ok && PH(17)) {
      for (uint q = 0; q < NTG; ++q) M = max(M, scf[O_TLf + q * 20u]);
      for (uint q = 0; q < NTG; ++q) S += scf[O_TLf + q * 20u + 1u] * metal::exp(scf[O_TLf + q * 20u] - M);
      for (uint q = 0; q < NTG * 8u; ++q) {
        float x = scf[O_TLf + (q / 8u) * 20u + 2u + q % 8u]; int xi = ((device int*)(scf + O_TLf))[(q / 8u) * 20u + 10u + q % 8u];
        if (x > fv[7] || (x == fv[7] && xi < fi[7])) { uint p = 7; while (p > 0u && (x > fv[p - 1u] || (x == fv[p - 1u] && xi < fi[p - 1u]))) { fv[p] = fv[p - 1u]; fi[p] = fi[p - 1u]; --p; } fv[p] = x; fi[p] = xi; }
      }
    }
    float lse = M + metal::precise::log(S);
    for (uint r = 0; r < 8u; ++r) { out_tis[r] = fi[r]; out_tp[r] = metal::precise::exp(fv[r] - lse); }
    out_pop[0] = int(t); out_pop[1] = fi[0]; out_pop[2] = int(J); out_pop[3] = ok ? 0 : 1; out_pop[4] = int(nb);
    fst_out[0] = int(t + 1u); fst_out[1] = fi[0] & 0x3FF; fst_out[2] = ok ? 0 : 1;
    for (uint r = 3; r < 8u; ++r) fst_out[r] = 0;
  }
"""

STREAM = r"""
  const uint gid = thread_position_in_grid.x, gsz = threads_per_grid.x;
  uint4 acc = uint4(0);
  const device uint4* a = (const device uint4*)dw;
  for (uint i = gid; i < NDW4; i += gsz) acc ^= a[i];
  const device uint4* b = (const device uint4*)ds;
  for (uint i = gid; i < NDS4; i += gsz) acc ^= b[i];
  const device uint4* c = (const device uint4*)db;
  for (uint i = gid; i < NDS4; i += gsz) acc ^= c[i];
  for (uint e = 0; e < 10u; ++e) {
    const device uint4* x = (const device uint4*)(ew + size_t(e * 7u) * EW_PER);
    for (uint i = gid; i < EW_PER / 4u; i += gsz) acc ^= x[i];
    const device uint4* y = (const device uint4*)(es + size_t(e * 7u) * ES_PER);
    for (uint i = gid; i < ES_PER / 8u; i += gsz) acc ^= y[i];
    const device uint4* z = (const device uint4*)(eb + size_t(e * 7u) * ES_PER);
    for (uint i = gid; i < ES_PER / 8u; i += gsz) acc ^= z[i];
  }
  const device uint4* h = (const device uint4*)hw;
  for (uint i = gid; i < NHW4; i += gsz) acc ^= h[i];
  const device uint4* hs4 = (const device uint4*)hs;
  for (uint i = gid; i < NHS4; i += gsz) acc ^= hs4[i];
  const device uint4* hb4 = (const device uint4*)hb;
  for (uint i = gid; i < NHS4; i += gsz) acc ^= hb4[i];
  uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (r == 0x9e3779b9u) out[gid & 255u] = int(r);
"""


def offsets():
    W, S, w, s = {}, {}, 0, 0
    for name, n, k in DENSE:
        W[name], S[name] = w, s
        w += n * k // 4
        s += n * k // 64
    return W, S, w, s


def scratch_layout(ntg):
    o, cur = {}, 0
    for name, n in (("EN", 2560), ("FH", 10240), ("HDN0", 512), ("HUP0", 10240), ("Q", 12288), ("K", 512), ("V", 512),
                    ("O", 6144), ("BLK", 2560), ("HYP", 10240), ("HDN1", 512), ("HUP1", 10240), ("MX2", 2560),
                    ("RL", 512), ("SEG", 640), ("SEU", 640), ("SY", 2560), ("EGU", 12800), ("EY", 25600),
                    ("HDN2", 512), ("HUP2", 10240), ("LG", 37120)):
        o[name] = cur
        cur += (n + 255) // 256 * 256
    o["SCf"] = (cur + 1) // 2
    cur = (o["SCf"] + 24 * 2048) * 2
    o["TOPf"] = cur // 2
    cur = (o["TOPf"] + 64) * 2
    o["TLf"] = cur // 2
    cur = (o["TLf"] + ntg * 20 + 64) * 2
    o["PRIV"] = (cur + 255) // 256 * 256
    return o, o["PRIV"] + ntg * 16384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntg", type=int, default=40)
    ap.add_argument("--nt", type=int, default=512)
    ap.add_argument("--s1", type=int, default=1024)
    ap.add_argument("--nh", type=int, default=36864)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--trees", type=int, default=4)
    ap.add_argument("--spin", type=int, default=400000)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    NTG, NT, S1, NH = a.ntg, a.nt, a.s1, a.nh
    W, S, nw32, ns16 = offsets()
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
    print(f"[pk_skel2] NTG {NTG} x {NT}, S1 {S1}, NH {NH}; bytes per pop {bytes_pop / 1e6:.1f} MB; floor at 292.7 GB/s "
          f"{floor:.3f} ms", flush=True)
    defs = "".join(f"#define W_{n} {W[n]}u\n#define S_{n} {S[n]}u\n" for n, _, _ in DENSE)
    defs += "".join(f"#define O_{k} {v}u\n" for k, v in sc.items())
    defs += f"#define EW_PER {EW_PER}u\n#define ES_PER {ES_PER}u\n#define NSG {NT // 32}u\n#define VPP {NT // 64}u\n"
    defs += f"#define NHEAD_C 24u\n#define NE_C {NE}\n#define NH_C {NH}u\n"
    SRC = build_src()

    def kern(work, mask=(1 << 19) - 2):
        src = defs + f"#define PHMASK {mask}u\n#define WORK {int(work)}\n#define NTG {NTG}u\n#define NT {NT}u\n#define SPIN {a.spin}u\n" + SRC
        return mx.fast.metal_kernel(
            name=f"pk_skel2_w{int(work)}_m{mask}_{NTG}_{NT}_{S1}_{NH}",
            input_names=["dw", "ds", "db", "ew", "es", "eb", "hw", "hs", "hb", "embt", "k1", "v1", "hsbuf", "tkv",
                         "scratch", "syncb", "fst", "meta"],
            output_names=["out_pop", "out_tis", "out_tp", "fst_out"], source=src, header=HDR)

    KW, KB = kern(1), kern(0)
    ks = mx.fast.metal_kernel(name="pk_stream2", input_names=["dw", "ds", "db", "ew", "es", "eb", "hw", "hs", "hb"],
                              output_names=["out"], header="#define NDW4 %du\n#define NDS4 %du\n#define EW_PER %du\n"
                              "#define ES_PER %du\n#define NHW4 %du\n#define NHS4 %du\n" % (
                                  dw.size // 4, ds.size // 8, EW_PER, ES_PER, hw.size // 4, hs.size // 8), source=STREAM)
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
    print(f"[pk_skel2] smoke: barrier fails {fails} of {len(pops)} pops; barriers counted {pops[-1][4]}; "
          f"top-8 ids {outs[1].tolist()[:4]}", flush=True)
    if a.ablate:
        tb = sorted(timed(lambda: tree(KB), 2)[0] / 30 * 1e3 for _ in range(5))[2]
        prev, rows = tb, []
        for k in range(1, 18):
            Kk = kern(1, sum(1 << j for j in range(1, k + 1)))
            timed(lambda: tree(Kk), 1)
            tk = sorted(timed(lambda: tree(Kk), 2)[0] / 30 * 1e3 for _ in range(5))[2]
            rows.append(dict(phase=k, us=1e3 * (tk - prev), cum_ms=tk))
            print(f"[pk_skel2] + P{k:2d}: {1e3 * (tk - prev):7.1f} us (phases 1..{k}: {tk:.3f} ms)", flush=True)
            prev = tk
        print(f"[pk_skel2] barriers only {tb:.3f} ms", flush=True)
        if a.out:
            json.dump(dict(barriers_ms=tb, rows=rows), open(a.out, "w"), indent=1)
        return
    res = dict(ntg=NTG, nt=NT, s1=S1, nh=NH, bytes_pop=bytes_pop, floor_ms=floor, fails_smoke=fails, skel=[], bar=[], stream=[])
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
    res["f_stream"] = res["skel_med"] - res["stream_med"]
    print(f"[pk_skel2] ms per pop (median of {a.rounds - 2} rounds, {a.trees} trees x 15 pops): skel {res['skel_med']:.3f}, "
          f"barriers only {res['bar_med']:.3f}, stream {res['stream_med']:.3f} ({bytes_pop / res['stream_med'] / 1e6:.1f} GB/s); "
          f"floor {floor:.3f}", flush=True)
    print(f"[pk_skel2] f = skel - floor = {res['f_line']:.3f} ms (M-46 skeleton line: GO <= 0.15, KILL >= 0.25); "
          f"skel - stream = {res['f_stream']:.3f} ms; barrier fails {fails}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
