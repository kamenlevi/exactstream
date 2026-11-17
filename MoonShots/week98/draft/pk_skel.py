"""PK SKELETON (MoonShots, Tue 17 Nov 2026; PK_PLAN.md, QUEUE M-46): GRANDMA lever 2's decisive test. One persistent
kernel per pop of the Q8H32 head, on synthetic weights of the EXACT real shapes, the real phase chain, and grid
barriers. Pre-registered (M-46): skeleton f >= 0.25 ms -> KILL the line; f <= 0.15 -> GO the bitwise build.

Phases (16 grid barriers per pop):
 P1  fuse: fc_embedding (Q8 2560x2560) + fc_hidden (Q8, 4 streams)
 P2  attention HC down (Q8 320x10240)          P3  HC up (Q8 10240x320)
 P4  q / k / v (Q8 12288 / 512 / 512 x 2560)
 P5  attention scores                         P6  softmax + PV (x sigmoid(gate))
 P7  o_proj (Q8 2560x6144)                    P8-P9  MLP HC down / up
 P10 router (Q8 512) | shared gate / up (Q8 640)
 P11 route (fp32 softmax, top-10; TG 0) | shared down (Q8 2560x640)
 P12 experts gate / up (10 x 4-bit g32 640x2560 x 2)
 P13 experts down (10 x 4-bit g32 2560x640)
 P14-P15 mixer HC down / up                   P16 screened head (4-bit g64, NH x 2560)
 P17 tail (TG 0): logsumexp, top-8, exp, a skeleton frontier
Every quantized GEMV is MLX's own qmv_fast_impl / qmv_impl (quantized.h as shipped), run as "virtual threadgroups",
so the per-output arithmetic is the stock's. Norms, attention, routing and the tail are simplified: they move the
real bytes, but their math is not bitwise yet.
Read-outs (ms per pop over 15-pop trees, one sync per tree):
  skel    the persistent pop kernel
  bar     the same kernel with the work skipped (WORK 0): the barriers and the launch only
  stream  one kernel streaming the same bytes, no barriers: the achievable byte floor
f = skel - bytes / 292.7 GB/s (the line), and also skel - stream (the floor as streamed here).
usage: ~/BigMoE/.venv/bin/python -I -B pk_skel.py [--ntg 40] [--nt 512] [--s1 1024] [--rounds 12] [--out <json>]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/bet7"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/draft"))
import mlx.core as mx                                      # noqa: E402
from fl_qmatch import QHEADER                              # noqa: E402  (quantized.h's helpers + qmv impls, as shipped)
import pk_qpf                                              # noqa: E402  (qmv_fast_impl + a prefetch, same arithmetic)

D, D4, NHEAD, HD = 2560, 10240, 24, 256
DENSE = [  # name, N (out), K (in): Q8 g64, the t30b_q8h32.safetensors shapes
    ("fc_emb", 2560, 2560), ("fc_hid", 2560, 2560), ("ahc_dn", 320, 10240), ("ahc_up", 10240, 320),
    ("q", 12288, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 6144),
    ("mhc_dn", 320, 10240), ("mhc_up", 10240, 320), ("router", 512, 2560), ("se_gate", 640, 2560),
    ("se_up", 640, 2560), ("se_down", 2560, 640), ("xhc_dn", 320, 10240), ("xhc_up", 10240, 320)]
NE, TOPE, FF = 128, 10, 640         # synthetic experts (the real head has 512; 128 x 3.07 MB keeps them out of the SLC)
NBAR = 16

HDR = r"""
struct QM { const device uint32_t* w; const device bfloat16_t* s; const device bfloat16_t* b; const device bfloat16_t* x;
            device bfloat16_t* y; uint K; uint N; uint kind; };       // kind 0 q8g64 fast, 1 q8g64 qmv, 2 q4g32 fast, 3 q4g32 qmv, 4 q4g64 fast
METAL_FUNC uint qm_rb(QM m) { return (m.kind <= 1u) ? m.K : m.K / 2u; }                 // weight bytes per row
METAL_FUNC uint qm_gb(QM m) { return (m.K / ((m.kind == 2u || m.kind == 3u) ? 32u : 64u)) * 2u; }   // scale bytes per row
METAL_FUNC void qm_call(QM m, uint v, uint sgp, uint lane) {
  uint3 t3 = uint3(0, v, 0);
  if (m.kind == 0u) QMVF<bfloat16_t, 64, 8>(m.w, m.s, m.b, m.x, m.y, int(m.K), int(m.N), t3, sgp, lane);
  else if (m.kind == 1u) qmv_impl<bfloat16_t, 64, 8>(m.w, m.s, m.b, m.x, m.y, int(m.K), int(m.N), t3, sgp, lane);
  else if (m.kind == 2u) QMVF<bfloat16_t, 32, 4>(m.w, m.s, m.b, m.x, m.y, int(m.K), int(m.N), t3, sgp, lane);
  else if (m.kind == 3u) qmv_impl<bfloat16_t, 32, 4>(m.w, m.s, m.b, m.x, m.y, int(m.K), int(m.N), t3, sgp, lane);
  else QMVF<bfloat16_t, 64, 4>(m.w, m.s, m.b, m.x, m.y, int(m.K), int(m.N), t3, sgp, lane);
}
// one 4-byte load per 64 bytes of the rows of vTGs [v0, v0 + n) of a segment list (weights, scales, biases)
METAL_FUNC uint pf_range(thread const QM* sg_, uint nseg, uint v0, uint n, uint hid, uint nh) {
  uint acc = 0u;
  for (uint q = 0; q < n; ++q) {
    uint v = v0 + q, i = 0;
    while (i < nseg && v >= sg_[i].N / 8u) { v -= sg_[i].N / 8u; ++i; }
    if (i >= nseg) break;
    QM m = sg_[i];
    uint rb = qm_rb(m), gb = qm_gb(m);
    const device uint* wp = (const device uint*)((const device uint8_t*)m.w + size_t(v) * 8u * rb);
    for (uint j = hid; j < 8u * rb / 64u; j += nh) acc ^= wp[j * 16u];
    const device uint* sp = (const device uint*)((const device uint8_t*)m.s + size_t(v) * 8u * gb);
    const device uint* bp = (const device uint*)((const device uint8_t*)m.b + size_t(v) * 8u * gb);
    for (uint j = hid; j < (8u * gb + 63u) / 64u; j += nh) acc ^= sp[j * 16u] ^ bp[j * 16u];
  }
  return acc;
}
// the phase: consumers (simdgroups < NCON, in pairs = stock virtual threadgroups) run round r while helpers load
// round r + 1 of this threadgroup's own vTGs (or the next phase's round 0); one threadgroup barrier per round
METAL_FUNC void run_phase(thread const QM* sg_, uint nseg, thread const QM* nx, uint nnx, uint tg, uint sg, uint lane,
                          uint tid, thread uint& pfacc) {
  uint total = 0;
  for (uint i = 0; i < nseg; ++i) total += sg_[i].N / 8u;
  uint rounds = (total + SLOTS - 1u) / SLOTS;
  for (uint r = 0; r < rounds; ++r) {
    if (sg < NCON) {
      uint v = r * SLOTS + tg * VPPC + (sg >> 1), i = 0;
      if (v < total) { while (v >= sg_[i].N / 8u) { v -= sg_[i].N / 8u; ++i; } qm_call(sg_[i], v, sg & 1u, lane); }
    } else if (HELP) {
      uint hid = tid - NCON * 32u, nh = NT_C - NCON * 32u;
      if (r + 1u < rounds) pfacc ^= pf_range(sg_, nseg, (r + 1u) * SLOTS + tg * VPPC, VPPC, hid, nh);
      else if (nnx) pfacc ^= pf_range(nx, nnx, tg * VPPC, VPPC, hid, nh);
    }
    threadgroup_barrier(mem_flags::mem_none);
  }
}

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

// 4 x 2560: x = a + b (+ c) with bf16 rounding per add; optional raw copy; normed per stream -> out (device)
METAL_FUNC void prep4(const device bfloat16_t* a, const device bfloat16_t* b, const device bfloat16_t* c, uint nadd,
                      device bfloat16_t* raw, device bfloat16_t* out, uint tid, uint nt, threadgroup float* red) {
  float4 acc = 0.0f;
  for (uint q = tid; q < 2560u; q += nt) {                    // q = 4-column chunk over the 4 x 2560 values
    uint i = q * 4u, c0 = i % 2560u, r = i / 2560u;
    float4 x = float4(*(const device vec<bfloat16_t, 4>*)(a + i));
    if (nadd > 1u) x = float4(vec<bfloat16_t, 4>(x + float4(*(const device vec<bfloat16_t, 4>*)(b + c0))));
    if (nadd > 2u) x = float4(vec<bfloat16_t, 4>(x + float4(*(const device vec<bfloat16_t, 4>*)(c + c0))));
    if (raw) *(device vec<bfloat16_t, 4>*)(raw + i) = vec<bfloat16_t, 4>(x);
    *(device vec<bfloat16_t, 4>*)(out + i) = vec<bfloat16_t, 4>(x);
    float sq = dot(x, x);
    acc[r] += sq;
  }
  for (uint r = 0; r < 4u; ++r) { float v = simd_sum(acc[r]); if ((tid & 31u) == 0u) red[r * 8u + (tid >> 5) % 8u] = 0.0f; acc[r] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* part = red;                              // 4 rows x up to 16 simdgroups -> uses red[0..63]
  if ((tid & 31u) == 0u) for (uint r = 0; r < 4u; ++r) part[r * 16u + (tid >> 5)] = acc[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4u) { float sacc = 0.0f; for (uint q = 0; q < nt / 32u; ++q) sacc += part[tid * 16u + q]; part[64u + tid] = metal::precise::rsqrt(sacc / 2560.0f + 1e-6f); }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint q = tid; q < 2560u; q += nt) {
    uint i = q * 4u, r = i / 2560u;
    float4 x = float4(*(const device vec<bfloat16_t, 4>*)(out + i)) * part[64u + r];
    *(device vec<bfloat16_t, 4>*)(out + i) = vec<bfloat16_t, 4>(x);
  }
  threadgroup_barrier(mem_flags::mem_device);
}

// rows x n bf16 (device) -> normed bf16 (device, this threadgroup's slot); fp32 sum of squares per row (skeleton order)
METAL_FUNC void rms_rows(const device bfloat16_t* x, device bfloat16_t* out, uint rows, uint n, uint tid, uint nt,
                         threadgroup float* red) {
  for (uint r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (uint i = tid; i < n; i += nt) { float v = float(x[r * n + i]); acc += v * v; }
    acc = simd_sum(acc);
    if ((tid & 31u) == 0u) red[tid >> 5] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) { float a = (tid < nt / 32u) ? red[tid] : 0.0f; a = simd_sum(a); if (tid == 0u) red[32] = a; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(red[32] / float(n) + 1e-6f);
    for (uint i = tid; i < n; i += nt) out[r * n + i] = bfloat16_t(float(x[r * n + i]) * inv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
"""

SRC = r"""
  const uint tg = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint t = uint(meta[0]), base = uint(meta[1]), S1 = uint(meta[2]);
  device atomic_uint* ctr = (device atomic_uint*)syncb;
  device bfloat16_t* scr = (device bfloat16_t*)scratch;
  device float* scf = (device float*)scratch;
  device bfloat16_t* mine = scr + O_PRIV + tg * 16384u;          // this threadgroup's private input slots
  threadgroup uint tflag;
  threadgroup float red[72];
  threadgroup float qn[NHEAD_C * 256];
  threadgroup float stv[128];
  threadgroup int sti[128];
  device bfloat16_t* hsw = (device bfloat16_t*)hsbuf;
  device bfloat16_t* tkw = (device bfloat16_t*)tkv;
  QM TP1[5] = {QM{dw + W_fc_emb, ds + S_fc_emb, db + S_fc_emb, mine, scr + O_EN, 2560u, 2560u, 0u}, QM{dw + W_fc_hid, ds + S_fc_hid, db + S_fc_hid, mine + 2560u, scr + O_FH + 0u, 2560u, 2560u, 0u}, QM{dw + W_fc_hid, ds + S_fc_hid, db + S_fc_hid, mine + 5120u, scr + O_FH + 2560u, 2560u, 2560u, 0u}, QM{dw + W_fc_hid, ds + S_fc_hid, db + S_fc_hid, mine + 7680u, scr + O_FH + 5120u, 2560u, 2560u, 0u}, QM{dw + W_fc_hid, ds + S_fc_hid, db + S_fc_hid, mine + 10240u, scr + O_FH + 7680u, 2560u, 2560u, 0u}};
  QM TP2[1] = {QM{dw + W_ahc_dn, ds + S_ahc_dn, db + S_ahc_dn, mine, scr + O_HDN0, 10240u, 320u, 0u}};
  QM TP3[1] = {QM{dw + W_ahc_up, ds + S_ahc_up, db + S_ahc_up, mine + 10240u, scr + O_HUP0, 320u, 10240u, 1u}};
  QM TP4[3] = {QM{dw + W_q, ds + S_q, db + S_q, mine + 12288u, scr + O_Q, 2560u, 12288u, 0u}, QM{dw + W_k, ds + S_k, db + S_k, mine + 12288u, scr + O_K, 2560u, 512u, 0u}, QM{dw + W_v, ds + S_v, db + S_v, mine + 12288u, scr + O_V, 2560u, 512u, 0u}};
  QM TP7[1] = {QM{dw + W_o, ds + S_o, db + S_o, scr + O_O, scr + O_BLK, 6144u, 2560u, 0u}};
  QM TP8[1] = {QM{dw + W_mhc_dn, ds + S_mhc_dn, db + S_mhc_dn, mine, scr + O_HDN1, 10240u, 320u, 0u}};
  QM TP9[1] = {QM{dw + W_mhc_up, ds + S_mhc_up, db + S_mhc_up, mine + 10240u, scr + O_HUP1, 320u, 10240u, 1u}};
  QM TP10[3] = {QM{dw + W_router, ds + S_router, db + S_router, mine + 12288u, scr + O_RL, 2560u, 512u, 0u}, QM{dw + W_se_gate, ds + S_se_gate, db + S_se_gate, mine + 12288u, scr + O_SEG, 2560u, 640u, 0u}, QM{dw + W_se_up, ds + S_se_up, db + S_se_up, mine + 12288u, scr + O_SEU, 2560u, 640u, 0u}};
  QM TP11[1] = {QM{dw + W_se_down, ds + S_se_down, db + S_se_down, mine + 13000u, scr + O_SY, 640u, 2560u, 1u}};
  QM TP14[1] = {QM{dw + W_xhc_dn, ds + S_xhc_dn, db + S_xhc_dn, mine, scr + O_HDN2, 10240u, 320u, 0u}};
  QM TP15[1] = {QM{dw + W_xhc_up, ds + S_xhc_up, db + S_xhc_up, mine + 10240u, scr + O_HUP2, 320u, 10240u, 1u}};
  QM T16[1] = {QM{hw, hs, hb, mine + 12288u, scr + O_LG, 2560u, NH_C, 4u}};
  QM T12[20];
  QM T13[10];
  uint pfacc = 0u;
  uint nb = 0;
  bool ok = true;
  const uint J = min(t + 1u, 16u), L = S1 + J;
#define PH(k) (WORK && ((PHMASK >> (k)) & 1u))
#define BAR() { nb += 1u; if (ok) ok = gbar(ctr, (base + nb) * NTG, tid, tflag, SPIN); }
  const uint par = (t == 0u) ? 0u : uint(fst[0]), tok = (t == 0u) ? 1u : uint(fst[1]) % 1024u;   // fst chains the trees too

  // ---------------- P1 fuse: rms(emb row), rms(parent's 4 streams); fc_embedding + fc_hidden (Q8) ----------------
  if (PH(1)) {
    rms_rows(embt + tok * 2560u, mine, 1u, 2560u, tid, NT, red);
    rms_rows(hsbuf + par * 10240u, mine + 2560u, 4u, 2560u, tid, NT, red);
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(TP1, 5u, TP2, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P2 attention HC down: x4 = fh + en; rms per stream; Q8 320 x 10240 ----------------
  if (ok && PH(2)) {
    prep4(scr + O_FH, scr + O_EN, scr, 2u, (device bfloat16_t*)0, mine, tid, NT, red);
    run_phase(TP2, 1u, TP3, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P3 HC up: silu(d / 4) -> Q8 10240 x 320 (qmv) ----------------
  if (ok && PH(3)) {
    for (uint i = tid; i < 320u; i += NT) { float d = float(scr[O_HDN0 + i]) * 0.25f; mine[10240u + i] = bfloat16_t(d / (1.0f + metal::exp(-d))); }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(TP3, 1u, TP4, 3u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P4 mixed = mean_s sigmoid(up) * normed; q / k / v (Q8) ----------------
  if (ok && PH(4)) {
    for (uint c = tid; c < 2560u; c += NT) {
      float a = 0.0f;
      for (uint s = 0; s < 4u; ++s) { float u = float(scr[O_HUP0 + s * 2560u + c]); a += float(mine[s * 2560u + c]) / (1.0f + metal::exp(-u)); }
      mine[12288u + c] = bfloat16_t(a * 0.25f);
    }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(TP4, 3u, TP7, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P5 attention scores: prefix K1 (S1 keys) + J tree columns; q normed per head ----------------
  if (ok && PH(5)) {
    if (tg == 0u) for (uint i = tid; i < 1024u; i += NT) tkw[t * 1024u + i] = (i < 512u) ? scr[O_K + i] : scr[O_V + i - 512u];
    for (uint i = tid; i < NHEAD_C * 256u; i += NT) qn[i] = float(scr[O_Q + (i / 256u) * 512u + (i % 256u)]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = tg * NSG + sg; w < 2u * L; w += NTG * NSG) {          // one simdgroup per (kv group, key)
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
  // ---------------- P6 softmax + PV for (kv group, 32-dim block), 12 heads at once; x sigmoid(gate) ----------------
  if (ok && PH(6)) {
    threadgroup float* part = qn;                                    // 16 x 32 partial sums (qn is free here)
    for (uint task = tg; task < 192u; task += NTG) {
      uint h = task / 8u, d0 = (task % 8u) * 32u, g = h / 12u;
      const device float* sc = scf + O_SCf + h * 2048u;
      float m = -INFINITY;
      for (uint j = tid; j < L; j += NT) m = max(m, sc[j]);
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
        for (uint kk = 0; kk < 32u; ++kk) {
          uint jj = c * 32u + kk;
          if (jj >= L) break;
          float pk = simd_shuffle(pj, kk);
          const device bfloat16_t* vr = (jj < S1) ? (v1 + (g * S1 + jj) * 256u)
                                        : ((jj + 1u == L) ? (scr + O_V + g * 256u) : (tkv + (jj - S1) * 1024u + 512u + g * 256u));
          acc += pk * float(vr[d0 + lane]);
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
  // ---------------- P7 o_proj (Q8 2560 x 6144) ----------------
  if (ok && PH(7)) run_phase(TP7, 1u, TP8, 1u, tg, sg, lane, tid, pfacc);
  BAR();
  // ---------------- P8 MLP HC down: hyper' = x4 + blk; rms per stream ----------------
  if (ok && PH(8)) {
    prep4(scr + O_FH, scr + O_EN, scr + O_BLK, 3u, tg == 0u ? scr + O_HYP : (device bfloat16_t*)0, mine, tid, NT, red);
    run_phase(TP8, 1u, TP9, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P9 MLP HC up ----------------
  if (ok && PH(9)) {
    for (uint i = tid; i < 320u; i += NT) { float d = float(scr[O_HDN1 + i]) * 0.25f; mine[10240u + i] = bfloat16_t(d / (1.0f + metal::exp(-d))); }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(TP9, 1u, TP10, 3u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P10 mixed2; router (Q8 512) | shared gate / up (Q8 640) ----------------
  if (ok && PH(10)) {
    for (uint c = tid; c < 2560u; c += NT) {
      float a = 0.0f;
      for (uint s = 0; s < 4u; ++s) { float u = float(scr[O_HUP1 + s * 2560u + c]); a += float(mine[s * 2560u + c]) / (1.0f + metal::exp(-u)); }
      mine[12288u + c] = bfloat16_t(a * 0.25f);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tg == 0u) for (uint c = tid; c < 2560u; c += NT) scr[O_MX2 + c] = mine[12288u + c];
    run_phase(TP10, 3u, TP11, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P11 route (TG 0: fp32 softmax over 512, top-10) | shared down (Q8 2560 x 640, qmv) ----------------
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
      for (uint r = 0; r < 10u; ++r) {                                 // 10 rounds of a simd argmax (value, then index)
        float bv = -1.0f; uint bi = 0;
        for (uint e = 0; e < 16u; ++e) if (v[e] > bv) { bv = v[e]; bi = lane * 16u + e; }
        float mv = simd_max(bv);
        uint mi = simd_min(bv == mv ? bi : 0xFFFFFFFFu);
        if (lane == 0u) { ((device int*)(scf + O_TOPf))[r] = int(mi % uint(NE_C)); scf[O_TOPf + 16u + r] = mv; }
        if (mi / 16u == lane) v[mi % 16u] = -2.0f;
      }
    }
    for (uint i = tid; i < 640u; i += NT) {
      float gg = float(scr[O_SEG + i]), uu = float(scr[O_SEU + i]);
      mine[13000u + i] = bfloat16_t(gg / (1.0f + metal::exp(-gg)) * uu);
    }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(TP11, 1u, TP11, 0u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P12 experts gate / up (10 x 4-bit g32, 640 x 2560 each) ----------------
  if (ok && PH(12)) {
    for (uint e = 0; e < 10u; ++e) {
      uint ex = (uint(((device int*)(scf + O_TOPf))[e]) + e * 13u) % uint(NE_C);
      for (uint gu = 0; gu < 2u; ++gu)
        T12[e * 2u + gu] = QM{ew + size_t(ex) * EW_PER + gu * 204800u, es + size_t(ex) * ES_PER + gu * 51200u,
                              eb + size_t(ex) * ES_PER + gu * 51200u, scr + O_MX2, scr + O_EGU + e * 1280u + gu * 640u, 2560u, 640u, 2u};
      T13[e] = QM{ew + size_t(ex) * EW_PER + 409600u, es + size_t(ex) * ES_PER + 102400u, eb + size_t(ex) * ES_PER + 102400u,
                  mine + e * 640u, scr + O_EY + e * 2560u, 640u, 2560u, 3u};
    }
    run_phase(T12, 20u, T13, 10u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P13 experts down (10 x 4-bit g32, 2560 x 640 each, qmv) ----------------
  if (ok && PH(13)) {
    for (uint i = tid; i < TOPE_C * 640u; i += NT) {
      uint e = i / 640u, c = i % 640u;
      float gg = float(scr[O_EGU + e * 1280u + c]), uu = float(scr[O_EGU + e * 1280u + 640u + c]);
      mine[i] = bfloat16_t(gg / (1.0f + metal::exp(-gg)) * uu);
    }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(T13, 10u, TP14, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P14 h = sum_e sc_e y_e + sy; hyper'' = hyper' + h; mixer HC down; TG 0 writes the node state ----------------
  if (ok && PH(14)) {
    for (uint q = tid; q < 640u; q += NT) {                         // h = sy + sum_e sc_e y_e (4 columns per thread)
      uint c0 = q * 4u;
      float4 h = float4(*(const device vec<bfloat16_t, 4>*)(scr + O_SY + c0));
      for (uint e = 0; e < TOPE_C; ++e) h += scf[O_TOPf + 16u + e] * float4(*(const device vec<bfloat16_t, 4>*)(scr + O_EY + e * 2560u + c0));
      *(device vec<bfloat16_t, 4>*)(mine + 12288u + c0) = vec<bfloat16_t, 4>(h);
    }
    threadgroup_barrier(mem_flags::mem_device);
    prep4(scr + O_HYP, mine + 12288u, scr, 2u, tg == 0u ? hsw + (t + 1u) * 10240u : (device bfloat16_t*)0, mine, tid, NT, red);
    run_phase(TP14, 1u, TP15, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P15 mixer HC up ----------------
  if (ok && PH(15)) {
    for (uint i = tid; i < 320u; i += NT) { float d = float(scr[O_HDN2 + i]) * 0.25f; mine[10240u + i] = bfloat16_t(d / (1.0f + metal::exp(-d))); }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(TP15, 1u, T16, 1u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  // ---------------- P16 x_head; screened head (4-bit g64, NH x 2560) ----------------
  if (ok && PH(16)) {
    for (uint c = tid; c < 2560u; c += NT) {
      float a = 0.0f;
      for (uint s = 0; s < 4u; ++s) { float u = float(scr[O_HUP2 + s * 2560u + c]); a += float(mine[s * 2560u + c]) / (1.0f + metal::exp(-u)); }
      mine[12288u + c] = bfloat16_t(a * 0.25f);
    }
    threadgroup_barrier(mem_flags::mem_device);
    run_phase(T16, 1u, T16, 0u, tg, sg, lane, tid, pfacc);
  }
  BAR();
  if (tg != 0u && HELP && ok) pfacc ^= pf_range(TP1, 5u, tg * VPPC, VPPC, tid, NT_C);
  if (pfacc == 0x9e3779b9u && t > 999999u) scr[0] = bfloat16_t(0.0f);
  // ---------------- P17 tail (TG 0): logsumexp, top-8 (value desc, index asc), exp; a skeleton frontier ----------------
  if (tg == 0u) {
    float m = -INFINITY, s = 0.0f;
    float bv[8]; int bi[8];
    for (uint r = 0; r < 8u; ++r) { bv[r] = -INFINITY; bi[r] = 0x7FFFFFFF; }
    if (ok && PH(17)) {
      for (uint i = tid; i < NH_C; i += NT) {
        float x = float(scr[O_LG + i]);
        float mn = max(m, x);
        s = s * metal::exp(m - mn) + metal::exp(x - mn);
        m = mn;
        if (x > bv[7]) {                                                 // equal values keep the lower index first
          uint p = 7;
          while (p > 0u && x > bv[p - 1u]) { bv[p] = bv[p - 1u]; bi[p] = bi[p - 1u]; --p; }
          bv[p] = x; bi[p] = int(i);
        }
      }
    }
    float mm = simd_max(m);
    s = simd_sum(m == -INFINITY ? 0.0f : s * metal::exp(m - mm));
    if (lane == 0u) { red[sg] = mm; red[20u + sg] = s; }
    uint ptr = 0;
    for (uint r = 0; r < 8u; ++r) {                                      // the simdgroup's top-8: (max value, min index)
      float v = -INFINITY; int ii = 0x7FFFFFFF;
      for (uint q = 0; q < 8u; ++q) if (q == ptr) { v = bv[q]; ii = bi[q]; }
      float mv = simd_max(v);
      int mi = simd_min(v == mv ? ii : 0x7FFFFFFF);
      if (v == mv && ii == mi) ptr += 1u;
      if (lane == 0u) { stv[sg * 8u + r] = mv; sti[sg * 8u + r] = mi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float M = -INFINITY, SS = 0.0f;
      for (uint q = 0; q < NSG; ++q) M = max(M, red[q]);
      for (uint q = 0; q < NSG; ++q) SS += red[20u + q] * metal::exp(red[q] - M);
      float lse = M + metal::precise::log(SS);
      float fv[8]; int fi[8];
      for (uint r = 0; r < 8u; ++r) { fv[r] = -INFINITY; fi[r] = 0x7FFFFFFF; }
      for (uint q = 0; q < NSG * 8u; ++q) {
        float x = stv[q]; int xi = sti[q];
        if (x > fv[7] || (x == fv[7] && xi < fi[7])) {
          uint p = 7;
          while (p > 0u && (x > fv[p - 1u] || (x == fv[p - 1u] && xi < fi[p - 1u]))) { fv[p] = fv[p - 1u]; fi[p] = fi[p - 1u]; --p; }
          fv[p] = x; fi[p] = xi;
        }
      }
      for (uint r = 0; r < 8u; ++r) { out_tis[r] = fi[r]; out_tp[r] = metal::precise::exp(fv[r] - lse); }
      out_pop[0] = int(t); out_pop[1] = fi[0]; out_pop[2] = int(J); out_pop[3] = ok ? 0 : 1; out_pop[4] = int(nb);
      fst_out[0] = int(t + 1u); fst_out[1] = fi[0] & 0x3FF; fst_out[2] = ok ? 0 : 1;
      for (uint r = 3; r < 8u; ++r) fst_out[r] = 0;
    }
  }
"""

STREAM = r"""
  // the same bytes, one pass, no barriers: dense blob, the 10 selected experts, the head
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
        w += n * k // 4                                     # 8-bit: 4 values per uint32
        s += n * k // 64                                    # g64: one scale / bias per 64 values
    return W, S, w, s


def scratch_layout(ntg):
    o, cur = {}, 0
    for name, n in (("EN", 2560), ("FH", 10240), ("HDN0", 512), ("HUP0", 10240), ("Q", 12288), ("K", 512), ("V", 512),
                    ("O", 6144), ("BLK", 2560), ("HYP", 10240), ("HDN1", 512), ("HUP1", 10240), ("MX2", 2560),
                    ("RL", 512), ("SEG", 640), ("SEU", 640), ("SY", 2560), ("EGU", 10 * 1280), ("EY", 10 * 2560),
                    ("HDN2", 512), ("HUP2", 10240), ("LG", 36864 + 256)):
        o[name] = cur                                       # bf16 element offsets
        cur += (n + 255) // 256 * 256
    o["SCf"] = (cur + 1) // 2                               # float offsets (scores 24 x 2048, top-10)
    cur = (o["SCf"] + 24 * 2048) * 2
    o["TOPf"] = cur // 2
    cur = (o["TOPf"] + 64) * 2
    o["PRIV"] = (cur + 255) // 256 * 256
    total = o["PRIV"] + ntg * 16384
    return o, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntg", type=int, default=40)
    ap.add_argument("--nt", type=int, default=512)
    ap.add_argument("--s1", type=int, default=1024)
    ap.add_argument("--nh", type=int, default=36864)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--trees", type=int, default=4)
    ap.add_argument("--spin", type=int, default=400000)
    ap.add_argument("--out", default="")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--pf", type=int, default=0)
    ap.add_argument("--ncon", type=int, default=12)
    ap.add_argument("--help-pf", dest="help", type=int, default=1)
    ap.add_argument("--pfd", type=int, default=2)
    a = ap.parse_args()
    NTG, NT, S1, NH = a.ntg, a.nt, a.s1, a.nh
    W, S, nw32, ns16 = offsets()
    sc, scr_n = scratch_layout(NTG)
    EW_PER, ES_PER = 614400, 153600                         # per expert: gate | up | down (uint32), scales / biases (bf16)
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
    print(f"[pk_skel] NTG {NTG} x {NT}, S1 {S1}, NH {NH}; bytes per pop {bytes_pop / 1e6:.1f} MB "
          f"(dense {(dw.nbytes + ds.nbytes + db.nbytes) / 1e6:.1f}, experts {10 * (EW_PER * 4 + 2 * ES_PER * 2) / 1e6:.1f}, "
          f"head {(hw.nbytes + hs.nbytes + hb.nbytes) / 1e6:.1f}); floor at 292.7 GB/s {bytes_pop / 292.7e6:.3f} ms", flush=True)

    defs = "".join(f"#define W_{n} {W[n]}u\n#define S_{n} {S[n]}u\n" for n, _, _ in DENSE)
    defs += "".join(f"#define O_{k} {v}u\n" for k, v in sc.items())
    defs += f"#define EW_PER {EW_PER}u\n#define ES_PER {ES_PER}u\n#define NSG {NT // 32}u\n#define VPP {NT // 64}u\n"
    defs += f"#define NHEAD_C 24u\n#define NE_C {NE}\n#define TOPE_C 10u\n#define NH_C {NH}u\n"

    def kern(work, mask=0x3FFFF):
        src = defs + f"#define PHMASK {mask}u\n#define WORK {int(work)}\n#define NTG {NTG}u\n#define NT {NT}u\n#define SPIN {a.spin}u\n" + SRC
        hdr = (("#define QMVF qmv_fast_pf_impl\n" if a.pf else "#define QMVF qmv_fast_impl\n") +
               f"#define NCON {a.ncon}u\n#define VPPC {a.ncon // 2}u\n#define SLOTS {NTG * (a.ncon // 2)}u\n"
               f"#define NT_C {NT}u\n#define HELP {a.help}u\n")
        return mx.fast.metal_kernel(
            name=f"pk_skel_c{a.ncon}h{a.help}_w{int(work)}_m{mask}_pf{a.pf}{a.pfd}_{NTG}_{NT}_{S1}_{NH}",
            input_names=["dw", "ds", "db", "ew", "es", "eb", "hw", "hs", "hb", "embt", "k1", "v1", "hsbuf", "tkv",
                         "scratch", "syncb", "fst", "meta"],
            output_names=["out_pop", "out_tis", "out_tp", "fst_out"], source=src, header=pk_qpf.header(a.pfd) + hdr + HDR)

    KW, KB = kern(1), kern(0)
    ks = mx.fast.metal_kernel(name="pk_stream", input_names=["dw", "ds", "db", "ew", "es", "eb", "hw", "hs", "hb"],
                              output_names=["out"], header="#define NDW4 %du\n#define NDS4 %du\n#define EW_PER %du\n"
                              "#define ES_PER %du\n#define NHW4 %du\n#define NHS4 %du\n" % (
                                  dw.size // 4, ds.size // 8, EW_PER, ES_PER, hw.size // 4, hs.size // 8),
                              source=STREAM)
    state = {"base": 0, "sync": mx.zeros((64,), dtype=mx.uint32), "fst": mx.zeros((8,), dtype=mx.int32)}

    def tree(K):
        fst = state["fst"]                                  # the previous tree's last state: one PK kernel at a time
        outs = []
        for t in range(15):
            meta = mx.array([t, state["base"], S1, 0], dtype=mx.int32)
            o_pop, o_tis, o_tp, fst = K(inputs=[dw, ds, db, ew, es, eb, hw, hs, hb, embt, k1, v1, hsbuf, tkv, scratch,
                                               state["sync"], fst, meta],
                                       grid=(NTG * NT, 1, 1), threadgroup=(NT, 1, 1),
                                       output_shapes=[(8,), (8,), (8,), (8,)],
                                       output_dtypes=[mx.int32, mx.int32, mx.float32, mx.int32])
            state["base"] += NBAR
            outs += [o_pop, o_tis, o_tp]
        state["fst"] = fst
        return outs

    def stream():
        return ks(inputs=[dw, ds, db, ew, es, eb, hw, hs, hb], grid=(65536, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(256,)], output_dtypes=[mx.int32])

    def timed(f, n):
        mx.synchronize()
        t0 = time.perf_counter()
        outs = []
        for _ in range(n):
            outs += f()
        mx.eval(outs)
        mx.synchronize()
        return time.perf_counter() - t0, outs

    if a.ablate:
        timed(lambda: tree(KW), 1)
        tb = sorted(timed(lambda: tree(KB), 2)[0] / 30 * 1e3 for _ in range(5))[2]
        tall = sorted(timed(lambda: tree(KW), 2)[0] / 30 * 1e3 for _ in range(5))[2]
        rows, prev = [], tb
        for k in range(1, 18):
            Kk = kern(1, sum(1 << j for j in range(1, k + 1)))   # phases 1..k (cumulative)
            timed(lambda: tree(Kk), 1)
            tk = sorted(timed(lambda: tree(Kk), 2)[0] / 30 * 1e3 for _ in range(5))[2]
            rows.append((k, tk - prev))
            print(f"[pk_skel] + P{k:2d}: {1e3 * (tk - prev):7.1f} us (phases 1..{k}: {tk:.3f} ms)", flush=True)
            prev = tk
        print(f"[pk_skel] barriers only {tb:.3f} ms; all phases {tall:.3f} ms", flush=True)
        return
    # smoke: one tree, check the barrier status
    _, outs = timed(lambda: tree(KW), 1)
    pops = [o.tolist() for o in outs[0::3]]
    fails = sum(p[3] for p in pops)
    print(f"[pk_skel] smoke: barrier fails {fails} of {len(pops)} pops; barriers counted {pops[-1][4]}; "
          f"top-8 ids {outs[1].tolist()[:4]} tp {[round(x, 4) for x in outs[2].tolist()[:4]]}", flush=True)
    if fails:
        print("[pk_skel] FAIL: a grid barrier timed out (threadgroups not co-resident?)", flush=True)
        state["sync"] = mx.zeros((64,), dtype=mx.uint32)
        state["base"] = 0
    res = dict(ntg=NTG, nt=NT, s1=S1, nh=NH, bytes_pop=bytes_pop, floor_ms=bytes_pop / 292.7e6, fails_smoke=fails,
               skel=[], bar=[], stream=[])
    for r in range(a.rounds):
        order = ["skel", "bar", "stream"] if r % 2 == 0 else ["stream", "bar", "skel"]
        for arm in order:
            if arm == "skel":
                dt, outs = timed(lambda: tree(KW), a.trees)
                res["skel"].append(dt / (15 * a.trees) * 1e3)
                fails += sum(int(o.tolist()[3]) for o in outs[0::3])
            elif arm == "bar":
                dt, _ = timed(lambda: tree(KB), a.trees)
                res["bar"].append(dt / (15 * a.trees) * 1e3)
            else:
                dt, _ = timed(lambda: [stream()], 15 * a.trees)
                res["stream"].append(dt / (15 * a.trees) * 1e3)
    res["fails_total"] = fails
    for arm in ("skel", "bar", "stream"):
        v = sorted(res[arm][2:])
        res[arm + "_med"] = v[len(v) // 2]
    res["f_line"] = res["skel_med"] - res["floor_ms"]
    res["f_stream"] = res["skel_med"] - res["stream_med"]
    print(f"[pk_skel] ms per pop (median of {a.rounds - 2} rounds, {a.trees} trees x 15 pops): skel {res['skel_med']:.3f}, "
          f"barriers only {res['bar_med']:.3f}, stream {res['stream_med']:.3f} ({bytes_pop / res['stream_med'] / 1e6:.1f} GB/s); "
          f"floor (292.7) {res['floor_ms']:.3f}", flush=True)
    print(f"[pk_skel] f = skel - floor = {res['f_line']:.3f} ms (line: PASS <= 0.10, GO <= 0.15, KILL >= 0.25); "
          f"skel - stream = {res['f_stream']:.3f} ms; barrier fails {fails}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
