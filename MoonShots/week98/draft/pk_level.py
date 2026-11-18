"""PK v2: the PERSISTENT tree level of r57's live Q8H32 head (MoonShots, Wed 18 Nov 2026; ALL_LEVERS item 1).
One dispatch per pop (20 TGs x 1024 threads, 16 grid barriers): the node's whole forward (forward_static's ops), its
screened head, the tail (logsumexp, top-K, p) and the frontier step, all bitwise to ra_sf.draft_sf by construction
(each phase: pk_ops' bit-tested replicas of MLX 0.32.2's stock kernels; pk_qbit / pk_hcbit / pk_attnbit / pk_moebit).
A per-tree setup dispatch does the root row's top-K and step 0. The host reads the pops and path products once per tree
and builds tree16_draft.draft's dict exactly as draft_sf does.
Falls back (returns None) when the one-pass attention form does not hold (prefix + tree >= 1024 keys) or a barrier timed
out (the caller then runs draft_sf: the same bits).
Phases: P1 fuse | P2-P3 attention HC | P4 q / k / v | P5 attention | P6 o_proj | P7-P8 MLP HC | P9 router, shared
gate / up, the fp32 gate's partials | P10 route | shared down | P11 experts gate / up | P12 experts down | P13 MoE
combine | P14-P15 mixer HC (P14 writes the node's hidden) | P16 head + top-8 per TG | P17 tail + frontier (TG 0)."""
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/MoonShots/week98/bet7"))
import mlx.core as mx  # noqa: E402
import pk_ops  # noqa: E402
import sf_tree as SFT  # noqa: E402
from fl_exact import RMS_HEADER  # noqa: E402
import pk_skel2 as S2  # noqa: E402

NTG, NT = 20, 1024
NBAR = 16
NS0 = 32768                                                   # base screen rows; + extra_cap slots
GBAR = S2.HDR[S2.HDR.index("METAL_FUNC bool gbar("):S2.HDR.index("// 4 x 2560: x = a + b")]
GBAR = GBAR.replace("uint spin) {\n  threadgroup_barrier(mem_flags::mem_device);",
                    "uint spin) {\n  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);   // every writer releases\n"
                    "  threadgroup_barrier(mem_flags::mem_device);")
assert "every writer releases" in GBAR, "gbar text changed"
HDR = (pk_ops.HEADER + RMS_HEADER + pk_ops.HC + pk_ops.ATTN + pk_ops.MOE + SFT.HDR + pk_ops.TAIL + GBAR)


class Layout:
    """scratch regions: byte offsets aligned to 256; per region its element offset in its own type's view"""

    SIZES = {"u64": 8, "i32": 4, "f32": 4, "bf16": 2}

    def __init__(self):
        self.cur, self.defs = 0, {}

    def add(self, name, typ, n):
        self.cur = (self.cur + 255) // 256 * 256
        sz = self.SIZES[typ]
        self.defs[name] = (typ, self.cur // sz)
        self.cur += n * sz

    def header(self):
        return "".join(f"#define O_{k} {off}u\n" for k, (typ, off) in self.defs.items())


def make_layout(n, K, cap):
    L = Layout()
    C = K * n
    for nm, ty, sz in (("fk", "u64", C), ("npp", "u64", n), ("fpar", "i32", C), ("ftok", "i32", C), ("fdep", "i32", C),
                       ("nf", "i32", 4), ("ndep", "i32", n), ("nanc", "i32", n * n), ("pops", "i32", n * 4),
                       ("info", "i32", 16), ("inds", "i32", 16), ("tk8i", "i32", NTG * 8), ("tisx", "i32", 16),
                       ("lg", "f32", NS0 + cap), ("tk8v", "f32", NTG * 8), ("tvs", "f32", 16), ("tps", "f32", 16),
                       ("spart", "f32", 16), ("ip1", "f32", 32), ("ip2", "f32", 32), ("ip3", "f32", 32),
                       ("HS", "bf16", (n + 1) * 10240), ("kd", "bf16", n * 512), ("vd", "bf16", n * 512),
                       ("en", "bf16", 2560), ("fh", "bf16", 10240), ("x1", "bf16", 10240), ("m1a", "bf16", 320),
                       ("mixed1", "bf16", 2560), ("inj1", "bf16", 16), ("qf", "bf16", 12288), ("kf", "bf16", 512),
                       ("vf", "bf16", 512), ("gated", "bf16", 6144), ("bo1", "bf16", 2560), ("hw2", "bf16", 10240),
                       ("m1b", "bf16", 320), ("mixed2", "bf16", 2560), ("inj2", "bf16", 16), ("rl", "bf16", 512),
                       ("sh", "bf16", 640), ("wts", "bf16", 16), ("segv", "bf16", 16), ("sout", "bf16", 2560),
                       ("eh", "bf16", 6400), ("yout", "bf16", 25600), ("bo2", "bf16", 2560), ("m1c", "bf16", 320),
                       ("xhead", "bf16", 2560), ("ones4", "bf16", 16), ("dmy", "bf16", 64),
                       ("PRIV", "bf16", NTG * 16384)):
        L.add(nm, ty, sz)
    return L


MAIN = r"""
  const uint tg = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint nsg = NT / 32u, gsg = tg * nsg + sg, nall = NTG * nsg;
  const int t = meta[0];
  const uint base = uint(meta[1]), Lc = uint(meta[2]), khs = uint(meta[5]), n_extra = uint(meta[6]);
  const int n = meta[3], KK = meta[4];
  device atomic_uint* ctr = (device atomic_uint*)syncb;
  device ulong* S64 = (device ulong*)scr;
  device int* S32 = (device int*)scr;
  device float* SF = (device float*)scr;
  device bfloat16_t* SB = (device bfloat16_t*)scr;
  threadgroup uint tflag;
  threadgroup float part[96];
  threadgroup float inv[4];
  threadgroup bfloat16_t qr[256], kr[256], qn[256], kn[256];
  threadgroup float cs[64], sn[64], outs_[32 * 32], maxs_[32], sums_[32];
  threadgroup float lmax[32], lnorm[32], lse_s[1];
  threadgroup bfloat16_t Pr[512], tvr[10];
  threadgroup uint tir[10], sgk[40];
  threadgroup float stv[NSG * 8u];
  threadgroup int sti[NSG * 8u];
  threadgroup float cvv[NTG * 8u];
  threadgroup int cii[NTG * 8u];
  float tv = -INFINITY;
  int ti = 0x7FFFFFFF;
  uint nb = 0;
  bool ok = true;
#define BAR() { nb += 1u; if (ok) ok = gbar(ctr, (base + nb) * NTG, lid, tflag, SPIN); }
  const int par = S32[O_info + 0], tok = S32[O_info + 1], d0 = S32[O_info + 2];
  const uint anc = uint(S32[O_info + 4]);
  device bfloat16_t* mine = SB + O_PRIV + tg * 16384u;
  const device bfloat16_t* hpar = SB + O_HS + uint(par + 1) * 10240u;

  // P1 fuse: every TG its emb row, en_in, hn; fc_embedding (8-bit, 1 row per simdgroup) | fc_hidden x4 (qmv_wide)
  pk_fuse_prep(ew, es, eb, uint(tok), hpar, sm + SM_pne, sm + SM_pnh, fs[FS_eps + 0], fs[FS_eps + 1], mine, mine + 2560u,
               mine + 5120u, part, lid, NT, sg, nsg, lane);
  for (uint u = gsg; u < 2560u + 640u; u += nall) {
    if (u < 2560u) qmv_pf<bfloat16_t, 64, 8, 1, 2, true, false>(dw + W_fc_emb, ds + S_fc_emb, db + S_fc_emb, mine + 2560u,
                                                                SB + O_en, 2560, int(u), lane, tv, ti);
    else { uint v = u - 2560u; qmv_wide_impl<bfloat16_t, 64, 8, 4, 8>(dw + W_fc_hid, ds + S_fc_hid, db + S_fc_hid,
             mine + 5120u, SB + O_fh, 2560, 2560, 4, uint3(0, v >> 1, 0), v & 1u, lane); }
  }
  BAR();
  // P2 / P3 attention HC: hyper' = fh + en (the fuse's add, as a pending write with inject 1), inject
  if (ok) pk_hc_a<true, true>(SB + O_fh, SB + O_en, SB + O_ones4, sm + SM_ahn, fs[FS_eps + 2], dw + W_ahc_dn, ds + S_ahc_dn,
                              db + S_ahc_dn, sm + SM_ahi, mine, SB + O_x1, SB + O_m1a, SF + O_ip1, part, inv, tg, NTG,
                              NTG - 1u, lid, NT, sg, nsg, lane);
  BAR();
  if (ok) pk_hc_b<true>(SB + O_m1a, dw + W_ahc_up, ds + S_ahc_up, db + S_ahc_up, mine, SB + O_mixed1, SF + O_ip1,
                        SB + O_inj1, tg, NTG, sg, nsg, lane);
  BAR();
  // P4 q / k / v
  if (ok) for (uint u = gsg; u < 12288u / 2u + 1024u; u += nall) {
    if (u < 6144u) qmv_pf<bfloat16_t, 64, 8, 2, 2, true, false>(dw + W_q, ds + S_q, db + S_q, SB + O_mixed1, SB + O_qf,
                                                                2560, int(u) * 2, lane, tv, ti);
    else if (u < 6656u) qmv_pf<bfloat16_t, 64, 8, 1, 2, true, false>(dw + W_k, ds + S_k, db + S_k, SB + O_mixed1, SB + O_kf,
                                                                     2560, int(u - 6144u), lane, tv, ti);
    else qmv_pf<bfloat16_t, 64, 8, 1, 2, true, false>(dw + W_v, ds + S_v, db + S_v, SB + O_mixed1, SB + O_vf, 2560,
                                                      int(u - 6656u), lane, tv, ti);
  }
  BAR();
  // P5 attention (heads over TGs); the node's rotated k / its v -> drafted row t
  if (ok) {
    for (uint h = tg; h < 24u; h += NTG)
      pk_attn_head(h, SB + O_qf, SB + O_kf, SB + O_vf, kc, vc, khs, SB + O_kd, SB + O_vd, Lc, uint(t), anc,
                   Lc + uint(d0), sm + SM_qn, sm + SM_kn, fs + FS_invf, fs[FS_eps + 5], SB + O_gated, SB + O_kd + uint(t) * 512u,
                   part, qr, kr, cs, sn, qn, kn, outs_, maxs_, sums_, lid, sg, lane, nsg);
    if (tg == NTG - 1u) for (uint i = lid; i < 512u; i += NT) SB[O_vd + uint(t) * 512u + i] = SB[O_vf + i];
  }
  BAR();
  // P6 o_proj
  if (ok) for (uint u = gsg; u < 1280u; u += nall)
    qmv_pf<bfloat16_t, 64, 8, 2, 2, true, false>(dw + W_o, ds + S_o, db + S_o, SB + O_gated, SB + O_bo1, 6144, int(u) * 2,
                                                 lane, tv, ti);
  BAR();
  // P7 / P8 MLP HC (pending: the attention block's write)
  if (ok) pk_hc_a<true, true>(SB + O_x1, SB + O_bo1, SB + O_inj1, sm + SM_mhn, fs[FS_eps + 3], dw + W_mhc_dn, ds + S_mhc_dn,
                              db + S_mhc_dn, sm + SM_mhi, mine, SB + O_hw2, SB + O_m1b, SF + O_ip2, part, inv, tg, NTG,
                              NTG - 1u, lid, NT, sg, nsg, lane);
  BAR();
  if (ok) pk_hc_b<true>(SB + O_m1b, dw + W_mhc_up, ds + S_mhc_up, db + S_mhc_up, mine, SB + O_mixed2, SF + O_ip2,
                        SB + O_inj2, tg, NTG, sg, nsg, lane);
  BAR();
  // P9 router | shared gate / up + swiglu | the fp32 gate's partials (last TG, simdgroups 0..15)
  if (ok) {
    if (tg == NTG - 1u && sg < 16u) pk_seg_partial(SB + O_mixed2, fs + FS_segw, SF + O_spart, sg, lane);
    const uint wid = (tg == NTG - 1u) ? (sg >= 16u ? tg * nsg + sg - 16u : 0xFFFFFFFFu) : gsg;
    for (uint u = wid; u < 512u + 640u; u += nall - 16u) {
      if (u < 512u) qmv_pf<bfloat16_t, 64, 8, 1, 2, true, false>(dw + W_router, ds + S_router, db + S_router, SB + O_mixed2,
                                                                 SB + O_rl, 2560, int(u), lane, tv, ti);
      else {
        uint j = u - 512u;
        bfloat16_t g[1], uu[1];
        qmv_rows<bfloat16_t, 64, 8, 1, true>(dw + W_se_gate, ds + S_se_gate, db + S_se_gate, SB + O_mixed2, 2560, int(j), lane, g);
        qmv_rows<bfloat16_t, 64, 8, 1, true>(dw + W_se_up, ds + S_se_up, db + S_se_up, SB + O_mixed2, 2560, int(j), lane, uu);
        if (lane == 0u) SB[O_sh + j] = fl_silu_jit(g[0]) * uu[0];
      }
    }
  }
  BAR();
  // P10 route + the gate's value (TG 0) | shared down (the other TGs)
  if (ok) {
    if (tg == 0u) {
      pk_route(SB + O_rl, (device uint*)(S32 + O_inds), SB + O_wts, lmax, lnorm, Pr, tir, tvr, sgk, lid, lane, sg);
      if (sg == 0u) { bfloat16_t s = pk_seg_value(SF + O_spart, lane); if (lane == 0u) SB[O_segv] = s; }
    } else {
      for (uint u = (tg - 1u) * nsg + sg; u < 2560u; u += (NTG - 1u) * nsg)
        qmv_pf<bfloat16_t, 64, 8, 1, 1, false, false>(dw + W_se_down, ds + S_se_down, db + S_se_down, SB + O_sh, SB + O_sout,
                                                      640, int(u), lane, tv, ti);
    }
  }
  BAR();
  // P11 experts gate / up + swiglu (4-bit g32, 2 rows per simdgroup)
  if (ok) for (uint u = gsg; u < 3200u; u += nall) {
    uint e = u / 320u, j = (u % 320u) * 2u;
    size_t ex = uint(S32[O_inds + e]);
    bfloat16_t g[2], uu[2];
    qmv_rows<bfloat16_t, 32, 4, 2, true>(egw + ex * (640 * 320), egs + ex * (640 * 80), egb + ex * (640 * 80), SB + O_mixed2,
                                         2560, int(j), lane, g);
    qmv_rows<bfloat16_t, 32, 4, 2, true>(euw + ex * (640 * 320), eus + ex * (640 * 80), eub + ex * (640 * 80), SB + O_mixed2,
                                         2560, int(j), lane, uu);
    if (lane == 0u) { SB[O_eh + e * 640u + j] = fl_silu_jit(g[0]) * uu[0]; SB[O_eh + e * 640u + j + 1u] = fl_silu_jit(g[1]) * uu[1]; }
  }
  BAR();
  // P12 experts down (4-bit g32, qmv K 640, 4 rows per simdgroup)
  if (ok) for (uint u = gsg; u < 6400u; u += nall) {
    uint e = u / 640u, r0 = (u % 640u) * 4u;
    size_t ex = uint(S32[O_inds + e]);
    qmv_pf<bfloat16_t, 32, 4, 4, 1, false, false>(edw + ex * (2560 * 80), eds + ex * (2560 * 20), edb + ex * (2560 * 20),
                                                  SB + O_eh + e * 640u, SB + O_yout + e * 2560u, 640, int(r0), lane, tv, ti);
  }
  BAR();
  // P13 the MoE block's output
  if (ok) for (uint c = tg * NT + lid; c < 2560u; c += NTG * NT)
    SB[O_bo2 + c] = pk_moe_out(SB + O_yout, SB + O_wts, SB[O_segv], SB + O_sout, c);
  BAR();
  // P14 / P15 mixer HC: the pending write = the node's hidden h (TG 0 writes HS[t + 1]); x_head
  if (ok) pk_hc_a<true, false>(SB + O_hw2, SB + O_bo2, SB + O_inj2, sm + SM_xhn, fs[FS_eps + 4], dw + W_xhc_dn, ds + S_xhc_dn,
                               db + S_xhc_dn, sm + SM_ahi, mine, SB + O_HS + uint(t + 1) * 10240u, SB + O_m1c, SF + O_ip3,
                               part, inv, tg, NTG, NTG - 1u, lid, NT, sg, nsg, lane);
  BAR();
  if (ok) pk_hc_b<false>(SB + O_m1c, dw + W_xhc_up, ds + S_xhc_up, db + S_xhc_up, mine, SB + O_xhead, SF + O_ip3,
                         SB + O_dmy, tg, NTG, sg, nsg, lane);
  BAR();
  // P16 the screened head (4-bit g64, 2 rows per simdgroup): base rows, the used extra slots (+ 0.0f, as + xmask);
  // the unused slots -inf; every simdgroup keeps a top-8 (value desc, index asc), each TG merges its lists
  if (ok) {
    const uint nb2 = NS0 / 2u, ne2 = (n_extra + 1u) / 2u;
    for (uint u = gsg; u < nb2 + ne2; u += nall) {
      bfloat16_t v2[2];
      if (u < nb2) {
        qmv_rows<bfloat16_t, 64, 4, 2, true>(hw, hs, hb, SB + O_xhead, 2560, int(u) * 2, lane, v2);
        for (int r = 0; r < 2; r++) {
          uint row = u * 2u + uint(r);
          float v = float(v2[r]);
          if (lane == 0u) SF[O_lg + row] = v;
          top8_insert(tv, ti, v, int(row), lane);
        }
      } else {
        uint j0 = (u - nb2) * 2u;
        qmv_rows<bfloat16_t, 64, 4, 2, true>(xw, xs, xb, SB + O_xhead, 2560, int(j0), lane, v2);
        for (int r = 0; r < 2; r++) {
          uint j = j0 + uint(r);
          if (j < n_extra) {
            float v = float(v2[r]) + 0.0f;
            if (lane == 0u) SF[O_lg + NS0 + j] = v;
            top8_insert(tv, ti, v, int(NS0 + j), lane);
          }
        }
      }
    }
    for (uint j = n_extra + tg * NT + lid; j < uint(XCAP); j += NTG * NT) SF[O_lg + NS0 + j] = -INFINITY;
    if (lane < 8u) { stv[sg * 8u + lane] = tv; sti[sg * 8u + lane] = ti; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float cv[NSG / 4u]; int ci[NSG / 4u];                    // NSG * 8 candidates over 32 lanes
      for (uint q = 0; q < NSG / 4u; ++q) { cv[q] = stv[lane * (NSG / 4u) + q]; ci[q] = sti[lane * (NSG / 4u) + q]; }
      for (uint r = 0; r < 8u; ++r) {
        float bv = -INFINITY; int bi = 0x7FFFFFFF; uint bq = 99u;
        for (uint q = 0; q < NSG / 4u; ++q) if (cv[q] > bv || (cv[q] == bv && ci[q] < bi)) { bv = cv[q]; bi = ci[q]; bq = q; }
        float mv = simd_max(bv);
        int mi = simd_min(bv == mv ? bi : 0x7FFFFFFF);
        if (bv == mv && bi == mi && bq < 99u) { cv[bq] = -INFINITY; ci[bq] = 0x7FFFFFFF; }
        if (lane == 0u) { SF[O_tk8v + tg * 8u + r] = mv; S32[O_tk8i + tg * 8u + r] = mi; }
      }
    }
  }
  BAR();
  // P17 tail (TG 0): logsumexp (stock order), the top-K by rank over the TGs' lists, p = exp(v - lse), ids, step t + 1
  if (tg == 0u) {
    float lse = pk_lse(SF + O_lg, int(NS0 + XCAP), lmax, lnorm, lse_s, lid, NT, lane, sg);
    if (lid < NTG * 8u) { cvv[lid] = SF[O_tk8v + lid]; cii[lid] = S32[O_tk8i + lid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < NTG * 8u) {
      float v = cvv[lid]; int i = cii[lid]; uint rk = 0;
      for (uint q = 0; q < NTG * 8u; ++q) rk += (cvv[q] > v || (cvv[q] == v && cii[q] < i)) ? 1u : 0u;
      if (rk < uint(KK)) { stv[rk] = v; sti[rk] = i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      int tis[16]; float tp[16];
      for (int c = 0; c < KK; ++c) {
        float dlt = stv[c] - lse;
        tp[c] = metal::precise::exp(dlt);
        tis[c] = imap[sti[c]];
      }
      if (t + 1 < n) {
        pk_step((device ulong*)(S64 + O_fk), S32 + O_fpar, S32 + O_ftok, S32 + O_fdep, S32 + O_nf, (device ulong*)(S64 + O_npp),
                S32 + O_ndep, S32 + O_nanc, S32 + O_pops, tis, tp, t + 1, KK, KK * n, n);
        const int s1 = t + 1;
        int dd = S32[O_pops + s1 * 4 + 2];
        uint ab = 0u;
        for (int j = 0; j < dd; ++j) { int a = S32[O_nanc + s1 * n + j]; if (a >= 0) ab |= (1u << uint(a)); }
        S32[O_info + 0] = S32[O_pops + s1 * 4 + 0]; S32[O_info + 1] = S32[O_pops + s1 * 4 + 1];
        S32[O_info + 2] = dd; S32[O_info + 3] = S32[O_pops + s1 * 4 + 3]; S32[O_info + 4] = int(ab);
      }
      S32[O_info + 8] = max(S32[O_info + 8], ok ? 0 : 1);
    }
  }
  if (tg == 0u && lid == 0u) dep[0] = ok ? 0 : 1;
"""

SETUP = r"""
  // one TG of 1024: re-init the frontier, HS[0] = the root's hidden, the root row's top-K + p, step 0 (node 0's info)
  const uint lid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int n = meta[3], KK = meta[4];
  device ulong* S64 = (device ulong*)scr;
  device int* S32 = (device int*)scr;
  device float* SF = (device float*)scr;
  device bfloat16_t* SB = (device bfloat16_t*)scr;
  threadgroup float lmax[32], lnorm[32], lse_s[1];
  threadgroup float stv[32 * 8];
  threadgroup int sti[32 * 8];
  for (uint i = lid; i < uint(KK * n); i += 1024u) { S64[O_fk + i] = 0ul; S32[O_fpar + i] = -1; S32[O_ftok + i] = 0; S32[O_fdep + i] = 0; }
  for (uint i = lid; i < uint(n); i += 1024u) { S64[O_npp + i] = 0ul; S32[O_ndep + i] = 0; }
  for (uint i = lid; i < uint(n * n); i += 1024u) S32[O_nanc + i] = -1;
  if (lid == 0u) { S32[O_nf] = 0; S32[O_info + 8] = 0; }
  for (uint i = lid; i < 10240u; i += 1024u) SB[O_HS + i] = h0[i];
  for (uint i = lid; i < 16u; i += 1024u) SB[O_ones4 + i] = static_cast<bfloat16_t>(1.0f);
  threadgroup_barrier(mem_flags::mem_device);
  float lse = pk_lse(root, int(NS0 + XCAP), lmax, lnorm, lse_s, lid, 1024u, lane, sg);
  float tv = -INFINITY; int ti = 0x7FFFFFFF;
  for (uint base = 0; base < uint(NS0 + XCAP); base += 32u * 32u) {            // simdgroups take 32-row chunks
    uint row = base + sg * 32u + lane;
    for (uint r = 0; r < 32u; ++r) {
      uint rr = base + sg * 32u + r;
      if (rr < uint(NS0 + XCAP)) top8_insert(tv, ti, root[rr], int(rr), lane);
    }
    (void)row;
  }
  if (lane < 8u) { stv[sg * 8u + lane] = tv; sti[sg * 8u + lane] = ti; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float fv[16];
  threadgroup int fi[16];
  if (lid < 256u) {
    float v = stv[lid]; int i = sti[lid]; uint rk = 0;
    for (uint q = 0; q < 256u; ++q) rk += (stv[q] > v || (stv[q] == v && sti[q] < i)) ? 1u : 0u;
    if (rk < uint(KK)) { fv[rk] = v; fi[rk] = i; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    int tis[16]; float tp[16];
    for (int c = 0; c < KK; ++c) { float dlt = fv[c] - lse; tp[c] = metal::precise::exp(dlt); tis[c] = imap[fi[c]]; }
    pk_step((device ulong*)(S64 + O_fk), S32 + O_fpar, S32 + O_ftok, S32 + O_fdep, S32 + O_nf, (device ulong*)(S64 + O_npp),
            S32 + O_ndep, S32 + O_nanc, S32 + O_pops, tis, tp, 0, KK, KK * n, n);
    S32[O_info + 0] = S32[O_pops + 0]; S32[O_info + 1] = S32[O_pops + 1]; S32[O_info + 2] = S32[O_pops + 2];
    S32[O_info + 3] = S32[O_pops + 3]; S32[O_info + 4] = 0;
    dep[0] = 0;
  }
"""


class PK:
    """the persistent level bound to r57's live head (after r57_draft.install(..., 'q8h32'))"""

    def __init__(self, lm, n=15, k=8, extra_cap=4096, spin=2000000):
        self.lm, self.n, self.k, self.cap, self.spin = lm, n, k, extra_cap, spin
        head = lm.mtp
        L0 = head.layers[0]
        att, moe = L0.self_attn, L0.mlp
        ahc, mhc, xhc = L0.attn_hyper_connection, L0.mlp_hyper_connection, head.hyper_connection_mixer
        se = moe.shared_expert
        mods = [("fc_emb", head.fc_embedding), ("fc_hid", head.fc_hidden), ("q", att.q_proj), ("k", att.k_proj),
                ("v", att.v_proj), ("o", att.o_proj), ("router", moe.gate), ("se_gate", se.gate_proj),
                ("se_up", se.up_proj), ("se_down", se.down_proj), ("ahc_dn", ahc.input_mix_weight_down),
                ("ahc_up", ahc.input_mix_weight_up), ("mhc_dn", mhc.input_mix_weight_down),
                ("mhc_up", mhc.input_mix_weight_up), ("xhc_dn", xhc.input_mix_weight_down),
                ("xhc_up", xhc.input_mix_weight_up)]
        defs, wl, sl, bl, wo, so = [], [], [], [], 0, 0
        for nm, m in mods:
            assert m.bits == 8 and m.group_size == 64, (nm, m.bits, m.group_size)
            defs.append(f"#define W_{nm} {wo}u\n#define S_{nm} {so}u\n")
            wl.append(m.weight.reshape(-1)); sl.append(m.scales.reshape(-1)); bl.append(m.biases.reshape(-1))
            wo += m.weight.size; so += m.scales.size
        self.dw, self.ds, self.db = mx.concatenate(wl), mx.concatenate(sl), mx.concatenate(bl)
        smalls = [("pnh", head.pre_fc_norm_hidden.weight), ("pne", head.pre_fc_norm_embedding.weight),
                  ("ahn", ahc.hc_norm.weight), ("mhn", mhc.hc_norm.weight), ("xhn", xhc.hc_norm.weight),
                  ("qn", att.q_norm.weight), ("kn", att.k_norm.weight), ("ahi", ahc.block_inject_weight.weight),
                  ("mhi", mhc.block_inject_weight.weight)]
        sml, smo = [], 0
        for nm, a in smalls:
            assert a.dtype == mx.bfloat16, (nm, a.dtype)
            defs.append(f"#define SM_{nm} {smo}u\n")
            sml.append(a.reshape(-1)); smo += a.size
        self.sm = mx.concatenate(sml)
        eps = [head.pre_fc_norm_embedding.eps, head.pre_fc_norm_hidden.eps, ahc.hc_norm.eps, mhc.hc_norm.eps,
               xhc.hc_norm.eps, att.q_norm.eps, 0.0, 0.0]
        segw = moe.shared_expert_gate.weight
        assert segw.dtype == mx.float32 and segw.size == 2560, (segw.dtype, segw.shape)
        self.fs = mx.concatenate([att._inv_freq.astype(mx.float32).reshape(-1), segw.reshape(-1), mx.array(eps, mx.float32)])
        defs.append(f"#define FS_invf 0u\n#define FS_segw {att._inv_freq.size}u\n#define FS_eps {att._inv_freq.size + 2560}u\n")
        sw = moe.switch_mlp
        assert int(moe.top_k) == 10 and sw.gate_proj.bits == 4 and sw.gate_proj.group_size == 32
        self.ex = [sw.gate_proj.weight, sw.gate_proj.scales, sw.gate_proj.biases, sw.up_proj.weight, sw.up_proj.scales,
                   sw.up_proj.biases, sw.down_proj.weight, sw.down_proj.scales, sw.down_proj.biases]
        et = lm.model.embed_tokens
        assert et.bits == 4 and et.group_size == 64
        self.emb = [et.weight.view(mx.uint8).reshape(-1), et.scales.reshape(-1), et.biases.reshape(-1)]
        self.L = make_layout(n, k, extra_cap)
        self.scr = mx.zeros(((self.L.cur + 15) // 16 * 4,), mx.uint32)
        self.sync = mx.zeros((64,), mx.uint32)
        mx.eval(self.dw, self.ds, self.db, self.sm, self.fs, self.scr, self.sync)
        self.base = 0
        cdefs = ("".join(defs) + self.L.header() + f"#define NTG {NTG}u\n#define NT {NT}u\n#define NSG {NT // 32}u\n"
                 f"#define SPIN {spin}u\n#define NS0 {NS0}u\n#define XCAP {extra_cap}u\n")
        self.kmain = mx.fast.metal_kernel(
            name=f"pk_level_{n}_{k}_{extra_cap}",
            input_names=["dw", "ds", "db", "sm", "fs", "egw", "egs", "egb", "euw", "eus", "eub", "edw", "eds", "edb", "hw",
                         "hs", "hb", "xw", "xs", "xb", "ew", "es", "eb", "kc", "vc", "imap", "scr", "syncb", "meta", "chain"],
            output_names=["dep"], source=MAIN, header=cdefs + HDR)
        self.ksetup = mx.fast.metal_kernel(name=f"pk_setup_{n}_{k}_{extra_cap}",
                                           input_names=["root", "h0", "imap", "scr", "meta"], output_names=["dep"],
                                           source=SETUP, header=cdefs + HDR)

    def draft(self, view, root_tok, sh, imap, sync_each=False):
        """tree16_draft.draft's dict for the head state `view` (cache, logits [1, 1, NS] screened, h), or None (fallback)"""
        import tree16_draft as T16D
        n, k = self.n, self.k
        qc = view.cache[0]
        Lc = int(qc.offset)
        if Lc + n >= 1024:
            return None                                       # the stock 2-pass attention form: not in this level
        kb, vb = qc.kv.keys, qc.kv.values
        cap_kv = int(kb.shape[2])
        root = view.logits[0, -1].astype(mx.float32)
        imap = imap.astype(mx.int32)
        assert root.size == NS0 + self.cap and imap.size == NS0 + self.cap
        hs, xs = [sh.w, sh.s, sh.b], [sh.xw, sh.xs, sh.xb]
        meta0 = mx.array([0, 0, Lc, n, k, cap_kv * 256, int(sh.n_extra), 0], mx.int32)
        dep = self.ksetup(inputs=[root, view.h.reshape(-1), imap, self.scr, meta0], grid=(1024, 1, 1),
                          threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.int32])[0]
        outs = [dep]
        for t in range(n - 1):
            meta = mx.array([t, self.base, Lc, n, k, cap_kv * 256, int(sh.n_extra), 0], mx.int32)
            dep = self.kmain(inputs=[self.dw, self.ds, self.db, self.sm, self.fs] + self.ex + hs + xs + self.emb +
                             [kb.reshape(-1), vb.reshape(-1), imap, self.scr, self.sync, meta, dep],
                             grid=(NTG * NT, 1, 1), threadgroup=(NT, 1, 1), output_shapes=[(1,)],
                             output_dtypes=[mx.int32])[0]
            self.base += NBAR
            outs.append(dep)
            if sync_each:
                mx.eval(dep)
        mx.eval(outs)
        S32 = np.array(self.scr.view(mx.int32))
        o = self.L.defs
        fails = int(sum(int(np.array(d)[0]) for d in outs[1:]))
        if fails:
            return None
        P_ = S32[o["pops"][1]:o["pops"][1] + 4 * n].reshape(n, 4)
        npp = np.array(self.scr.view(mx.uint64))[o["npp"][1]:o["npp"][1] + n].view(np.float64)
        par = [int(x) for x in P_[:, 0]]
        tok = [int(x) for x in P_[:, 1]]
        dep_ = [int(x) + 1 for x in P_[:, 2]]
        pth = [float(x) for x in npp[:n]]
        pdr = [pth[i] / (pth[par[i]] if par[i] >= 0 else 1.0) for i in range(n)]
        out = T16D.preorder(int(root_tok), par, tok, dep_, pth, pdr)
        out["bad"] = int(P_[:, 3].max())
        return out
