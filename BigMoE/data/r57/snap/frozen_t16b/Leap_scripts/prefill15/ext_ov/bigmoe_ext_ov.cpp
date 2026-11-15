// Leap 11-11 (prefill 1.5K): a COPY of BigMoE's engine/ext/bigmoe_ext.cpp as module bigmoe_ext_ov. Only change:
// read_rebuilt / read_rebuilt_layout release the GIL (nb::gil_scoped_release) and require evaluated destinations,
// so a prefill's main thread can build the next graphs while the read runs. Everything else is the original.
// bigmoe_ext: exact SSD expert streaming for MLX without a per-layer CPU<->GPU sync.
//
// Measured on M5 Pro (swift/eventwait.swift): a GPU-signalled MTLSharedEvent costs ~140 us
// before a dependent wait is released, but a CPU-signalled wait is free and a GPU store to
// shared memory is visible to a spinning CPU thread at kernel end. So per layer:
//   GPU: router -> tiny kernel copies expert ids + sequence number into a shared buffer
//        -> gate(): commit the command buffer, next buffer encodeWait(event, v)
//   CPU (native thread, here): spin on the sequence number, make the experts resident
//        (LRU, preadv straight into Metal-backed slot pools), write slot ids, signal v.
//   GPU: expert matmul reads the slot ids and the pools. Nothing is substituted: exact.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <list>
#include <memory>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <dispatch/dispatch.h>
#include <time.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <mach/mach_time.h>

#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/event.h"
#include "mlx/backend/metal/device.h"

namespace nb = nanobind;
using namespace mlx::core;

static double now_s() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

// ------------------------------------------------------------------ expert cache
struct Piece { int fd; int64_t off; int64_t per; uint8_t* pool; };

struct Layer {
  int C = 0, E = 0;
  std::vector<Piece> pieces;            // 9: gate/up/down x weight/scales/biases
  std::vector<int32_t> table;           // expert -> slot or -1
  std::vector<int32_t> slot_owner;      // slot -> expert or -1
  std::list<int32_t> lru;               // experts, most recent at front
  std::vector<std::list<int32_t>::iterator> where;
  std::vector<int32_t> free_slots;
  // mid-token prefetch: predicted experts go into spare slots that no published table references;
  // they join the cache only when the CPU serves that layer's real routing (promote in ensure()).
  int C_main = 0, spare = 0;
  std::unordered_map<int32_t, int32_t> pf;       // expert -> spare slot (loading or loaded)
  std::list<int32_t> pf_order;                    // oldest first
  std::vector<std::atomic<int>>* ready = nullptr; // per slot: 1 = bytes in place
  std::vector<int32_t> pred; int32_t pred_seq = -1;
  int rec_fmt = 1;                                  // 1 = records/ (9 pieces), 3 = records3 [MSB-3 | LSB-1] (DESIGN_3bit.md)
  int64_t msb_sz = 0;
  // records3 converted in place (fmt 3 readers only; docs/PLAN_records3_convert.md): 0 row-major, 1 native, 2 native+p11;
  // rows per projection, the record's 15 piece offsets (index msb_piece_off + lsb_piece_off), today's row-major sizes
  int layout = 0; int64_t rows3[3] = {0, 0, 0}, loff[15] = {}, rm_msb = 0, rm_len = 0;
  int direct_n = -1;                                // BIGMOE_RECORDS_DIRECT: read the first n pieces straight in, the rest
                                                    // via a bounce (-1 = all straight in; native+p11+pad's LSB part: 3)
  int64_t part_off = 0, part_len = 0;              // fmt 1 on a part of a record (tier pools: records3 MSB or LSB part)
  int pom_fd = -1; int64_t pom_recsz = 0, seg1 = 0, seg2 = 0;   // R-44 b (BIGMOE_POM): records3p beside the records3m part
  int pom_r2 = 0, lp1 = 0, rank_k = 0;              // POM's R2 (base pool) / LP1 (sidecar); ids per routing row (gate ranks)
  std::vector<uint8_t> two;                        // POM: per slot, 1 = 2-plane (seg1 only; never published)
  std::vector<double> lfu; double lfu_w = 1.0;        // BIGMOE_EVICT_LFU: decayed use counts (per gated call)
  std::vector<float> prior;                            // BIGMOE_EVICT_PRIOR: popularity per expert (set_prior)
  int rec_fd = -1; int64_t recsz = 0;               // BIGMOE_RECORDS: one record per expert (9 pieces back to back)  // look-ahead prediction for this layer (from L-1's row)
  // MoonShots scheduler (BIGMOE_PF_JIT): slots reserved for prefetches whose read is not issued yet (ready = 0, in pf),
  // each prefetch's (pass predicted for, class), and this layer's predicted set for the pass (victims skip it)
  std::vector<int32_t> queued;
  std::unordered_map<int32_t, std::pair<int32_t, int>> meta;
  std::vector<int32_t> pmark;                     // expert -> last pass it was predicted for (victims skip this pass's)
  // S10 forecast log: per expert, the pass it was last forecast for and at which distances (bit d-1), the pass a read
  // was reserved for it and from which distance; fpass = last pass any row forecast this layer
  std::vector<int32_t> fseq, rseq; std::vector<uint8_t> fmask, rdist; int32_t fpass = -1;
  // BIGMOE_VTIER (BigMoE 11-03; Leap's PURGE-1 PASS: purgeable Metal memory goes back before compression or swap): a victim
  // tier behind the LRU pool in one purgeable shared MTLBuffer (set_vtier). An evicted expert's bytes stay in its slot as a
  // ghost until a demand load reuses the slot; then they go into the tier (its own LRU), and a later miss on that expert is
  // copied back instead of read from the SSD. The slab is non-volatile only while a call uses it; a previous state of Empty
  // at the pin means macOS purged it, and the tier index is dropped. Exact: a cache only. Every other taker of a free slot
  // (prefetch, JIT reserve, admission) clears the slot's ghost, since it overwrites the bytes.
  std::vector<int32_t> ghost;                    // slot -> evicted expert whose bytes are still in it, or -1
  MTL::Buffer* vt_buf = nullptr; uint8_t* vt_mem = nullptr; int vt_n = 0; int64_t vt_slot = 0;
  std::vector<int32_t> vt_table, vt_owner, vt_free; std::list<int32_t> vt_lru;
  std::vector<std::list<int32_t>::iterator> vt_where; bool vt_pinned = false;
  int64_t vt_h = 0, vt_i = 0;                       // this layer's tier hits / copies in (NewIdeas' per-layer replay)
  // lever 2 (BIGMOE_FC_F / FS): per tier slot 1 = bytes in place (0 while a forecast read writes it), 1 = a forecast's
  // entry not yet used; forecast reads in flight (the slab stays pinned meanwhile); each piece's offset in a tier slot
  std::vector<std::atomic<int>>* vt_rdy = nullptr; std::vector<uint8_t> vt_fc; std::atomic<int>* vt_fly = nullptr;
  std::vector<int64_t> vt_off;
};
static int32_t g_cur_seq = -1;                      // pass being served (for the miss split)

struct Stats { std::atomic<int64_t> uses{0}, misses{0}, stalled{0}, gated{0}; std::atomic<int64_t> io_ns{0}, wait_ns{0}, proc_ns{0};
               std::atomic<int64_t> pf_issued{0}, pf_used{0}, pf_waited_ns{0}, slow_calls{0}, slow_ns{0}, pf_victims{0},
               pf_ontime{0}, pf_late{0}, miss_pred_unissued{0}, miss_unpred{0},    // demand-miss split
               pf_reserved{0}, pf_cancel{0}, pf_qdemand{0}, pf_gated_off{0}, pf_nofit{0}, pub_jit{0}; };   // scheduler

static std::vector<Layer> g_layers;
static std::mutex g_mu;
static Stats g_st;
// demand bursts and the SSD's idle gap before each (MoonShots' open question: do the engine's bursts follow idle gaps?)
static std::atomic<double> g_last_rd_end{0.0};
static std::atomic<int64_t> g_bursts{0}, g_burst_g025{0}, g_burst_g1{0};
static std::atomic<int64_t> g_vt_hits{0}, g_vt_ins{0}, g_vt_purged{0}, g_vt_drops{0};   // BIGMOE_VTIER
static std::atomic<int64_t> g_fc_issued{0}, g_fc_landed{0}, g_fc_hits{0}, g_fc_late{0}, g_fc_wait_ns{0}, g_fc_drops{0},
                            g_fc_skip{0}, g_fc_busy_bursts{0}, g_fc_busy_n{0};   // lever 2 (BIGMOE_FC_F / FS)
static std::atomic<int> g_fc_fly_all{0};
// S10 read / coverage trace (BIGMOE_TRACE): every expert read (demand or prefetch, issue -> bytes in place) and, per gated
// call, how its experts were found (resident, prefetched on time / late, queued read cancelled into demand, demand miss
// predicted-but-unissued / unpredicted). Read with ext.trace_reads() / ext.trace_cov().
static bool g_rtrace = getenv("BIGMOE_TRACE") != nullptr;
struct RdEv { int L, e, tag; double t0, t1, tp0, tp1; };   // tag 0 demand, 1 prefetch, 2 joint; S0: tp0 / tp1 = the pread
struct CovEv { int L, seq, need, hit, pf_ready, pf_late, q_demand, pred_unissued, unpred; double t; };
static std::mutex g_rmu;
static std::vector<RdEv> g_rd;
static std::vector<CovEv> g_cov;
static thread_local int t_rd_tag = 0;
static thread_local double t_p0 = 0, t_p1 = 0;            // S0 (BIGMOE_TRACE): the last read_record's start / end

#include <sys/uio.h>
// one expert record -> 9 destinations with a single preadv (16 KB-aligned record, contiguous pieces); short reads
// continue from where they stopped
static void read_record(int fd, int64_t off, struct iovec* iov, int n) {
  int i = 0;
  if (g_rtrace) t_p0 = now_s();
  while (i < n) {
    ssize_t r = preadv(fd, iov + i, n - i, off);
    if (r <= 0) { fprintf(stderr, "bigmoe: record preadv failed\n"); abort(); }
    off += r;
    while (i < n && r >= (ssize_t)iov[i].iov_len) { r -= iov[i].iov_len; i++; }
    if (i < n && r > 0) { iov[i].iov_base = (uint8_t*)iov[i].iov_base + r; iov[i].iov_len -= r; }
  }
  if (g_rtrace) t_p1 = now_s();
}

// records3 -> 4-bit pieces, bit-exact (exact mode on the plane records): q = 2m + l per weight, s = s3 / 2.
// 32 weights per step: 3 words of 3-bit codes (MLX bits=3, little-endian bitstream) + 1 word of low bits -> 4 words.
static inline uint32_t spread8(uint32_t x, uint32_t l) {   // 8 weights: 24 code bits + 8 low bits -> 8 nibbles
  uint32_t a = (x & 0xFFF) | ((x & 0xFFF000) << 4);
  a = (a & 0x003F003F) | ((a & 0x0FC00FC0) << 2);
  a = (a & 0x07070707) | ((a & 0x38383838) << 1);
  uint32_t y = l & 0xFF; y = (y | (y << 12)) & 0x000F000F; y = (y | (y << 6)) & 0x03030303; y = (y | (y << 3)) & 0x11111111;
  return (a << 1) | y;
}
static void rebuild_w4(const uint32_t* __restrict w3, const uint32_t* __restrict l1, uint32_t* __restrict out, int64_t n32) {
  for (int64_t k = 0; k < n32; k++) {
    uint32_t a = w3[3 * k], b = w3[3 * k + 1], c = w3[3 * k + 2], l = l1[k];
    out[4 * k + 0] = spread8(a & 0xFFFFFF, l);
    out[4 * k + 1] = spread8((a >> 24) | ((b & 0xFFFF) << 8), l >> 8);
    out[4 * k + 2] = spread8((b >> 16) | ((c & 0xFF) << 16), l >> 16);
    out[4 * k + 3] = spread8(c >> 8, l >> 24);
  }
}
static void halve_bf16(const uint16_t* in, uint16_t* out, int64_t n) {   // exact for normals; converter checked s3/2 == s
  for (int64_t i = 0; i < n; i++) { uint16_t h = in[i]; out[i] = (h & 0x7FFF) == 0 ? h : (uint16_t)(h - 0x80); }
}
// R-44 (a) M-01 (BIGMOE_M01=1 on records3m/, scripts/m01_convert.py): records3 with its per-group fields re-packed at a
// fixed width, exact. Each part (MSB-3 or LSB-1) holds the code planes of the 3 projections first, then per projection the
// packed fields, then an 8-byte header ('M', '1', d0 x 3, base x 3). MSB fields of G groups: A[G] = s3 sign | s3 mantissa;
// B[G] = b3 mantissa | (s3 exponent - base) bit 0 << 7; N[G/2] nibbles (2j low, 2j+1 high) = (s3 exponent - base) bits
// 1-2 | (b3 exponent - s3 exponent - d0) << 2; b3's sign is the opposite of s3's (converter-checked). LSB fields:
// A[G] = b sign | b mantissa; E[3G/8] = 3-bit (b exponent - base), 8 per 3 bytes, little-endian. Each part unpacks alone.
static const bool g_m01 = getenv("BIGMOE_M01") != nullptr;
static void m01_s3b3(const uint8_t* A, const uint8_t* B, const uint8_t* N, int64_t G, unsigned base, int d0, uint16_t* s3,
                     uint16_t* b3) {
  for (int64_t j = 0; j < G / 2; j++) {
    const unsigned n = N[j];
    for (int h = 0; h < 2; h++) {
      const int64_t i = 2 * j + h;
      const unsigned nb = (n >> (4 * h)) & 0xF, a = A[i], b = B[i], es = base + ((b >> 7) | ((nb & 3) << 1));
      s3[i] = (uint16_t)(((a & 0x80) << 8) | (es << 7) | (a & 0x7F));
      if (b3) b3[i] = (uint16_t)((((a & 0x80) ^ 0x80) << 8) | (((es + d0 + (nb >> 2)) & 0xFF) << 7) | (b & 0x7F));
    }
  }
}
static void m01_b(const uint8_t* A, const uint8_t* E, int64_t G, unsigned base, uint16_t* out) {
  for (int64_t j = 0; j < G / 8; j++) {
    const uint32_t w = E[3 * j] | ((uint32_t)E[3 * j + 1] << 8) | ((uint32_t)E[3 * j + 2] << 16);
    for (int k = 0; k < 8; k++) {
      const unsigned a = A[8 * j + k], ex = base + ((w >> (3 * k)) & 7);
      out[8 * j + k] = (uint16_t)(((a & 0x80) << 8) | (ex << 7) | (a & 0x7F));
    }
  }
}
struct M01Off { int64_t W3[3], L1[3], G[3], cw[3], cl[3], fm[3], fl[3], hm, hl; };
// One M-01 part (rec_fmt 1 tier pools: 9 pieces = the MSB-3 part, 6 = the LSB-1 part) or full record (rec_fmt 3: rebuilt to
// the 4-bit pieces, as read_expert does for records3) from the bounce buffer into its destinations.
static void m01_expert(const Layer* lp, const uint8_t* rb, uint8_t* const* dst, bool par) {
  const bool full = lp->rec_fmt == 3, lsb_only = !full && lp->pieces.size() == 6;
  M01Off off; int64_t om = 0, ol = 0;
  for (int p = 0; p < 3; p++) {
    if (full) { off.L1[p] = lp->pieces[3 * p].per / 4; off.W3[p] = 3 * off.L1[p]; off.G[p] = lp->pieces[3 * p + 1].per / 2; }
    else if (lsb_only) { off.L1[p] = lp->pieces[2 * p].per; off.W3[p] = 3 * off.L1[p]; off.G[p] = lp->pieces[2 * p + 1].per / 2; }
    else { off.W3[p] = lp->pieces[3 * p].per; off.L1[p] = off.W3[p] / 3; off.G[p] = lp->pieces[3 * p + 1].per / 2; }
    off.cw[p] = om; om += off.W3[p]; off.cl[p] = ol; ol += off.L1[p];
  }
  for (int p = 0; p < 3; p++) {
    off.fm[p] = om; om += 2 * off.G[p] + off.G[p] / 2; off.fl[p] = ol; ol += off.G[p] + 3 * off.G[p] / 8;
  }
  off.hm = om; off.hl = ol;
  const uint8_t* msb = lsb_only ? nullptr : rb;
  const uint8_t* lsb = full ? rb + lp->msb_sz : (lsb_only ? rb : nullptr);
  if ((msb && (msb[om] != 'M' || msb[om + 1] != '1')) || (lsb && (lsb[ol] != 'M' || lsb[ol + 1] != '1'))) {
    fprintf(stderr, "bigmoe: BIGMOE_M01 is set but this record is not M-01 (records3m/)\n"); abort();
  }
  const M01Off* o = &off;
  auto proj = ^(size_t p) {
    const int64_t g = o->G[p];
    if (full) {
      rebuild_w4((const uint32_t*)(msb + o->cw[p]), (const uint32_t*)(lsb + o->cl[p]), (uint32_t*)dst[3 * p],
                 lp->pieces[3 * p].per / 16);
      uint16_t* s = (uint16_t*)dst[3 * p + 1];
      m01_s3b3(msb + o->fm[p], msb + o->fm[p] + g, msb + o->fm[p] + 2 * g, g, msb[o->hm + 5 + p], (int8_t)msb[o->hm + 2 + p],
               s, nullptr);
      halve_bf16(s, s, g);
      m01_b(lsb + o->fl[p], lsb + o->fl[p] + g, g, lsb[o->hl + 5 + p], (uint16_t*)dst[3 * p + 2]);
    } else if (lsb_only) {
      memcpy(dst[2 * p], lsb + o->cl[p], o->L1[p]);
      m01_b(lsb + o->fl[p], lsb + o->fl[p] + g, g, lsb[o->hl + 5 + p], (uint16_t*)dst[2 * p + 1]);
    } else {
      memcpy(dst[3 * p], msb + o->cw[p], o->W3[p]);
      m01_s3b3(msb + o->fm[p], msb + o->fm[p] + g, msb + o->fm[p] + 2 * g, g, msb[o->hm + 5 + p], (int8_t)msb[o->hm + 2 + p],
               (uint16_t*)dst[3 * p + 1], (uint16_t*)dst[3 * p + 2]);
    }
  };
  if (par) dispatch_apply(3, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), proj);
  else for (size_t p = 0; p < 3; p++) proj(p);
}
// R-44 b POM (BIGMOE_POM=R2 with BIGMOE_M01; scripts/m03_convert.py -> records3p/, MoonShots' M-03): a base-pool miss whose
// uses in the call are all at gate rank >= R2 reads only records3p's seg1 (the top two code planes, s3 and the 2-plane bias)
// into a 2-plane slot: 3-bit codes 2 m2 + 0 and bias2 in the b3 piece, which is the Eval suite's tier-2 math exactly
// (s3 * 2 m2 == s4 * 4 m2). A later use at rank < R2 reads seg2 (plane 3 + b3) and upgrades the slot in place; a miss with
// a rank < R2 use reads records3m's MSB part as before. 2-plane slots are never published in the residency tables, so
// OoO's phase 1 never computes them and an upgrade never races the GPU. Prefetches take their mode from the forecast
// (bit 20 of a prediction = predicted rank < R2). BIGMOE_POM_LP1 (MoonShots' M-07): a sidecar miss whose uses are all at
// rank 1 reads nothing and inserts nothing; those uses get slot -2 and the fused GLU runs them at tier 3.
static const int g_pom = getenv("BIGMOE_POM") ? atoi(getenv("BIGMOE_POM")) : 0;
static const bool g_lp1 = getenv("BIGMOE_POM_LP1") != nullptr;
// 0 2-plane reads, 1 full reads (POM layers), 2 upgrades, 3 uses at rank >= R2, 4 of them on 2-plane slots, 5 rank-1
// sidecar uses, 6 of them without the LSB (LP1), 7 2-plane prefetch reservations, 8 sidecar reads skipped (LP1)
static std::atomic<int64_t> g_pom_n[9];
static inline uint32_t spread2to3(uint32_t x) {   // 8 2-bit fields (16 bits) -> 8 3-bit fields holding 2 m (bits 3k+1..3k+2)
  uint32_t a = (x & 0xFFu) | ((x & 0xFF00u) << 4);
  a = (a & 0x00F00Fu) | ((a & 0x0F00F0u) << 2);
  a = (a & 0x0C30C3u) | ((a & 0x30C30Cu) << 1);
  return a << 1;
}
static inline uint32_t spread1to3(uint32_t y) {   // 8 bits -> bit k at 3k
  y &= 0xFFu;
  y = (y & 0x0Fu) | ((y & 0xF0u) << 8);
  y = (y & 0x3003u) | ((y & 0xC00Cu) << 4);
  y = (y & 0x041041u) | ((y & 0x082082u) << 2);
  return y;
}
// q2 planes (2 bits per weight, 4 per byte, little-endian) -> MLX 3-bit codes 2 m2 + 0
static void pom_codes(const uint8_t* __restrict q2, uint8_t* __restrict out, int64_t nw) {
  for (int64_t k = 0; k < nw / 8; k++) {
    const uint32_t c = spread2to3((uint32_t)q2[2 * k] | ((uint32_t)q2[2 * k + 1] << 8));
    out[3 * k] = (uint8_t)c; out[3 * k + 1] = (uint8_t)(c >> 8); out[3 * k + 2] = (uint8_t)(c >> 16);
  }
}
static void pom_or_p3(uint8_t* __restrict codes, const uint8_t* __restrict p3, int64_t nw) {   // upgrade: 2 m2 + p3
  for (int64_t k = 0; k < nw / 8; k++) {
    const uint32_t y = spread1to3(p3[k]);
    codes[3 * k] |= (uint8_t)y; codes[3 * k + 1] |= (uint8_t)(y >> 8); codes[3 * k + 2] |= (uint8_t)(y >> 16);
  }
}
// b3 from seg2's fields: sign = NOT s3's, exponent = s3's + d3 + dd (dd = B3 bit 7 | H bit << 1), mantissa = B3 & 0x7F
static void pom_b3(const uint8_t* B3, const uint8_t* H, const uint16_t* s3, int64_t G, int d3, uint16_t* out) {
  for (int64_t i = 0; i < G; i++) {
    const unsigned b = B3[i], s = s3[i], dd = (b >> 7) | (((unsigned)(H[i >> 3] >> (i & 7)) & 1u) << 1);
    out[i] = (uint16_t)(((s & 0x8000u) ^ 0x8000u) | (((((s >> 7) & 0xFFu) + d3 + dd) & 0xFFu) << 7) | (b & 0x7Fu));
  }
}
struct PomOff { int64_t NW[3], G[3], c1[3], c2[3], f1[3], f2[3], h1, h2; };
static PomOff pom_off(const Layer* lp) {          // seg1 / seg2 layout from the 9 MSB piece sizes (m03_convert.py)
  PomOff o; int64_t a = 0, b = 0;
  for (int p = 0; p < 3; p++) {
    o.NW[p] = lp->pieces[3 * p].per * 8 / 3; o.G[p] = lp->pieces[3 * p + 1].per / 2;
    o.c1[p] = a; a += o.NW[p] / 4; o.c2[p] = b; b += o.NW[p] / 8;
  }
  for (int p = 0; p < 3; p++) { o.f1[p] = a; a += 2 * o.G[p] + o.G[p] / 2; o.f2[p] = b; b += o.G[p] + o.G[p] / 8; }
  o.h1 = a; o.h2 = b;
  return o;
}
// Page-aligned bounce buffers for the one-read-per-expert loads, shared by all threads. They were thread_local, which
// leaked: GCD retires idle workers and a thread_local pointer has no destructor, so a long-lived server lost about one
// buffer per worker per idle gap (NewIdeas / Fit12 R-45T, 10-29). The pool holds at most the peak number of reads in
// flight; a buffer grows to the largest part it has carried, as the thread-local ones did.
static std::mutex g_bounce_mu;
static std::vector<std::pair<uint8_t*, int64_t>> g_bounce;
struct Bounce {
  uint8_t* p = nullptr; int64_t cap = 0;
  explicit Bounce(int64_t len) {
    {
      std::lock_guard<std::mutex> lk(g_bounce_mu);
      if (!g_bounce.empty()) { p = g_bounce.back().first; cap = g_bounce.back().second; g_bounce.pop_back(); }
    }
    if (cap < len) { free(p); if (posix_memalign((void**)&p, 16384, len)) abort(); cap = len; }
  }
  ~Bounce() { std::lock_guard<std::mutex> lk(g_bounce_mu); g_bounce.push_back({p, cap}); }
  Bounce(const Bounce&) = delete;
  Bounce& operator=(const Bounce&) = delete;
};
// One expert's 2-plane read (mode 1: seg1 -> a 2-plane slot) or upgrade (mode 2: seg2 -> the slot's codes and b3)
static void pom_read(const Layer* lp, int e, uint8_t* const* dst, int mode, bool par) {
  const int64_t len = mode == 1 ? lp->seg1 : lp->seg2, off = (int64_t)e * lp->pom_recsz + (mode == 1 ? 0 : lp->seg1);
  Bounce bb(len);
  struct iovec one = {bb.p, (size_t)len};
  read_record(lp->pom_fd, off, &one, 1);
  const PomOff o = pom_off(lp);
  const PomOff* op = &o;
  const uint8_t* rb = bb.p;                        // the blocks capture this pointer (bb is returned when we return)
  const uint8_t* h = rb + (mode == 1 ? o.h1 : o.h2);
  if (h[0] != 'P' || h[1] != (mode == 1 ? '2' : '3')) { fprintf(stderr, "bigmoe: BIGMOE_POM: not a records3p segment\n"); abort(); }
  auto proj = ^(size_t p) {
    const int64_t g = op->G[p];
    if (mode == 1) {
      pom_codes(rb + op->c1[p], dst[3 * p], op->NW[p]);
      const uint8_t* f = rb + op->f1[p];
      m01_s3b3(f, f + g, f + 2 * g, g, h[5 + p], (int8_t)h[2 + p], (uint16_t*)dst[3 * p + 1], (uint16_t*)dst[3 * p + 2]);
    } else {
      pom_or_p3(dst[3 * p], rb + op->c2[p], op->NW[p]);
      const uint8_t* f = rb + op->f2[p];
      pom_b3(f, f + g, (const uint16_t*)dst[3 * p + 1], g, (int8_t)h[2 + p], (uint16_t*)dst[3 * p + 2]);
    }
  };
  if (par) dispatch_apply(3, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), proj);
  else for (size_t p = 0; p < 3; p++) proj(p);
}
static inline int32_t pub_slot(const Layer& ly, int64_t e) {   // the slot a residency table may show: never a 2-plane one
  if (e >= ly.E) return -1;
  const int32_t s = ly.table[e];
  return (s >= 0 && !ly.two.empty() && ly.two[s]) ? -1 : s;
}
// records3 converted in place (Leap's format, 11-03): every piece in NAX6-N's native order, chunk (t*KG + g)*32 + r = the
// row-major chunk (t*32 + r)*KG + g (a chunk = one group of 64 weights: 6 w3 words, 2 l1 words, or one scale). Under p11
// (layout 2) the s3 / b3 / b pieces hold 11-bit codes: block b = t*KG + g is 11 words (32 codes, LSB-first, code r at
// bits 11r..11r+10), word 11 * blocks = the piece's base exponent; bf16 = (c & 0x400) << 5 | ((c & 0x3FF) + (base << 7)).
// native_rowmajor rebuilds today's row-major record (MSB per projection w3 s3 b3, LSB from rm_msb per projection l1 b,
// zero padding), so the rebuild in read_expert stays as it is.
static void unnative(const uint8_t* src, uint8_t* dst, int64_t O, int64_t KG, int64_t chunk) {
  for (int64_t t = 0; t < O / 32; t++)
    for (int64_t g = 0; g < KG; g++)
      for (int64_t r = 0; r < 32; r++)
        memcpy(dst + ((t * 32 + r) * KG + g) * chunk, src + ((t * KG + g) * 32 + r) * chunk, chunk);
}
static void unp11(const uint8_t* src, uint16_t* dst, int64_t O, int64_t KG) {
  const uint32_t* w = (const uint32_t*)src;
  const int64_t nb = O / 32 * KG;
  const uint32_t hdr = w[11 * nb];
  if (hdr >> 8) { fprintf(stderr, "bigmoe: a p11 piece without its header (the store is not native+p11?)\n"); abort(); }
  const uint32_t base = hdr << 7;
  for (int64_t b = 0; b < nb; b++) {
    const uint32_t* q = w + 11 * b;
    uint16_t* o = dst + (b / KG) * 32 * KG + b % KG;
    for (int r = 0; r < 32; r++) {
      const int bit = 11 * r, wi = bit >> 5, sh = bit & 31;
      uint32_t c = q[wi] >> sh;
      if (sh > 21) c |= q[wi + 1] << (32 - sh);
      o[r * KG] = (uint16_t)(((c & 0x400) << 5) | ((c & 0x3FF) + base));
    }
  }
}
static void native_rowmajor(const Layer* lp, const uint8_t* rec, uint8_t* out) {
  int64_t om = 0, ol = lp->rm_msb;
  for (int p = 0; p < 3; p++) {
    const int64_t w4 = lp->pieces[3 * p].per, sb = lp->pieces[3 * p + 1].per, w3 = w4 / 4 * 3, l1 = w4 / 4;
    const int64_t O = lp->rows3[p], KG = sb / 2 / O;
    unnative(rec + lp->loff[3 * p], out + om, O, KG, w3 / (O * KG)); om += w3;
    for (int q = 1; q <= 2; q++, om += sb) {
      if (lp->layout == 2) unp11(rec + lp->loff[3 * p + q], (uint16_t*)(out + om), O, KG);
      else unnative(rec + lp->loff[3 * p + q], out + om, O, KG, 2);
    }
    unnative(rec + lp->loff[9 + 2 * p], out + ol, O, KG, l1 / (O * KG)); ol += l1;
    if (lp->layout == 2) unp11(rec + lp->loff[10 + 2 * p], (uint16_t*)(out + ol), O, KG);
    else unnative(rec + lp->loff[10 + 2 * p], out + ol, O, KG, 2);
    ol += sb;
  }
  memset(out + om, 0, lp->rm_msb - om);
  memset(out + ol, 0, lp->rm_len - ol);
}
// a converted layer's layout for the fmt 3 readers: rows per projection + the 15 piece offsets; today's row-major sizes
static void set_layout_(Layer& ly, int layout, const std::vector<int64_t>& rows, const std::vector<int64_t>& offs) {
  if (layout < 0 || layout > 2 || (layout && (rows.size() != 3 || offs.size() != 15)) || (layout && g_m01)) {
    fprintf(stderr, "bigmoe: bad records3 layout %d (rows %zu, offsets %zu; not with BIGMOE_M01)\n", layout, rows.size(),
            offs.size());
    abort();
  }
  ly.layout = layout;
  if (!layout) return;
  int64_t m = 0, l = 0;
  for (int p = 0; p < 3; p++) {
    const int64_t w4 = ly.pieces[3 * p].per, sb = ly.pieces[3 * p + 1].per;
    ly.rows3[p] = rows[p];
    if (rows[p] % 32 || (sb / 2) % rows[p]) { fprintf(stderr, "bigmoe: layout rows %lld do not tile\n", (long long)rows[p]); abort(); }
    m += w4 / 4 * 3 + 2 * sb; l += w4 / 4 + sb;
  }
  for (int i = 0; i < 15; i++) ly.loff[i] = offs[i];
  ly.rm_msb = (m + 16383) / 16384 * 16384; ly.rm_len = ly.rm_msb + (l + 16383) / 16384 * 16384;
}
// One expert from its record into 9 destinations (gate/up/down x w/s/b, 4-bit layout) via a pooled aligned bounce.
static void read_expert(const Layer* lp, int e, uint8_t* const* dst, bool par = true) {
  int64_t len = lp->part_len ? lp->part_len : lp->recsz;
  Bounce bb(len);
  uint8_t* buf = bb.p;
  struct iovec one = {buf, (size_t)len};
  read_record(lp->rec_fd, (int64_t)e * lp->recsz + lp->part_off, &one, 1);
  if (g_m01 && (lp->rec_fmt == 3 || lp->part_len)) { m01_expert(lp, buf, dst, par); return; }
  if (lp->rec_fmt == 1) {
    // BIGMOE_COPY_PAR (wall 1, MoonShots' M-29: a lone base record's copy tail 122 -> 81 us): with few jobs in the call
    // (par), the pieces go to their slots on three threads, one per projection (3 or 2 pieces each); same bytes, same slots
    static const bool cpar = getenv("BIGMOE_COPY_PAR") != nullptr;
    const size_t np = lp->pieces.size();
    if (cpar && par && np % 3 == 0) {
      const uint8_t* rb = buf;
      dispatch_apply(3, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t p) {
        int64_t o = 0;
        for (size_t i = 0; i < p * (np / 3); i++) o += lp->pieces[i].per;
        for (size_t i = p * (np / 3); i < (p + 1) * (np / 3); i++) { memcpy(dst[i], rb + o, lp->pieces[i].per); o += lp->pieces[i].per; }
      });
      return;
    }
    int64_t o = 0;
    for (size_t i = 0; i < np; i++) { memcpy(dst[i], buf + o, lp->pieces[i].per); o += lp->pieces[i].per; }
    return;
  }
  // records3: MSB part = per proj (w3, s3, b3), LSB part at msb_sz = per proj (l1, b); the 3 projections in parallel.
  // par = false when many experts load at once (b5: nested dispatch_apply under a wide verify miss list starved the
  // loader into spin-gate timeouts): the job's own thread rebuilds all 3 projections.
  const uint8_t* rb = buf;
  int64_t msb = lp->msb_sz;
  std::unique_ptr<Bounce> rmb;                     // a converted layer: back to today's row-major record first
  if (lp->layout) { rmb.reset(new Bounce(lp->rm_len)); native_rowmajor(lp, buf, rmb->p); rb = rmb->p; msb = lp->rm_msb; }
  auto proj = ^(size_t p) {
    int64_t w4 = lp->pieces[3 * p].per, sb = lp->pieces[3 * p + 1].per, w3 = w4 / 4 * 3, l1 = w4 / 4;
    const uint8_t* m = rb + (int64_t)p * (w3 + 2 * sb);
    const uint8_t* l = rb + msb + (int64_t)p * (l1 + sb);
    rebuild_w4((const uint32_t*)m, (const uint32_t*)l, (uint32_t*)dst[3 * p], w4 / 16);
    halve_bf16((const uint16_t*)(m + w3), (uint16_t*)dst[3 * p + 1], sb / 2);
    memcpy(dst[3 * p + 2], l + l1, sb);
  };
  if (par) dispatch_apply(3, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), proj);
  else for (size_t p = 0; p < 3; p++) proj(p);
}

static void vt_copy(const Layer& ly, int s, int t, bool to_tier);   // BIGMOE_VTIER (below, with the tier's helpers)
// modes (R-44 b POM, base pools only): per job 0 = the record part as before, 1 = 2-plane read, 2 = upgrade in place
static void load_jobs(Layer& ly, const std::vector<std::pair<int, int>>& jobs, const std::vector<uint8_t>* modes = nullptr,
                      const std::vector<std::pair<int, int>>* vtj = nullptr, const std::vector<int>* tdst = nullptr) {
  if (ly.rec_fd >= 0) {                             // record path: one read per expert
    Layer* lp = &ly; const auto* jp = &jobs; const std::vector<uint8_t>* mp = modes; const auto* vp = vtj;
    const std::vector<int>* tp = tdst;               // lever 2: job j's bytes go to tier slot (*tp)[j] (pieces as vt_copy)
    static bool direct = getenv("BIGMOE_RECORDS_DIRECT") != nullptr;   // preadv straight into the 9 pool slots
    const int tag = t_rd_tag, Li = (int)(lp - g_layers.data());
    dispatch_apply(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t j) {
      auto [e, s] = (*jp)[j];
      const int ss = s, tt = tp ? (*tp)[j] : -1;
      auto D = [lp, ss, tt](int i) -> uint8_t* {     // piece i's destination: pool slot ss, or tier slot tt
        return tt >= 0 ? lp->vt_mem + (int64_t)tt * lp->vt_slot + lp->vt_off[i] : lp->pieces[i].pool + (int64_t)ss * lp->pieces[i].per; };
      if (vp) {                                     // BIGMOE_VTIER: the slot's ghost into the tier first; a hit copies back
        auto [ts, tv] = (*vp)[j];
        if (tv >= 0) vt_copy(*lp, s, tv, true);
        if (ts >= 0) {                              // lever 2: a forecast read still writing ts -> wait for its bytes
          if (lp->vt_rdy && !(*lp->vt_rdy)[ts].load(std::memory_order_acquire)) {
            const double tw = now_s(); while (!(*lp->vt_rdy)[ts].load(std::memory_order_acquire)) {}
            g_fc_wait_ns += (int64_t)((now_s() - tw) * 1e9);
          }
          vt_copy(*lp, s, ts, false); return;
        }
      }
      const double t_r0 = g_rtrace ? now_s() : 0;
      int n = (int)lp->pieces.size();
      if (direct && lp->rec_fmt == 1 && !(g_m01 && lp->part_len) && lp->direct_n >= 0 && lp->direct_n < n) {
        // a split part (native+p11+pad's LSB: three 4 KB-aligned l1 planes, then the packed b pieces): ONE preadv, the
        // aligned leading pieces straight into their slots, the rest into an aligned bounce, then copied into theirs
        const int dn = lp->direct_n;
        int64_t tail = 0;
        for (int i = dn; i < n; i++) tail += lp->pieces[i].per;
        Bounce tb(tail);
        struct iovec iov[16];
        for (int i = 0; i < dn; i++) { iov[i].iov_base = D(i); iov[i].iov_len = lp->pieces[i].per; }
        iov[dn].iov_base = tb.p; iov[dn].iov_len = (size_t)tail;
        read_record(lp->rec_fd, (int64_t)e * lp->recsz + lp->part_off, iov, dn + 1);
        int64_t o = 0;
        for (int i = dn; i < n; i++) { memcpy(D(i), tb.p + o, lp->pieces[i].per); o += lp->pieces[i].per; }
        if (g_rtrace) { double t_r1 = now_s(); std::lock_guard<std::mutex> lk(g_rmu); g_rd.push_back({Li, e, tag, t_r0, t_r1, t_p0, t_p1}); }
        return;
      }
      if (direct && lp->rec_fmt == 1 && !(g_m01 && lp->part_len)) {   // (M-01 parts unpack from the bounce)
        struct iovec iov[16];
        for (int i = 0; i < n; i++) { iov[i].iov_base = D(i); iov[i].iov_len = lp->pieces[i].per; }
        read_record(lp->rec_fd, (int64_t)e * lp->recsz + lp->part_off, iov, n);
        if (g_rtrace) { double t_r1 = now_s(); std::lock_guard<std::mutex> lk(g_rmu); g_rd.push_back({Li, e, tag, t_r0, t_r1, t_p0, t_p1}); }
        return;
      }
      // default: ONE page-aligned read into a per-thread bounce buffer (Scout: aligned single reads are the fastest
      // pattern; unaligned F_NOCACHE destinations are slow), then the pieces go into their pool slots (records3: rebuilt)
      uint8_t* d[16];
      for (int i = 0; i < n; i++) d[i] = D(i);
      const int md = mp ? (*mp)[j] : 0;
      if (md) pom_read(lp, e, d, md, jp->size() < 4);
      else read_expert(lp, e, d, jp->size() < 4);
      if (g_rtrace) { double t_r1 = now_s(); std::lock_guard<std::mutex> lk(g_rmu); g_rd.push_back({Li, e, tag, t_r0, t_r1, t_p0, t_p1}); }
    });
    return;
  }
  // every (expert, piece) read in flight at once
  size_t n = jobs.size() * ly.pieces.size();
  dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
    auto [e, s] = jobs[i / ly.pieces.size()];
    const Piece& p = ly.pieces[i % ly.pieces.size()];
    int64_t done = 0;
    while (done < p.per) {
      ssize_t r = pread(p.fd, p.pool + (int64_t)s * p.per + done, p.per - done, p.off + (int64_t)e * p.per + done);
      if (r <= 0) { fprintf(stderr, "bigmoe: pread failed L? e=%d\n", e); abort(); }
      done += r;
    }
  });
}

static int g_spare = getenv("BIGMOE_SPARE") ? atoi(getenv("BIGMOE_SPARE")) : 0;
static bool g_pf_publish = getenv("BIGMOE_PF_NOPUBLISH") == nullptr;
// BIGMOE_PF_VICTIM=N (MoonShots): while layer L stalls (GPU held at L's gate, so it has not read L+1/L+2's tables),
// prefetch predicted experts into the slots of LRU-tail victims, up to N outstanding per layer, with no spare pool.
// The victim leaves the CPU table and this pass's published table *before* L is released, so the GPU never reads
// a slot whose bytes are being replaced; a wrong guess only evicts the LRU tail early.
static int g_pf_victim = getenv("BIGMOE_PF_VICTIM") ? atoi(getenv("BIGMOE_PF_VICTIM")) : 0;
// BIGMOE_PF_SIDE=m (BigMoE 11-02, R-57 at 45 GB: the 4-bit LSB sidecars take 60-70 % of the misses): the non-JIT look-ahead
// also prefetches L+1's sidecar planes for the m most likely ids of its ranked list (rank 0-1 experts sit near the top),
// into the sidecar's spare slots (BIGMOE_SPARE). Needs tier families (ext.set_families); 0 = off, the default.
static int g_pf_side = getenv("BIGMOE_PF_SIDE") ? atoi(getenv("BIGMOE_PF_SIDE")) : 0;
// Replay oracle (MoonShots runahead bound): BIGMOE_ORACLE_REC=file records every gated (pass, layer)'s unique ids;
// BIGMOE_ORACLE_PLAY=file replays them: while L stalls, L+1's *true* ids go to the victim prefetch, with the reads
// issued BIGMOE_ORACLE_DELAY_US after (the realistic runahead latency). Same tokens and passes -> same (seq, L) keys.
static const char* g_orec = getenv("BIGMOE_ORACLE_REC");
static const char* g_oplay = getenv("BIGMOE_ORACLE_PLAY");
static int g_odelay_us = getenv("BIGMOE_ORACLE_DELAY_US") ? atoi(getenv("BIGMOE_ORACLE_DELAY_US")) : 0;
static std::unordered_map<int64_t, std::vector<int32_t>> g_or;
static int g_fam0 = 0;                               // tier layouts: first sidecar row (ext.set_families), 0 = one family
static std::vector<uint8_t*> g_stage[2];
static int g_stage_cap = 0;
static dispatch_group_t g_sgrp[2] = {nullptr, nullptr};
static void oracle_save() {
  FILE* f = fopen(g_orec, "wb"); if (!f) return;
  for (auto& [k, v] : g_or) { int64_t n = v.size(); fwrite(&k, 8, 1, f); fwrite(&n, 8, 1, f); fwrite(v.data(), 4, n, f); }
  fclose(f); fprintf(stderr, "[bigmoe] oracle: %zu (pass, layer) rows -> %s\n", g_or.size(), g_orec);
}
static void oracle_load() {
  FILE* f = fopen(g_oplay, "rb"); if (!f) { fprintf(stderr, "bigmoe: no oracle file %s\n", g_oplay); abort(); }
  int64_t k, n;
  while (fread(&k, 8, 1, f) == 1 && fread(&n, 8, 1, f) == 1) { std::vector<int32_t> v(n); fread(v.data(), 4, n, f); g_or[k] = v; }
  fclose(f); fprintf(stderr, "[bigmoe] oracle: %zu rows from %s\n", g_or.size(), g_oplay);
}

// ---------------------------------------------------------------- MoonShots prefetch scheduler (all flag-gated, off by default)
// BIGMOE_PF_JIT=1: publish layer X's residency table for pass s just in time, when the CPU serves X-1 of pass s, instead
//   of right after serving X in pass s-1 (layer 0 is never published: always the gate). Between X's use in one pass and
//   that publication no table the GPU will read references X's slots, so a prefetch may take X's LRU-tail slot (or
//   recycle a finished prefetch) at ANY time, not only while some layer stalls. Rules, all under g_mu: slots are only
//   reserved for layers X > the layer being served, before X is published; a published layer loses no slot until it is
//   served. Reserving happens when a prediction arrives; the read is issued later, nearest layer first.
// BIGMOE_PF_QD=n        at most n speculative reads in flight (0 = issue at once). Nothing is issued while a demand
//                       read runs (the loader thread is the only issuer), so demand reads always go first.
// BIGMOE_PF_THETA=p     feedback throttle: a prediction class (distance x confidence bucket) whose measured precision
//                       (used when its layer is served / reserved) is below p is not reserved (1 in 16 still is).
// BIGMOE_PF_BUDGET=n    at most n reservations per pass.
// BIGMOE_PF_EARLY=1     reserve and issue before serving the layer's own demand reads (default: right after release).
// BIGMOE_ORACLE_DIST=D  with BIGMOE_ORACLE_PLAY: serving L reserves the recorded true ids of layers L+1..L+D (bound).
// Per-layer cap of prefetches in flight or unused: spare + BIGMOE_PF_VICTIM (default 12 in JIT mode).
// BIGMOE_PRED_LINES=k (0..4, default 1 = the 4-line row): k prediction lines after line 3 ([seq, 31 ids] each), so a row
// carries 30 + 31k predictions (k = 3: 123, e.g. top-12 of 4 verify rows for L+1 plus L+2/L+3). gated.py sizes rows alike.
static int g_plines = std::max(0, std::min(4, getenv("BIGMOE_PRED_LINES") ? atoi(getenv("BIGMOE_PRED_LINES")) : 1));
// BIGMOE_MAXK > 61 (verify S=7..9 x top-10): one more id line at the end of the row ([seq, ids61..91]) and release-region
// entries of 3 lines (slots 0..30 | 31..61 | 62..92). The default (MAXK <= 61) layout is unchanged.
// R-55 GATE16 (MAXK 93..185): k = 2..4 id lines ([seq, ids61..91] [seq, ids92..122] ...) and entries of 2 + k lines, so a
// 16-row check (16 x top-10 = 160 ids, k = 4) takes the sync-free gate. MAXK <= 92 keeps the layout above.
static int xlines_of(const char* s) { int k = s ? atoi(s) : 0; return k > 61 ? std::min(4, (k - 61 + 30) / 31) : 0; }
static int g_xlines = xlines_of(getenv("BIGMOE_MAXK"));
static int g_rs = 32 * (2 + g_xlines);                // ints per release-region entry (64 | 96 | ... | 192)
static int g_row = 96 + 32 * g_plines + 32 * g_xlines, g_pcap = 30 + 31 * g_plines;
static inline int ipos(int i) { return i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * g_plines + 32 * ((i - 61) / 31) + 1 + (i - 61) % 31; }
// Gate handoff stamps (MoonShots ~/MoonShots/patches/gate_stamps, 10-25): bits 22-31 of every handoff word carry
// seq & 0x3FF (routing-row ids and preds GPU -> loader; release regions and residency tables loader -> GPU). A 128-byte
// line carries its seq tag only in word 0, and under another process's GPU load a reader saw a fresh tag next to a stale
// 64-byte sector (the loader read the previous post's ids from words 16+ and computed with the wrong experts). Readers
// check every word's stamp. Payloads fit in bits 0-21 (expert ids, slots; -1 travels as 0x3FFFFF). 10 bits = 1024
// passes, far beyond how old a stale word can be.
static inline uint32_t stamp_of(int32_t seq) { return ((uint32_t)seq & 0x3FFu) << 22; }
static inline int32_t stamp_enc(int32_t v, int32_t seq) { return (int32_t)(((uint32_t)v & 0x3FFFFFu) | stamp_of(seq)); }
static inline bool stamp_ok(int32_t w, int32_t seq) { return ((uint32_t)w & 0xFFC00000u) == stamp_of(seq); }
static inline int32_t stamp_dec(int32_t w) { int32_t v = (int32_t)((uint32_t)w & 0x3FFFFFu); return v == 0x3FFFFF ? -1 : v; }
static std::atomic<int64_t> g_stamp_calls{0}, g_stamp_rereads{0}, g_stamp_ns{0};
static inline int pidx(int i) { return i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31; }   // prediction i's int
static bool g_jit = getenv("BIGMOE_PF_JIT") != nullptr;
static int g_qd = getenv("BIGMOE_PF_QD") ? atoi(getenv("BIGMOE_PF_QD")) : 0;
static double g_theta = getenv("BIGMOE_PF_THETA") ? atof(getenv("BIGMOE_PF_THETA")) : 0.0;
static int g_budget = getenv("BIGMOE_PF_BUDGET") ? atoi(getenv("BIGMOE_PF_BUDGET")) : 0;
static bool g_early = getenv("BIGMOE_PF_EARLY") != nullptr;
static int g_odist = getenv("BIGMOE_ORACLE_DIST") ? atoi(getenv("BIGMOE_ORACLE_DIST")) : 0;
struct PfReq { int L; int32_t e, seq, s; float prec; int d; uint8_t mode = 0; };   // mode: POM read (load_jobs)
static std::vector<PfReq> g_pq;                      // reserved, not issued (under g_mu)
static std::atomic<int> g_inflight{0}, g_pq_n{0};
// BIGMOE_PF_QD_FAR=n (S10, NewIdeas): per-distance queue: reads forecast for the next layer (d = 1) bypass QD, deeper
// ones (d >= 2) at most n in flight (0 = no limit). Unset: one queue, QD for all (the default, unchanged).
static int g_qd_far = getenv("BIGMOE_PF_QD_FAR") ? atoi(getenv("BIGMOE_PF_QD_FAR")) : -1;
static std::atomic<int> g_inflight_far{0}, g_pq_near{0};
// S10 per-distance log (JIT): misses at layers that had forecasts this pass; of them, forecast at distance d (raw, before
// any gate), forecast >= 2 layers ahead; reads issued per distance and how many of them their layer then needed
static std::atomic<int64_t> g_fc_miss{0}, g_fc_lead2{0}, g_fc_d[4], g_iss_d[4], g_iss_used_d[4], g_fc_u[4], g_fc_ev[4];
static inline bool pq_can(const PfReq& q) {
  if (g_qd_far < 0) return g_qd <= 0 || g_inflight.load() < g_qd;
  return q.d <= 1 || g_qd_far == 0 || g_inflight_far.load() < g_qd_far;
}
static inline bool pq_ready() {                    // anything issuable now (loader idle loop)
  if (g_qd_far < 0) return g_qd <= 0 || g_inflight.load() < g_qd;
  return g_pq_near.load() > 0 || g_qd_far == 0 || g_inflight_far.load() < g_qd_far;
}
static inline void mark_resize(Layer& ly) {
  if ((int)ly.fseq.size() < ly.E) { ly.fseq.assign(ly.E, -1); ly.rseq.assign(ly.E, -1); ly.fmask.assign(ly.E, 0); ly.rdist.assign(ly.E, 0); }
}
static std::vector<int32_t> g_pub, g_srv;            // per layer: pass its table was last published for / last served in
static int32_t g_pass = -1; static int g_pass_res = 0, g_or_upto = -1;
constexpr int NCLS = 10;                             // (distance 1..3) x (confidence bucket 0..2), + oracle
static double g_cls_iss[NCLS], g_cls_used[NCLS];
static int64_t g_cls_res[NCLS], g_cls_skip[NCLS];
static double cls_prec(int c) { return (g_cls_used[c] + 1.0) / (g_cls_iss[c] + 2.0); }
// S10b trace (NewIdeas QUEUE.md row S10b): BIGMOE_S10B_MISS=file records, for every (pass, layer) the loader serves, its
// unique ids, each flagged if it was not resident when the row was seen (the S10 log's miss: a prefetched expert not yet
// promoted counts as a miss). int32 stream of [seq, L, n, (e | miss << 16 | kind << 17) x n], written at exit. Off by
// default. kind (b33, S10c) splits a miss as ensure() is about to see it: 0 demand (or resident), 1 prefetched with its
// bytes ready (on time), 2 prefetch issued but not ready (late), 3 prefetch still queued (ensure() cancels it to demand).
static const char* g_s10b = getenv("BIGMOE_S10B_MISS");
static std::vector<int32_t> g_s10b_v;
static void s10b_save() {
  FILE* f = fopen(g_s10b, "wb"); if (!f) return;
  fwrite(g_s10b_v.data(), 4, g_s10b_v.size(), f); fclose(f);
  fprintf(stderr, "[bigmoe] s10b: %zu ints -> %s\n", g_s10b_v.size(), g_s10b);
}
static void s10b_rec(int L, int32_t seq, const int32_t* ids, int n) {   // caller holds g_mu
  static bool reg = (std::atexit(s10b_save), true); (void)reg;
  Layer& ly = g_layers.at(L);
  size_t h = g_s10b_v.size();
  g_s10b_v.push_back(seq); g_s10b_v.push_back(L); g_s10b_v.push_back(0);
  for (int i = 0; i < n; i++) {
    int32_t e = ids[i]; bool dup = e < 0 || e >= ly.E;          // a bad id aborts in ensure() right after
    for (size_t j = h + 3; j < g_s10b_v.size() && !dup; j++) dup = (g_s10b_v[j] & 0xFFFF) == e;
    if (dup) continue;
    int k = 0;
    if (ly.table[e] < 0) {
      auto pit = ly.pf.find(e);
      if (std::find(ly.queued.begin(), ly.queued.end(), e) != ly.queued.end()) k = 3;
      else if (pit != ly.pf.end()) k = (*ly.ready)[pit->second].load(std::memory_order_acquire) ? 1 : 2;
    }
    g_s10b_v.push_back(e | ((ly.table[e] < 0 ? 1 : 0) << 16) | (k << 17));
  }
  g_s10b_v[h + 2] = (int32_t)(g_s10b_v.size() - h - 3);
}
static void pq_drop(int L, int32_t e) {
  for (size_t i = 0; i < g_pq.size(); i++)
    if (g_pq[i].L == L && g_pq[i].e == e) { if (g_pq[i].d <= 1) g_pq_near--; g_pq[i] = g_pq.back(); g_pq.pop_back(); break; }
  g_pq_n = (int)g_pq.size();
}

static bool in(const std::vector<int32_t>& v, int32_t e) { return std::find(v.begin(), v.end(), e) != v.end(); }

// BIGMOE_EVICT_LFU=d (NewIdeas I9; trace replay on v3 at 24 GB: d = 0.99 -> -8..-11 % demand misses vs LRU): victims are
// the lowest decayed use count (each gated call adds w, w /= d per call) instead of the LRU tail. Residency only: exact.
static double g_lfu_decay = getenv("BIGMOE_EVICT_LFU") ? atof(getenv("BIGMOE_EVICT_LFU")) : 0.0;
// BIGMOE_EVICT_PRIOR=b (SEED45, MoonShots M-13's pointer; cap45 replay at C 224-256: LRU + prior b 30 -> -9.4..-9.8 %
// misses per kept token vs LRU): victims are the lowest last-use stamp (gated calls of the layer) + b * prior[e] (the
// share of calibration checks that needed e at this layer, set_prior); with BIGMOE_EVICT_LFU the decayed count + b * w *
// prior[e]. Residency only: exact. 0 (default) keeps the LRU tail / plain LFU.
static double g_prior_b = getenv("BIGMOE_EVICT_PRIOR") ? atof(getenv("BIGMOE_EVICT_PRIOR")) : 0.0;
static inline bool prior_on(Layer& ly) {             // lfu holds the stamps (LRU) or counts (LFU) the victims read
  if (g_prior_b <= 0) return false;
  if (ly.lfu.empty()) ly.lfu.assign(ly.E, 0.0);
  return true;
}
static std::list<int32_t>::iterator lfu_victim(Layer& ly, const std::vector<int32_t>* need, int skip_seq) {
  thread_local std::vector<uint8_t> mark;           // need as a byte mask (in() per candidate is too slow here)
  if ((int)mark.size() < ly.E) mark.assign(ly.E, 0);
  if (need) for (int e : *need) mark[e] = 1;
  auto best = ly.lru.end(); double bs = 0;
  for (auto it = ly.lru.begin(); it != ly.lru.end(); ++it) {
    if (mark[*it]) continue;
    if (skip_seq >= 0 && ly.pmark[*it] == skip_seq) continue;
    double sc = ly.lfu[*it];
    if (g_prior_b > 0 && !ly.prior.empty()) sc += g_prior_b * (g_lfu_decay > 0 ? ly.lfu_w : 1.0) * ly.prior[*it];
    if (best == ly.lru.end() || sc < bs) { best = it; bs = sc; }
  }
  if (need) for (int e : *need) mark[e] = 0;
  return best;
}

// keep at most C_main experts resident: evict LRU experts not needed now (their slots become free)
static void make_room(Layer& ly, const std::vector<int32_t>& need) {
  // prefetches beyond the spare pool (victim mode) occupy main slots
  while ((int)ly.lru.size() + std::max(0, (int)ly.pf.size() - ly.spare) >= ly.C_main && !ly.lru.empty()) {
    auto it = ly.lru.end();
    bool found = false;
    if (!ly.lfu.empty()) { it = lfu_victim(ly, &need, -1); found = it != ly.lru.end(); }
    else while (it != ly.lru.begin()) { --it; if (!in(need, *it)) { found = true; break; } }
    if (!found) break;                               // everything resident is needed now: the pf-drop path frees a slot
    int v = *it; ly.lru.erase(it); int s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1; ly.free_slots.push_back(s);
    if (ly.vt_n) ly.ghost[s] = v;                     // BIGMOE_VTIER: v's bytes stay in s until the slot is reused
  }
}
static void vt_pin(Layer& ly) {                     // BIGMOE_VTIER: non-volatile for this call; purged -> drop the index
  if (!ly.vt_n || ly.vt_pinned) return;
  if (ly.vt_buf->setPurgeableState(MTL::PurgeableStateNonVolatile) == MTL::PurgeableStateEmpty) {
    for (int32_t e : ly.vt_lru) ly.vt_table[e] = -1;
    ly.vt_lru.clear(); ly.vt_free.clear();
    for (int t = ly.vt_n - 1; t >= 0; t--) { ly.vt_owner[t] = -1; ly.vt_free.push_back(t); }
    g_vt_purged++;
  }
  ly.vt_pinned = true;
}
static void vt_unpin(Layer& ly) {
  if (!ly.vt_pinned) return;
  if (ly.vt_fly && ly.vt_fly->load(std::memory_order_acquire) > 0) return;   // lever 2: a forecast read still writes
  ly.vt_buf->setPurgeableState(MTL::PurgeableStateVolatile); ly.vt_pinned = false;
}
static int vt_take(Layer& ly, const std::vector<int32_t>& busy) {   // a tier slot: a free one, else drop the tier's
  if (!ly.vt_free.empty()) { int t = ly.vt_free.back(); ly.vt_free.pop_back(); return t; }   // oldest not copied now
  for (auto it = ly.vt_lru.end(); it != ly.vt_lru.begin();) {
    --it; int e = *it, t = ly.vt_table[e];
    if (std::find(busy.begin(), busy.end(), t) != busy.end()) continue;
    if (ly.vt_rdy && !(*ly.vt_rdy)[t].load(std::memory_order_acquire)) continue;   // lever 2: still being written
    if (!ly.vt_fc.empty() && ly.vt_fc[t]) { ly.vt_fc[t] = 0; g_fc_drops++; }   // a forecast never used
    ly.vt_lru.erase(it); ly.vt_table[e] = -1; ly.vt_owner[t] = -1; g_vt_drops++; return t;
  }
  return -1;
}
static void vt_copy(const Layer& ly, int s, int t, bool to_tier) {   // pool slot s <-> tier slot t, piece by piece
  uint8_t* base = ly.vt_mem + (int64_t)t * ly.vt_slot;
  for (const auto& pc : ly.pieces) {
    uint8_t* slot = pc.pool + (int64_t)s * pc.per;
    if (to_tier) memcpy(base, slot, pc.per); else memcpy(slot, base, pc.per);
    base += pc.per;
  }
}

// ids -> slots (in place into out). Loads misses (blocking). Caller holds g_mu.
static std::atomic<int64_t> g_qom_skip{0}, g_qom_opt{0};   // QOM: optional-only misses not read / optional uses seen
static void ensure(int L, const int32_t* ids_in, int n, int32_t* out) {
  Layer& ly = g_layers.at(L);
  // QOM (BigMoE Wed 11-04; Breakthrough's keep-set rule, ~/Breakthrough/keepset): bit 21 of an id marks an OPTIONAL use (a
  // sure row's rank 2-9 expert outside the check's loads). An expert whose every use is optional is served from the pool
  // when resident (or a prefetch), else not read at all: its slot comes back -2 and the GPU drops it (weight 0). No bit 21,
  // no change. The id is the low 20 bits.
  std::vector<int32_t> ids_m(ids_in, ids_in + n); bool anyopt = false;
  for (int i = 0; i < n; i++) if ((ids_m[i] >> 21) & 1) { anyopt = true; ids_m[i] &= ~(1 << 21); }
  const int32_t* ids = ids_m.data();
  std::vector<uint8_t> req;                        // QOM: per needed expert, 1 = some use is not optional
  int cov_q = 0;                                    // BIGMOE_TRACE: needed experts whose queued prefetch was cancelled
  std::vector<int32_t> need; need.reserve(n);
  const bool rk = (ly.pom_r2 || ly.lp1) && ly.rank_k > 0;   // R-44 b: each needed expert's best gate rank in this call
  std::vector<int8_t> mr;                         // (the ids are rank-ordered rows of rank_k)
  for (int i = 0; i < n; i++) {
    int e = ids[i];
    if (e < 0 || e >= ly.E) { fprintf(stderr, "bigmoe: bad expert id %d at layer %d\n", e, L); abort(); }
    auto it = std::find(need.begin(), need.end(), e);
    if (it == need.end()) { need.push_back(e); if (rk) mr.push_back(127); if (anyopt) req.push_back(0); it = need.end() - 1; }
    if (anyopt && !((ids_in[i] >> 21) & 1)) req[it - need.begin()] = 1;
    if (rk) { int8_t& m = mr[it - need.begin()]; m = (int8_t)std::min<int>(m, i % ly.rank_k); }
  }
  if ((int)need.size() > ly.C_main) { fprintf(stderr, "bigmoe: %zu experts > C=%d at layer %d\n", need.size(), ly.C_main, L); abort(); }
  if (g_lfu_decay > 0) {
    if (ly.lfu.empty()) ly.lfu.assign(ly.E, 0.0);
    ly.lfu_w /= g_lfu_decay;
    if (ly.lfu_w > 1e200) { for (auto& v : ly.lfu) v /= ly.lfu_w; ly.lfu_w = 1.0; }
    for (int e : need) ly.lfu[e] += ly.lfu_w;
  } else if (prior_on(ly)) {                        // BIGMOE_EVICT_PRIOR on LRU: last-use stamps, one tick per call
    ly.lfu_w += 1.0;
    for (int e : need) ly.lfu[e] = ly.lfu_w;
  }
  if (g_jit && !ly.fseq.empty() && ly.fpass == g_cur_seq) {   // S10 log: this layer had forecasts this pass (fpass and
                                                 // g_cur_seq both start at -1: the pool warm-up's ensure() calls)
    for (int e : need) {
      if (ly.table[e] >= 0) continue;
      g_fc_miss++;
      uint8_t m = ly.fseq[e] == g_cur_seq ? ly.fmask[e] : 0;
      for (int d = 1; d <= 3; d++) if (m & (1u << (d - 1))) g_fc_d[d]++;
      if (m & ~1u) g_fc_lead2++;
      if (ly.rseq[e] == g_cur_seq && ly.pf.count(e) && !in(ly.queued, e)) g_iss_used_d[ly.rdist[e]]++;   // issued, used
    }
  }
  if (g_jit) {
    // feedback: score this pass's predictions for L (reserved, issued or not) against what L really needs
    for (auto it = ly.meta.begin(); it != ly.meta.end();) {
      if (it->second.first == g_cur_seq) {
        int c = it->second.second;
        if (g_cls_iss[c] > 4096) { g_cls_iss[c] *= 0.5; g_cls_used[c] *= 0.5; }
        g_cls_iss[c] += 1; if (in(need, it->first)) g_cls_used[c] += 1;
      }
      if (it->second.first <= g_cur_seq) it = ly.meta.erase(it); else ++it;
    }
    // reads still queued: cancel them; a needed one becomes an ordinary demand miss below (its slot is free again)
    for (int32_t e : ly.queued) {
      if (g_rtrace && in(need, e)) cov_q++;
      int s = ly.pf[e]; pq_drop(L, e);
      ly.pf.erase(e); ly.pf_order.remove(e); ly.free_slots.push_back(s);
      if (in(need, e)) g_st.pf_qdemand++; else g_st.pf_cancel++;
    }
    ly.queued.clear();
  }
  std::vector<std::pair<int, int>> jobs;
  std::vector<uint8_t> modes;                     // POM: per job 0 full, 1 2-plane, 2 upgrade
  std::vector<std::pair<int, int>> vtj;           // BIGMOE_VTIER: per job (tier slot copied back, tier slot the ghost goes to)
  std::vector<int32_t> vt_busy, vt_out;           // tier slots copied this call (no other copy may take them) / freed after
  int n_miss = 0, c_hit = 0, c_rdy = 0, c_late = 0, c_pu = 0, c_un = 0;
  for (size_t j = 0; j < need.size(); j++) {
    const int e = need[j];
    const bool full = !ly.pom_r2 || mr[j] < ly.pom_r2;   // POM: a use at rank < R2 needs all three planes
    if (ly.table[e] >= 0) {
      c_hit++;
      if (ly.pom_r2 && full && ly.two[ly.table[e]]) { jobs.push_back({e, ly.table[e]}); modes.push_back(2); vtj.push_back({-1, -1}); }
      continue;
    }
    if (ly.lp1 && rk && mr[j] >= 1 && !ly.pf.count(e)) continue;   // LP1: rank-1 uses only -> no read, no insert
    if (anyopt && !req[j] && !ly.pf.count(e)) { g_qom_skip++; continue; }   // QOM: optional-only miss: not read (-2)
    n_miss++;
    auto pit = ly.pf.find(e);
    if (pit != ly.pf.end()) {                       // prefetched: wait for its bytes, then promote
      int s = pit->second;
      if ((*ly.ready)[s].load(std::memory_order_acquire)) { g_st.pf_ontime++; c_rdy++; } else { g_st.pf_late++; c_late++; }
      double t = now_s(); while (!(*ly.ready)[s].load(std::memory_order_acquire)) {} g_st.pf_waited_ns += (int64_t)((now_s() - t) * 1e9);
      ly.pf.erase(pit); ly.pf_order.remove(e);
      make_room(ly, need);
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
      g_st.pf_used++;
      if (ly.pom_r2 && full && ly.two[s]) { jobs.push_back({e, s}); modes.push_back(2); vtj.push_back({-1, -1}); }   // a 2-plane prefetch
      continue;
    }
    make_room(ly, need);
    if (ly.free_slots.empty()) {                    // all free slots hold prefetches: drop the oldest one not needed now
      for (int pass = 0; pass < 2 && ly.free_slots.empty(); pass++)   // pass 0: finished ones; pass 1: wait for one in flight
        for (auto it = ly.pf_order.begin(); it != ly.pf_order.end(); ++it) {
          if (in(need, *it)) continue;
          int s = ly.pf[*it];
          if (pass == 1) while (!(*ly.ready)[s].load(std::memory_order_acquire)) {}
          if ((*ly.ready)[s].load()) { ly.pf.erase(*it); ly.pf_order.erase(it); ly.free_slots.push_back(s); break; }
        }
      if (ly.free_slots.empty()) { fprintf(stderr, "bigmoe: no free slot at layer %d\n", L); abort(); }
    }
    int s = ly.free_slots.back(); ly.free_slots.pop_back();
    int ts = -1, tv = -1;
    if (ly.vt_n) {                                  // BIGMOE_VTIER: a tier hit copies back; the slot's ghost goes in first
      vt_pin(ly);
      if (ly.vt_table[e] >= 0) {
        ts = ly.vt_table[e]; ly.vt_table[e] = -1; ly.vt_owner[ts] = -1; ly.vt_lru.erase(ly.vt_where[e]);
        vt_busy.push_back(ts); vt_out.push_back(ts); g_vt_hits++; ly.vt_h++;
        if (!ly.vt_fc.empty() && ly.vt_fc[ts]) {      // lever 2: a forecast read's entry used (landed or still landing)
          ly.vt_fc[ts] = 0; g_fc_hits++;
          if (ly.vt_rdy && !(*ly.vt_rdy)[ts].load(std::memory_order_acquire)) g_fc_late++;
        }
      }
      const int v = ly.ghost[s];                     // (never one needed now: this call loads those itself)
      if (v >= 0 && ly.table[v] < 0 && ly.vt_table[v] < 0 && !in(need, v) && (tv = vt_take(ly, vt_busy)) >= 0) {
        ly.vt_table[v] = tv; ly.vt_owner[tv] = v; ly.vt_lru.push_front(v); ly.vt_where[v] = ly.vt_lru.begin();
        vt_busy.push_back(tv); g_vt_ins++; ly.vt_i++;
      }
      ly.ghost[s] = -1;
    }
    vtj.push_back({ts, tv});
    if (ly.pred_seq == g_cur_seq && in(ly.pred, e)) { g_st.miss_pred_unissued++; c_pu++; } else { g_st.miss_unpred++; c_un++; }
    ly.table[e] = s; ly.slot_owner[s] = e; jobs.push_back({e, s}); modes.push_back(ly.pom_r2 && !full ? 1 : 0);
    if (ly.pom_r2) ly.two[s] = !full;
    ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
  }
  g_st.uses += need.size(); g_st.misses += n_miss; g_st.stalled += !jobs.empty();
  if (anyopt) for (int i = 0; i < n; i++) g_qom_opt += (ids_in[i] >> 21) & 1;
  if (g_rtrace) { std::lock_guard<std::mutex> lk(g_rmu);
                  g_cov.push_back({L, g_cur_seq, (int)need.size(), c_hit, c_rdy, c_late, cov_q, c_pu, c_un, now_s()}); }
  if (!jobs.empty()) {
    double t = now_s(), gap = t - g_last_rd_end.load(std::memory_order_relaxed);   // the SSD's idle time before this burst
    g_bursts++; if (gap >= 250e-6) g_burst_g025++; if (gap >= 1e-3) g_burst_g1++;
    { const int fl = g_fc_fly_all.load(std::memory_order_relaxed); if (fl > 0) { g_fc_busy_bursts++; g_fc_busy_n += fl; } }
    load_jobs(ly, jobs, ly.pom_r2 ? &modes : nullptr, ly.vt_n ? &vtj : nullptr);
    double t1 = now_s(); g_st.io_ns += (int64_t)((t1 - t) * 1e9); g_last_rd_end.store(t1, std::memory_order_relaxed);
  }
  if (ly.vt_n) { for (int t : vt_out) ly.vt_free.push_back(t); vt_unpin(ly); }   // BIGMOE_VTIER: volatile again
  if (ly.pom_r2)
    for (size_t k = 0; k < jobs.size(); k++) {
      if (modes[k] == 2) ly.two[jobs[k].second] = 0;
      g_pom_n[modes[k] == 1 ? 0 : modes[k] == 2 ? 2 : 1]++;
    }
  for (int e : need) { if (ly.table[e] < 0) continue; ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); }
  for (int i = 0; i < n; i++) out[i] = ly.table[ids[i]] >= 0 ? ly.table[ids[i]] : -2;   // -2: LP1, no LSB (tier 3)
  if (ly.pom_r2 && rk)
    for (int i = 0; i < n; i++) if (i % ly.rank_k >= ly.pom_r2) { g_pom_n[3]++; if (ly.two[ly.table[ids[i]]]) g_pom_n[4]++; }
  if (ly.lp1 && rk)
    for (int i = 0; i < n; i++) if (i % ly.rank_k == 1) { g_pom_n[5]++; if (ly.table[ids[i]] < 0) g_pom_n[6]++; }
}

// tables are declared later; prefetch adds ready experts to the table the GPU reads for `seq`
static void table_add(int L, int seq, int e, int s);
static void table_remove(int L, int seq, int e);
// start async reads of predicted experts into spare slots. Caller holds g_mu.
static void prefetch(int L, const int32_t* ids, int n, int seq, bool victim_ok = false) {
  Layer& ly = g_layers.at(L);
  int cap = ly.spare + (victim_ok && ly.C_main < ly.E ? g_pf_victim : 0);
  if (cap <= 0) return;
  for (int i = 0; i < n; i++) {
    int e = ids[i];
    if (e < 0 || e >= ly.E || ly.table[e] >= 0 || ly.pf.count(e)) continue;
    if ((int)ly.pf.size() >= cap) {                 // recycle the oldest finished prefetch
      bool freed = false;
      for (auto it = ly.pf_order.begin(); it != ly.pf_order.end(); ++it) {
        int s = ly.pf[*it];
        if ((*ly.ready)[s].load()) { ly.pf.erase(*it); ly.pf_order.erase(it); ly.free_slots.push_back(s); freed = true; break; }
      }
      if (!freed) return;
    }
    int s;
    if (!ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); if (ly.vt_n) ly.ghost[s] = -1; }
    else if (victim_ok && g_pf_victim > 0 && !ly.lru.empty()) {
      int v = ly.lru.back(); ly.lru.pop_back(); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      table_remove(L, seq, v); g_st.pf_victims++;
    } else return;
    ly.pf[e] = s; ly.pf_order.push_back(e); (*ly.ready)[s].store(0);
    if (!ly.two.empty()) ly.two[s] = 0;            // (POM needs the JIT scheduler; this path reads full parts)
    g_st.pf_issued++;
    Layer* lp = &ly;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(g_oplay ? g_odelay_us : 0) * 1000),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      load_jobs(*lp, {{e, s}});
      g_last_rd_end.store(now_s(), std::memory_order_relaxed);
      (*lp->ready)[s].store(1, std::memory_order_release);
      // additive update of this pass's published table: before the GPU reads it -> fast path
      // with the spare slot; after -> slow path as before. Never removes or moves an entry.
      if (g_pf_publish) table_add(L, seq, e, s);
    });
  }
}

// Lever 2 (BIGMOE_FC_F / BIGMOE_FC_FS; BigMoE Wed 11-04, FableGod push/WALL1_D1_SPEC.md section 8): right after layer L's
// release, read the first F ids of L+1's posted forecast (the ranked look-ahead list) that are neither resident nor in the
// tier as whole records INTO THE VOLATILE TIER (BIGMOE_VTIER), one GCD block each (trace tag 3); FS = the same for L+1's
// sidecar. The GPU never reads the tier, so a wrong guess evicts nothing a check needs (R-57's KILL: wrong guesses evicted
// LRU experts); a demand miss that finds its expert there copies it back, waiting for a read still in flight. A slot being
// written is never taken (vt_take) and the slab stays pinned meanwhile (vt_unpin), so a purge cannot drop it mid-write.
// Exact: a cache only. Off unless F or FS > 0.
static int g_fc_f = getenv("BIGMOE_FC_F") ? atoi(getenv("BIGMOE_FC_F")) : 0;
static int g_fc_fs = getenv("BIGMOE_FC_FS") ? atoi(getenv("BIGMOE_FC_FS")) : 0;
static void fc_issue(int L, const int32_t* ids, int n, int F) {   // under g_mu
  if (F <= 0 || L < 0 || L >= (int)g_layers.size()) return;
  Layer& ly = g_layers.at(L);
  if (!ly.vt_n || !ly.vt_rdy || ly.rec_fd < 0) return;
  const std::vector<int32_t> none;
  for (int i = 0, k = 0; i < n && k < F; i++) {
    int e = ids[i];
    if (e < 0) continue;
    e &= 0xFFFFF;
    if (e >= ly.E) continue;
    if (ly.table[e] >= 0 || ly.vt_table[e] >= 0 || ly.pf.count(e)) { g_fc_skip++; continue; }
    vt_pin(ly);
    const int t = vt_take(ly, none);
    if (t < 0) break;
    ly.vt_table[e] = t; ly.vt_owner[t] = e; ly.vt_lru.push_front(e); ly.vt_where[e] = ly.vt_lru.begin();
    ly.vt_fc[t] = 1; (*ly.vt_rdy)[t].store(0, std::memory_order_release);
    ly.vt_fly->fetch_add(1); g_fc_fly_all++; g_fc_issued++; k++;
    Layer* lp = &ly;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      t_rd_tag = 3;                                  // trace: 3 = forecast (lever 2)
      const std::vector<std::pair<int, int>> jobs{{e, 0}}; const std::vector<int> td{t};
      load_jobs(*lp, jobs, nullptr, nullptr, &td);
      t_rd_tag = 0;
      g_last_rd_end.store(now_s(), std::memory_order_relaxed);
      (*lp->vt_rdy)[t].store(1, std::memory_order_release);
      lp->vt_fly->fetch_sub(1, std::memory_order_acq_rel); g_fc_fly_all--; g_fc_landed++;
    });
  }
}

// ------------------------------------------------------------------ GPU gate
static int g_maxk = 0, g_nl = 0;
static bool g_nowait = getenv("BIGMOE_NOWAIT") != nullptr;
static bool g_trace = getenv("BIGMOE_TRACE") != nullptr;
struct TraceEv { int L; int64_t seq; int kind; double t0, t1; };   // kind 0: gated cmd buffer GPU start/end, 1: loader seen/signal
static std::mutex g_tmu;
static std::vector<TraceEv> g_tr;
static bool g_inline = getenv("BIGMOE_INLINE") != nullptr;   // test only: mechanics without the wait
static std::vector<MTL::SharedEvent*> g_events;   // one per layer, value = pass seq (out-of-order safe)
// W1-PICKUP (Leap: ~/Leap/TABLE_350.md 'W1-PICKUP', patches/w1_pickup; MLX's MLX_METAL_FAST_SYNCH pattern). Both OFF by
// default: today's kernels and release are the fallback (with the flags off the kernels preprocess to today's code).
// BIGMOE_SPIN_COHERENT=1: the spin is one simdgroup polling ONE row (pair 0 of its (layer, pass) region) with
//   coherent(system) loads and system-scope fences; the release writes that row only. Each spin keeps its own row: the
//   region is keyed by (seq, L), and JOINT's sidecar spin has its own L (side0 + L). Harness [K]: release 0.2 us, pickup
//   2.9 us flat (50 us - 2 ms waits) vs 6.5 us + 17-74 us today. BIGMOE_SPIN_COH_CAP = polls before the timeout
//   (default 1000000, ~0.25 s at ~0.25 us per poll; today's 128 back-off pairs end after ~0.2 s).
// BIGMOE_POST_SYS=1: post_ids writes the routing row through a coherent(system) pointer with system-scope fences, so the
//   loader sees it in ~7 us. Device-scope stores reach the CPU only when a later dispatch flushes them [K]: ~20 us after
//   the next real dispatch, ms if only the spin follows. Same words, same order.
static bool g_spin_coh = getenv("BIGMOE_SPIN_COHERENT") && atoi(getenv("BIGMOE_SPIN_COHERENT"));
static bool g_post_sys = getenv("BIGMOE_POST_SYS") && atoi(getenv("BIGMOE_POST_SYS"));
static const char* k_sys_scope = R"(#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
)";
static const char* k_spin_coh = R"(
      using namespace metal;
      // W1-PICKUP: dep = [ok, slots...] from post_ids. ok -> copy; else one simdgroup polls pair 0 of this (pass,
      // layer)'s release region with coherent(system) loads (it sees the CPU's stores without fresh lines or back-off);
      // a stale word polls again; timeout -> status[0]++ (the pass is reported invalid), as today.
      #ifndef COUNT_FAST
      #define COUNT_FAST 1
      #endif
      kernel void bigmoe_spin(device const int* dep [[buffer(0)]], volatile coherent(system) device int* reg [[buffer(1)]],
                              constant int4& m [[buffer(2)]], device atomic_int* status [[buffer(3)]],
                              device int* out [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
        const int seq = m.y, n = m.z, nl = RS / 32;
        if (dep[0] == 1) {
          for (int i = (int)lane; i < n; i += 32) out[i] = dep[1 + i];
          if (COUNT_FAST && lane == 0) atomic_fetch_add_explicit(status + 16, 1, memory_order_relaxed);
          return;
        }
        const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
        for (int it = 0; it < COH_CAP; it++) {
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
          if (!simd_all(reg[32 * min((int)lane, nl - 1)] == seq)) continue;           // every line's tag
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
          bool good = true;
          for (int i = (int)lane; i < n; i += 32) {
            uint w = (uint)(i < 31 ? reg[1 + i] : i < 62 ? reg[32 + 1 + (i - 31)] : i < 93 ? reg[64 + 1 + (i - 62)] : reg[32 * (i / 31) + 1 + i % 31]);
            good = good && (w & 0xFFC00000u) == STAMP;
            out[i] = (w & 0x3FFFFFu) >= 0x3FFFFEu ? (int)(w & 0x3FFFFFu) - 0x400000 : (int)(w & 0x3FFFFFu);   // -1, -2 (LP1)
          }
          if (simd_all(good)) return;
          if (lane == 0) atomic_fetch_add_explicit(status + 17, 1, memory_order_relaxed);   // fresh tags, a stale word
        }
        if (lane == 0) atomic_fetch_add_explicit(status, 1, memory_order_relaxed);
        for (int i = (int)lane; i < n; i += 32) out[i] = 0;
      })";

static array* g_route_arr = nullptr;
static array* g_tab_arr = nullptr;
static int32_t* g_tab = nullptr;     // residency tables published a pass ahead: per (pass, layer) region of
static int64_t g_tab_n = 0, g_tab_lines = 0;   // g_tab_lines 128-byte lines = [seq, 31 slot entries (-1 = absent)]
static MTL::Buffer* g_route_mtl = nullptr;   // own buffer outside MLX's residency set (BIGMOE_OWNROW)
static array* g_post_bump = nullptr;         // >max MB per buffer: MLX commits right after post_ids
static MTL::ComputePipelineState* post_pso() {
  static MTL::ComputePipelineState* pso = nullptr;
  if (!pso) {
    auto* dev = metal::device(Device::gpu).mtl_device();
    std::string srcs = "#define PLINES " + std::to_string(g_plines) + "\n#define XLINES " + std::to_string(g_xlines) + "\n" + R"(
      #include <metal_stdlib>
      using namespace metal;
      #ifndef POST_SYS            // W1-PICKUP: BIGMOE_POST_SYS writes the row at system scope (CPU-visible at once)
      #define ROW_T device atomic_int*
      #define RST(IDX, VAL) atomic_store_explicit(&row[IDX], VAL, memory_order_relaxed)
      #define RSCOPE thread_scope_device
      #else
      #define ROW_T volatile coherent(system) device int*
      #define RST(IDX, VAL) (row[IDX] = (VAL))
      #define RSCOPE thread_scope_system
      #endif
      // writes [seq, n, ids0..29 | seq, ids30..39 | seq, np, preds0..29 | PLINES x (seq, 31 more preds)]
      kernel void bigmoe_post_ids(device const int* ids [[buffer(0)]], ROW_T route [[buffer(1)]],
                                  constant int4& m [[buffer(2)]], device int* dep [[buffer(3)]],
                                  device const int* tab [[buffer(5)]], device const int* preds [[buffer(6)]],
                                  constant int& np [[buffer(7)]], uint t [[thread_position_in_grid]]) {
        // dep = [ok, slot0..slot{n-1}]: residency table published a pass ahead, read in parallel
        threadgroup atomic_int ok;
        if (t == 0) atomic_store_explicit(&ok, m.w, memory_order_relaxed);     // m.w = 1 if a table exists
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int n = m.z;
        if (m.w && (int)t < n) {
          int e = ids[t] & 0xFFFFF, li = e / 31;   // bit 21 = QOM's optional mark (the loader reads it)
          const volatile device int* tb = (const volatile device int*)tab;
          int tag = tb[li * 32], w = tb[li * 32 + 1 + (e - li * 31)];
          bool fresh = tag == m.y && (((uint)w) & 0xFFC00000u) == ((((uint)m.y) & 0x3FFu) << 22);   // stamped entry
          int sl = (((uint)w) & 0x3FFFFFu) == 0x3FFFFFu ? -1 : (int)(((uint)w) & 0x3FFFFFu);
          if (!fresh || sl < 0) atomic_store_explicit(&ok, 0, memory_order_relaxed);
          dep[1 + t] = fresh ? sl : -1;            // OoO (MoEPrefetch): partial-hit row, -1 = wait for the loader
        }
        if (!m.w && (int)t < n) dep[1 + t] = -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t != 0) return;
        dep[0] = atomic_load_explicit(&ok, memory_order_relaxed);
        ROW_T row = route + m.x * (96 + 32 * PLINES + 32 * XLINES);
        const uint STAMP = (((uint)m.y) & 0x3FFu) << 22;               // every id / pred word carries seq & 0x3FF
        RST(65, np);                 // line 3: next-layer predictions
        int tot = min(30 + 31 * PLINES, (np & 255) + ((np >> 8) & 255) + ((np >> 16) & 255));   // n(L+1) | n(L+2) << 8 | n(L+3) << 16
        for (int i = 0; i < tot; i++)
          RST(i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31, (int)((((uint)preds[i]) & 0x3FFFFFu) | STAMP));
        RST(1, n);
        for (int i = 0; i < n; i++) RST(i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * PLINES + 32 * ((i - 61) / 31) + 1 + (i - 61) % 31, (int)((((uint)ids[i]) & 0x3FFFFFu) | STAMP));
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, RSCOPE);
        for (int j = 0; j < PLINES + XLINES; j++) RST(96 + 32 * j, m.y);
        RST(64, m.y);
        RST(32, m.y);
        RST(0, m.y);
#ifdef POST_SYS
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);   // publish now
#endif
      })";
    if (g_post_sys) srcs = std::string(k_sys_scope) + "#define POST_SYS 1\n" + srcs;   // W1-PICKUP
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(srcs.c_str(), NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) { fprintf(stderr, "bigmoe: post kernel compile failed\n"); abort(); }
    pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("bigmoe_post_ids", NS::UTF8StringEncoding)), &err);
  }
  return pso;
}

static array* g_reg_arr = nullptr;
static array* g_status_arr = nullptr;
static int g_pairs_i = 128;
static MTL::ComputePipelineState* spin_pso() {
  static MTL::ComputePipelineState* pso = nullptr;
  if (!pso) {
    auto* dev = metal::device(Device::gpu).mtl_device();
    std::string srcs = std::string(getenv("BIGMOE_NOCOUNT") ? "#define COUNT_FAST 0\n" : "") + "#define RS " + std::to_string(g_rs) + "\n"
                       + "#define SPIN_G1 " + std::string(getenv("BIGMOE_SPIN_G1") ? getenv("BIGMOE_SPIN_G1") : "1.1") + "f\n"
                       + "#define SPIN_G2 " + std::string(getenv("BIGMOE_SPIN_G2") ? getenv("BIGMOE_SPIN_G2") : "1.1") + "f\n"
                       + "#define SPIN_KNEE " + std::string(getenv("BIGMOE_SPIN_KNEE") ? getenv("BIGMOE_SPIN_KNEE") : "0") + ".0f\n" + R"(
      #include <metal_stdlib>
      using namespace metal;
      // dep = [ok, slots...] from post_ids. ok -> copy; else poll fresh line pairs of this (pass, layer)'s
      // release region with back-off; timeout -> status[0]++ (the pass is reported invalid).
      #ifndef COUNT_FAST
      #define COUNT_FAST 1
      #endif
      kernel void bigmoe_spin(device const int* dep [[buffer(0)]], device const int* reg [[buffer(1)]],
                              constant int4& m [[buffer(2)]], device atomic_int* status [[buffer(3)]],
                              device int* out [[buffer(4)]], uint t [[thread_position_in_grid]]) {
        if (t != 0) return;
        int seq = m.y, n = m.z, pairs = m.w;
        if (dep[0] == 1) { for (int i = 0; i < n; i++) out[i] = dep[1 + i]; if (COUNT_FAST) atomic_fetch_add_explicit(status + 16, 1, memory_order_relaxed); return; }
        float x = 1.0f, d = 32.0f; int got = -1;
        for (int p = 0; p < pairs; p++) {
          const volatile device int* l0 = (const volatile device int*)(reg + p * RS);
          bool tags = l0[0] == seq && l0[32] == seq && (RS == 64 || l0[64] == seq);
          for (int j = 3; j < RS / 32 && tags; j++) tags = l0[32 * j] == seq;     // R-55: entries of > 3 lines
          if (tags) {   // and every word stamped
            const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
            bool good = true;
            for (int i = 0; i < n && good; i++) {
              uint w = (uint)(i < 31 ? l0[1 + i] : i < 62 ? l0[32 + 1 + (i - 31)] : i < 93 ? l0[64 + 1 + (i - 62)] : l0[32 * (i / 31) + 1 + i % 31]);
              good = (w & 0xFFC00000u) == STAMP;
              out[i] = (w & 0x3FFFFFu) >= 0x3FFFFEu ? (int)(w & 0x3FFFFFu) - 0x400000 : (int)(w & 0x3FFFFFu);   // -1, -2 (LP1)
            }
            if (good) { got = p; break; }
            atomic_fetch_add_explicit(status + 17, 1, memory_order_relaxed);   // fresh tags, a stale word: next pair
          }
          for (int j = 0; j < (int)d; j++) x = x * 1.0000001f + 1e-7f;
          d = d < SPIN_KNEE ? d * SPIN_G1 : min(d * SPIN_G2, 400000.0f);   // fine polls first, then fast growth (Scout SC5)
        }
        if (got < 0) { atomic_fetch_add_explicit(status, 1, memory_order_relaxed); for (int i = 0; i < n; i++) out[i] = 0; return; }
        if (x == 12345.0f) out[0] = 0;                // out[] was decoded from the stamped pair above
      })";
    if (g_spin_coh)   // W1-PICKUP: one simdgroup polls ONE row with system-scope loads
      srcs = std::string(k_sys_scope) + (getenv("BIGMOE_NOCOUNT") ? "#define COUNT_FAST 0\n" : "") + "#define RS " + std::to_string(g_rs) + "\n"
             + "#define COH_CAP " + std::string(getenv("BIGMOE_SPIN_COH_CAP") ? getenv("BIGMOE_SPIN_COH_CAP") : "1000000") + "\n" + k_spin_coh;
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(srcs.c_str(), NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) { fprintf(stderr, "bigmoe: spin kernel compile failed\n"); abort(); }
    pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("bigmoe_spin", NS::UTF8StringEncoding)), &err);
  }
  return pso;
}
class SpinWait : public Primitive {
 public:
  SpinWait(Stream s, int L, int seq, int n) : Primitive(s), L_(L), seq_(seq), n_(n) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override { throw std::runtime_error("gpu only"); }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override;
  DEFINE_NAME(SpinWait)
  bool is_equivalent(const Primitive&) const override { return false; }
  int L_, seq_, n_;
};

// post_ids(ids[int32 n], layer, seq): GPU writes the routing row into the route buffer through a
// genuinely writable binding (stores via const_cast of a read-only input only became
// CPU-visible at command-buffer end). Returns a 1-element dependency array.
class PostIds : public Primitive {
 public:
  PostIds(Stream s, int L, int seq, int n, int64_t tab_off, int np) : Primitive(s), L_(L), seq_(seq), n_(n), np_(np), tab_off_(tab_off) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override { throw std::runtime_error("gpu only"); }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& enc = metal::get_command_encoder(stream());
    auto& o = out[0];
    o.set_data(allocator::malloc(o.nbytes()));
    enc.set_compute_pipeline_state(post_pso());
    enc.set_input_array(in[0], 0);
    if (g_route_mtl) {
      enc.set_buffer(g_route_mtl, 1);       // setBuffer also makes it resident for this encoder
    } else {
      enc.set_output_array(*g_route_arr, 1);
    }
    int m[4] = {L_, seq_, n_, (g_tab_arr && tab_off_ >= 0) ? 1 : 0};
    enc.set_bytes(m, 4, 2);
    enc.set_output_array(o, 3);
    if (g_tab_arr) enc.set_input_array(*g_tab_arr, 5, std::max<int64_t>(tab_off_, 0) * 4); else enc.set_input_array(in[0], 5);
    int np = np_;
    enc.set_input_array(in[1], 6);
    enc.set_bytes(np, 7);
    if (g_post_bump) enc.set_input_array(*g_post_bump, 4);   // unused by the kernel; triggers MLX's commit
    int nt = g_xlines > 1 ? 192 : g_xlines ? 128 : 64;                       // R-55: up to 185 ids
    enc.dispatch_threads(MTL::Size(nt, 1, 1), MTL::Size(nt, 1, 1));   // one thread per id (table lookup)
  }
  DEFINE_NAME(PostIds)
  bool is_equivalent(const Primitive&) const override { return false; }
 private:
  int L_, seq_, n_, np_;
  int64_t tab_off_;
};
static void table_add(int L, int seq, int e, int s) {
  if (!g_tab) return;
  { const Layer& ly = g_layers.at(L); if (!ly.two.empty() && ly.two[s]) return; }   // POM: 2-plane slots stay unpublished
  int32_t* tb = g_tab + ((int64_t)((uint64_t)seq * g_nl + L) % g_tab_n) * g_tab_lines * 32;
  int li = e / 31;
  if (__atomic_load_n(tb + li * 32, __ATOMIC_ACQUIRE) != seq) return;   // that table is not this pass's
  __atomic_store_n(tb + li * 32 + 1 + (e - li * 31), s, __ATOMIC_RELEASE);
}
static void table_remove(int L, int seq, int e) {   // caller guarantees the GPU has not read (seq, L) yet
  if (!g_tab) return;
  int32_t* tb = g_tab + ((int64_t)((uint64_t)seq * g_nl + L) % g_tab_n) * g_tab_lines * 32;
  int li = e / 31;
  __atomic_store_n(tb + li * 32 + 1 + (e - li * 31), -1, __ATOMIC_RELAXED);   // fenced by the release that follows
}
static int64_t tab_offset(int L, int seq) {      // int32 offset of (pass, layer)'s published table
  return g_tab ? ((int64_t)((uint64_t)seq * g_nl + L) % g_tab_n) * g_tab_lines * 32 : -1;
}

// ---- prefetch scheduler (BIGMOE_PF_JIT). Everything below runs under g_mu, on the loader thread.
// (seq, X)'s table = the CPU table plus finished prefetches, then the line tags. The GPU reads it at X's post_ids.
static void publish_jit(int X, int32_t seq) {
  g_pub[X] = seq;
  if (!g_tab) return;
  Layer& ly = g_layers.at(X);
  int32_t* tb = g_tab + ((int64_t)((uint64_t)seq * g_nl + X) % g_tab_n) * g_tab_lines * 32;
  for (int64_t li = 0; li < g_tab_lines; li++)
    for (int j = 0; j < 31; j++) { int64_t e = li * 31 + j; tb[li * 32 + 1 + j] = stamp_enc(pub_slot(ly, e), seq); }
  for (auto& [e, s] : ly.pf)                       // a queued (unissued) prefetch has ready = 0
    if ((*ly.ready)[s].load(std::memory_order_acquire) && (ly.two.empty() || !ly.two[s])) tb[(e / 31) * 32 + 1 + e % 31] = stamp_enc(s, seq);
  __atomic_thread_fence(__ATOMIC_RELEASE);
  for (int64_t li = 0; li < g_tab_lines; li++) __atomic_store_n(tb + li * 32, seq, __ATOMIC_RELAXED);
  g_st.pub_jit++;
}

static void issue(int L, int32_t e, int s, int32_t seq, int d, uint8_t mode = 0) {
  Layer* lp = &g_layers.at(L);
  const bool far = d > 1;
  g_inflight++; g_st.pf_issued++; g_iss_d[std::min(std::max(d, 1), 3)]++;
  if (far) g_inflight_far++;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    t_rd_tag = 1;
    if (mode) { std::vector<uint8_t> mv{mode}; load_jobs(*lp, {{e, s}}, &mv); } else load_jobs(*lp, {{e, s}});
    t_rd_tag = 0;
    if (g_pf_publish) table_add(L, seq, e, s);     // before ready: a slot is only recycled once ready, so never under it
    (*lp->ready)[s].store(1, std::memory_order_release);
    g_inflight--;
    if (far) g_inflight_far--;
  });
}

// issue queued reads, nearest layer first, then the most precise class, while fewer than QD are in flight
static void pump() {
  for (;;) {                                       // the best issuable request (one queue: any, while under QD)
    const size_t NONE = (size_t)-1;
    size_t b = NONE;
    for (size_t i = 0; i < g_pq.size(); i++) {
      const PfReq& q = g_pq[i];
      if (!pq_can(q)) continue;
      if (b == NONE) { b = i; continue; }
      const PfReq& c = g_pq[b];
      if (q.seq < c.seq || (q.seq == c.seq && (q.L < c.L || (q.L == c.L && q.prec > c.prec)))) b = i;
    }
    if (b == NONE) break;
    PfReq r = g_pq[b]; g_pq[b] = g_pq.back(); g_pq.pop_back();
    if (r.d <= 1) g_pq_near--;
    auto& qv = g_layers.at(r.L).queued;
    auto it = std::find(qv.begin(), qv.end(), r.e);
    if (it == qv.end()) { fprintf(stderr, "bigmoe: queued read %d/%d lost\n", r.L, r.e); abort(); }
    qv.erase(it);
    issue(r.L, r.e, r.s, r.seq, r.d, r.mode);
  }
  g_pq_n = (int)g_pq.size();
}

// reserve a slot of layer X for expert e, predicted for pass seq, and queue its read. X is never the layer being
// served or one already published for seq, so no table the GPU will read references the slot.
static bool reserve(int X, int32_t e, int32_t seq, int cls, int d, uint8_t mode = 0) {
  Layer& ly = g_layers.at(X);
  if (e < 0 || e >= ly.E || ly.table[e] >= 0) return false;
  if (ly.pf.count(e)) {                            // already prefetched or queued: keep the better class
    auto m = ly.meta.find(e);
    if (m != ly.meta.end() && m->second.first == seq) m->second.second = std::min(m->second.second, cls);
    else ly.meta[e] = {seq, cls};
    return false;
  }
  if (g_pub[X] == seq && g_srv[X] != seq) { fprintf(stderr, "bigmoe: reserve on published layer %d\n", X); abort(); }
  int cap = ly.spare + (g_pf_victim > 0 ? g_pf_victim : 12);
  if ((int)ly.pf.size() >= cap) {                  // recycle the oldest finished prefetch not predicted for this pass
    bool freed = false;
    for (auto it = ly.pf_order.begin(); it != ly.pf_order.end(); ++it) {
      int s = ly.pf[*it];
      auto m = ly.meta.find(*it);
      if (m != ly.meta.end() && m->second.first == seq) continue;
      if ((*ly.ready)[s].load(std::memory_order_acquire)) {
        ly.meta.erase(*it); ly.pf.erase(*it); ly.pf_order.erase(it); ly.free_slots.push_back(s); freed = true; break;
      }
    }
    if (!freed) { g_st.pf_nofit++; return false; }
  }
  int s = -1;
  if (!ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); if (ly.vt_n) ly.ghost[s] = -1; }
  else if (!ly.lfu.empty()) {                     // BIGMOE_EVICT_LFU: the lowest-score expert not predicted for this pass
    auto it = lfu_victim(ly, nullptr, seq);
    if (it != ly.lru.end()) { int v = *it; ly.lru.erase(it); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1; g_st.pf_victims++; }
    if (s < 0) { g_st.pf_nofit++; return false; }
  }
  else {                                           // victim: from the LRU tail, skipping experts predicted for this pass
    auto it = ly.lru.end(); int scan = 0;
    while (it != ly.lru.begin() && scan++ < 16) {
      --it;
      if (ly.pmark[*it] == seq) continue;
      int v = *it; ly.lru.erase(it); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1; g_st.pf_victims++;
      break;
    }
    if (s < 0) { g_st.pf_nofit++; return false; }
  }
  ly.pf[e] = s; ly.pf_order.push_back(e); (*ly.ready)[s].store(0); ly.queued.push_back(e);
  ly.meta[e] = {seq, cls};
  mark_resize(ly); ly.rseq[e] = seq; ly.rdist[e] = (uint8_t)std::min(std::max(d, 1), 3);
  if (ly.pom_r2) { ly.two[s] = mode == 1; if (mode == 1) g_pom_n[7]++; } else mode = 0;   // POM: set before the read
  g_pq.push_back({X, e, seq, s, (float)cls_prec(cls), d, mode}); g_pq_n = (int)g_pq.size();
  if (d <= 1) g_pq_near++;
  g_st.pf_reserved++; g_cls_res[cls]++; g_pass_res++;
  return true;
}

// layer L's row carries predictions for L+1..L+3 (row[65] = n1 | n2 << 8 | n3 << 16 | ranked << 24, ids from row[66]);
// with the oracle, the recorded true ids of L+1..L+D are reserved instead
static void schedule(int L, int32_t seq, volatile int32_t* row) {
  if (seq != g_pass) { g_pass = seq; g_pass_res = 0; g_or_upto = L; }
  if (g_oplay && g_odist > 0) {
    static bool ld = (oracle_load(), true); (void)ld;
    // per layer, the pass its true ids were last reserved for. Tier layouts (set_families) have two row families, base
    // 0..fam0-1 and sidecar fam0..: each row looks ahead within its own family (one high-water mark per pass would let
    // the first sidecar row end the base rows' look-ahead).
    static std::vector<int32_t> or_seq(g_nl, -1);
    int hi = std::min(L + g_odist, (g_fam0 > 0 && L < g_fam0 ? g_fam0 : g_nl) - 1);
    for (int X = L + 1; X <= hi; X++) {
      if (or_seq[X] == seq) continue;
      or_seq[X] = seq;
      auto it = g_or.find((int64_t)seq * 1024 + X);
      if (it != g_or.end()) {
        Layer& ly = g_layers.at(X);
        for (int32_t e : it->second) ly.pmark[e] = seq;   // victims never come from X's own true set
        for (int32_t e : it->second) reserve(X, e, seq, NCLS - 1, X - L);
      }
    }
    return;
  }
  int np = row[65], nd[3] = {np & 255, (np >> 8) & 255, (np >> 16) & 255};
  bool ranked = (np >> 24) & 1;                    // one unique list per distance, most likely first
  int tot = std::min(g_pcap, nd[0] + nd[1] + nd[2]);
  int32_t pr[160];
  for (int i = 0; i < tot; i++) pr[i] = stamp_dec(row[pidx(i)]);
  static std::vector<int32_t> slot_of(1024, -1);    // expert -> index in u (loader thread only)
  for (int d = 1, off = 0; d <= 3; off += nd[d - 1], d++) {
    int n = std::min(nd[d - 1], tot - off), X = L + d;
    if (n <= 0 || X >= g_nl) continue;
    const int32_t* p = pr + off;
    Layer& ly = g_layers.at(X);
    // unique experts; confidence bucket = rank in a ranked list, else how many verify rows predicted it
    int32_t u[160]; int cnt[160], first[160], nu = 0; uint8_t fl[160];
    for (int i = 0; i < n; i++) {
      int e = p[i] < 0 ? -1 : p[i] & 0xFFFFF;      // R-44 b: bit 20 = POM flag (predicted rank < R2), not part of the id
      if (e < 0 || e >= ly.E) continue;
      int j = slot_of[e];
      if (j < 0) { j = nu++; slot_of[e] = j; u[j] = e; cnt[j] = 0; first[j] = i; fl[j] = 0; }
      cnt[j]++; fl[j] |= (uint8_t)((p[i] >> 20) & 1);
    }
    for (int j = 0; j < nu; j++) slot_of[u[j]] = -1;
    mark_resize(ly); ly.fpass = seq;               // S10 log: raw forecasts, before any gate
    g_fc_u[d] += nu; g_fc_ev[d]++;
    for (int j = 0; j < nu; j++) {
      int e = u[j];
      if (ly.fseq[e] != seq) { ly.fseq[e] = seq; ly.fmask[e] = 0; }
      ly.fmask[e] |= (uint8_t)(1u << (d - 1));
    }
    int order[160]; for (int j = 0; j < nu; j++) order[j] = j;
    std::stable_sort(order, order + nu, [&](int a, int b) { return ranked ? first[a] < first[b] : cnt[a] > cnt[b]; });
    for (int j = 0; j < nu; j++) ly.pmark[u[j]] = seq;   // all of them before any victim
    for (int k = 0; k < nu; k++) {
      int j = order[k];
      if (u[j] < 0 || u[j] >= ly.E) continue;
      int bucket = ranked ? (first[j] < 8 ? 0 : first[j] < 24 ? 1 : 2) : (cnt[j] >= 3 ? 0 : cnt[j] == 2 ? 1 : 2);
      int cls = (d - 1) * 3 + bucket;
      if (ly.table[u[j]] >= 0) continue;           // resident: a hit, nothing to read
      if (g_budget > 0 && g_pass_res >= g_budget) continue;
      if (g_theta > 0 && cls_prec(cls) < g_theta && (g_cls_skip[cls]++ & 15)) { g_st.pf_gated_off++; continue; }
      reserve(X, u[j], seq, cls, d, ly.pom_r2 && !fl[j] ? 1 : 0);   // POM: 2-plane unless predicted at rank < R2
    }
  }
}
static array post_ids(const array& ids, const array& preds, int L, int seq, int np) {
  return array({1 + (int)ids.size()}, int32,
               std::make_shared<PostIds>(default_stream(Device::gpu), L, seq, (int)ids.size(), tab_offset(L, seq), np), {ids, preds});
}

static MTL::ComputePipelineState* copy_pso() {
  static MTL::ComputePipelineState* pso = nullptr;
  if (!pso) {
    auto* dev = metal::device(Device::gpu).mtl_device();
    const char* src = R"(
      #include <metal_stdlib>
      using namespace metal;
      kernel void bigmoe_ring_copy(device const int* ring [[buffer(0)]], device int* out [[buffer(1)]],
                                   constant int& n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
        if ((int)t < n) out[t] = ring[t];
      })";
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(src, NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) { fprintf(stderr, "bigmoe: kernel compile failed\n"); abort(); }
    auto* fn = lib->newFunction(NS::String::string("bigmoe_ring_copy", NS::UTF8StringEncoding));
    pso = dev->newComputePipelineState(fn, &err);
  }
  return pso;
}

class CpuGate : public Primitive {
 public:
  CpuGate(Stream s, int L, uint64_t v, int64_t off) : Primitive(s), L_(L), v_(v), off_(off) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("[CpuGate] GPU only");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& enc = metal::get_command_encoder(stream());
    // The id-copy kernel before this gate carried a 64 MB dummy input, so MLX already committed
    // its command buffer -> the GPU runs it and the CPU sees the ids. We are at the start of a
    // fresh buffer: wait, then copy the CPU-written ring row into our own output ON THE GPU,
    // inside this buffer after the wait. Consumers depend on that GPU-produced output, so
    // MLX's own fences order them after it even if MLX commits before they are encoded
    // (a wait does not hold back later command buffers).
    enc.end_encoding();
    if (!g_nowait) enc.get_command_buffer()->encodeWait(g_events.at(L_), v_);
    if (g_trace) {
      int L = L_; int64_t v = (int64_t)v_; double t_enc = now_s();
      enc.get_command_buffer()->addScheduledHandler([L, v, t_enc](MTL::CommandBuffer*) {
        std::lock_guard<std::mutex> lk(g_tmu); g_tr.push_back({L, v, 2, t_enc, now_s()});
      });
      enc.get_command_buffer()->addCompletedHandler([L, v](MTL::CommandBuffer* cb) {
        std::lock_guard<std::mutex> lk(g_tmu); g_tr.push_back({L, v, 0, cb->GPUStartTime(), cb->GPUEndTime()});
      });
    }
    auto& o = out[0];
    o.set_data(allocator::malloc(o.nbytes()));
    int n = (int)o.size();
    enc.set_compute_pipeline_state(copy_pso());
    enc.set_input_array(in[0], 0, off_ * 4);
    enc.set_output_array(o, 1);
    enc.set_bytes(n, 2);
    enc.dispatch_threads(MTL::Size(n, 1, 1), MTL::Size(std::min(n, 64), 1, 1));
  }
  DEFINE_NAME(CpuGate)
  bool is_equivalent(const Primitive&) const override { return false; }
 private:
  int L_;
  uint64_t v_;
  int64_t off_;
};

// flush(x): ends MLX's current compute encoder so the preceding kernels' writes (the routing
// row) reach CPU-visible memory without committing the command buffer; returns x unchanged.
class EndEncoder : public Primitive {
 public:
  explicit EndEncoder(Stream s) : Primitive(s) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override { throw std::runtime_error("gpu only"); }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    metal::get_command_encoder(stream()).end_encoding();
    out[0].copy_shared_buffer(in[0]);
  }
  DEFINE_NAME(EndEncoder)
  bool is_equivalent(const Primitive&) const override { return false; }
};
static array flush_enc(const array& x) {
  return array(x.shape(), x.dtype(), std::make_shared<EndEncoder>(default_stream(Device::gpu)), {x});
}

// gate(slots_buf, dep, layer, seq): returns an array aliasing slots_buf, ordered after `dep`,
// whose consumers run only once the CPU has signalled layer's event at `seq`.
static int64_t g_ring_rows = 0;
static int64_t ring_off(int L, uint64_t seq) { return (int64_t)((seq * g_nl + L) % g_ring_rows) * g_maxk; }
static array gate(const array& ring, const array& dep, int layer, uint64_t seq, int n) {
  auto s = default_stream(Device::gpu);
  return array({n}, int32, std::make_shared<CpuGate>(s, layer, seq, ring_off(layer, seq)), {ring, dep});
}

// ------------------------------------------------------------------ CPU loader thread
// Routing buffer (int32), written by the GPU id-copy kernel: per layer L a row
// [seq, n, ids[MAXK]]; seq = pass number + 1, written last. The loader scans the rows
// for new seqs (abandoned graphs simply never write), loads, fills the layer's slot-id
// array and signals event value seq*NL + L + 1 -- exactly what that layer's gate waits on.
static int32_t* g_route = nullptr;
static int32_t* g_ring = nullptr;
static int32_t* g_dbg = nullptr;     // optional: ids the loader acted on, per ring row
static int32_t* g_reg = nullptr;     // spin mode: release regions (pairs of 128-byte lines)
static int64_t g_reg_n = 0, g_pairs = 0;
static std::thread g_thread;
static std::atomic<bool> g_run{false};
// BIGMOE_SSD_KEEPALIVE_US=n (MoonShots M-20 SSDWAKE, ~/MoonShots/data/ssdwake.txt e6f2cd6; the orchestrator 11-02 22:1x): the
// internal SSD charges +50-65 us on the next read burst after idle gaps of only 0.25-1 ms, and one 4 KB F_NOCACHE read every
// ~50 us removes it. A thread issues one such read (a random 4 KB block of a random record file) every n us while the loader
// served a call in the last 50 ms (decoding), and naps otherwise. Reads only: exact. 0 = off (the default).
static int g_ka_us = getenv("BIGMOE_SSD_KEEPALIVE_US") ? atoi(getenv("BIGMOE_SSD_KEEPALIVE_US")) : 0;
// BIGMOE_IDLE_MS=n (Breakthrough 11-02 22:3x: a process waiting in pausewait spins the loader at ~1 core): after n ms with
// no served call the idle loader naps 200 us per pass instead of only yielding (calls within a request come far less than
// 50 ms apart, so decoding never sees it; the first call after a long idle waits <= ~0.2 ms). 0 = always spin (default).
static double g_idle_s = (getenv("BIGMOE_IDLE_MS") ? atof(getenv("BIGMOE_IDLE_MS")) : 0.0) * 1e-3;
static std::atomic<double> g_last_serve{0.0};
static std::atomic<int64_t> g_ka_reads{0};
static std::thread g_ka_thread;
static void keepalive_main() {
  while (g_run.load(std::memory_order_relaxed) && g_last_serve.load() == 0.0) usleep(1000);   // every layer exists by then
  std::vector<std::pair<int, int64_t>> fl;          // (fd, 4 KB blocks) of each layer's record file
  { std::lock_guard<std::mutex> lk(g_mu);
    for (auto& ly : g_layers) if (ly.rec_fd >= 0 && ly.recsz > 0) fl.push_back({ly.rec_fd, (int64_t)ly.E * ly.recsz / 4096}); }
  void* buf = nullptr;
  if (fl.empty() || posix_memalign(&buf, 16384, 4096)) return;
  uint64_t x = 0x9E3779B97F4A7C15ull;
  while (g_run.load(std::memory_order_relaxed)) {
    if (now_s() - g_last_serve.load(std::memory_order_relaxed) > 0.05) { usleep(2000); continue; }
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    const auto& f = fl[x % fl.size()];
    if (pread(f.first, buf, 4096, (int64_t)((x >> 16) % (uint64_t)std::max<int64_t>(1, f.second)) * 4096) == 4096) g_ka_reads++;
    usleep(g_ka_us);
  }
  free(buf);
}
static int g_delay_us = getenv("BIGMOE_DELAY_US") ? atoi(getenv("BIGMOE_DELAY_US")) : 0;

// JOINT (port of Fit12 B1 v2, ~/Fit12/patches/b1v2_joint_v3f.diff, to v3's plain rows): each tier layer posts two gate
// rows, the MSB-3 base row L and the LSB-1 sidecar row side0 + L, and this thread served them one after the other, so the
// sidecar's reads began only after the base row was released: a layer paid the per-call cost (GCD wake, SSD first byte,
// burst tail, release pickup) twice. With ext.set_joint(side0, nbase) (BIGMOE_JOINT=1; tier's OoO path posts the sidecar
// row right after the base row, before either spin), serving base row L first starts the sidecar row's missing ids
// reading in the background, into the slots make_room frees for them (never an id the row needs), then serves the base
// row as before. The sidecar row is served as before: ensure() finds those reads, waits for them and promotes them.
// Same ids, same bytes in the slots the GPU reads: tokens unchanged (slots may differ; results do not depend on them).
static int g_side0 = 0, g_nbase = 0;
static double g_joint_wait = 30e-6;
static std::atomic<int64_t> g_jt_tried{0}, g_jt_paired{0}, g_jt_reads{0}, g_jt_wait_ns{0};

// caller holds g_mu, serving base row L of pass seq: start the sidecar row's missing ids reading. Returns 1 if paired.
static int joint_side(int L, int32_t seq) {
  if (g_side0 <= 0 || L >= g_nbase) return 0;
  int P = g_side0 + L;
  if (P >= g_nl) return 0;
  g_jt_tried++;
  volatile int32_t* row = g_route + (int64_t)P * g_row;
  double t0 = now_s();
  bool ok = false;
  for (;;) {                                       // the sidecar is posted right after the base row: wait briefly
    if (row[0] == seq) {
      ok = row[32] == seq && row[64] == seq;
      for (int j = 0; j < g_plines + g_xlines && ok; j++) ok = row[96 + 32 * j] == seq;
      if (ok) break;
    }
    if (now_s() - t0 > g_joint_wait) break;
  }
  g_jt_wait_ns += (int64_t)((now_s() - t0) * 1e9);
  if (!ok) return 0;
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  int n = row[1];
  if (n <= 0 || n > g_maxk) return 0;
  Layer& ly = g_layers.at(P);
  std::vector<int32_t> need; std::vector<int8_t> mr;   // LP1: best gate rank per expert (rank_k ids per row)
  for (int i = 0; i < n; i++) {
    int32_t w = row[ipos(i)];
    if (!stamp_ok(w, seq)) return 0;               // a torn sector: leave the row to its own service
    int e = (int)((uint32_t)w & 0x3FFFFFu);
    if (e < 0 || e >= ly.E) return 0;
    auto it = std::find(need.begin(), need.end(), e);
    if (it == need.end()) { need.push_back(e); mr.push_back(127); it = need.end() - 1; }
    if (ly.lp1 && ly.rank_k > 0) { int8_t& m = mr[it - need.begin()]; m = (int8_t)std::min<int>(m, i % ly.rank_k); }
  }
  g_jt_paired++;
  Layer* lp = &ly;
  auto start = [lp](int e, int s) {
    g_jt_reads++;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
      t_rd_tag = 2;                                // trace: 2 = joint (a demand read issued one row early)
      load_jobs(*lp, {{e, s}});
      t_rd_tag = 0;
      (*lp->ready)[s].store(1, std::memory_order_release);
    });
  };
  for (size_t j = 0; j < need.size(); j++) {
    const int e = need[j];
    if (ly.table[e] >= 0) continue;
    if (ly.lp1 && ly.rank_k > 0 && mr[j] >= 1 && !ly.pf.count(e)) { g_pom_n[8]++; continue; }   // LP1: no read
    auto pit = ly.pf.find(e);
    if (pit != ly.pf.end()) {                      // a JIT read reserved for it but not issued yet: issue it now (it
      auto q = std::find(ly.queued.begin(), ly.queued.end(), e);   // would be cancelled into a demand read at P's turn)
      if (q != ly.queued.end()) { ly.queued.erase(q); pq_drop(P, e); start(e, pit->second); }
      continue;                                    // in flight or done: ensure() waits for it and promotes it
    }
    make_room(ly, need);
    if (ly.free_slots.empty()) break;              // everything resident is needed: ensure() takes it from here
    int s = ly.free_slots.back(); ly.free_slots.pop_back(); if (ly.vt_n) ly.ghost[s] = -1;
    ly.pf[e] = s; ly.pf_order.push_back(e); (*ly.ready)[s].store(0);
    start(e, s);
  }
  return 1;
}

// b32 (S10b deployable score, NewIdeas GO 10-26 09:08): BIGMOE_S10B_RES=file records, at each JIT prefetch decision (right
// before schedule(L, seq), whose filter is table[e] >= 0 = resident, nothing to read), the residency of the base pools
// T = L+1..L+3: int32 stream of [seq, L, nT, (T, w_0..w_{W-1}) x nT], bit e % 32 of w_{e / 32} = expert e resident,
// W = (E + 31) / 32. Base rows only (L < nbase: the LSB sidecars are no forecast targets). Written at exit. Off by default.
static const char* g_s10b_res = getenv("BIGMOE_S10B_RES");
static std::vector<int32_t> g_s10b_rv;
static void s10b_res_save() {
  FILE* f = fopen(g_s10b_res, "wb"); if (!f) return;
  fwrite(g_s10b_rv.data(), 4, g_s10b_rv.size(), f); fclose(f);
  fprintf(stderr, "[bigmoe] s10b res: %zu ints -> %s\n", g_s10b_rv.size(), g_s10b_res);
}
static void s10b_res(int L, int32_t seq) {   // caller holds g_mu
  static bool reg = (std::atexit(s10b_res_save), true); (void)reg;
  int nb = g_nbase > 0 ? g_nbase : g_nl;
  if (L >= nb) return;
  size_t h = g_s10b_rv.size();
  g_s10b_rv.push_back(seq); g_s10b_rv.push_back(L); g_s10b_rv.push_back(0);
  for (int T = L + 1; T <= L + 3 && T < nb; T++) {
    const Layer& ly = g_layers.at(T);
    g_s10b_rv.push_back(T);
    size_t w0 = g_s10b_rv.size();
    g_s10b_rv.resize(w0 + (ly.E + 31) / 32, 0);
    for (int e = 0; e < ly.E; e++)
      if (ly.table[e] >= 0) g_s10b_rv[w0 + e / 32] |= (int32_t)(1u << (e % 32));
    g_s10b_rv[h + 2]++;
  }
}

static void loader_main() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   // keep the spinner on a P-core
  std::vector<int32_t> last(g_nl, 0);
  double t_idle = now_s();
  while (g_run.load(std::memory_order_relaxed)) {
    bool any = false;
    for (int L = 0; L < g_nl; L++) {
      volatile int32_t* row = g_route + (int64_t)L * g_row;   // 3 (+ BIGMOE_PRED_LINES) cache lines, each with its own seq
      int32_t seq = row[0];
      if (seq == last[L] || seq == 0) continue;
      if (g_side0 > 0 && L >= g_side0 && L - g_side0 < g_nbase) {   // JOINT: a sidecar row waits while its base row of
        volatile int32_t* br = g_route + (int64_t)(L - g_side0) * g_row;   // the same pass is posted but not served yet:
        if (br[0] == seq && last[L - g_side0] != seq) continue;           // the base row's service starts its reads
      }
      for (bool done = false; !done;) {            // all lines complete
        done = row[32] == seq && row[64] == seq;
        for (int j = 0; j < g_plines + g_xlines && done; j++) done = row[96 + 32 * j] == seq;
      }
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      any = true; last[L] = seq;
      double t_seen = now_s(), t_e0 = 0, t_e1 = 0;   // S0: t_e0 / t_e1 = ensure() start / end
      g_last_serve.store(t_seen, std::memory_order_relaxed);   // BIGMOE_SSD_KEEPALIVE_US: decoding
      g_st.wait_ns += (int64_t)((now_s() - t_idle) * 1e9);
      int n = row[1];
      bool slow_call = false, victim_done = false;
      if (n < 0 || n > g_maxk) { fprintf(stderr, "bigmoe: bad n=%d at layer %d\n", n, L); abort(); }
      {
        std::lock_guard<std::mutex> lk(g_mu);
        int32_t ids_copy[256];
        {                                            // fresh line tags != fresh sectors: every id word must carry
          double t_st = 0; int64_t rr = 0;           // this post's stamp (re-read until it does), then strip it
          for (;;) {
            bool good = true;
            for (int i = 0; i < n; i++) { ids_copy[i] = row[ipos(i)]; good = good && stamp_ok(ids_copy[i], seq); }
            if (good) break;
            if (!rr++) t_st = now_s();
            if (now_s() - t_st > 0.25) { fprintf(stderr, "bigmoe: stamp wait > 250 ms at layer %d seq %d\n", L, seq); break; }
          }
          if (rr) { g_stamp_calls++; g_stamp_rereads += rr; g_stamp_ns += (int64_t)((now_s() - t_st) * 1e9); }
          for (int i = 0; i < n; i++) ids_copy[i] = (int32_t)((uint32_t)ids_copy[i] & 0x3FFFFFu);
        }
        if (g_dbg) memcpy(g_dbg + ring_off(L, (uint64_t)seq), ids_copy, n * 4);
        int64_t m0 = g_st.misses.load(), j0 = g_st.stalled.load();
        g_cur_seq = seq;
        if (g_s10b) s10b_rec(L, seq, ids_copy, n);
        if (g_jit) { g_srv[L] = seq; if (g_early) { if (g_s10b_res) s10b_res(L, seq); schedule(L, seq, row); pump(); } }
        if (g_side0 > 0 && L < g_nbase) joint_side(L, seq);   // JOINT: the sidecar row's reads first, in the background
        t_e0 = g_trace ? now_s() : 0;
        ensure(L, ids_copy, n, g_ring + ring_off(L, (uint64_t)seq));
        t_e1 = g_trace ? now_s() : 0;
        if (L + 1 < g_nl) {                         // remember L+1's look-ahead set for the miss split
          int np0 = row[65], k1 = std::min(np0 & 255, g_pcap);
          Layer& nx = g_layers.at(L + 1); nx.pred.assign(k1, 0);
          for (int i = 0; i < k1; i++) { int32_t v = stamp_dec(row[pidx(i)]); nx.pred[i] = v < 0 ? v : v & 0xFFFFF; }
          nx.pred_seq = seq;
        }
        slow_call = g_st.misses.load() != m0;
        if (g_orec) {
          static bool reg = (std::atexit(oracle_save), true); (void)reg;
          std::vector<int32_t> u;
          for (int i = 0; i < n; i++) if (!in(u, ids_copy[i])) u.push_back(ids_copy[i]);
          g_or[(int64_t)seq * 1024 + L] = u;
        }
        // a real load ran, so the GPU is held at this gate: victim prefetch for L+1/L+2 is safe until release
        if (!g_jit && g_pf_victim > 0 && g_st.stalled.load() != j0) {
          if (g_oplay) {
            static bool ld = (oracle_load(), true); (void)ld;
            auto it = g_or.find((int64_t)seq * 1024 + L + 1);
            if (it != g_or.end() && L + 1 < g_nl) prefetch(L + 1, it->second.data(), (int)it->second.size(), seq, true);
          } else {
          int np = row[65], n1 = std::min(np & 255, g_pcap), n2 = std::min((np >> 8) & 255, g_pcap - n1);
          int32_t pr[160];
          for (int i = 0; i < n1 + n2; i++) pr[i] = stamp_dec(row[pidx(i)]);
          if (n1 && L + 1 < g_nl) prefetch(L + 1, pr, n1, seq, true);
          if (n2 && L + 2 < g_nl) prefetch(L + 2, pr + n1, n2, seq, true);
          }
          victim_done = true;
        }
      }
      __atomic_thread_fence(__ATOMIC_RELEASE);
      if (g_delay_us > 0) { double t0 = now_s(); while (now_s() - t0 < g_delay_us * 1e-6) {} }   // test only
      g_st.gated++;
      const double t_r0 = g_trace ? now_s() : 0;    // S0: the release writes start
      if (g_reg) {   // spin mode: write [seq, slots...] into every line pair of this (layer, pass) region
        int32_t* reg = g_reg + ((int64_t)((uint64_t)seq * g_nl + L) % g_reg_n) * g_pairs * g_rs;
        const int32_t* sl = g_ring + ring_off(L, (uint64_t)seq);
        const int64_t npairs = g_spin_coh ? 1 : g_pairs;   // W1-PICKUP: the coherent spin polls pair 0 only
        for (int64_t p = 0; p < npairs; p++) {
          int32_t* l0 = reg + p * g_rs; int32_t* l1 = l0 + 32; int32_t* l2 = l0 + 64;
          for (int i = 0; i < n && i < 31; i++) l0[1 + i] = stamp_enc(sl[i], seq);     // stamped words
          for (int i = 31; i < n && i < 62; i++) l1[1 + (i - 31)] = stamp_enc(sl[i], seq);
          for (int i = 62; i < n && i < 93; i++) l2[1 + (i - 62)] = stamp_enc(sl[i], seq);
          for (int i = 93; i < n; i++) l0[32 * (i / 31) + 1 + i % 31] = stamp_enc(sl[i], seq);   // R-55: lines 3..5
        }
        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (int64_t p = 0; p < npairs; p++) {
          int32_t* l0 = reg + p * g_rs;
          for (int j = g_rs / 32 - 1; j >= 3; j--) __atomic_store_n(l0 + 32 * j, seq, __ATOMIC_RELAXED);   // R-55
          if (g_rs >= 96) __atomic_store_n(l0 + 64, seq, __ATOMIC_RELAXED);
          __atomic_store_n(l0 + 32, seq, __ATOMIC_RELAXED); __atomic_store_n(l0, seq, __ATOMIC_RELAXED);
        }
      } else {
        g_events[L]->setSignaledValue((uint64_t)seq);
      }
      double t_rel = now_s();                      // GPU released: bookkeeping below is off the critical path
      if (slow_call) { g_st.slow_calls++; g_st.slow_ns += (int64_t)((t_rel - t_seen) * 1e9); }
      if (g_tab && !g_jit) {   // publish layer L's residency for the next pass (valid until that pass serves L)
        Layer& ly = g_layers.at(L);
        int32_t nseq = seq + 1;
        int32_t* tb = g_tab + ((int64_t)((uint64_t)nseq * g_nl + L) % g_tab_n) * g_tab_lines * 32;
        for (int64_t li = 0; li < g_tab_lines; li++)
          for (int j = 0; j < 31; j++) { int64_t e = li * 31 + j; tb[li * 32 + 1 + j] = stamp_enc(pub_slot(ly, e), nseq); }
        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (int64_t li = 0; li < g_tab_lines; li++) __atomic_store_n(tb + li * 32, nseq, __ATOMIC_RELAXED);
      }
      double t_now = now_s();
      g_st.proc_ns += (int64_t)((t_now - t_seen) * 1e9);
      int np = row[65];                            // predicted experts -> async prefetch:
      int n1 = std::min(np & 255, g_pcap), n2 = std::min((np >> 8) & 255, g_pcap - n1);   // first for L+1, then for L+2
      if (g_jit) {   // after the release (off the critical path): reserve L+1..L+3's predictions, publish L+1, issue
        // safe even if the GPU is already at L+1's post_ids: it then reads an old tag and waits at L+1's gate
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_early) { if (g_s10b_res) s10b_res(L, seq); schedule(L, seq, row); }
        if (L + 1 < g_nl) publish_jit(L + 1, seq);
        pump();
      } else if (n1 + n2 > 0 && !victim_done) {
        int32_t pr[160];
        for (int i = 0; i < n1 + n2; i++) pr[i] = stamp_dec(row[pidx(i)]);
        std::lock_guard<std::mutex> lk(g_mu);
        if ((g_fc_f > 0 || g_fc_fs > 0) && n1) {     // lever 2: L+1's forecast into the volatile tier (base row L only)
          const int nb = g_fam0 > 0 ? g_fam0 : g_nl;
          if (L + 1 < nb) {
            fc_issue(L + 1, pr, n1, g_fc_f);
            if (g_fam0 > 0 && g_fam0 + L + 1 < g_nl) fc_issue(g_fam0 + L + 1, pr, n1, g_fc_fs);
          }
        }
        if (n1 && L + 1 < g_nl) prefetch(L + 1, pr, n1, seq);
        if (n2 && L + 2 < g_nl) prefetch(L + 2, pr + n1, n2, seq);
        if (g_pf_side > 0 && n1 && g_fam0 > 0 && L + 1 < g_fam0 && g_fam0 + L + 1 < g_nl)   // BIGMOE_PF_SIDE
          prefetch(g_fam0 + L + 1, pr, std::min(n1, g_pf_side), seq);
      }
      if (g_trace) { std::lock_guard<std::mutex> lk(g_tmu); g_tr.push_back({L, (int64_t)seq, 1, t_seen, t_rel});
                     if (slow_call) g_tr.push_back({L, (int64_t)seq, 4, t_seen, t_rel});   // 4 = slow (had misses)
                     if (t_e1 > 0) g_tr.push_back({L, (int64_t)seq, 5, t_e0, t_e1});   // S0: ensure() (plan + loads)
                     g_tr.push_back({L, (int64_t)seq, 6, t_r0, t_rel}); }              // S0: the release writes
      t_idle = t_now;
    }
    if (!any) {
      if (g_jit && g_pq_n.load(std::memory_order_relaxed) > 0 && pq_ready()) {
        std::lock_guard<std::mutex> lk(g_mu); pump();   // a speculative read finished: issue the next one
      }
      if (g_idle_s > 0 && now_s() - t_idle > g_idle_s) usleep(200);   // BIGMOE_IDLE_MS: idle for a while: nap
      else __builtin_arm_yield();
    }
  }
}

static int g_spinvar = getenv("BIGMOE_SPINVAR") ? atoi(getenv("BIGMOE_SPINVAR")) : 0;
static MTL::ComputePipelineState* copyonly_pso() {
  static MTL::ComputePipelineState* pso = nullptr;
  if (!pso) {
    auto* dev = metal::device(Device::gpu).mtl_device();
    const char* src = R"(
      #include <metal_stdlib>
      using namespace metal;
      kernel void bigmoe_copyonly(device const int* dep [[buffer(0)]], constant int4& m [[buffer(2)]], device int* out [[buffer(4)]],
                                  uint t [[thread_position_in_grid]]) { if ((int)t < m.z) out[t] = dep[1 + t]; })";
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(src, NS::UTF8StringEncoding), nullptr, &err);
    pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("bigmoe_copyonly", NS::UTF8StringEncoding)), &err);
  }
  return pso;
}
void SpinWait::eval_gpu(const std::vector<array>& in, std::vector<array>& out) {
  auto& enc = metal::get_command_encoder(stream());
  auto& o = out[0];
  o.set_data(allocator::malloc(o.nbytes()));
  if (g_spinvar) {   // timing experiments: 1 = copy-only kernel, no region/status bindings (wrong on misses)
    enc.set_compute_pipeline_state(copyonly_pso());   // 3 = copy-only kernel but with the region+status bindings
    enc.set_input_array(in[0], 0);
    if (g_spinvar == 3) {
      int64_t ro = ((int64_t)((uint64_t)seq_ * g_nl + L_) % g_reg_n) * g_pairs * g_rs;
      enc.set_input_array(*g_reg_arr, 1, ro * 4); enc.set_input_array(*g_status_arr, 3);
    }
    int m[4] = {L_, seq_, n_, 0}; enc.set_bytes(m, 4, 2);
    enc.set_output_array(o, 4);
    enc.dispatch_threads(MTL::Size(n_, 1, 1), MTL::Size(n_, 1, 1));
    return;
  }
  int64_t reg_off = ((int64_t)((uint64_t)seq_ * g_nl + L_) % g_reg_n) * g_pairs * g_rs;
  enc.set_compute_pipeline_state(spin_pso());
  enc.set_input_array(in[0], 0);
  enc.set_input_array(*g_reg_arr, 1, reg_off * 4);
  int m[4] = {L_, seq_, n_, (int)g_pairs};
  enc.set_bytes(m, 4, 2);
  // status is bound as an INPUT: registering a persistent buffer as an output every layer made MLX
  // fence each spin against the previous one (~45 us/layer). The kernel's atomics need no tracking.
  if (getenv("BIGMOE_STATUS_OUT")) enc.set_output_array(*g_status_arr, 3); else enc.set_input_array(*g_status_arr, 3);
  enc.set_output_array(o, 4);
  const int nt = g_spin_coh ? 32 : 1;                // W1-PICKUP: the coherent spin is one simdgroup
  enc.dispatch_threads(MTL::Size(nt, 1, 1), MTL::Size(nt, 1, 1));
}
static array spin_wait(const array& dep, int L, int seq, int n) {
  return array({n}, int32, std::make_shared<SpinWait>(default_stream(Device::gpu), L, seq, n), {dep});
}

NB_MODULE(bigmoe_ext_ov, m) {
  nb::module_::import_("mlx.core");
  m.def("add_layer", [](int C, int E, std::vector<int> fds, std::vector<int64_t> offs, std::vector<int64_t> pers,
                        std::vector<array> pools) {
    Layer ly; ly.C = C; ly.E = E;
    for (size_t i = 0; i < fds.size(); i++) {
      auto& p = pools[i]; p.eval();
      ly.pieces.push_back({fds[i], offs[i], pers[i], p.data<uint8_t>()});
    }
    ly.table.assign(E, -1); ly.slot_owner.assign(C, -1); ly.where.resize(E);
    ly.pmark.assign(E, -1);
    ly.spare = (C < E) ? std::min(g_spare, C / 4) : 0; ly.C_main = C - ly.spare;
    ly.ready = new std::vector<std::atomic<int>>(C);
    for (int s = C - 1; s >= 0; s--) ly.free_slots.push_back(s);
    g_layers.push_back(std::move(ly));
    return (int)g_layers.size() - 1;
  });
  // ---- PF1 prefill staging (MoonShots): double-buffered reads of a layer's non-resident experts into staging
  // buffers while the previous layer computes; the scratch pool is then built from [pool; staging] in the same
  // sorted expert order as the plain path, so the bytes and kernels are unchanged.
  m.def("stage_bufs", [](int b, std::vector<array> bufs, int cap) {
    g_stage[b].clear();                             // Python owns the buffers (freed after prefill)
    for (auto& a : bufs) { a.eval(); g_stage[b].push_back(a.data<uint8_t>()); }
    g_stage_cap = cap;
    if (!g_sgrp[b]) g_sgrp[b] = dispatch_group_create();
  });
  m.def("stage_start", [](int b, int L) {          // returns the staged expert ids (ascending); reads run async
    std::vector<int32_t> ids;
    { std::lock_guard<std::mutex> lk(g_mu);
      Layer& ly = g_layers.at(L);
      for (int e = 0; e < ly.E; e++) if (ly.table[e] < 0) ids.push_back(e); }   // incl. prefetched (not resident)
    if ((int)ids.size() > g_stage_cap) { fprintf(stderr, "bigmoe: stage %zu > cap %d\n", ids.size(), g_stage_cap); abort(); }
    Layer* lp = &g_layers.at(L);
    std::vector<uint8_t*> dst = g_stage[b];
    auto idv = std::make_shared<std::vector<int32_t>>(ids);
    dispatch_group_async(g_sgrp[b], dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      if (lp->rec_fd >= 0) {
        dispatch_apply(idv->size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
          int n = (int)lp->pieces.size();
          if (lp->rec_fmt == 3 || (g_m01 && lp->part_len)) {   // records3: bounce + rebuild into the 4-bit staging layout
            uint8_t* d[16];
            for (int i = 0; i < n; i++) d[i] = dst[i] + (int64_t)k * lp->pieces[i].per;
            read_expert(lp, (*idv)[k], d, false);
            return;
          }
          struct iovec iov[16];
          for (int i = 0; i < n; i++) { iov[i].iov_base = dst[i] + (int64_t)k * lp->pieces[i].per; iov[i].iov_len = lp->pieces[i].per; }
          read_record(lp->rec_fd, (int64_t)(*idv)[k] * lp->recsz + lp->part_off, iov, n);
        });
        return;
      }
      size_t np = lp->pieces.size(), n = idv->size() * np;
      dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        int k = (int)(i / np), e = (*idv)[k];
        const Piece& p = lp->pieces[i % np];
        int64_t done = 0;
        while (done < p.per) {
          ssize_t r = pread(p.fd, dst[i % np] + (int64_t)k * p.per + done, p.per - done, p.off + (int64_t)e * p.per + done);
          if (r <= 0) { fprintf(stderr, "bigmoe: stage pread failed e=%d\n", e); abort(); }
          done += r;
        }
      });
    });
    return ids;
  });
  // PF_FULLSCR: build a full-E scratch pool (expert id order) for layer L from its slot pool (resident experts) and staging
  // buffer b (the rest), by parallel memcpy into caller-owned arrays (GPU must be done with them). Returns #staged used.
  m.def("scratch_full", [](int L, int b, std::vector<array> dst, std::vector<int32_t> sids) {
    Layer* lp = &g_layers.at(L);
    std::vector<int64_t> src(lp->E);                  // >= 0: pool slot; < 0: -(1 + staging index)
    { std::lock_guard<std::mutex> lk(g_mu);
      for (int e = 0; e < lp->E; e++) {
        if (lp->table[e] >= 0) { src[e] = lp->table[e]; continue; }
        auto it = std::lower_bound(sids.begin(), sids.end(), e);
        if (it == sids.end() || *it != e) { fprintf(stderr, "bigmoe: scratch_full: expert %d neither resident nor staged\n", e); abort(); }
        src[e] = -(1 + (int64_t)(it - sids.begin()));
      } }
    std::vector<uint8_t*> d; for (auto& a : dst) { a.eval(); d.push_back(a.data<uint8_t>()); }
    std::vector<uint8_t*> stg = g_stage[b];
    size_t np = lp->pieces.size(); const int64_t* sp = src.data(); uint8_t* const* dp = d.data(); uint8_t* const* gp = stg.data();
    dispatch_apply((size_t)lp->E * np, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
      int e = (int)(i / np); size_t k = i % np; const Piece& p = lp->pieces[k];
      const uint8_t* from = sp[e] >= 0 ? p.pool + sp[e] * p.per : gp[k] + (-sp[e] - 1) * p.per;
      memcpy(dp[k] + (int64_t)e * p.per, from, p.per);
    });
  });
  // PF_FULLSCR (layer-major prefill): when a layer is done, copy its most-used experts (cand, priority order) from the
  // full-E scratch (index = expert id) into its pool: free slots first, then over LRU-tail experts not in `used`.
  m.def("adopt_full", [](int L, std::vector<array> srcs, std::vector<int32_t> used, std::vector<int32_t> cand) {
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    const bool pr = prior_on(ly), st = pr && g_lfu_decay <= 0;   // BIGMOE_EVICT_PRIOR: stamps + prior-aware victims
    for (int e : used) if (ly.table[e] >= 0) { ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); if (st) ly.lfu[e] = ly.lfu_w; }
    std::vector<uint8_t*> src; for (auto& a : srcs) src.push_back(a.data<uint8_t>());
    std::vector<std::pair<int, int>> jobs;
    for (int e : cand) {
      if (ly.table[e] >= 0) continue;
      int s;
      int extra = std::max(0, (int)ly.pf.size() - ly.spare);
      if ((int)ly.lru.size() + extra < ly.C_main && !ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); if (ly.vt_n) ly.ghost[s] = -1; }
      else if (pr) {
        auto it = lfu_victim(ly, &used, -1);
        if (it == ly.lru.end()) break;
        int v = *it; ly.lru.erase(it); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      } else {
        if (ly.lru.empty()) break;
        int v = ly.lru.back();
        if (in(used, v)) break;
        ly.lru.pop_back(); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      }
      jobs.push_back({s, e});
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); if (st) ly.lfu[e] = ly.lfu_w;
    }
    size_t np = ly.pieces.size(); Layer* lp = &ly; const auto* jp = &jobs; uint8_t* const* sp = src.data();
    dispatch_apply(jobs.size() * np, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
      auto [s, e] = (*jp)[i / np]; const Piece& p = lp->pieces[i % np];
      memcpy(p.pool + (int64_t)s * p.per, sp[i % np] + (int64_t)e * p.per, p.per);
    });
    return (int)jobs.size();
  });
  m.def("set_records", [](int L, std::string path, int64_t recsz) {   // BIGMOE_RECORDS=1: this layer reads records
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "bigmoe: cannot open %s\n", path.c_str()); abort(); }
    fcntl(fd, F_NOCACHE, 1);
    std::lock_guard<std::mutex> lk(g_mu);
    g_layers.at(L).rec_fd = fd; g_layers.at(L).recsz = recsz;
  });
  m.def("set_records3", [](int L, std::string path, int64_t recsz, int64_t msb_sz) {   // exact mode on records3
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "bigmoe: cannot open %s\n", path.c_str()); abort(); }
    fcntl(fd, F_NOCACHE, 1);
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L); ly.rec_fd = fd; ly.recsz = recsz; ly.rec_fmt = 3; ly.msb_sz = msb_sz;
  });
  m.def("m01_on", [] { return g_m01; });   // R-44 a: Python refuses BIGMOE_M01 on an ext without the M-01 unpack
  m.def("set_pom", [](int L, std::string path, int64_t recsz, int64_t seg1, int64_t seg2, int r2, int k) {
    // R-44 b: base layer L (records3m MSB part already set) also reads records3p's seg1 / seg2; ids are rows of k ranks
    if (!g_pom || !g_m01 || !g_jit) { fprintf(stderr, "bigmoe: set_pom needs BIGMOE_POM, BIGMOE_M01 and BIGMOE_PF_JIT\n"); abort(); }
    if (seg1 % 16384 || seg2 % 16384 || recsz % 16384) { fprintf(stderr, "bigmoe: unaligned records3p\n"); abort(); }
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "bigmoe: cannot open %s\n", path.c_str()); abort(); }
    fcntl(fd, F_NOCACHE, 1);
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    if (ly.pieces.size() != 9 || !ly.part_len) { fprintf(stderr, "bigmoe: set_pom on a layer that is not a base pool\n"); abort(); }
    ly.pom_fd = fd; ly.pom_recsz = recsz; ly.seg1 = seg1; ly.seg2 = seg2; ly.pom_r2 = r2; ly.rank_k = k;
    ly.two.assign(ly.slot_owner.size(), 0);
  });
  m.def("set_vtier", [](int L, int slots) -> long long {   // BIGMOE_VTIER: a purgeable victim tier of `slots` records at layer
    std::lock_guard<std::mutex> lk(g_mu);              // L (record pools only); returns its bytes (0 = not set)
    Layer& ly = g_layers.at(L);
    if (ly.rec_fd < 0 || slots <= 0 || ly.vt_n) return 0;
    int64_t sb = 0; for (auto& pc : ly.pieces) sb += pc.per;
    MTL::Buffer* b = metal::device(Device::gpu).mtl_device()->newBuffer((size_t)(sb * slots), MTL::ResourceStorageModeShared);
    if (!b) { fprintf(stderr, "bigmoe: vtier: no %lld bytes at layer %d\n", (long long)(sb * slots), L); return 0; }
    b->setPurgeableState(MTL::PurgeableStateVolatile);  // empty and reclaimable until a call pins it
    ly.vt_buf = b; ly.vt_mem = (uint8_t*)b->contents(); ly.vt_slot = sb;
    ly.vt_table.assign(ly.E, -1); ly.vt_owner.assign(slots, -1); ly.vt_where.resize(ly.E);
    ly.vt_free.clear(); for (int t = slots - 1; t >= 0; t--) ly.vt_free.push_back(t);
    ly.ghost.assign(ly.C, -1);
    ly.vt_rdy = new std::vector<std::atomic<int>>(slots);   // lever 2: every slot's bytes in place (no read in flight)
    for (auto& r : *ly.vt_rdy) r.store(1);
    ly.vt_fc.assign(slots, 0); ly.vt_fly = new std::atomic<int>(0);
    ly.vt_off.clear(); { int64_t o = 0; for (auto& pc : ly.pieces) { ly.vt_off.push_back(o); o += pc.per; } }
    ly.vt_n = slots;                                    // last: ensure() / make_room() key on it
    return (long long)(sb * slots);
  });
  m.def("fc_test", [](int L, std::vector<int32_t> ids, int F) -> int {   // lever 2, test only: L's forecast reads now
    std::lock_guard<std::mutex> lk(g_mu); const int64_t i0 = g_fc_issued.load();
    fc_issue(L, ids.data(), (int)ids.size(), F); return (int)(g_fc_issued.load() - i0); });
  m.def("fc_wait", []() { while (g_fc_fly_all.load(std::memory_order_acquire) > 0) usleep(100); });   // test only
  m.def("vtier_purge", [](int L) {                   // test only: discard layer L's tier the way a macOS purge does
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    if (ly.vt_n && !ly.vt_pinned) ly.vt_buf->setPurgeableState(MTL::PurgeableStateEmpty);
  });
  m.def("set_prior", [](int L, std::vector<float> p) {   // BIGMOE_EVICT_PRIOR: layer L's popularity, E values in [0, 1]
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    if ((int)p.size() != ly.E) { fprintf(stderr, "bigmoe: set_prior needs %d values at layer %d\n", ly.E, L); abort(); }
    ly.prior = std::move(p);
  });
  m.def("set_lp1", [](int L, int k) {              // R-44 b LP1: sidecar layer L, k ids (4-bit ranks) per row
    if (!g_lp1) { fprintf(stderr, "bigmoe: set_lp1 needs BIGMOE_POM_LP1\n"); abort(); }
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L); ly.lp1 = 1; ly.rank_k = k;
  });
  m.def("pom_stats", [] { std::vector<int64_t> v; for (auto& a : g_pom_n) v.push_back(a.load()); return v; });
  m.def("bounce_stats", [] {   // the shared bounce pool now: [buffers, bytes] (all idle between calls)
    std::lock_guard<std::mutex> lk(g_bounce_mu);
    int64_t b = 0; for (auto& x : g_bounce) b += x.second;
    return std::vector<int64_t>{(int64_t)g_bounce.size(), b};
  });
  m.def("set_records_part", [](int L, std::string path, int64_t recsz, int64_t part_off, int64_t part_len) {
    // tier pools (DESIGN_3bit.md): this layer's pieces lie back to back in [part_off, part_off + part_len) of each record
    // (16 KB-aligned offset and length), e.g. records3's MSB-3 part for the base pool or its LSB-1 part for the sidecar
    if (part_off % 16384 || part_len % 16384) { fprintf(stderr, "bigmoe: unaligned record part\n"); abort(); }
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "bigmoe: cannot open %s\n", path.c_str()); abort(); }
    fcntl(fd, F_NOCACHE, 1);
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L); ly.rec_fd = fd; ly.recsz = recsz; ly.rec_fmt = 1; ly.part_off = part_off; ly.part_len = part_len;
  });
  // plan-mode exact prefill: experts ids[k] from records3, rebuilt to the 4-bit pieces, into row rows[k] of the 9
  // caller-owned arrays dst (GPU must be done with them). Parallel over experts; same rebuild as BIGMOE_RECORDS=3.
  // read_rebuilt_layout: the same on a converted layer (layout 1 native / 2 native+p11, rows per projection, the index's
  // 15 piece offsets), rebuilt through today's row-major record, so the 4-bit rows are today's bit for bit.
  static auto rebuilt = [](std::string path, int64_t recsz, int64_t msb_sz, std::vector<int64_t> pers4,
                           std::vector<int32_t> ids, std::vector<int32_t> rows, std::vector<array> dst, int layout,
                           std::vector<int64_t> rows3, std::vector<int64_t> offs) {
    static std::unordered_map<std::string, int> fds;
    int fd;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = fds.find(path);
      if (it == fds.end()) { fd = open(path.c_str(), O_RDONLY); if (fd < 0) abort(); fcntl(fd, F_NOCACHE, 1); fds[path] = fd; }
      else fd = it->second; }
    Layer ly; ly.rec_fd = fd; ly.recsz = recsz; ly.rec_fmt = 3; ly.msb_sz = msb_sz;
    for (auto p : pers4) ly.pieces.push_back({-1, 0, p, nullptr});
    set_layout_(ly, layout, rows3, offs);
    std::vector<uint8_t*> d;
    // ov copy (Leap 11-11): called with the GIL released, so no MLX scheduling here: dst must already be evaluated
    for (auto& a : dst) { if (!a.is_available()) { fprintf(stderr, "bigmoe_ext_ov: a dst array is not evaluated\n"); abort(); }
                          d.push_back(a.data<uint8_t>()); }
    const Layer* lp = &ly; const auto* idp = &ids; const auto* rp = &rows; const auto* dp = &d;
    dispatch_apply(ids.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t k) {
      uint8_t* t[16];
      for (size_t i = 0; i < lp->pieces.size(); i++) t[i] = (*dp)[i] + (int64_t)(*rp)[k] * lp->pieces[i].per;
      read_expert(lp, (*idp)[k], t, false);
    });
  };
  m.def("read_rebuilt", [](std::string path, int64_t recsz, int64_t msb_sz, std::vector<int64_t> pers4,
                           std::vector<int32_t> ids, std::vector<int32_t> rows, std::vector<array> dst) {
    rebuilt(path, recsz, msb_sz, pers4, ids, rows, dst, 0, {}, {});
  }, nb::call_guard<nb::gil_scoped_release>());
  m.def("read_rebuilt_layout", [](std::string path, int64_t recsz, int64_t msb_sz, std::vector<int64_t> pers4,
                                  std::vector<int32_t> ids, std::vector<int32_t> rows, std::vector<array> dst, int layout,
                                  std::vector<int64_t> rows3, std::vector<int64_t> offs) {
    rebuilt(path, recsz, msb_sz, pers4, ids, rows, dst, layout, rows3, offs);
  }, nb::call_guard<nb::gil_scoped_release>());
  m.def("layout_max", [] { return 2; });
  m.def("set_direct_split", [](int L, int n) {   // native+p11+pad: the LSB part's 3 aligned l1 planes straight in, b via bounce
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    if (n < 0 || n > (int)ly.pieces.size()) { fprintf(stderr, "bigmoe: set_direct_split %d of %zu pieces\n", n, ly.pieces.size()); abort(); }
    ly.direct_n = n;
  });   // the highest records3 layout the fmt 3 readers take (Python's guard)
  m.def("record_rowmajor", [](std::string path, int64_t recsz, int64_t msb_sz, std::vector<int64_t> pers4, int layout,
                              std::vector<int64_t> rows3, std::vector<int64_t> offs, int e, array out) {
    // test: expert e's record of a converted layer, back to today's row-major record, into out (uint8, >= rm_len bytes)
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) abort();
    fcntl(fd, F_NOCACHE, 1);
    Layer ly; ly.rec_fd = fd; ly.recsz = recsz; ly.rec_fmt = 3; ly.msb_sz = msb_sz;
    for (auto p : pers4) ly.pieces.push_back({-1, 0, p, nullptr});
    set_layout_(ly, layout, rows3, offs);
    out.eval();
    if (!layout || (int64_t)out.nbytes() < ly.rm_len) abort();
    Bounce bb(recsz);
    struct iovec one = {bb.p, (size_t)recsz};
    read_record(fd, (int64_t)e * recsz, &one, 1);
    close(fd);
    native_rowmajor(&ly, bb.p, out.data<uint8_t>());
    return ly.rm_len;
  });
  m.def("rebuild_test", [](array w3, array l1, array out) {   // unit test of the records3 rebuild (CPU arrays)
    w3.eval(); l1.eval(); out.eval();
    rebuild_w4(w3.data<uint32_t>(), l1.data<uint32_t>(), out.data<uint32_t>(), (int64_t)out.size() / 4);
  });
  m.def("stage_wait", [](int b) { if (g_sgrp[b]) dispatch_group_wait(g_sgrp[b], DISPATCH_TIME_FOREVER); });
  m.def("lookup", [](int L, std::vector<int32_t> ids) {   // slot or -1, no loading
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    std::vector<int32_t> out; for (int e : ids) out.push_back(ly.table[e]);
    return out;
  });
  // after the layer's call ran (GPU done with the pool): touch the used resident experts, then copy staged experts
  // (in priority order) into free slots or over LRU-tail experts this chunk did not use.
  m.def("adopt", [](int L, int b, std::vector<int32_t> used, std::vector<int32_t> cand, std::vector<int32_t> sidx) {
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    const bool pr = prior_on(ly), st = pr && g_lfu_decay <= 0;   // BIGMOE_EVICT_PRIOR: stamps + prior-aware victims
    for (int e : used) if (ly.table[e] >= 0) { ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); if (st) ly.lfu[e] = ly.lfu_w; }
    int adopted = 0;
    std::vector<std::pair<int, int>> jobs;            // (pool slot, staging index)
    for (size_t k = 0; k < cand.size(); k++) {
      int e = cand[k];
      if (ly.table[e] >= 0) continue;
      int s;
      int extra = std::max(0, (int)ly.pf.size() - ly.spare);
      if ((int)ly.lru.size() + extra < ly.C_main && !ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); if (ly.vt_n) ly.ghost[s] = -1; }
      else if (pr) {
        auto it = lfu_victim(ly, &used, -1);
        if (it == ly.lru.end()) break;
        int v = *it; ly.lru.erase(it); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      } else {
        if (ly.lru.empty()) break;
        int v = ly.lru.back();
        if (in(used, v)) break;                        // the tail is used this chunk: the pool holds only used experts
        ly.lru.pop_back(); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      }
      jobs.push_back({s, sidx[k]});
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); adopted++;
      if (st) ly.lfu[e] = ly.lfu_w;
    }
    // the copies run in parallel (a single-threaded memcpy of ~0.7 GB per layer cost ~4 s per prompt)
    size_t np = ly.pieces.size();
    Layer* lp = &ly; const auto* jp = &jobs; uint8_t* const* src = g_stage[b].data();
    dispatch_apply(jobs.size() * np, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
      auto [s, k] = (*jp)[i / np];
      const Piece& p = lp->pieces[i % np];
      memcpy(p.pool + (int64_t)s * p.per, src[i % np] + (int64_t)k * p.per, p.per);
    });
    return adopted;
  });
  m.def("ensure", [](int L, array ids) {           // synchronous path (prefill, debugging)
    ids = astype(ids, int32); ids.eval();
    std::vector<int32_t> out(ids.size());
    { std::lock_guard<std::mutex> lk(g_mu); ensure(L, ids.data<int32_t>(), (int)ids.size(), out.data()); }
    return array(out.begin(), ids.shape(), int32);
  });
  m.def("start", [](array route, array ring, int nl, int maxk) {
    route.eval(); ring.eval(); g_route = route.data<int32_t>(); g_ring = ring.data<int32_t>();
    g_route_arr = new array(route);
    if (!getenv("BIGMOE_NOPOSTCOMMIT")) { g_post_bump = new array(zeros({64 << 20}, uint8)); g_post_bump->eval(); }
    if (getenv("BIGMOE_OWNROW")) {
      g_route_mtl = metal::device(Device::gpu).mtl_device()->newBuffer(route.nbytes(), MTL::ResourceStorageModeShared);
      memset(g_route_mtl->contents(), 0, route.nbytes());
      g_route = (int32_t*)g_route_mtl->contents();
    }
    g_nl = nl; g_maxk = maxk; g_ring_rows = (int64_t)ring.size() / maxk;
    g_pub.assign(nl, -1); g_srv.assign(nl, -1);
    while ((int)g_events.size() < g_nl) { auto* e = metal::device(Device::gpu).mtl_device()->newSharedEvent(); e->setSignaledValue(0); g_events.push_back(e); }
    g_run = true; g_thread = std::thread(loader_main);
    if (g_ka_us > 0) g_ka_thread = std::thread(keepalive_main);
  });
  m.def("stop", []() { g_run = false; if (g_thread.joinable()) g_thread.join(); if (g_ka_thread.joinable()) g_ka_thread.join(); });
  m.def("gate", &gate);
  m.def("flush", &flush_enc);
  m.def("post_ids", &post_ids);
  m.def("trace", []() { std::lock_guard<std::mutex> lk(g_tmu); std::vector<std::vector<double>> o;
    for (auto& e : g_tr) o.push_back({(double)e.L, (double)e.seq, (double)e.kind, e.t0, e.t1}); g_tr.clear(); return o; });
  m.def("set_families", [](int fam0) {             // tier layouts: rows fam0.. are sidecars (the oracle's look-ahead)
    std::lock_guard<std::mutex> lk(g_mu); g_fam0 = fam0; });
  m.def("set_joint", [](int side0, int nbase, double wait_us) {   // JOINT: sidecar row of base layer L = side0 + L
    std::lock_guard<std::mutex> lk(g_mu); g_side0 = side0; g_nbase = nbase; g_joint_wait = wait_us * 1e-6; });
  m.def("trace_reads", []() { std::lock_guard<std::mutex> lk(g_rmu); std::vector<std::vector<double>> o;
    for (auto& r : g_rd) o.push_back({(double)r.L, (double)r.e, (double)r.tag, r.t0, r.t1, r.tp0, r.tp1}); g_rd.clear(); return o; });
  m.def("trace_cov", []() { std::lock_guard<std::mutex> lk(g_rmu); std::vector<std::vector<double>> o;
    for (auto& c : g_cov) o.push_back({(double)c.L, (double)c.seq, (double)c.need, (double)c.hit, (double)c.pf_ready,
                                       (double)c.pf_late, (double)c.q_demand, (double)c.pred_unissued, (double)c.unpred, c.t});
    g_cov.clear(); return o; });
  m.def("spin_wait", &spin_wait);
  m.def("set_status", [](array st) { st.eval(); g_status_arr = new array(st); });
  m.def("set_regions", [](array reg, int64_t pairs) { reg.eval(); g_reg = reg.data<int32_t>(); g_pairs = pairs; g_reg_arr = new array(reg);
    g_reg_n = (int64_t)reg.size() / (pairs * g_rs); });
  m.def("set_tables", [](array tab, int64_t lines) { tab.eval(); g_tab = tab.data<int32_t>(); g_tab_lines = lines;
    g_tab_arr = new array(tab);
    g_tab_n = (int64_t)tab.size() / (lines * 32); });
  m.def("set_debug", [](array dbg) { dbg.eval(); g_dbg = dbg.data<int32_t>(); });
  m.def("ring_off", [](int L, uint64_t seq) { return ring_off(L, seq); });
  m.def("stats", []() {
    nb::dict d; d["uses"] = g_st.uses.load(); d["misses"] = g_st.misses.load(); d["stalled"] = g_st.stalled.load();
    d["gated"] = g_st.gated.load(); d["io_s"] = g_st.io_ns.load() * 1e-9; d["cpu_wait_s"] = g_st.wait_ns.load() * 1e-9; d["proc_s"] = g_st.proc_ns.load() * 1e-9;
    d["pf_issued"] = g_st.pf_issued.load(); d["pf_used"] = g_st.pf_used.load(); d["pf_waited_s"] = g_st.pf_waited_ns.load() * 1e-9;
    d["fc_miss"] = g_fc_miss.load(); d["fc_lead2"] = g_fc_lead2.load();
    for (int k = 1; k <= 3; k++) {
      d[("fc_d" + std::to_string(k)).c_str()] = g_fc_d[k].load();
      d[("iss_d" + std::to_string(k)).c_str()] = g_iss_d[k].load();
      d[("iss_used_d" + std::to_string(k)).c_str()] = g_iss_used_d[k].load();
      d[("fc_u" + std::to_string(k)).c_str()] = g_fc_u[k].load(); d[("fc_ev" + std::to_string(k)).c_str()] = g_fc_ev[k].load();
    }
    d["joint_tried"] = g_jt_tried.load(); d["joint_paired"] = g_jt_paired.load(); d["joint_reads"] = g_jt_reads.load();
    d["joint_wait_s"] = g_jt_wait_ns.load() * 1e-9;
    d["stamp_calls"] = g_stamp_calls.load(); d["stamp_rereads"] = g_stamp_rereads.load(); d["stamp_wait_s"] = g_stamp_ns.load() * 1e-9;
    d["slow_calls"] = g_st.slow_calls.load(); d["slow_s"] = g_st.slow_ns.load() * 1e-9; d["pf_victims"] = g_st.pf_victims.load();
    d["ka_reads"] = g_ka_reads.load(); d["bursts"] = g_bursts.load(); d["burst_gap025"] = g_burst_g025.load();   // M-20
    d["burst_gap1"] = g_burst_g1.load();
    d["vt_hits"] = g_vt_hits.load(); d["vt_ins"] = g_vt_ins.load(); d["vt_purged"] = g_vt_purged.load();   // BIGMOE_VTIER
    d["vt_drops"] = g_vt_drops.load();
    d["fc_issued"] = g_fc_issued.load(); d["fc_landed"] = g_fc_landed.load(); d["fc_hits"] = g_fc_hits.load();   // lever 2
    d["fc_late"] = g_fc_late.load(); d["fc_wait_s"] = g_fc_wait_ns.load() * 1e-9; d["fc_drops"] = g_fc_drops.load();
    d["fc_skip"] = g_fc_skip.load(); d["fc_busy_bursts"] = g_fc_busy_bursts.load(); d["fc_busy_n"] = g_fc_busy_n.load();
    d["fc_f"] = g_fc_f; d["fc_fs"] = g_fc_fs;
    d["qom_skip"] = g_qom_skip.load(); d["qom_opt"] = g_qom_opt.load();   // QOM: optional-only misses not read / optional uses
    { int64_t vb = 0, vr = 0; std::lock_guard<std::mutex> lk(g_mu);
      for (auto& ly : g_layers) if (ly.vt_n) { vb += (int64_t)ly.vt_n * ly.vt_slot; vr += (int64_t)(ly.vt_n - (int)ly.vt_free.size()) * ly.vt_slot; }
      d["vt_bytes"] = vb; d["vt_used_bytes"] = vr;        // the tier's size / its filled part (volatile: not in phys_footprint)
      std::vector<int64_t> hl, il; for (auto& ly : g_layers) { hl.push_back(ly.vt_h); il.push_back(ly.vt_i); }
      d["vt_hits_layer"] = hl; d["vt_ins_layer"] = il; }  // per ext layer (cumulative; the harness takes run deltas)
    d["pf_ontime"] = g_st.pf_ontime.load(); d["pf_late"] = g_st.pf_late.load(); d["miss_pred_unissued"] = g_st.miss_pred_unissued.load(); d["miss_unpred"] = g_st.miss_unpred.load();
    d["pf_reserved"] = g_st.pf_reserved.load(); d["pf_cancel"] = g_st.pf_cancel.load(); d["pf_qdemand"] = g_st.pf_qdemand.load();
    d["pf_gated_off"] = g_st.pf_gated_off.load(); d["pf_nofit"] = g_st.pf_nofit.load(); d["pub_jit"] = g_st.pub_jit.load();
    std::vector<std::vector<double>> cls;            // per class: reserved, scored, used, precision (d1 b0..2, d2, d3, oracle)
    for (int c = 0; c < NCLS; c++) cls.push_back({(double)g_cls_res[c], g_cls_iss[c], g_cls_used[c], cls_prec(c)});
    d["pf_cls"] = cls; return d;
  });
  m.def("reset_stats", []() { g_ka_reads = 0; g_bursts = 0; g_burst_g025 = 0; g_burst_g1 = 0; g_vt_hits = 0; g_vt_ins = 0; g_vt_purged = 0; g_vt_drops = 0; g_st.uses = 0; g_st.misses = 0; g_st.stalled = 0; g_st.gated = 0; g_st.io_ns = 0; g_st.wait_ns = 0; g_st.proc_ns = 0; g_st.pf_issued = 0; g_st.pf_used = 0; g_st.pf_waited_ns = 0; g_st.slow_calls = 0; g_st.slow_ns = 0; g_st.pf_victims = 0; g_st.pf_ontime = 0; g_st.pf_late = 0; g_st.miss_pred_unissued = 0; g_st.miss_unpred = 0;
    g_st.pf_reserved = 0; g_st.pf_cancel = 0; g_st.pf_qdemand = 0; g_st.pf_gated_off = 0; g_st.pf_nofit = 0; g_st.pub_jit = 0;
    for (int c = 0; c < NCLS; c++) g_cls_res[c] = 0; });   // the class precisions keep learning across resets
}
