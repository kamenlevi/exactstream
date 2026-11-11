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
  int64_t part_off = 0, part_len = 0;              // fmt 1 on a part of a record (tier pools: records3 MSB or LSB part)
  int rec_fd = -1; int64_t recsz = 0;               // BIGMOE_RECORDS: one record per expert (9 pieces back to back)  // look-ahead prediction for this layer (from L-1's row)
  // MoonShots scheduler (BIGMOE_PF_JIT): slots reserved for prefetches whose read is not issued yet (ready = 0, in pf),
  // each prefetch's (pass predicted for, class), and this layer's predicted set for the pass (victims skip it)
  std::vector<int32_t> queued;
  std::unordered_map<int32_t, std::pair<int32_t, int>> meta;
  std::vector<int32_t> pmark;                     // expert -> last pass it was predicted for (victims skip this pass's)
};
static int32_t g_cur_seq = -1;                      // pass being served (for the miss split)

struct Stats { std::atomic<int64_t> uses{0}, misses{0}, stalled{0}, gated{0}; std::atomic<int64_t> io_ns{0}, wait_ns{0}, proc_ns{0};
               std::atomic<int64_t> pf_issued{0}, pf_used{0}, pf_waited_ns{0}, slow_calls{0}, slow_ns{0}, pf_victims{0},
               pf_ontime{0}, pf_late{0}, miss_pred_unissued{0}, miss_unpred{0},    // demand-miss split
               pf_reserved{0}, pf_cancel{0}, pf_qdemand{0}, pf_gated_off{0}, pf_nofit{0}, pub_jit{0}; };   // scheduler

static std::vector<Layer> g_layers;
static std::mutex g_mu;
static Stats g_st;

#include <sys/uio.h>
// one expert record -> 9 destinations with a single preadv (16 KB-aligned record, contiguous pieces); short reads
// continue from where they stopped
static void read_record(int fd, int64_t off, struct iovec* iov, int n) {
  int i = 0;
  while (i < n) {
    ssize_t r = preadv(fd, iov + i, n - i, off);
    if (r <= 0) { fprintf(stderr, "bigmoe: record preadv failed\n"); abort(); }
    off += r;
    while (i < n && r >= (ssize_t)iov[i].iov_len) { r -= iov[i].iov_len; i++; }
    if (i < n && r > 0) { iov[i].iov_base = (uint8_t*)iov[i].iov_base + r; iov[i].iov_len -= r; }
  }
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
// One expert from its record into 9 destinations (gate/up/down x w/s/b, 4-bit layout) via a per-thread aligned bounce.
static void read_expert(const Layer* lp, int e, uint8_t* const* dst, bool par = true) {
  thread_local uint8_t* buf = nullptr; thread_local int64_t cap = 0;
  int64_t len = lp->part_len ? lp->part_len : lp->recsz;
  if (cap < len) { free(buf); if (posix_memalign((void**)&buf, 16384, len)) abort(); cap = len; }
  struct iovec one = {buf, (size_t)len};
  read_record(lp->rec_fd, (int64_t)e * lp->recsz + lp->part_off, &one, 1);
  if (lp->rec_fmt == 1) {
    int64_t o = 0;
    for (size_t i = 0; i < lp->pieces.size(); i++) { memcpy(dst[i], buf + o, lp->pieces[i].per); o += lp->pieces[i].per; }
    return;
  }
  // records3: MSB part = per proj (w3, s3, b3), LSB part at msb_sz = per proj (l1, b); the 3 projections in parallel.
  // The block must use a local copy of buf: blocks do not capture thread_locals, so worker threads would read their own.
  // par = false when many experts load at once (b5: nested dispatch_apply under a wide verify miss list starved the
  // loader into spin-gate timeouts): the job's own thread rebuilds all 3 projections.
  const uint8_t* rb = buf;
  auto proj = ^(size_t p) {
    int64_t w4 = lp->pieces[3 * p].per, sb = lp->pieces[3 * p + 1].per, w3 = w4 / 4 * 3, l1 = w4 / 4;
    const uint8_t* m = rb + (int64_t)p * (w3 + 2 * sb);
    const uint8_t* l = rb + lp->msb_sz + (int64_t)p * (l1 + sb);
    rebuild_w4((const uint32_t*)m, (const uint32_t*)l, (uint32_t*)dst[3 * p], w4 / 16);
    halve_bf16((const uint16_t*)(m + w3), (uint16_t*)dst[3 * p + 1], sb / 2);
    memcpy(dst[3 * p + 2], l + l1, sb);
  };
  if (par) dispatch_apply(3, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), proj);
  else for (size_t p = 0; p < 3; p++) proj(p);
}

static void load_jobs(Layer& ly, const std::vector<std::pair<int, int>>& jobs) {
  if (ly.rec_fd >= 0) {                             // record path: one read per expert
    Layer* lp = &ly; const auto* jp = &jobs;
    static bool direct = getenv("BIGMOE_RECORDS_DIRECT") != nullptr;   // preadv straight into the 9 pool slots
    dispatch_apply(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t j) {
      auto [e, s] = (*jp)[j];
      int n = (int)lp->pieces.size();
      if (direct && lp->rec_fmt == 1) {
        struct iovec iov[16];
        for (int i = 0; i < n; i++) { iov[i].iov_base = lp->pieces[i].pool + (int64_t)s * lp->pieces[i].per; iov[i].iov_len = lp->pieces[i].per; }
        read_record(lp->rec_fd, (int64_t)e * lp->recsz + lp->part_off, iov, n);
        return;
      }
      // default: ONE page-aligned read into a per-thread bounce buffer (Scout: aligned single reads are the fastest
      // pattern; unaligned F_NOCACHE destinations are slow), then the pieces go into their pool slots (records3: rebuilt)
      uint8_t* d[16];
      for (int i = 0; i < n; i++) d[i] = lp->pieces[i].pool + (int64_t)s * lp->pieces[i].per;
      read_expert(lp, e, d, jp->size() < 4);
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
// Replay oracle (MoonShots runahead bound): BIGMOE_ORACLE_REC=file records every gated (pass, layer)'s unique ids;
// BIGMOE_ORACLE_PLAY=file replays them: while L stalls, L+1's *true* ids go to the victim prefetch, with the reads
// issued BIGMOE_ORACLE_DELAY_US after (the realistic runahead latency). Same tokens and passes -> same (seq, L) keys.
static const char* g_orec = getenv("BIGMOE_ORACLE_REC");
static const char* g_oplay = getenv("BIGMOE_ORACLE_PLAY");
static int g_odelay_us = getenv("BIGMOE_ORACLE_DELAY_US") ? atoi(getenv("BIGMOE_ORACLE_DELAY_US")) : 0;
static std::unordered_map<int64_t, std::vector<int32_t>> g_or;
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
static int g_xlines = (getenv("BIGMOE_MAXK") && atoi(getenv("BIGMOE_MAXK")) > 61) ? 1 : 0;
static int g_rs = g_xlines ? 96 : 64;                 // ints per release-region entry
static int g_row = 96 + 32 * g_plines + 32 * g_xlines, g_pcap = 30 + 31 * g_plines;
static inline int ipos(int i) { return i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * g_plines + 1 + (i - 61); }
// [MoonShots v3h, Sun 25 Oct] seq stamps on every handoff word: routing-row ids and preds (GPU -> loader), release
// regions and residency tables (loader -> GPU); bits 22-31 = seq & 0x3FF. A 128-byte line carries its seq tag only in
// word 0, and under another process's GPU load a reader saw a fresh tag next to a stale 64-byte sector (the loader read the
// previous post's ids from words 16+ of routing-row lines: data/b8mr/handoff_*.json in ~/MoonShots). Readers now check
// every word's stamp, not just the line tags. Payloads fit in bits 0-21 (expert ids < 2^20 plus flag bits 20-21; slots,
// positions; -1 travels as 0x3FFFFF). Wrap: 10 bits = 1024 gate passes; a stale word comes from an earlier post /
// publication of the same row, region or table a few passes back (MTP draft steps advance the pass too), far below 1024.
static inline uint32_t stamp_of(int32_t seq) { return ((uint32_t)seq & 0x3FFu) << 22; }
static inline int32_t stamp_enc(int32_t v, int32_t seq) { return (int32_t)(((uint32_t)v & 0x3FFFFFu) | stamp_of(seq)); }
static inline bool stamp_ok(int32_t w, int32_t seq) { return ((uint32_t)w & 0xFFC00000u) == stamp_of(seq); }
static inline int32_t stamp_dec(int32_t w) { int32_t v = (int32_t)((uint32_t)w & 0x3FFFFFu); return v == 0x3FFFFF ? -1 : v; }
static std::atomic<int64_t> g_stamp_calls{0}, g_stamp_rereads{0}, g_stamp_ns{0}, g_stamp_joint{0};
static inline int pidx(int i) { return i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31; }   // prediction i's int
static bool g_jit = getenv("BIGMOE_PF_JIT") != nullptr;
static int g_qd = getenv("BIGMOE_PF_QD") ? atoi(getenv("BIGMOE_PF_QD")) : 0;
static double g_theta = getenv("BIGMOE_PF_THETA") ? atof(getenv("BIGMOE_PF_THETA")) : 0.0;
static int g_budget = getenv("BIGMOE_PF_BUDGET") ? atoi(getenv("BIGMOE_PF_BUDGET")) : 0;
static bool g_early = getenv("BIGMOE_PF_EARLY") != nullptr;
static int g_odist = getenv("BIGMOE_ORACLE_DIST") ? atoi(getenv("BIGMOE_ORACLE_DIST")) : 0;
struct PfReq { int L; int32_t e, seq, s; float prec; };
static std::vector<PfReq> g_pq;                      // reserved, not issued (under g_mu)
static std::atomic<int> g_inflight{0}, g_pq_n{0};
static std::vector<int32_t> g_pub, g_srv;            // per layer: pass its table was last published for / last served in
static int32_t g_pass = -1; static int g_pass_res = 0, g_or_upto = -1;
constexpr int NCLS = 10;                             // (distance 1..3) x (confidence bucket 0..2), + oracle
static double g_cls_iss[NCLS], g_cls_used[NCLS];
static int64_t g_cls_res[NCLS], g_cls_skip[NCLS];
static double cls_prec(int c) { return (g_cls_used[c] + 1.0) / (g_cls_iss[c] + 2.0); }
static void pq_drop(int L, int32_t e) {
  for (size_t i = 0; i < g_pq.size(); i++) if (g_pq[i].L == L && g_pq[i].e == e) { g_pq[i] = g_pq.back(); g_pq.pop_back(); break; }
  g_pq_n = (int)g_pq.size();
}

static bool in(const std::vector<int32_t>& v, int32_t e) { return std::find(v.begin(), v.end(), e) != v.end(); }

// keep at most C_main experts resident: evict LRU experts not needed now (their slots become free)
static void make_room(Layer& ly, const std::vector<int32_t>& need) {
  // prefetches beyond the spare pool (victim mode) occupy main slots
  while ((int)ly.lru.size() + std::max(0, (int)ly.pf.size() - ly.spare) >= ly.C_main && !ly.lru.empty()) {
    auto it = ly.lru.end();
    bool found = false;
    while (it != ly.lru.begin()) { --it; if (!in(need, *it)) { found = true; break; } }
    if (!found) break;                               // everything resident is needed now: the pf-drop path frees a slot
    int v = *it; ly.lru.erase(it); int s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1; ly.free_slots.push_back(s);
  }
}

// ids -> slots (in place into out). Loads misses (blocking). Caller holds g_mu.
static void ensure(int L, const int32_t* ids, int n, int32_t* out, const std::vector<int32_t>* protect = nullptr) {
  Layer& ly = g_layers.at(L);
  std::vector<int32_t> need; need.reserve(n);
  for (int i = 0; i < n; i++) {
    int e = ids[i];
    if (e < 0 || e >= ly.E) { fprintf(stderr, "bigmoe: bad expert id %d at layer %d\n", e, L); abort(); }
    if (!in(need, e)) need.push_back(e);
  }
  std::vector<int32_t> keep = need;              // [Breakthrough v3f] experts make_room must not evict (the whole row)
  if (protect) for (int e : *protect) if (!in(keep, e)) keep.push_back(e);
  if ((int)need.size() > ly.C_main) { fprintf(stderr, "bigmoe: %zu experts > C=%d at layer %d\n", need.size(), ly.C_main, L); abort(); }
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
      int s = ly.pf[e]; pq_drop(L, e);
      ly.pf.erase(e); ly.pf_order.remove(e); ly.free_slots.push_back(s);
      if (in(need, e)) g_st.pf_qdemand++; else g_st.pf_cancel++;
    }
    ly.queued.clear();
  }
  std::vector<std::pair<int, int>> jobs;
  int n_miss = 0;
  for (int e : need) {
    if (ly.table[e] >= 0) continue;
    n_miss++;
    auto pit = ly.pf.find(e);
    if (pit != ly.pf.end()) {                       // prefetched: wait for its bytes, then promote
      int s = pit->second;
      if ((*ly.ready)[s].load(std::memory_order_acquire)) g_st.pf_ontime++; else g_st.pf_late++;
      double t = now_s(); while (!(*ly.ready)[s].load(std::memory_order_acquire)) {} g_st.pf_waited_ns += (int64_t)((now_s() - t) * 1e9);
      ly.pf.erase(pit); ly.pf_order.remove(e);
      make_room(ly, keep);
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
      g_st.pf_used++;
      continue;
    }
    make_room(ly, keep);
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
    if (ly.pred_seq == g_cur_seq && in(ly.pred, e)) g_st.miss_pred_unissued++; else g_st.miss_unpred++;
    ly.table[e] = s; ly.slot_owner[s] = e; jobs.push_back({e, s});
    ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
  }
  g_st.uses += need.size(); g_st.misses += n_miss; g_st.stalled += !jobs.empty();
  if (!jobs.empty()) { double t = now_s(); load_jobs(ly, jobs); g_st.io_ns += (int64_t)((now_s() - t) * 1e9); }
  for (int e : need) { ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); }
  for (int i = 0; i < n; i++) out[i] = ly.table[ids[i]];
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
    if (!ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); }
    else if (victim_ok && g_pf_victim > 0 && !ly.lru.empty()) {
      int v = ly.lru.back(); ly.lru.pop_back(); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      table_remove(L, seq, v); g_st.pf_victims++;
    } else return;
    ly.pf[e] = s; ly.pf_order.push_back(e); (*ly.ready)[s].store(0);
    g_st.pf_issued++;
    Layer* lp = &ly;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(g_oplay ? g_odelay_us : 0) * 1000),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      load_jobs(*lp, {{e, s}});
      (*lp->ready)[s].store(1, std::memory_order_release);
      // additive update of this pass's published table: before the GPU reads it -> fast path
      // with the spare slot; after -> slow path as before. Never removes or moves an entry.
      if (g_pf_publish) table_add(L, seq, e, s);
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
      // writes [seq, n, ids0..29 | seq, ids30..39 | seq, np, preds0..29 | PLINES x (seq, 31 more preds)]
      kernel void bigmoe_post_ids(device const int* ids [[buffer(0)]], device atomic_int* route [[buffer(1)]],
                                  constant int4& m [[buffer(2)]], device int* dep [[buffer(3)]],
                                  device const int* tab [[buffer(5)]], device const int* preds [[buffer(6)]],
                                  constant int& np [[buffer(7)]], uint t [[thread_position_in_grid]]) {
        // dep = [ok, slot0..slot{n-1}]: residency table published a pass ahead, read in parallel
        threadgroup atomic_int ok;
        if (t == 0) atomic_store_explicit(&ok, m.w, memory_order_relaxed);     // m.w = 1 if a table exists
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int n = m.z & 255, nw = (m.z >> 8) & 255;  // [Breakthrough v3f] m.z = n | nw << 8 | nk << 16 (nw 255: flag mode)
        if (m.w && (int)t < n) {
          int e = ids[t] & 0xFFFFF, li = e / 31;   // strip the GPU flags (bit 20 wait, bit 21 late)
          const volatile device int* tb = (const volatile device int*)tab;
          int tag = tb[li * 32], w = tb[li * 32 + 1 + (e - li * 31)];
          bool fresh = tag == m.y && (((uint)w) & 0xFFC00000u) == ((((uint)m.y) & 0x3FFu) << 22);   // [v3h] stamped entry
          int sl = (((uint)w) & 0x3FFFFFu) == 0x3FFFFFu ? -1 : (int)(((uint)w) & 0x3FFFFFu);
          bool must = nw != 255 || ((ids[t] >> 20) & 3) != 0;   // flag mode: the fast path needs only the waited and
          if ((!fresh || sl < 0) && must) atomic_store_explicit(&ok, 0, memory_order_relaxed);   // late ids resident
          dep[1 + t] = fresh ? sl : -1;            // OoO (MoEPrefetch): partial-hit row, -1 = wait for the loader
        }
        if (!m.w && (int)t < n) dep[1 + t] = -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t != 0) return;
        dep[0] = atomic_load_explicit(&ok, memory_order_relaxed);
        device atomic_int* row = route + m.x * (96 + 32 * PLINES + 32 * XLINES);
        const uint STAMP = (((uint)m.y) & 0x3FFu) << 22;               // [v3h] every id / pred word carries seq & 0x3FF
        atomic_store_explicit(&row[65], np, memory_order_relaxed);                 // line 3: next-layer predictions
        int tot = min(30 + 31 * PLINES, (np & 255) + ((np >> 8) & 255) + ((np >> 16) & 255));   // n(L+1) | n(L+2) << 8 | n(L+3) << 16
        for (int i = 0; i < tot; i++)
          atomic_store_explicit(&row[i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31], (int)((((uint)preds[i]) & 0x3FFFFFu) | STAMP), memory_order_relaxed);
        atomic_store_explicit(&row[1], m.z, memory_order_relaxed);     // packed n | nw << 8 | nk << 16 (plain rows: n)
        for (int i = 0; i < n; i++) atomic_store_explicit(&row[i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * PLINES + 1 + (i - 61)], (int)((((uint)ids[i]) & 0x3FFFFFu) | STAMP), memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        for (int j = 0; j < PLINES + XLINES; j++) atomic_store_explicit(&row[96 + 32 * j], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[64], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[32], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[0], m.y, memory_order_relaxed);
      })";
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
          if (l0[0] == seq && l0[32] == seq && (RS == 64 || l0[64] == seq)) {   // [v3h] and every word stamped
            const uint STAMP = (((uint)seq) & 0x3FFu) << 22;
            bool good = true;
            for (int i = 0; i < n && good; i++) {
              uint w = (uint)(i < 31 ? l0[1 + i] : i < 62 ? l0[32 + 1 + (i - 31)] : l0[64 + 1 + (i - 62)]);
              good = (w & 0xFFC00000u) == STAMP;
              out[i] = (w & 0x3FFFFFu) == 0x3FFFFFu ? -1 : (int)(w & 0x3FFFFFu);
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
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(srcs.c_str(), NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) { fprintf(stderr, "bigmoe: spin kernel compile failed\n"); abort(); }
    pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("bigmoe_spin", NS::UTF8StringEncoding)), &err);
  }
  return pso;
}
class SpinWait : public Primitive {
 public:
  SpinWait(Stream s, int L, int seq, int n, int phase = 0) : Primitive(s), L_(L), seq_(seq), n_(n), phase_(phase) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override { throw std::runtime_error("gpu only"); }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override;
  DEFINE_NAME(SpinWait)
  bool is_equivalent(const Primitive&) const override { return false; }
  int L_, seq_, n_, phase_ = 0;   // [v3f] phase 1 = the phase-B region (late experts)
};

// post_ids(ids[int32 n], layer, seq): GPU writes the routing row into the route buffer through a
// genuinely writable binding (stores via const_cast of a read-only input only became
// CPU-visible at command-buffer end). Returns a 1-element dependency array.
class PostIds : public Primitive {
 public:
  PostIds(Stream s, int L, int seq, int n, int64_t tab_off, int np, int wk = 0) : Primitive(s), L_(L), seq_(seq), n_(n), np_(np), wk_(wk), tab_off_(tab_off) {}
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
    int m[4] = {L_, seq_, n_ | (wk_ << 8), (g_tab_arr && tab_off_ >= 0) ? 1 : 0};   // [v3f] wk = nw | nk << 8
    enc.set_bytes(m, 4, 2);
    enc.set_output_array(o, 3);
    if (g_tab_arr) enc.set_input_array(*g_tab_arr, 5, std::max<int64_t>(tab_off_, 0) * 4); else enc.set_input_array(in[0], 5);
    int np = np_;
    enc.set_input_array(in[1], 6);
    enc.set_bytes(np, 7);
    if (g_post_bump) enc.set_input_array(*g_post_bump, 4);   // unused by the kernel; triggers MLX's commit
    enc.dispatch_threads(MTL::Size(g_xlines ? 128 : 64, 1, 1), MTL::Size(g_xlines ? 128 : 64, 1, 1));   // one thread per id (table lookup)
  }
  DEFINE_NAME(PostIds)
  bool is_equivalent(const Primitive&) const override { return false; }
 private:
  int L_, seq_, n_, np_, wk_;
  int64_t tab_off_;
};
static void table_add(int L, int seq, int e, int s) {
  if (!g_tab) return;
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
    for (int j = 0; j < 31; j++) { int64_t e = li * 31 + j; tb[li * 32 + 1 + j] = stamp_enc(e < ly.E ? ly.table[e] : -1, seq); }
  for (auto& [e, s] : ly.pf)                       // a queued (unissued) prefetch has ready = 0
    if ((*ly.ready)[s].load(std::memory_order_acquire)) tb[(e / 31) * 32 + 1 + e % 31] = stamp_enc(s, seq);
  __atomic_thread_fence(__ATOMIC_RELEASE);
  for (int64_t li = 0; li < g_tab_lines; li++) __atomic_store_n(tb + li * 32, seq, __ATOMIC_RELAXED);
  g_st.pub_jit++;
}

static void issue(int L, int32_t e, int s, int32_t seq) {
  Layer* lp = &g_layers.at(L);
  g_inflight++; g_st.pf_issued++;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    load_jobs(*lp, {{e, s}});
    if (g_pf_publish) table_add(L, seq, e, s);     // before ready: a slot is only recycled once ready, so never under it
    (*lp->ready)[s].store(1, std::memory_order_release);
    g_inflight--;
  });
}

// issue queued reads, nearest layer first, then the most precise class, while fewer than QD are in flight
static void pump() {
  while (!g_pq.empty() && (g_qd <= 0 || g_inflight.load() < g_qd)) {
    size_t b = 0;
    for (size_t i = 1; i < g_pq.size(); i++) {
      const PfReq &q = g_pq[i], &c = g_pq[b];
      if (q.seq < c.seq || (q.seq == c.seq && (q.L < c.L || (q.L == c.L && q.prec > c.prec)))) b = i;
    }
    PfReq r = g_pq[b]; g_pq[b] = g_pq.back(); g_pq.pop_back();
    auto& qv = g_layers.at(r.L).queued;
    auto it = std::find(qv.begin(), qv.end(), r.e);
    if (it == qv.end()) { fprintf(stderr, "bigmoe: queued read %d/%d lost\n", r.L, r.e); abort(); }
    qv.erase(it);
    issue(r.L, r.e, r.s, r.seq);
  }
  g_pq_n = (int)g_pq.size();
}

// reserve a slot of layer X for expert e, predicted for pass seq, and queue its read. X is never the layer being
// served or one already published for seq, so no table the GPU will read references the slot.
static bool reserve(int X, int32_t e, int32_t seq, int cls) {
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
  if (!ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); }
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
  g_pq.push_back({X, e, seq, s, (float)cls_prec(cls)}); g_pq_n = (int)g_pq.size();
  g_st.pf_reserved++; g_cls_res[cls]++; g_pass_res++;
  return true;
}

// layer L's row carries predictions for L+1..L+3 (row[65] = n1 | n2 << 8 | n3 << 16 | ranked << 24, ids from row[66]);
// with the oracle, the recorded true ids of L+1..L+D are reserved instead
static void schedule(int L, int32_t seq, volatile int32_t* row) {
  if (seq != g_pass) { g_pass = seq; g_pass_res = 0; g_or_upto = L; }
  if (g_oplay && g_odist > 0) {
    static bool ld = (oracle_load(), true); (void)ld;
    for (int X = std::max(L + 1, g_or_upto + 1); X <= std::min(L + g_odist, g_nl - 1); X++) {
      auto it = g_or.find((int64_t)seq * 1024 + X);
      if (it != g_or.end()) {
        Layer& ly = g_layers.at(X);
        for (int32_t e : it->second) ly.pmark[e] = seq;   // victims never come from X's own true set
        for (int32_t e : it->second) reserve(X, e, seq, NCLS - 1);
      }
      g_or_upto = X;
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
    int32_t u[160]; int cnt[160], first[160], nu = 0;
    for (int i = 0; i < n; i++) {
      int e = p[i];
      if (e < 0 || e >= ly.E) continue;
      int j = slot_of[e];
      if (j < 0) { j = nu++; slot_of[e] = j; u[j] = e; cnt[j] = 0; first[j] = i; }
      cnt[j]++;
    }
    for (int j = 0; j < nu; j++) slot_of[u[j]] = -1;
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
      reserve(X, u[j], seq, cls);
    }
  }
}
static array post_ids(const array& ids, const array& preds, int L, int seq, int np, int wk = 0) {
  return array({1 + (int)ids.size()}, int32,
               std::make_shared<PostIds>(default_stream(Device::gpu), L, seq, (int)ids.size(), tab_offset(L, seq), np, wk), {ids, preds});
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
// [Breakthrough v3f] frugal on the tier engine: flag-mode rows (GPU flags on the routing-row ids: bit 20 = read and wait,
// bit 21 = late-eligible, unflagged = used only if resident, never read). With BIGMOE_DEFER=M the loader releases phase A
// (waited + resident; late ids -1) once the waited ids are in, then reads the late ids and publishes up to M (position,
// slot) pairs as phase B in a second region buffer, which the GPU consumes half a layer later.
static int g_defer = getenv("BIGMOE_DEFER") ? atoi(getenv("BIGMOE_DEFER")) : 0;
static std::atomic<int64_t> g_def_rows{0}, g_def_late{0}, g_def_over{0}, g_def_waited{0};
static int32_t* g_regB = nullptr;    // phase-B regions: same layout as g_reg
static array* g_regB_arr = nullptr;
static void write_region(int L, int seq, const int32_t* sl, int n, bool phaseB) {
  int32_t* reg = ((phaseB && g_regB) ? g_regB : g_reg) + ((int64_t)((uint64_t)seq * g_nl + L) % g_reg_n) * g_pairs * g_rs;
  for (int64_t p = 0; p < g_pairs; p++) {
    int32_t* l0 = reg + p * g_rs; int32_t* l1 = l0 + 32; int32_t* l2 = l0 + 64;
    for (int i = 0; i < n && i < 31; i++) l0[1 + i] = stamp_enc(sl[i], seq);     // [v3h] stamped words
    for (int i = 31; i < n && i < 62; i++) l1[1 + (i - 31)] = stamp_enc(sl[i], seq);
    for (int i = 62; i < n; i++) l2[1 + (i - 62)] = stamp_enc(sl[i], seq);
  }
  __atomic_thread_fence(__ATOMIC_RELEASE);
  for (int64_t p = 0; p < g_pairs; p++) {
    int32_t* l0 = reg + p * g_rs;
    if (g_rs == 96) __atomic_store_n(l0 + 64, seq, __ATOMIC_RELAXED);
    __atomic_store_n(l0 + 32, seq, __ATOMIC_RELAXED); __atomic_store_n(l0, seq, __ATOMIC_RELAXED);
  }
}
static std::thread g_thread;
static std::atomic<bool> g_run{false};
static int g_delay_us = getenv("BIGMOE_DELAY_US") ? atoi(getenv("BIGMOE_DELAY_US")) : 0;

// [Fit12 B1] (2026-10-24) BIGMOE_JOINT=1: one read burst per layer. bt_b12's frugal block posts two gate rows per layer,
// base (MSB-3, row L) and LSB sidecar (row BIGMOE_JOINT_SIDE0 + L), and this thread served them one after the other: the
// sidecar's reads began only after the base row was released, so a layer paid the per-call cost twice (Fit12 C1: ~85 of
// 99 calls per pass stall, ~1.26 ms for ~13.4 MB each). With BIGMOE_JOINT=1 (bigmoe_v3f.gated then posts both rows before
// the first wait), on seeing base row L the loader waits up to BIGMOE_JOINT_WAIT_US for the sidecar row of the same pass
// and starts its missing waited ids reading in the background, into the slots make_room frees for them (LRU tail, never an
// id the row protects), before the base row's own reads. The sidecar row is then served as before: ensure() waits for
// those reads and promotes them. Same evictions, same ids per slot, same bytes: the GPU's results do not change.
// BIGMOE_CALLLOG=path: one line per served row (layer, pass, misses, paired, seen / released times, partner wait), at stop.
static bool g_joint = false;
static int g_side0 = 0;
static double g_joint_wait = 30e-6;
static std::atomic<int64_t> g_jt_tried{0}, g_jt_paired{0}, g_jt_reads{0}, g_jt_late{0}, g_jt_wait_ns{0};
struct CallRec { int32_t L, seq, misses, paired; double t_seen, t_rel, t_wait; };
static std::vector<CallRec> g_clog;
static const char* g_clog_path = nullptr;

static void clog_dump() {
  if (!g_clog_path || g_clog.empty()) return;
  FILE* f = fopen(g_clog_path, "w");
  if (!f) return;
  fprintf(f, "L,seq,misses,paired,t_seen,t_rel,t_wait\n");
  for (auto& r : g_clog) fprintf(f, "%d,%d,%d,%d,%.7f,%.7f,%.7f\n", r.L, r.seq, r.misses, r.paired, r.t_seen, r.t_rel, r.t_wait);
  fclose(f);
  g_clog.clear();
}

// caller holds g_mu, serving base row L of pass seq: start the sidecar row's missing waited ids reading in the background.
// Returns 1 if the sidecar row was there (and had no late ids); *wait = time spent looking for it.
static int joint_side(int L, int32_t seq, double* wait) {
  int P = g_side0 + L;
  *wait = 0;
  if (P >= g_nl) return 0;
  g_jt_tried++;
  volatile int32_t* row = g_route + (int64_t)P * g_row;
  double t0 = now_s();
  bool ok = false;
  for (;;) {
    if (row[0] == seq) {
      ok = row[32] == seq && row[64] == seq;
      for (int j = 0; j < g_plines + g_xlines && ok; j++) ok = row[96 + 32 * j] == seq;
      if (ok) break;
    }
    if (now_s() - t0 > g_joint_wait) break;
  }
  *wait = now_s() - t0;
  g_jt_wait_ns += (int64_t)(*wait * 1e9);
  if (!ok) return 0;
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  int raw = row[1];
  int n = raw & 255, nw = (raw >> 8) & 255;
  if (nw != 255 || n <= 0 || n > g_maxk) return 0;  // flag-mode rows only
  Layer& ly = g_layers.at(P);
  std::vector<int32_t> wl, keep;                     // as the flag-mode service: waited ids, then the resident ids it
  for (int i = 0; i < n; i++) {                      // lists while slots remain (make_room never evicts those)
    int32_t v = row[ipos(i)];
    if (!stamp_ok(v, seq)) { g_stamp_joint++; return 0; }   // [v3h] a stale word: the row's own service handles it
    v = (int32_t)((uint32_t)v & 0x3FFFFFu);
    if ((v >> 21) & 1) { g_jt_late++; return 0; }    // late ids: phase A/B service, left to the row itself
    int e = v & 0xFFFFF;
    if (e < 0 || e >= ly.E) return 0;
    if (((v >> 20) & 1) && !in(wl, e)) wl.push_back(e);
  }
  keep = wl;
  int room = ly.C_main - (int)keep.size();
  for (int i = 0; i < n && room > 0; i++) {
    int32_t v = row[ipos(i)];
    if (!stamp_ok(v, seq)) { g_stamp_joint++; return 0; }   // [v3h]
    int e = v & 0xFFFFF;
    if (ly.table[e] >= 0 && !in(keep, e)) { keep.push_back(e); room--; }
  }
  g_jt_paired++;
  Layer* lp = &ly;
  for (int e : wl) {
    if (ly.table[e] >= 0 || ly.pf.count(e)) continue;
    make_room(ly, keep);
    if (ly.free_slots.empty()) break;                // everything resident is protected: ensure() takes it from here
    int s = ly.free_slots.back(); ly.free_slots.pop_back();
    ly.pf[e] = s; ly.pf_order.push_back(e); (*ly.ready)[s].store(0);
    g_jt_reads++;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
      load_jobs(*lp, {{e, s}});
      (*lp->ready)[s].store(1, std::memory_order_release);
    });
  }
  return 1;
}

static void loader_main() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   // keep the spinner on a P-core
  g_joint = getenv("BIGMOE_JOINT") && atoi(getenv("BIGMOE_JOINT")) > 0;   // [Fit12 B1] read here: set after import
  g_side0 = getenv("BIGMOE_JOINT_SIDE0") ? atoi(getenv("BIGMOE_JOINT_SIDE0")) : 0;
  if (getenv("BIGMOE_JOINT_WAIT_US")) g_joint_wait = atof(getenv("BIGMOE_JOINT_WAIT_US")) * 1e-6;
  if (g_joint && g_side0 <= 0) { fprintf(stderr, "bigmoe: BIGMOE_JOINT needs BIGMOE_JOINT_SIDE0\n"); g_joint = false; }
  g_clog_path = getenv("BIGMOE_CALLLOG");
  if (g_clog_path) g_clog.reserve(1 << 20);
  std::vector<int32_t> last(g_nl, 0);
  double t_idle = now_s();
  while (g_run.load(std::memory_order_relaxed)) {
    bool any = false;
    for (int L = 0; L < g_nl; L++) {
      volatile int32_t* row = g_route + (int64_t)L * g_row;   // 3 (+ BIGMOE_PRED_LINES) cache lines, each with its own seq
      int32_t seq = row[0];
      if (seq == last[L] || seq == 0) continue;
      if (g_joint && L >= g_side0 && L - g_side0 < g_side0 - 1) {   // [Fit12 B1 v2] a sidecar row waits for its base row
        int Bl = L - g_side0;                                        // of the same pass when that one is posted and not
        volatile int32_t* br = g_route + (int64_t)Bl * g_row;        // served yet: the base row's service starts the
        if (br[0] == seq && last[Bl] != seq) continue;               // sidecar's reads beside its own (v1 served the
      }                                                              // sidecar first whenever the scan met it first)
      for (bool done = false; !done;) {            // all lines complete
        done = row[32] == seq && row[64] == seq;
        for (int j = 0; j < g_plines + g_xlines && done; j++) done = row[96 + 32 * j] == seq;
      }
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      any = true; last[L] = seq;
      double t_seen = now_s();
      g_st.wait_ns += (int64_t)((now_s() - t_idle) * 1e9);
      int raw = row[1];
      int n = raw & 255, nw = (raw >> 8) & 255, nk = (raw >> 16) & 255;   // [v3f] packed by post_ids (plain rows: n)
      bool flagmode = nw == 255;
      bool split_early = false; int32_t split_lb[160]; int split_lb_n = 0;
      bool slow_call = false, victim_done = false;
      int clog_miss = 0, clog_pair = 0; double clog_wait = 0;   // [Fit12 B1]
      if (n < 0 || n > g_maxk) { fprintf(stderr, "bigmoe: bad n=%d at layer %d\n", n, L); abort(); }
      {
        std::lock_guard<std::mutex> lk(g_mu);
        int32_t ids_copy[256];
        {                                            // [v3h] fresh line tags != fresh sectors: every id word must carry
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
        int32_t wl[256]; int nwl = 0; bool lf[256] = {false};
        if (flagmode)                                // [v3f] collect the waited (bit 20) and late (bit 21) ids, strip flags
          for (int i = 0; i < n; i++) {
            bool f = (ids_copy[i] >> 20) & 1;
            lf[i] = (ids_copy[i] >> 21) & 1;
            ids_copy[i] &= 0xFFFFF;
            bool seen = false;
            for (int j = 0; j < nwl && !seen; j++) seen = wl[j] == ids_copy[i];
            if (f && !seen) wl[nwl++] = ids_copy[i];
          }
        if (g_dbg) memcpy(g_dbg + ring_off(L, (uint64_t)seq), ids_copy, n * 4);
        int64_t m0 = g_st.misses.load(), j0 = g_st.stalled.load();
        g_cur_seq = seq;
        if (g_jit) { g_srv[L] = seq; if (g_early) { schedule(L, seq, row); pump(); } }
        if (flagmode) {                            // [v3f] frugal: read and wait for the flagged ids only
          Layer& ly = g_layers.at(L);
          int kk = std::min(nk ? nk : n, n);
          std::vector<int32_t> wnow(wl, wl + nwl), late, all;
          for (int i = 0; i < n; i++) {
            int e = ids_copy[i];
            if (!lf[i] || in(wnow, e) || in(late, e) || ly.table[e] >= 0) continue;
            if (g_defer > 0) late.push_back(e); else wnow.push_back(e);   // no phase B: late-eligible ids are waited for
          }
          int over = (int)late.size() - g_defer;   // more late ids than pairs: the first ones (rank order) are waited for
          if (over > 0) { wnow.insert(wnow.end(), late.begin(), late.begin() + over); late.erase(late.begin(), late.begin() + over); g_def_over += over; }
          // protect what fits: the waited and late ids, then resident ids in list order while slots remain (one-row calls
          // fit whole, which the late path needs: phase A may compute any resident id while the late reads run); an id
          // evicted for a waited read is released as -1, so the GPU never reads a reused slot
          for (int e : wnow) if (!in(all, e)) all.push_back(e);
          for (int e : late) if (!in(all, e)) all.push_back(e);
          int room = ly.C_main - (int)all.size();
          for (int i = 0; i < n && room > 0; i++)
            if (ly.table[ids_copy[i]] >= 0 && !in(all, ids_copy[i])) { all.push_back(ids_copy[i]); room--; }
          for (int e : wnow) g_def_waited += ly.table[e] < 0;
          int32_t tmp[256];
          if (g_joint && L < g_side0) clog_pair = joint_side(L, seq, &clog_wait);   // [Fit12 B1] sidecar reads first
          if (!wnow.empty()) ensure(L, wnow.data(), (int)wnow.size(), tmp, &all);
          for (int i = 0; i < kk; i++) {          // LRU-touch the resident ids the GPU may use (as fork125's ensure_wait)
            int e = ids_copy[i];
            if (in(wnow, e) || in(late, e) || ly.table[e] < 0) continue;
            ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
          }
          int32_t* ring = g_ring + ring_off(L, (uint64_t)seq);
          for (int i = 0; i < n; i++) ring[i] = in(late, ids_copy[i]) ? -1 : ly.table[ids_copy[i]];
          if (g_defer > 0 && g_reg) {
            write_region(L, seq, ring, n, false);  // phase A: the GPU goes on
            split_early = true;
            if (!late.empty()) ensure(L, late.data(), (int)late.size(), tmp, &all);
            int q = 0;
            for (int i = 0; i < n && q < g_defer; i++)
              if (lf[i] && in(late, ids_copy[i])) { split_lb[2 * q] = i; split_lb[2 * q + 1] = ly.table[ids_copy[i]]; q++; }
            for (; q < g_defer; q++) { split_lb[2 * q] = n; split_lb[2 * q + 1] = 0; }
            split_lb_n = 2 * g_defer;
            g_def_late += (int64_t)late.size();
          }
          g_def_rows++;
        }
        else ensure(L, ids_copy, n, g_ring + ring_off(L, (uint64_t)seq));
        if (L + 1 < g_nl) {                         // remember L+1's look-ahead set for the miss split
          int np0 = row[65], k1 = std::min(np0 & 255, g_pcap);
          Layer& nx = g_layers.at(L + 1); nx.pred.assign(k1, 0);
          for (int i = 0; i < k1; i++) nx.pred[i] = stamp_dec(row[pidx(i)]);
          nx.pred_seq = seq;
        }
        slow_call = g_st.misses.load() != m0;
        clog_miss = (int)(g_st.misses.load() - m0);
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
      if (g_reg && split_early) {                  // [v3f] phase A went out before the late reads: publish phase B
        write_region(L, seq, split_lb, split_lb_n, true);
      } else if (g_reg) {   // spin mode: write [seq, slots...] into every line pair of this (layer, pass) region
        write_region(L, seq, g_ring + ring_off(L, (uint64_t)seq), n, false);   // [v3h] same layout, stamped words
      } else {
        g_events[L]->setSignaledValue((uint64_t)seq);
      }
      double t_rel = now_s();                      // GPU released: bookkeeping below is off the critical path
      if (g_clog_path && g_clog.size() < (1u << 20)) g_clog.push_back({L, seq, clog_miss, clog_pair, t_seen, t_rel, clog_wait});
      if (slow_call) { g_st.slow_calls++; g_st.slow_ns += (int64_t)((t_rel - t_seen) * 1e9); }
      if (g_tab && !g_jit) {   // publish layer L's residency for the next pass (valid until that pass serves L)
        Layer& ly = g_layers.at(L);
        int32_t nseq = seq + 1;
        int32_t* tb = g_tab + ((int64_t)((uint64_t)nseq * g_nl + L) % g_tab_n) * g_tab_lines * 32;
        for (int64_t li = 0; li < g_tab_lines; li++)
          for (int j = 0; j < 31; j++) { int64_t e = li * 31 + j; tb[li * 32 + 1 + j] = stamp_enc(e < ly.E ? ly.table[e] : -1, nseq); }
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
        if (!g_early) schedule(L, seq, row);
        if (L + 1 < g_nl) publish_jit(L + 1, seq);
        pump();
      } else if (n1 + n2 > 0 && !victim_done) {
        int32_t pr[160];
        for (int i = 0; i < n1 + n2; i++) pr[i] = stamp_dec(row[pidx(i)]);
        std::lock_guard<std::mutex> lk(g_mu);
        if (n1 && L + 1 < g_nl) prefetch(L + 1, pr, n1, seq);
        if (n2 && L + 2 < g_nl) prefetch(L + 2, pr + n1, n2, seq);
      }
      if (g_trace) { std::lock_guard<std::mutex> lk(g_tmu); g_tr.push_back({L, (int64_t)seq, 1, t_seen, t_rel});
                     if (slow_call) g_tr.push_back({L, (int64_t)seq, 4, t_seen, t_rel}); }   // 4 = slow (had misses)
      t_idle = t_now;
    }
    if (!any) {
      if (g_jit && g_pq_n.load(std::memory_order_relaxed) > 0 && (g_qd <= 0 || g_inflight.load() < g_qd)) {
        std::lock_guard<std::mutex> lk(g_mu); pump();   // a speculative read finished: issue the next one
      }
      __builtin_arm_yield();
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
  if (phase_ && g_regB_arr) enc.set_input_array(*g_regB_arr, 1, reg_off * 4);   // [v3f] phase B: its own buffer
  else enc.set_input_array(*g_reg_arr, 1, reg_off * 4);
  int m[4] = {L_, seq_, n_, (int)g_pairs};
  enc.set_bytes(m, 4, 2);
  // status is bound as an INPUT: registering a persistent buffer as an output every layer made MLX
  // fence each spin against the previous one (~45 us/layer). The kernel's atomics need no tracking.
  if (getenv("BIGMOE_STATUS_OUT")) enc.set_output_array(*g_status_arr, 3); else enc.set_input_array(*g_status_arr, 3);
  enc.set_output_array(o, 4);
  enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
static array spin_wait(const array& dep, int L, int seq, int n, int phase = 0) {
  return array({n}, int32, std::make_shared<SpinWait>(default_stream(Device::gpu), L, seq, n, phase), {dep});
}

NB_MODULE(bigmoe_ext, m) {
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
  // [Fit12 PFR] reset_layer(L, pools): layer L's pool pieces now live in new buffers (same C, E, files, offsets, piece
  // sizes); every slot is empty, as right after add_layer. Only between generations (no pass in flight, loader idle);
  // call invalidate_tables() after the last reset so no pass reads a table published before it.
  m.def("reset_layer", [](int L, std::vector<array> pools) {
    std::lock_guard<std::mutex> lk(g_mu);
    Layer& ly = g_layers.at(L);
    if (pools.size() != ly.pieces.size()) throw std::runtime_error("reset_layer: piece count differs");
    for (size_t i = 0; i < pools.size(); i++) { auto& p = pools[i]; p.eval(); ly.pieces[i].pool = p.data<uint8_t>(); }
    std::fill(ly.table.begin(), ly.table.end(), -1);
    std::fill(ly.slot_owner.begin(), ly.slot_owner.end(), -1);
    std::fill(ly.pmark.begin(), ly.pmark.end(), -1);
    ly.lru.clear();
    ly.where.assign(ly.E, std::list<int32_t>::iterator());
    ly.free_slots.clear();
    for (int s = ly.C - 1; s >= 0; s--) ly.free_slots.push_back(s);
    ly.pf.clear(); ly.pf_order.clear(); ly.queued.clear(); ly.meta.clear();
    for (auto& r : *ly.ready) r.store(0);
    ly.pred.clear(); ly.pred_seq = -1;
  });
  m.def("invalidate_tables", []() {                 // [Fit12 PFR] every line's tag -1: matches no pass
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_tab) return;
    for (int64_t i = 0; i < g_tab_n * g_tab_lines; i++) __atomic_store_n(g_tab + i * 32, -1, __ATOMIC_RELEASE);
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
          if (lp->rec_fmt == 3) {                      // records3: bounce + rebuild into the 4-bit staging layout
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
    for (int e : used) if (ly.table[e] >= 0) { ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); }
    std::vector<uint8_t*> src; for (auto& a : srcs) src.push_back(a.data<uint8_t>());
    std::vector<std::pair<int, int>> jobs;
    for (int e : cand) {
      if (ly.table[e] >= 0) continue;
      int s;
      int extra = std::max(0, (int)ly.pf.size() - ly.spare);
      if ((int)ly.lru.size() + extra < ly.C_main && !ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); }
      else {
        if (ly.lru.empty()) break;
        int v = ly.lru.back();
        if (in(used, v)) break;
        ly.lru.pop_back(); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      }
      jobs.push_back({s, e});
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
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
  m.def("read_rebuilt", [](std::string path, int64_t recsz, int64_t msb_sz, std::vector<int64_t> pers4,
                           std::vector<int32_t> ids, std::vector<int32_t> rows, std::vector<array> dst) {
    // plan-mode exact prefill: experts ids[k] from records3, rebuilt to the 4-bit pieces, into row rows[k] of the 9
    // caller-owned arrays dst (GPU must be done with them). Parallel over experts; same rebuild as BIGMOE_RECORDS=3.
    static std::unordered_map<std::string, int> fds;
    int fd;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = fds.find(path);
      if (it == fds.end()) { fd = open(path.c_str(), O_RDONLY); if (fd < 0) abort(); fcntl(fd, F_NOCACHE, 1); fds[path] = fd; }
      else fd = it->second; }
    Layer ly; ly.rec_fd = fd; ly.recsz = recsz; ly.rec_fmt = 3; ly.msb_sz = msb_sz;
    for (auto p : pers4) ly.pieces.push_back({-1, 0, p, nullptr});
    std::vector<uint8_t*> d;
    for (auto& a : dst) { a.eval(); d.push_back(a.data<uint8_t>()); }
    const Layer* lp = &ly; const auto* idp = &ids; const auto* rp = &rows; const auto* dp = &d;
    dispatch_apply(ids.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t k) {
      uint8_t* t[16];
      for (size_t i = 0; i < lp->pieces.size(); i++) t[i] = (*dp)[i] + (int64_t)(*rp)[k] * lp->pieces[i].per;
      read_expert(lp, (*idp)[k], t, false);
    });
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
    for (int e : used) if (ly.table[e] >= 0) { ly.lru.erase(ly.where[e]); ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); }
    int adopted = 0;
    std::vector<std::pair<int, int>> jobs;            // (pool slot, staging index)
    for (size_t k = 0; k < cand.size(); k++) {
      int e = cand[k];
      if (ly.table[e] >= 0) continue;
      int s;
      int extra = std::max(0, (int)ly.pf.size() - ly.spare);
      if ((int)ly.lru.size() + extra < ly.C_main && !ly.free_slots.empty()) { s = ly.free_slots.back(); ly.free_slots.pop_back(); }
      else {
        if (ly.lru.empty()) break;
        int v = ly.lru.back();
        if (in(used, v)) break;                        // the tail is used this chunk: the pool holds only used experts
        ly.lru.pop_back(); s = ly.table[v]; ly.table[v] = -1; ly.slot_owner[s] = -1;
      }
      jobs.push_back({s, sidx[k]});
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin(); adopted++;
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
  });
  m.def("stop", []() { g_run = false; if (g_thread.joinable()) g_thread.join(); clog_dump(); });
  m.def("gate", &gate);
  m.def("flush", &flush_enc);
  m.def("post_ids", &post_ids, nb::arg("ids"), nb::arg("preds"), nb::arg("L"), nb::arg("seq"), nb::arg("np"), nb::arg("wk") = 0);
  m.def("set_regions_b", [](array reg) { reg.eval(); g_regB = reg.data<int32_t>(); g_regB_arr = new array(reg); });   // [v3f]
  m.def("defer_stats", []() { return std::vector<int64_t>{(int64_t)g_def_rows, (int64_t)g_def_waited, (int64_t)g_def_late, (int64_t)g_def_over}; });
  m.def("defer_reset", []() { g_def_rows = 0; g_def_waited = 0; g_def_late = 0; g_def_over = 0; });
  m.def("trace", []() { std::lock_guard<std::mutex> lk(g_tmu); std::vector<std::vector<double>> o;
    for (auto& e : g_tr) o.push_back({(double)e.L, (double)e.seq, (double)e.kind, e.t0, e.t1}); g_tr.clear(); return o; });
  m.def("spin_wait", &spin_wait, nb::arg("dep"), nb::arg("L"), nb::arg("seq"), nb::arg("n"), nb::arg("phase") = 0);
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
    d["joint_tried"] = g_jt_tried.load(); d["joint_paired"] = g_jt_paired.load(); d["joint_reads"] = g_jt_reads.load();
    d["joint_late"] = g_jt_late.load(); d["joint_wait_s"] = g_jt_wait_ns.load() * 1e-9;   // [Fit12 B1]
    d["stamp_calls"] = g_stamp_calls.load(); d["stamp_rereads"] = g_stamp_rereads.load();   // [v3h] routing-row re-reads
    d["stamp_wait_s"] = g_stamp_ns.load() * 1e-9; d["stamp_joint"] = g_stamp_joint.load();
    d["slow_calls"] = g_st.slow_calls.load(); d["slow_s"] = g_st.slow_ns.load() * 1e-9; d["pf_victims"] = g_st.pf_victims.load();
    d["pf_ontime"] = g_st.pf_ontime.load(); d["pf_late"] = g_st.pf_late.load(); d["miss_pred_unissued"] = g_st.miss_pred_unissued.load(); d["miss_unpred"] = g_st.miss_unpred.load();
    d["pf_reserved"] = g_st.pf_reserved.load(); d["pf_cancel"] = g_st.pf_cancel.load(); d["pf_qdemand"] = g_st.pf_qdemand.load();
    d["pf_gated_off"] = g_st.pf_gated_off.load(); d["pf_nofit"] = g_st.pf_nofit.load(); d["pub_jit"] = g_st.pub_jit.load();
    std::vector<std::vector<double>> cls;            // per class: reserved, scored, used, precision (d1 b0..2, d2, d3, oracle)
    for (int c = 0; c < NCLS; c++) cls.push_back({(double)g_cls_res[c], g_cls_iss[c], g_cls_used[c], cls_prec(c)});
    d["pf_cls"] = cls; return d;
  });
  m.def("reset_stats", []() { g_st.uses = 0; g_st.misses = 0; g_st.stalled = 0; g_st.gated = 0; g_st.io_ns = 0; g_st.wait_ns = 0; g_st.proc_ns = 0; g_st.pf_issued = 0; g_st.pf_used = 0; g_st.pf_waited_ns = 0; g_st.slow_calls = 0; g_st.slow_ns = 0; g_st.pf_victims = 0; g_st.pf_ontime = 0; g_st.pf_late = 0; g_st.miss_pred_unissued = 0; g_st.miss_unpred = 0;
    g_st.pf_reserved = 0; g_st.pf_cancel = 0; g_st.pf_qdemand = 0; g_st.pf_gated_off = 0; g_st.pf_nofit = 0; g_st.pub_jit = 0;
    for (int c = 0; c < NCLS; c++) g_cls_res[c] = 0; });   // the class precisions keep learning across resets
}
