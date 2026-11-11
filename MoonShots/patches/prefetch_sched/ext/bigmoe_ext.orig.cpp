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
#include <dispatch/dispatch.h>
#include <time.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
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
  std::vector<int32_t> pred; int32_t pred_seq = -1;  // look-ahead prediction for this layer (from L-1's row)
};
static int32_t g_cur_seq = -1;
// prediction i of a routing row: line 3 holds 0..29 (after [seq, np]), line 4 holds 30..60 (after its seq)
#define PRED(row, i) ((i) < 30 ? (row)[66 + (i)] : (row)[97 + ((i) - 30)])                      // pass being served (for the miss split)

struct Stats { std::atomic<int64_t> uses{0}, misses{0}, stalled{0}, gated{0}; std::atomic<int64_t> io_ns{0}, wait_ns{0}, proc_ns{0};
               std::atomic<int64_t> pf_issued{0}, pf_used{0}, pf_waited_ns{0}, slow_calls{0}, slow_ns{0}, pf_victims{0},
               pf_ontime{0}, pf_late{0}, miss_pred_unissued{0}, miss_unpred{0}; };   // demand-miss split

static std::vector<Layer> g_layers;
static std::mutex g_mu;
static Stats g_st;

static void load_jobs(Layer& ly, const std::vector<std::pair<int, int>>& jobs) {
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
static void ensure(int L, const int32_t* ids, int n, int32_t* out) {
  Layer& ly = g_layers.at(L);
  std::vector<int32_t> need; need.reserve(n);
  for (int i = 0; i < n; i++) {
    int e = ids[i];
    if (e < 0 || e >= ly.E) { fprintf(stderr, "bigmoe: bad expert id %d at layer %d\n", e, L); abort(); }
    if (!in(need, e)) need.push_back(e);
  }
  if ((int)need.size() > ly.C_main) { fprintf(stderr, "bigmoe: %zu experts > C=%d at layer %d\n", need.size(), ly.C_main, L); abort(); }
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
      make_room(ly, need);
      ly.table[e] = s; ly.slot_owner[s] = e; ly.lru.push_front(e); ly.where[e] = ly.lru.begin();
      g_st.pf_used++;
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
    const char* src = R"(
      #include <metal_stdlib>
      using namespace metal;
      // writes [seq, n, ids0..29 | seq, ids30..39] (two 128-byte lines, seq last in each)
      kernel void bigmoe_post_ids(device const int* ids [[buffer(0)]], device atomic_int* route [[buffer(1)]],
                                  constant int4& m [[buffer(2)]], device int* dep [[buffer(3)]],
                                  device const int* tab [[buffer(5)]], device const int* preds [[buffer(6)]],
                                  constant int& np [[buffer(7)]], uint t [[thread_position_in_grid]]) {
        // dep = [ok, slot0..slot{n-1}]: residency table published a pass ahead, read in parallel
        threadgroup atomic_int ok;
        if (t == 0) atomic_store_explicit(&ok, m.w, memory_order_relaxed);     // m.w = 1 if a table exists
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int n = m.z;
        if (m.w && (int)t < n) {
          int e = ids[t], li = e / 31;
          const volatile device int* tb = (const volatile device int*)tab;
          int tag = tb[li * 32], sl = tb[li * 32 + 1 + (e - li * 31)];
          if (tag != m.y || sl < 0) atomic_store_explicit(&ok, 0, memory_order_relaxed);
          dep[1 + t] = sl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t != 0) return;
        dep[0] = atomic_load_explicit(&ok, memory_order_relaxed);
        device atomic_int* row = route + m.x * 128;
        atomic_store_explicit(&row[65], np, memory_order_relaxed);                 // line 3: next-layer predictions
        int tot = min(61, (np & 255) + (np >> 8));                                  // np = n(L+1) | n(L+2) << 8
        for (int i = 0; i < tot; i++) atomic_store_explicit(&row[i < 30 ? 66 + i : 97 + (i - 30)], preds[i], memory_order_relaxed);
        atomic_store_explicit(&row[1], n, memory_order_relaxed);
        for (int i = 0; i < n; i++) atomic_store_explicit(&row[i < 30 ? 2 + i : 33 + (i - 30)], ids[i], memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_store_explicit(&row[96], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[64], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[32], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[0], m.y, memory_order_relaxed);
      })";
    NS::Error* err = nullptr;
    auto* lib = dev->newLibrary(NS::String::string(src, NS::UTF8StringEncoding), nullptr, &err);
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
    std::string srcs = std::string(getenv("BIGMOE_NOCOUNT") ? "#define COUNT_FAST 0\n" : "") + R"(
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
          const volatile device int* l0 = (const volatile device int*)(reg + p * 64);
          if (l0[0] == seq && l0[32] == seq) { got = p; break; }
          for (int j = 0; j < (int)d; j++) x = x * 1.0000001f + 1e-7f;
          d = min(d * 1.1f, 400000.0f);
        }
        if (got < 0) { atomic_fetch_add_explicit(status, 1, memory_order_relaxed); for (int i = 0; i < n; i++) out[i] = 0; return; }
        const volatile device int* l0 = (const volatile device int*)(reg + got * 64);
        for (int i = 0; i < n; i++) out[i] = i < 31 ? l0[1 + i] : l0[32 + 1 + (i - 31)];
        if (x == 12345.0f) out[0] = 0;
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
    enc.dispatch_threads(MTL::Size(64, 1, 1), MTL::Size(64, 1, 1));
  }
  DEFINE_NAME(PostIds)
  bool is_equivalent(const Primitive&) const override { return false; }
 private:
  int L_, seq_, n_, np_;
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
static int g_delay_us = getenv("BIGMOE_DELAY_US") ? atoi(getenv("BIGMOE_DELAY_US")) : 0;

static void loader_main() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   // keep the spinner on a P-core
  std::vector<int32_t> last(g_nl, 0);
  double t_idle = now_s();
  while (g_run.load(std::memory_order_relaxed)) {
    bool any = false;
    for (int L = 0; L < g_nl; L++) {
      volatile int32_t* row = g_route + (int64_t)L * 128;  // 4 cache lines, each with its own seq
      int32_t seq = row[0];
      if (seq == last[L] || seq == 0) continue;
      while (row[32] != seq || row[64] != seq || row[96] != seq) {}  // all lines complete
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      any = true; last[L] = seq;
      double t_seen = now_s();
      g_st.wait_ns += (int64_t)((now_s() - t_idle) * 1e9);
      int n = row[1];
      bool slow_call = false, victim_done = false;
      if (n < 0 || n > g_maxk) { fprintf(stderr, "bigmoe: bad n=%d at layer %d\n", n, L); abort(); }
      {
        std::lock_guard<std::mutex> lk(g_mu);
        int32_t ids_copy[256];
        for (int i = 0; i < n; i++) ids_copy[i] = row[i < 30 ? 2 + i : 33 + (i - 30)];
        if (g_dbg) memcpy(g_dbg + ring_off(L, (uint64_t)seq), ids_copy, n * 4);
        int64_t m0 = g_st.misses.load(), j0 = g_st.stalled.load();
        g_cur_seq = seq;
        ensure(L, ids_copy, n, g_ring + ring_off(L, (uint64_t)seq));
        if (L + 1 < g_nl) {                         // remember L+1's look-ahead set for the miss split
          int np0 = row[65], k1 = std::min(np0 & 255, 61);
          Layer& nx = g_layers.at(L + 1); nx.pred.assign(k1, 0);
          for (int i = 0; i < k1; i++) nx.pred[i] = PRED(row, i);
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
        if (g_pf_victim > 0 && g_st.stalled.load() != j0) {
          if (g_oplay) {
            static bool ld = (oracle_load(), true); (void)ld;
            auto it = g_or.find((int64_t)seq * 1024 + L + 1);
            if (it != g_or.end() && L + 1 < g_nl) prefetch(L + 1, it->second.data(), (int)it->second.size(), seq, true);
          } else {
          int np = row[65], n1 = std::min(np & 255, 61), n2 = std::min(np >> 8, 61 - n1);
          int32_t pr[64];
          for (int i = 0; i < n1 + n2; i++) pr[i] = PRED(row, i);
          if (n1 && L + 1 < g_nl) prefetch(L + 1, pr, n1, seq, true);
          if (n2 && L + 2 < g_nl) prefetch(L + 2, pr + n1, n2, seq, true);
          }
          victim_done = true;
        }
      }
      __atomic_thread_fence(__ATOMIC_RELEASE);
      if (g_delay_us > 0) { double t0 = now_s(); while (now_s() - t0 < g_delay_us * 1e-6) {} }   // test only
      g_st.gated++;
      if (g_reg) {   // spin mode: write [seq, slots...] into every line pair of this (layer, pass) region
        int32_t* reg = g_reg + ((int64_t)((uint64_t)seq * g_nl + L) % g_reg_n) * g_pairs * 64;
        const int32_t* sl = g_ring + ring_off(L, (uint64_t)seq);
        for (int64_t p = 0; p < g_pairs; p++) {
          int32_t* l0 = reg + p * 64; int32_t* l1 = l0 + 32;
          for (int i = 0; i < n && i < 31; i++) l0[1 + i] = sl[i];
          for (int i = 31; i < n; i++) l1[1 + (i - 31)] = sl[i];
        }
        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (int64_t p = 0; p < g_pairs; p++) {
          int32_t* l0 = reg + p * 64;
          __atomic_store_n(l0 + 32, seq, __ATOMIC_RELAXED); __atomic_store_n(l0, seq, __ATOMIC_RELAXED);
        }
      } else {
        g_events[L]->setSignaledValue((uint64_t)seq);
      }
      double t_rel = now_s();                      // GPU released: bookkeeping below is off the critical path
      if (slow_call) { g_st.slow_calls++; g_st.slow_ns += (int64_t)((t_rel - t_seen) * 1e9); }
      if (g_tab) {   // publish layer L's residency for the next pass (valid until that pass serves L)
        Layer& ly = g_layers.at(L);
        int32_t nseq = seq + 1;
        int32_t* tb = g_tab + ((int64_t)((uint64_t)nseq * g_nl + L) % g_tab_n) * g_tab_lines * 32;
        for (int64_t li = 0; li < g_tab_lines; li++)
          for (int j = 0; j < 31; j++) { int64_t e = li * 31 + j; tb[li * 32 + 1 + j] = e < ly.E ? ly.table[e] : -1; }
        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (int64_t li = 0; li < g_tab_lines; li++) __atomic_store_n(tb + li * 32, nseq, __ATOMIC_RELAXED);
      }
      double t_now = now_s();
      g_st.proc_ns += (int64_t)((t_now - t_seen) * 1e9);
      int np = row[65];                            // predicted experts -> async prefetch:
      int n1 = std::min(np & 255, 61), n2 = std::min(np >> 8, 61 - n1);   // first for L+1, then for L+2
      if (n1 + n2 > 0 && !victim_done) {
        int32_t pr[64];
        for (int i = 0; i < n1 + n2; i++) pr[i] = PRED(row, i);
        std::lock_guard<std::mutex> lk(g_mu);
        if (n1 && L + 1 < g_nl) prefetch(L + 1, pr, n1, seq);
        if (n2 && L + 2 < g_nl) prefetch(L + 2, pr + n1, n2, seq);
      }
      if (g_trace) { std::lock_guard<std::mutex> lk(g_tmu); g_tr.push_back({L, (int64_t)seq, 1, t_seen, t_rel});
                     if (slow_call) g_tr.push_back({L, (int64_t)seq, 4, t_seen, t_rel}); }   // 4 = slow (had misses)
      t_idle = t_now;
    }
    if (!any) __builtin_arm_yield();
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
      int64_t ro = ((int64_t)((uint64_t)seq_ * g_nl + L_) % g_reg_n) * g_pairs * 64;
      enc.set_input_array(*g_reg_arr, 1, ro * 4); enc.set_input_array(*g_status_arr, 3);
    }
    int m[4] = {L_, seq_, n_, 0}; enc.set_bytes(m, 4, 2);
    enc.set_output_array(o, 4);
    enc.dispatch_threads(MTL::Size(n_, 1, 1), MTL::Size(n_, 1, 1));
    return;
  }
  int64_t reg_off = ((int64_t)((uint64_t)seq_ * g_nl + L_) % g_reg_n) * g_pairs * 64;
  enc.set_compute_pipeline_state(spin_pso());
  enc.set_input_array(in[0], 0);
  enc.set_input_array(*g_reg_arr, 1, reg_off * 4);
  int m[4] = {L_, seq_, n_, (int)g_pairs};
  enc.set_bytes(m, 4, 2);
  // status is bound as an INPUT: registering a persistent buffer as an output every layer made MLX
  // fence each spin against the previous one (~45 us/layer). The kernel's atomics need no tracking.
  if (getenv("BIGMOE_STATUS_OUT")) enc.set_output_array(*g_status_arr, 3); else enc.set_input_array(*g_status_arr, 3);
  enc.set_output_array(o, 4);
  enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
static array spin_wait(const array& dep, int L, int seq, int n) {
  return array({n}, int32, std::make_shared<SpinWait>(default_stream(Device::gpu), L, seq, n), {dep});
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
    while ((int)g_events.size() < g_nl) { auto* e = metal::device(Device::gpu).mtl_device()->newSharedEvent(); e->setSignaledValue(0); g_events.push_back(e); }
    g_run = true; g_thread = std::thread(loader_main);
  });
  m.def("stop", []() { g_run = false; if (g_thread.joinable()) g_thread.join(); });
  m.def("gate", &gate);
  m.def("flush", &flush_enc);
  m.def("post_ids", &post_ids);
  m.def("trace", []() { std::lock_guard<std::mutex> lk(g_tmu); std::vector<std::vector<double>> o;
    for (auto& e : g_tr) o.push_back({(double)e.L, (double)e.seq, (double)e.kind, e.t0, e.t1}); g_tr.clear(); return o; });
  m.def("spin_wait", &spin_wait);
  m.def("set_status", [](array st) { st.eval(); g_status_arr = new array(st); });
  m.def("set_regions", [](array reg, int64_t pairs) { reg.eval(); g_reg = reg.data<int32_t>(); g_pairs = pairs; g_reg_arr = new array(reg);
    g_reg_n = (int64_t)reg.size() / (pairs * 64); });
  m.def("set_tables", [](array tab, int64_t lines) { tab.eval(); g_tab = tab.data<int32_t>(); g_tab_lines = lines;
    g_tab_arr = new array(tab);
    g_tab_n = (int64_t)tab.size() / (lines * 32); });
  m.def("set_debug", [](array dbg) { dbg.eval(); g_dbg = dbg.data<int32_t>(); });
  m.def("ring_off", [](int L, uint64_t seq) { return ring_off(L, seq); });
  m.def("stats", []() {
    nb::dict d; d["uses"] = g_st.uses.load(); d["misses"] = g_st.misses.load(); d["stalled"] = g_st.stalled.load();
    d["gated"] = g_st.gated.load(); d["io_s"] = g_st.io_ns.load() * 1e-9; d["cpu_wait_s"] = g_st.wait_ns.load() * 1e-9; d["proc_s"] = g_st.proc_ns.load() * 1e-9;
    d["pf_issued"] = g_st.pf_issued.load(); d["pf_used"] = g_st.pf_used.load(); d["pf_waited_s"] = g_st.pf_waited_ns.load() * 1e-9;
    d["slow_calls"] = g_st.slow_calls.load(); d["slow_s"] = g_st.slow_ns.load() * 1e-9; d["pf_victims"] = g_st.pf_victims.load();
    d["pf_ontime"] = g_st.pf_ontime.load(); d["pf_late"] = g_st.pf_late.load(); d["miss_pred_unissued"] = g_st.miss_pred_unissued.load(); d["miss_unpred"] = g_st.miss_unpred.load(); return d;
  });
  m.def("reset_stats", []() { g_st.uses = 0; g_st.misses = 0; g_st.stalled = 0; g_st.gated = 0; g_st.io_ns = 0; g_st.wait_ns = 0; g_st.proc_ns = 0; g_st.pf_issued = 0; g_st.pf_used = 0; g_st.pf_waited_ns = 0; g_st.slow_calls = 0; g_st.slow_ns = 0; g_st.pf_victims = 0; g_st.pf_ontime = 0; g_st.pf_late = 0; g_st.miss_pred_unissued = 0; g_st.miss_unpred = 0; });
}
