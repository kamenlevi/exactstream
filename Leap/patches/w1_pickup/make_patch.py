"""Build the W1-PICKUP patch for BigMoE's ext (Leap, Tue 2026-11-03; TABLE_350 'W1-PICKUP'): base/ext/bigmoe_ext.cpp
(BigMoE's committed HEAD, see base/BIGMOE_HEAD) -> new/ext/bigmoe_ext.cpp, then w1_pickup.patch (git style, a/ b/).
Every edit is an exact-text replacement with its expected count asserted. Both flags default OFF; with them off the
kernels preprocess to today's statements and the release loops run over all pairs as today."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "base/ext/bigmoe_ext.cpp")).read()


def rep(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:90], s.count(old))
    return s.replace(old, new)


FLAGS = r'''static std::vector<MTL::SharedEvent*> g_events;   // one per layer, value = pass seq (out-of-order safe)
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
      })";'''
src = rep(src, "static std::vector<MTL::SharedEvent*> g_events;   // one per layer, value = pass seq (out-of-order safe)", FLAGS)

# post_ids: the row's type, stores and fence through macros that preprocess to today's statements unless POST_SYS
a = src.index("kernel void bigmoe_post_ids(")
b = src.index('})";', a)
post = src[a:b]
post = rep(post, "device atomic_int* route [[buffer(1)]]", "ROW_T route [[buffer(1)]]")
post = rep(post, "device atomic_int* row = route + ", "ROW_T row = route + ")
post = rep(post, "atomic_store_explicit(&row[65], np, memory_order_relaxed);", "RST(65, np);")
post = rep(post, "atomic_store_explicit(&row[i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31], (int)((((uint)preds[i]) & 0x3FFFFFu) | STAMP), memory_order_relaxed);",
           "RST(i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31, (int)((((uint)preds[i]) & 0x3FFFFFu) | STAMP));")
post = rep(post, "atomic_store_explicit(&row[1], n, memory_order_relaxed);", "RST(1, n);")
post = rep(post, "atomic_store_explicit(&row[i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * PLINES + 32 * ((i - 61) / 31) + 1 + (i - 61) % 31], (int)((((uint)ids[i]) & 0x3FFFFFu) | STAMP), memory_order_relaxed);",
           "RST(i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * PLINES + 32 * ((i - 61) / 31) + 1 + (i - 61) % 31, (int)((((uint)ids[i]) & 0x3FFFFFu) | STAMP));")
post = rep(post, "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);",
           "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, RSCOPE);")
post = rep(post, "atomic_store_explicit(&row[96 + 32 * j], m.y, memory_order_relaxed);", "RST(96 + 32 * j, m.y);")
post = rep(post, "atomic_store_explicit(&row[64], m.y, memory_order_relaxed);", "RST(64, m.y);")
post = rep(post, "atomic_store_explicit(&row[32], m.y, memory_order_relaxed);", "RST(32, m.y);")
post = rep(post, "atomic_store_explicit(&row[0], m.y, memory_order_relaxed);",
           "RST(0, m.y);\n#ifdef POST_SYS\n        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);   // publish now\n#endif")
assert "&row[" not in post, "a row store was left unconverted"
src = src[:a] + post + src[b:]
src = rep(src, """      #include <metal_stdlib>
      using namespace metal;
      // writes [seq, n, ids0..29 | seq, ids30..39 | seq, np, preds0..29 | PLINES x (seq, 31 more preds)]""",
          """      #include <metal_stdlib>
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
      // writes [seq, n, ids0..29 | seq, ids30..39 | seq, np, preds0..29 | PLINES x (seq, 31 more preds)]""")
p0 = src.index("static MTL::ComputePipelineState* post_pso() {")
p1 = src.index("    NS::Error* err = nullptr;", p0)
src = src[:p1] + "    if (g_post_sys) srcs = std::string(k_sys_scope) + \"#define POST_SYS 1\\n\" + srcs;   // W1-PICKUP\n" + src[p1:]

# spin: the coherent kernel replaces the source when the flag is set (same name, same bindings)
s0 = src.index("static MTL::ComputePipelineState* spin_pso() {")
s1 = src.index("    NS::Error* err = nullptr;", s0)
src = src[:s1] + ("    if (g_spin_coh)   // W1-PICKUP: one simdgroup polls ONE row with system-scope loads\n"
                  "      srcs = std::string(k_sys_scope) + (getenv(\"BIGMOE_NOCOUNT\") ? \"#define COUNT_FAST 0\\n\" : \"\") + \"#define RS \" + std::to_string(g_rs) + \"\\n\"\n"
                  "             + \"#define COH_CAP \" + std::string(getenv(\"BIGMOE_SPIN_COH_CAP\") ? getenv(\"BIGMOE_SPIN_COH_CAP\") : \"1000000\") + \"\\n\" + k_spin_coh;\n") + src[s1:]
src = rep(src, """  enc.set_output_array(o, 4);
  enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}""", """  enc.set_output_array(o, 4);
  const int nt = g_spin_coh ? 32 : 1;                // W1-PICKUP: the coherent spin is one simdgroup
  enc.dispatch_threads(MTL::Size(nt, 1, 1), MTL::Size(nt, 1, 1));
}""")

# release: the coherent spin polls pair 0 only, so only pair 0 is written
src = rep(src, """        const int32_t* sl = g_ring + ring_off(L, (uint64_t)seq);
        for (int64_t p = 0; p < g_pairs; p++) {""", """        const int32_t* sl = g_ring + ring_off(L, (uint64_t)seq);
        const int64_t npairs = g_spin_coh ? 1 : g_pairs;   // W1-PICKUP: the coherent spin polls pair 0 only
        for (int64_t p = 0; p < npairs; p++) {""")
src = rep(src, """        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (int64_t p = 0; p < g_pairs; p++) {""", """        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (int64_t p = 0; p < npairs; p++) {""")

os.makedirs(os.path.join(HERE, "new/ext"), exist_ok=True)
open(os.path.join(HERE, "new/ext/bigmoe_ext.cpp"), "w").write(src)
d = subprocess.run(["diff", "-u", "--label", "a/ext/bigmoe_ext.cpp", "--label", "b/ext/bigmoe_ext.cpp",
                    os.path.join(HERE, "base/ext/bigmoe_ext.cpp"), os.path.join(HERE, "new/ext/bigmoe_ext.cpp")],
                   capture_output=True, text=True).stdout
open(os.path.join(HERE, "w1_pickup.patch"), "w").write(d)
print(f"patch: {d.count(chr(10))} lines, +{sum(1 for l in d.splitlines() if l.startswith('+') and not l.startswith('+++'))} "
      f"-{sum(1 for l in d.splitlines() if l.startswith('-') and not l.startswith('---'))}")
