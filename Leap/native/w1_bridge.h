// w1_bridge.h - W1-PICKUP (Leap, Tue 2026-11-03; TABLE_350 'W1-PICKUP'): the engine's spin-mode release, copied from
// ~/BigMoE/ext/bigmoe_ext.cpp loader_main at a0a5fbe (stamped slot words into every line pair of a (layer, pass) region,
// a release fence, then the tags from the highest line down to line 0), plus loads / stores / ticks for
// native/w1_pickup.swift.
#include <stdint.h>
#include <mach/mach_time.h>
static inline uint32_t w1_load(const void* p) { return __atomic_load_n((const uint32_t*)p, __ATOMIC_ACQUIRE); }
static inline void w1_store_rel(void* p, uint32_t v) { __atomic_store_n((uint32_t*)p, v, __ATOMIC_RELEASE); }
static inline uint64_t w1_ticks(void) { return mach_absolute_time(); }
static inline void w1_fence_rel(void) { __atomic_thread_fence(__ATOMIC_RELEASE); }
static inline int32_t w1_stamp_enc(int32_t v, int32_t seq) { return (int32_t)(((uint32_t)v & 0x3FFFFFu) | (((uint32_t)seq & 0x3FFu) << 22)); }
static inline void w1_release_pairs(int32_t* reg, int64_t pairs, int rs, const int32_t* sl, int n, int32_t seq) {
  for (int64_t p = 0; p < pairs; p++) {
    int32_t* l0 = reg + p * rs; int32_t* l1 = l0 + 32; int32_t* l2 = l0 + 64;
    for (int i = 0; i < n && i < 31; i++) l0[1 + i] = w1_stamp_enc(sl[i], seq);     // stamped words
    for (int i = 31; i < n && i < 62; i++) l1[1 + (i - 31)] = w1_stamp_enc(sl[i], seq);
    for (int i = 62; i < n && i < 93; i++) l2[1 + (i - 62)] = w1_stamp_enc(sl[i], seq);
    for (int i = 93; i < n; i++) l0[32 * (i / 31) + 1 + i % 31] = w1_stamp_enc(sl[i], seq);   // R-55: lines 3..5
  }
  __atomic_thread_fence(__ATOMIC_RELEASE);
  for (int64_t p = 0; p < pairs; p++) {
    int32_t* l0 = reg + p * rs;
    for (int j = rs / 32 - 1; j >= 3; j--) __atomic_store_n(l0 + 32 * j, seq, __ATOMIC_RELAXED);   // R-55
    if (rs >= 96) __atomic_store_n(l0 + 64, seq, __ATOMIC_RELAXED);
    __atomic_store_n(l0 + 32, seq, __ATOMIC_RELAXED); __atomic_store_n(l0, seq, __ATOMIC_RELAXED);
  }
}
