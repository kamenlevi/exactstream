#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
#define RS 192
#define COH_CAP 1000000

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
      }