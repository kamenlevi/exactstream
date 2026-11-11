#define PLINES 0
#define XLINES 4

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
        int n = m.z;
        if (m.w && (int)t < n) {
          int e = ids[t], li = e / 31;
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
        device atomic_int* row = route + m.x * (96 + 32 * PLINES + 32 * XLINES);
        const uint STAMP = (((uint)m.y) & 0x3FFu) << 22;               // every id / pred word carries seq & 0x3FF
        atomic_store_explicit(&row[65], np, memory_order_relaxed);                 // line 3: next-layer predictions
        int tot = min(30 + 31 * PLINES, (np & 255) + ((np >> 8) & 255) + ((np >> 16) & 255));   // n(L+1) | n(L+2) << 8 | n(L+3) << 16
        for (int i = 0; i < tot; i++)
          atomic_store_explicit(&row[i < 30 ? 66 + i : 97 + 32 * ((i - 30) / 31) + (i - 30) % 31], (int)((((uint)preds[i]) & 0x3FFFFFu) | STAMP), memory_order_relaxed);
        atomic_store_explicit(&row[1], n, memory_order_relaxed);
        for (int i = 0; i < n; i++) atomic_store_explicit(&row[i < 30 ? 2 + i : i < 61 ? 33 + (i - 30) : 96 + 32 * PLINES + 32 * ((i - 61) / 31) + 1 + (i - 61) % 31], (int)((((uint)ids[i]) & 0x3FFFFFu) | STAMP), memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        for (int j = 0; j < PLINES + XLINES; j++) atomic_store_explicit(&row[96 + 32 * j], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[64], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[32], m.y, memory_order_relaxed);
        atomic_store_explicit(&row[0], m.y, memory_order_relaxed);
      }