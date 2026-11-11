"""Minimal CPU<->GPU visibility test inside one running Metal kernel (MLX custom kernel).
GPU thread: write req=1, spin until flag!=0 (cap), record spin count.
CPU: async_eval, poll req, note when seen, set flag. Variants of the GPU load/store.
-> data/coherence.txt"""
import os, time, numpy as np, mlx.core as mx
VAR = """
#if V == 0
#define LD(p) atomic_load_explicit((device atomic_int*)(p), memory_order_relaxed)
#define ST(p,v) atomic_store_explicit((device atomic_int*)(p), v, memory_order_relaxed)
#elif V == 1
#define LD(p) atomic_fetch_add_explicit((device atomic_int*)(p), 0, memory_order_relaxed)
#define ST(p,v) atomic_exchange_explicit((device atomic_int*)(p), v, memory_order_relaxed)
#elif V == 3
#define LD(p) ({ atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system); atomic_load_explicit((device atomic_int*)(p), memory_order_relaxed); })
#define ST(p,v) { atomic_store_explicit((device atomic_int*)(p), v, memory_order_relaxed); atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system); }
#elif V == 4
#define LD(p) evict_ld(p, junk)
#define ST(p,v) { *(volatile device int*)(p) = v; evict_ld(p, junk); }
#else
#define LD(p) (*(volatile device int*)(p))
#define ST(p,v) (*(volatile device int*)(p) = v)
#endif
"""
EVICT = r"""
inline int evict_ld(device int* p, const device uint* j) {
  uint a = 0;
  for (uint i = 0; i < (64u<<20)/4; i += 32) a += j[i];   // 64 MB stride-128B sweep
  return *(volatile device int*)(p) + (int)(a & 0u);
}
"""
src = r"""
if (thread_position_in_grid.x == 0) {
  device int* f = const_cast<device int*>(flag);
  device int* r = const_cast<device int*>(req);
  ST(r, 1);
  #if FENCE
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  #endif
  uint s = 0;
  while (LD(f) == 0 && s < CAP) s++;
  out[0] = s;
}
"""
lines = []
junk = mx.zeros(((64 << 20) // 4,), mx.uint32); mx.eval(junk)
for V in (3, 4):  # buffers >4 KB so MLX binds them as device memory, not set_bytes copies
    for FENCE in (0, 1):
        try:
            k = mx.fast.metal_kernel(name=f"coh{V}{FENCE}", input_names=["flag", "req", "junk"], output_names=["out"],
                                     source=src, header=f"#define V {V}\n#define FENCE {FENCE}\n#define CAP (V == 4 ? 4000u : 20000000u)\n" + VAR + EVICT)
            flag = mx.zeros((16384,), mx.int32); req = mx.zeros((16384,), mx.int32); mx.eval(flag, req)
            F = np.frombuffer(memoryview(flag), np.int32); Q = np.frombuffer(memoryview(req), np.int32)
            res = []
            for rep in range(5):
                F[:] = 0; Q[:] = 0
                o = k(inputs=[flag, req, junk], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
                t0 = time.perf_counter(); mx.async_eval(o); seen = None
                while time.perf_counter() - t0 < 2.0:
                    if Q[0]:
                        seen = time.perf_counter() - t0; F[0] = 1; break
                mx.eval(o); done = time.perf_counter() - t0
                res.append((None if seen is None else round(seen * 1e3, 3), round(done * 1e3, 3), int(o.item())))
            lines.append(f"V{V} fence{FENCE}: (CPU saw req ms, kernel done ms, GPU spins) {res}")
        except Exception as e:
            lines.append(f"V{V} fence{FENCE}: ERROR {str(e)[:200]}")
        print(lines[-1], flush=True)
open(os.path.expanduser("~/BigMoE/data/coherence.txt"), "w").write("\n".join(lines) + "\n")
