"""gbar_test.py (Leap, Mon 16 Nov 2026 21:2x; CHECK25 module 4 feasibility): can one kernel run the HC read's three
phases (K1C -> K2B -> K3C) with GRID barriers between them, instead of three dependent launches?
A persistent kernel of P threadgroups runs NPH phases. In phase ph, each threadgroup publishes one value, then waits at
a grid barrier (one device atomic counter; thread 0 adds 1 and spins until the counter reaches (ph + 1) * P; the spin
is capped, so a group that never arrives shows up as a FAIL flag, not a hang). Then every threadgroup reads all P values
of that phase and checks them. Reads are plain loads (RD 0) or atomic loads (RD 1): Apple's per-core L1 may serve stale
lines across threadgroups.
Timing: the persistent kernel's us per barrier vs a chain of NPH dependent one-phase kernels (us per kernel boundary).
usage: [BigMoE venv] gbar_test.py   (seconds, < 100 MB; run under a share lock)"""
import os
import sys
import time

import mlx.core as mx

_SRC = r"""
    const uint tg = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
    device atomic_uint* c = (device atomic_uint*)ctr;
    uint fail = 0, bad = 0;
    threadgroup uint tfail;
#if TGB > 0
    threadgroup uint pad[TGB / 4];
    for (uint j = tid; j < TGB / 4; j += TPT) pad[j] = j ^ tg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pad[(tid * 37u) % (TGB / 4)] == 0xFFFFFFFFu) bad += 1000000u;
#endif
    for (int ph = 0; ph < NPH; ph++) {
        if (tid == 0) {
            const uint v = tg * 7919u + uint(ph) * 104729u + 1u;
#if RD == 1 || RD == 3
            atomic_store_explicit((device atomic_uint*)(buf + ph * P_C + tg), v, memory_order_relaxed);
#else
            buf[ph * P_C + tg] = v;
#endif
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
#if RD >= 2
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
            atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
            const uint target = uint(ph + 1) * P_C;
            uint it = 0;
            while (atomic_load_explicit(c, memory_order_relaxed) < target) { if (++it > SPIN) { fail = 1; break; } }
#if RD >= 2
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
            tfail = fail;
        }
        threadgroup_barrier(mem_flags::mem_device);
#if RD >= 2
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
        if (tfail) break;                     // fail fast: one capped spin per group at most, never a hang
        for (uint j = tid; j < P_C; j += TPT) {
#if RD == 1 || RD == 3
            const uint w = atomic_load_explicit((device atomic_uint*)(buf + ph * P_C + j), memory_order_relaxed);
#else
            const uint w = buf[ph * P_C + j];
#endif
            if (w != j * 7919u + uint(ph) * 104729u + 1u) bad++;
        }
    }
    bad = simd_sum(bad);
    if (thread_index_in_simdgroup == 0) atomic_fetch_add_explicit((device atomic_uint*)(res + 1), bad, memory_order_relaxed);
    if (tid == 0 && fail) atomic_fetch_add_explicit((device atomic_uint*)(res + 0), 1u, memory_order_relaxed);
"""

_STEP = r"""
    const uint i = thread_position_in_grid.x;
    if (i < N_C) y[i] = x[i] * 1.0001f + 1.0f;
"""


def _kern(rd, tgb=0):
    src = f"#define RD {int(rd)}\n#define TGB {int(tgb)}\n" + _SRC
    return mx.fast.metal_kernel(name=f"leap_gbar_rd{int(rd)}_tgb{int(tgb)}", input_names=["dummy"], output_names=["buf", "ctr", "res"],
                                source=src)


def run(P, T, NPH, rd, spin=1 << 18, tgb=0):
    (buf, ctr, res) = _kern(rd, tgb)(inputs=[mx.zeros((1,), dtype=mx.uint32)],
                                template=[("P_C", P), ("TPT", T), ("NPH", NPH), ("SPIN", spin)],
                                grid=(P * T, 1, 1), threadgroup=(T, 1, 1),
                                output_shapes=[(NPH * P,), (1,), (2,)], output_dtypes=[mx.uint32] * 3, init_value=0)
    return buf, ctr, res


def chain(N, NPH):
    k = mx.fast.metal_kernel(name="leap_gbar_step", input_names=["x"], output_names=["y"], source=_STEP)
    x = mx.zeros((N,), dtype=mx.float32)
    for _ in range(NPH):
        (x,) = k(inputs=[x], template=[("N_C", N)], grid=(N, 1, 1), threadgroup=(min(N, 256), 1, 1),
                 output_shapes=[(N,)], output_dtypes=[mx.float32])
    return x


def timeit(f, reps=7):
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        o = f()
        mx.eval(o)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    lines = []
    for P, T in ((20, 32), (20, 352), (40, 352), (80, 128)):
        for rd in (0, 1, 2, 3):
            for NPH in (2, 64):
                buf, ctr, res = run(P, T, NPH, rd)
                mx.eval(buf, ctr, res)
                r = [int(v) for v in res.tolist()]
                lines.append(f"  P {P:3d} x {T:3d} threads, {['plain', 'atomic', 'plain + fences', 'atomic + fences'][rd]:15s}, {NPH:2d} barriers: "
                             f"spin-cap FAILs {r[0]}, stale / wrong reads {r[1]} of {P * P * NPH} -> "
                             f"{'OK' if r == [0, 0] else 'BAD'}")
                print(lines[-1], flush=True)
    for P, T in ((20, 352), (40, 352), (80, 128)):
        t2 = timeit(lambda: run(P, T, 2, 2))
        tn = timeit(lambda: run(P, T, 1026, 2))
        lines.append(f"  TIME (ROUGH, shared GPU): grid barrier {1e6 * (tn - t2) / 1024:.2f} us each (P {P} x {T}, plain + fences, "
                     f"1,024 barriers)")
        print(lines[-1], flush=True)
    c2 = timeit(lambda: chain(20 * 352, 2))
    cn = timeit(lambda: chain(20 * 352, 1026))
    lines.append(f"  TIME (ROUGH, shared GPU): dependent kernel boundary {1e6 * (cn - c2) / 1024:.2f} us each (1,024 step "
                 f"kernels of 7,040 threads; includes ~Python per launch)")
    print(lines[-1], flush=True)
    p = os.path.expanduser("~/Leap/data/week98/check25/gbar_test.txt")
    open(p, "w").write("GBAR test: grid barriers inside one kernel (capped spins) vs dependent launches\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
