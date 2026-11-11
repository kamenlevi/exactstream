"""Cheapest decisive test for a persistent (grid-barrier) speed-of-light layer: the HC read pair (hc_down v2 -> hc_up v2)
as (a) 2 dependent dispatches vs (b) ONE persistent dispatch of G threadgroups with a device-atomic grid barrier between
the phases (bounded spin, so a non-co-resident grid fails instead of hanging; the last threadgroup out resets the
counter). Both run the same per-item code, so the outputs must be bitwise equal. Chains of 48 over 6 weight sets
(weights from DRAM), M = 1 and 4, G = 20..320. usage: persist_pair.py   (GPU seconds; quiet lock)"""
import time
import numpy as np
import mlx.core as mx

HDR = r"""
inline float silu_f(float x) { return x / (1.0f + metal::exp(-x)); }
inline float sigm_f(float x) { return 1.0f / (1.0f + metal::exp(-x)); }

template <int RB, int M>
inline void hcd_item(uint item, uint lane, uint k, const device bfloat16_t* xin, const device bfloat16_t* W,
                     const device bfloat16_t* nw, device float* y, device float* aux_out, threadgroup float* part) {
    const uint n0 = item * RB;
    float acc[RB][M], ss[M];
    for (int m = 0; m < M; ++m) { ss[m] = 0.0f; for (int r = 0; r < RB; ++r) acc[r][m] = 0.0f; }
    for (int j = 0; j < 10; ++j) {
        const uint c = k * 320 + j * 32 + lane;
        const uint4 nw4 = ((const device uint4*)nw)[c];
        const thread bfloat16_t* nv = (const thread bfloat16_t*)&nw4;
        float xv[M][8];
        for (int m = 0; m < M; ++m) {
            const uint4 x4 = ((const device uint4*)(xin + (size_t)m * 10240))[c];
            const thread bfloat16_t* xb = (const thread bfloat16_t*)&x4;
            for (int i = 0; i < 8; ++i) { const float v = (float)xb[i]; ss[m] += v * v; xv[m][i] = v * (float)nv[i]; }
        }
        for (int r = 0; r < RB; ++r) {
            if (n0 + r >= 324u) break;
            const uint4 w4 = ((const device uint4*)(W + (size_t)(n0 + r) * 10240))[c];
            const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
            for (int m = 0; m < M; ++m) { float a = 0.0f; for (int i = 0; i < 8; ++i) a += (float)wv[i] * xv[m][i]; acc[r][m] += a; }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float inv = metal::rsqrt(simd_sum(ss[m]) / 2560.0f + 1e-6f);
        if (item == 0 && lane == 0) aux_out[m * 4 + k] = inv;
        for (int r = 0; r < RB; ++r) { const float a = simd_sum(acc[r][m]) * inv; if (lane == 0) part[(k * RB + r) * M + m] = a; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = k * 32 + lane;
    if (t < (uint)(RB * M)) {
        const uint r = t / M, m = t % M, n = n0 + r;
        if (n < 324u) {
            float a = 0.0f;
            for (int g = 0; g < 4; ++g) a += part[(g * RB + r) * M + m];
            y[(size_t)m * 324 + n] = (n < 320) ? silu_f(a / 4.0f) : 2.0f * sigm_f(a / 4.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);              // part is reused by the next item
}

template <int M, typename AuxPtr>                                // aux_in: device, or constant when it has < 8 values
inline void hcu_item(uint sgi, uint lane, const device float* xin, const device bfloat16_t* W, const device bfloat16_t* nw,
                     const device bfloat16_t* stream, AuxPtr aux_in, device float* ym) {
    const uint r = lane / 4, q = lane % 4, kk = r % 4, j = r / 4;
    const uint d = sgi * 2 + j, n = kk * 2560 + d;
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    for (int i = 0; i < 10; ++i) {
        const uint c = q + 4 * i;
        const uint4 w4 = ((const device uint4*)(W + (size_t)n * 320))[c];
        const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
        for (int m = 0; m < M; ++m) {
            const device float* xm = xin + (size_t)m * 324 + c * 8;
            float a = 0.0f;
            for (int e = 0; e < 8; ++e) a += (float)wv[e] * xm[e];
            acc[m] += a;
        }
    }
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        a += simd_shuffle_xor(a, 1); a += simd_shuffle_xor(a, 2);
        float term = sigm_f(a) * (float)stream[(size_t)m * 10240 + n] * aux_in[m * 4 + kk] * (float)nw[n];
        term += simd_shuffle_xor(term, 4); term += simd_shuffle_xor(term, 8);
        if ((lane & 15u) == 0) ym[(size_t)m * 2560 + d] = term * 0.25f;
    }
}
"""

HCD2 = """
    threadgroup float part[4 * RB_C * M_C];
    hcd_item<RB_C, M_C>(threadgroup_position_in_grid.x, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
                        xin, W, nw, y, aux_out, part);
"""
HCU2 = """
    hcu_item<M_C>(threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
                  xin, W, nw, stream, aux_in, ym);
"""
PERS = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, k = simdgroup_index_in_threadgroup;
    threadgroup float part[4 * RB_C * M_C];
    threadgroup uint stop;
    device atomic_uint* cnt = (device atomic_uint*)cnt_in;
    for (uint it = tg; it < (324u + RB_C - 1) / RB_C; it += G_C)
        hcd_item<RB_C, M_C>(it, lane, k, xin, Wd, nw, y, aux_out, part);
    // grid barrier: every threadgroup arrives, then waits for all G_C (bounded spin)
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
        stop = 0;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
        atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
        uint spins = 0;
        while (atomic_load_explicit(cnt, memory_order_relaxed) < (uint)G_C)
            if (++spins > 200000u) { stop = 1; break; }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
    if (stop) { if (t == 0) fail[0] = 1u; return; }
    for (uint s = tg * 4 + k; s < 1280u; s += G_C * 4)
        hcu_item<M_C>(s, lane, y, Wu, nw, xin, aux_out, ym);
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0 && atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == 2u * G_C - 1u)
        atomic_store_explicit(cnt, 0u, memory_order_relaxed);        // last one out resets for the next dispatch
"""
k_hcd = mx.fast.metal_kernel(name="pp_hcd", input_names=["xin", "W", "nw", "dep"], output_names=["y", "aux_out"],
                             source=HCD2, header=HDR)
k_hcu = mx.fast.metal_kernel(name="pp_hcu", input_names=["xin", "W", "nw", "stream", "aux_in"], output_names=["ym"],
                             source=HCU2, header=HDR)
k_per = mx.fast.metal_kernel(name="pp_per", input_names=["xin", "Wd", "Wu", "nw", "cnt_in", "dep"],
                             output_names=["y", "aux_out", "ym", "fail"], source=PERS, header=HDR)
RB, NLP, N, REPS = 2, 6, 48, 5


def pair(x, Wd, Wu, nw, dep, M):
    y, aux = k_hcd(inputs=[x, Wd, nw, dep], template=[("RB_C", RB), ("M_C", M)], grid=(-(-324 // RB) * 128, 1, 1),
                   threadgroup=(128, 1, 1), output_shapes=[(M, 324), (M, 4)], output_dtypes=[mx.float32, mx.float32])
    return k_hcu(inputs=[y, Wu, nw, x, aux], template=[("M_C", M)], grid=(1280 * 32, 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[(M, 2560)], output_dtypes=[mx.float32])[0]


def persistent(x, Wd, Wu, nw, cnt, dep, M, G):
    out = k_per(inputs=[x, Wd, Wu, nw, cnt, dep], template=[("RB_C", RB), ("M_C", M), ("G_C", G)],
                grid=(G * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(M, 324), (M, 4), (M, 2560), (1,)],
                output_dtypes=[mx.float32, mx.float32, mx.float32, mx.uint32])
    return out[2], out[3]


def chain(fn, seed, n=N):
    """us per link of a chain of n dependent calls: GPU side (graph built before the clock) and Python build time"""
    best, bld = 1e9, 1e9
    for rep in range(REPS + 1):
        d = seed
        t0 = time.perf_counter()
        for i in range(n):
            d = fn(d, i)
        t1 = time.perf_counter()
        mx.eval(d)
        if rep:
            best, bld = min(best, time.perf_counter() - t1), min(bld, t1 - t0)
    return best / n * 1e6, bld / n * 1e6


def compile_graph():
    """lazy graphs of every kernel instance (for compile_check.py)"""
    r = lambda *s: mx.zeros(s, mx.bfloat16)
    w = (r(324, 10240), r(10240, 320), r(10240))
    for M in (1, 4):
        x, d = r(M, 10240), mx.zeros((M, 2560), mx.float32)
        pair(x, *w, d, M)
        for G in (20, 40, 80, 160, 320):
            persistent(x, *w, mx.zeros((64,), mx.uint32), d, M, G)


if __name__ == "__main__":
    r = lambda *s: (mx.random.normal(s) * 0.02).astype(mx.bfloat16)
    W = [(r(324, 10240), r(10240, 320), r(10240) + 1) for _ in range(NLP)]
    mx.eval(W)
    cnt = mx.zeros((64,), mx.uint32)
    mx.eval(cnt)
    hop = mx.fast.metal_kernel(name="pp_hop", input_names=["x"], output_names=["y"],
                               source="uint i = thread_position_in_grid.x; y[i] = x[i] + 1.0f;")
    x0 = mx.zeros((4096,), mx.float32)
    mx.eval(x0)
    for n in (48, 400):
        th, bh = chain(lambda d, i: hop(inputs=[d], grid=(4096, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(4096,)],
                                        output_dtypes=[mx.float32])[0], x0, n)
        print(f"tiny dependent hop, chain of {n}: {th:.2f} us GPU side per hop (+ Python graph build {bh:.2f} us per call)", flush=True)
    for M in (1, 4):
        X = [(mx.random.normal((M, 10240)) * 0.5).astype(mx.bfloat16) for _ in range(NLP)]
        mx.eval(X)
        seed = mx.zeros((M, 2560), mx.float32)
        mx.eval(seed)
        t2, b2 = chain(lambda d, i: pair(X[i % NLP], *W[i % NLP], d, M), seed)
        ref = pair(X[0], *W[0], seed, M)
        mx.eval(ref)
        print(f"M={M}: HC read as 2 dependent dispatches: {t2:.1f} us per pair (GPU side; Python graph build {b2:.1f} us)", flush=True)
        for G in (20, 40, 80, 160, 320):
            ym, fl = persistent(X[0], *W[0], cnt, seed, M, G)
            mx.eval(ym, fl)
            same = mx.array_equal(ym, ref).item()
            c0 = int(np.array(cnt)[0])
            if not same or c0 != 0:
                print(f"    persistent G={G}: FAIL (bitwise {same}, counter {c0}, fail flag {int(np.array(fl)[0])})", flush=True)
                break
            tp, bp = chain(lambda d, i: persistent(X[i % NLP], *W[i % NLP], cnt, d, M, G)[0], seed)
            print(f"    persistent G={G:3d} x 128 threads, 1 dispatch + 1 grid barrier: {tp:.1f} us (bitwise equal, "
                  f"counter reset ok, {t2 - tp:+.1f} us vs 2 dispatches)", flush=True)
