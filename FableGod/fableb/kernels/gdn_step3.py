"""gdn_step3.py (Fable B, Fri 2026-11-06, Leap's ask after the chunk form's KILL): DRAFT of a 16-row gated-delta step that
keeps the reference's grid (mlx_lm gated_delta_step: grid (32, 128, 48), threadgroup (32, 4, 1), one simdgroup per state
row, lane = 4 state columns) and is cheaper per row step. NOT RUN ON A GPU.
Where the per-row time goes [P]: 6,144 simdgroups x 16 rows in ~45 us above the 2-row cost. With ~220 simdgroups in
flight that is ~115 ns per (simdgroup, row). The chunk form's KILL showed that making the reductions independent does
not help, so the cost is the amount of work per row step, not the chain. Per row step the reference does: 11 loads (g,
beta, v, 4 k, 4 q) with 8 bf16 -> float conversions, 12 multiply-adds, and TWO simd_sum reductions. The reductions are
the only cross-lane operations, so they are the first suspect.
VAR 1: one reduction per row instead of two. With s = g S (the decayed state row):
    a = sum(s k), b = sum(s q), c = sum(k q)   ->   ONE simd_sum of a float3
    delta = (v - a) beta;   S <- s + k delta;   out = b + delta c         (out = sum((s + k delta) q), re-associated)
VAR 2: the same with c = k_t . q_t read from an input (kq [S, 16 key heads] fp32; the prework kernel already has q and k
    of a head in one threadgroup and can emit it): one simd_sum of a float2, 4 fewer multiply-adds per lane.
VAR 0: the reference loop (the control in the same harness).
ABL (timing only, results are garbage): 1 = no simd_sum (lane-local values), 2 = k / q from constants (no loads),
    3 = both. They say which part a row step pays for, as NAX-ABL did for the expert kernel.
PF (added after Leap's ablations, which showed neither the reductions nor the k / q loads carry the per-row cost:
    us per call 73.9 with all, 79.8 without simd_sum, 78.6 with constant k / q, 69.4 without both [K]):
    PF & 1 = v, g, beta of all rows fetched before the loop; PF & 2 = k, q too; PF & 4 = y parked in threadgroup memory
    and stored once after the loop. The suspects: a device load of v[t] on the dependent chain of every row (each row's
    v sits 12 KB from the last, so every row step starts with a cold line), and 6,144 simdgroups writing single bf16
    values into shared cache lines of y on every row. PF changes no value and no expression: VAR 0 with any PF is
    bitwise == the reference, so a PF win is an EXACT gain. Threadgroup memory: 8.8 KB at bf16.
Bits: VAR 1 / 2 re-associate out, so FAST (see emulate: ~1e-7 relative, well under one bf16 step on most outputs);
the state update itself is the reference's expression, so the state stays bitwise.
Line [G]: PASS <= 46 us chained at S 16 (-0.43 ms per 16-row check), KILL > 53. If ABL 1 does not drop the time by
>= 25 %, the reductions are not the cost and VAR 1 / 2 cannot pass: stop there."""
import numpy as np

HDR = ""
SRC = r"""
    constexpr int Hv = 48, Hk = 16, Dk = 128, Dv = 128, n_per_t = Dk / 32;
    const uint n = thread_position_in_grid.z;                  // value head (B = 1)
    const uint hk_idx = n / (Hv / Hk);
    const uint dk_idx = thread_position_in_threadgroup.x;
    const uint dv_idx = thread_position_in_grid.y;
    const device T* q_ = q + hk_idx * Dk + n_per_t * dk_idx;
    const device T* k_ = k + hk_idx * Dk + n_per_t * dk_idx;
    const device T* v_ = v + n * Dv + dv_idx;
    device T* y_ = y + n * Dv + dv_idx;
    const device float* i_state = state_in + (n * Dv + dv_idx) * Dk + n_per_t * dk_idx;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) state[i] = i_state[i];
    // PF (Leap's abl 4): what a row step reads from device memory is fetched BEFORE the t loop, into threadgroup memory.
    //   PF & 1: v, g, beta of all rows;  PF & 2: k, q of all rows (this key head);  PF & 4: y parked, stored after the loop.
    // Every value and every expression is the reference's, so any PF with VAR 0 is bitwise == the reference.
    const uint sgi = thread_position_in_threadgroup.y;         // simdgroup in the threadgroup (4 state rows per threadgroup)
    threadgroup float vt[4 * 16];
    threadgroup float gt[16];
    threadgroup float bt[16];
    threadgroup T kt[16 * Dk];
    threadgroup T qt[16 * Dk];
    threadgroup float yt[4 * 16];
#if (PF & 1)
    if (dk_idx < (uint)S_C) {
        vt[sgi * 16 + dk_idx] = static_cast<float>(v_[dk_idx * Hv * Dv]);
        if (sgi == 0) { gt[dk_idx] = g[dk_idx * Hv + n]; bt[dk_idx] = static_cast<float>(beta[dk_idx * Hv + n]); }
    }
#endif
#if (PF & 2)
    for (uint e = sgi * 32 + dk_idx; e < (uint)(S_C * Dk); e += 128u) {
        kt[e] = k[(e / Dk) * Hk * Dk + hk_idx * Dk + e % Dk];
        qt[e] = q[(e / Dk) * Hk * Dk + hk_idx * Dk + e % Dk];
    }
#endif
#if (PF & 3)
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

    for (int t = 0; t < S_C; ++t) {
#if (PF & 1)
        const float gv = gt[t], bv = bt[t], vv = vt[sgi * 16 + t];
#else
        const float gv = g[t * Hv + n];
        const float bv = static_cast<float>(beta[t * Hv + n]);
        const float vv = static_cast<float>(v_[t * Hv * Dv]);
#endif
        float kf[n_per_t], qf[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
#if (ABL & 2)
            kf[i] = 0.01f * float(dk_idx + i + 1); qf[i] = 0.02f * float(dk_idx + i + 1);
#elif (PF & 2)
            kf[i] = static_cast<float>(kt[t * Dk + n_per_t * dk_idx + i]); qf[i] = static_cast<float>(qt[t * Dk + n_per_t * dk_idx + i]);
#else
            kf[i] = static_cast<float>(k_[t * Hk * Dk + i]); qf[i] = static_cast<float>(q_[t * Hk * Dk + i]);
#endif
        }
#if VAR == 0
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) { state[i] = state[i] * gv; kv_mem += state[i] * kf[i]; }
#if !(ABL & 1)
        kv_mem = simd_sum(kv_mem);
#endif
        const float delta = (vv - kv_mem) * bv;
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) { state[i] = state[i] + kf[i] * delta; out += state[i] * qf[i]; }
#if !(ABL & 1)
        out = simd_sum(out);
#endif
#else
#if VAR == 1
        float3 r = float3(0.0f);
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * gv;
            r += float3(state[i] * kf[i], state[i] * qf[i], kf[i] * qf[i]);
        }
#if !(ABL & 1)
        r = simd_sum(r);
#endif
        const float cq = r.z;
#else
        float2 r = float2(0.0f);
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * gv;
            r += float2(state[i] * kf[i], state[i] * qf[i]);
        }
#if !(ABL & 1)
        r = simd_sum(r);
#endif
        const float cq = kq[t * Hk + hk_idx];
#endif
        const float delta = (vv - r.x) * bv;
        for (int i = 0; i < n_per_t; ++i) state[i] = state[i] + kf[i] * delta;
        const float out = r.y + delta * cq;
#endif
#if (PF & 4)
        if (dk_idx == 0) yt[sgi * 16 + t] = out;
#else
        if (dk_idx == 0) y_[t * Hv * Dv] = static_cast<T>(out);
#endif
    }
#if (PF & 4)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dk_idx < (uint)S_C) y_[dk_idx * Hv * Dv] = static_cast<T>(yt[sgi * 16 + dk_idx]);   // lane t stores row t
#endif
    device float* o_state = state_out + (n * Dv + dv_idx) * Dk + n_per_t * dk_idx;
    for (int i = 0; i < n_per_t; ++i) o_state[i] = state[i];
"""
IN = ["q", "k", "v", "g", "beta", "state_in", "kq"]
OUT = ["y", "state_out"]
TYPES = {"g": "float", "state_in": "float", "kq": "float", "state_out": "float"}


def gdn_step3(q, k, v, g, beta, state, S, var=1, abl=0, kq=None, pf=0):
    """needs MLX + a GPU (Leap runs it). q, k [S, 16, 128] bf16, v [S, 48, 128] bf16, g [S, 48] fp32, beta [S, 48] bf16,
    state [48, 128, 128] fp32; var 2 needs kq [S, 16] fp32 = sum(k * q, -1) in fp32"""
    import mlx.core as mx
    T = q.dtype
    if kq is None:
        kq = mx.zeros((S, 16), mx.float32)
    kern = mx.fast.metal_kernel(name=f"fb_gdn_step3_v{var}_a{abl}_p{pf}", input_names=IN, output_names=OUT,
                                source=f"#define VAR {var}\n#define ABL {abl}\n#define PF {pf}\n" + SRC)
    return kern(inputs=[q.reshape(-1), k.reshape(-1), v.reshape(-1), g.reshape(-1), beta.reshape(-1), state.reshape(-1),
                        kq.reshape(-1)], template=[("T", T), ("S_C", S)], grid=(32, 128, 48), threadgroup=(32, 4, 1),
                output_shapes=[(S * 48 * 128,), (48 * 128 * 128,)], output_dtypes=[T, mx.float32])


def emulate(S0, q, k, v, g, beta):
    """VAR 1 / 2's arithmetic for one head in fp32 (lane sums as numpy sums): returns out, state"""
    f = np.float32
    S = S0.astype(f).copy(); o = np.zeros_like(v, dtype=f)
    for t in range(len(g)):
        S = S * f(g[t])
        a, b, c = S @ k[t], S @ q[t], f(k[t] @ q[t])
        d = (v[t] - a) * f(beta[t])
        S = S + np.outer(d, k[t]).astype(f)
        o[t] = b + d * c
    return o, S


if __name__ == "__main__":
    import importlib.util, os
    sp = importlib.util.spec_from_file_location("gdn_chunk", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_chunk.py"))
    GC = importlib.util.module_from_spec(sp); sp.loader.exec_module(GC)
    rng = np.random.default_rng(5)
    wo = ws = 0.0; flips = tot = 0
    for trial in range(200):
        n = int(rng.integers(1, 17))
        S0 = rng.normal(0, 0.3, (128, 128)).astype(np.float32)
        k = rng.normal(0, 1, (n, 128)); k = (k / np.linalg.norm(k, axis=1, keepdims=True)).astype(np.float32)
        q = rng.normal(0, 1, (n, 128)); q = (q / np.linalg.norm(q, axis=1, keepdims=True) * 0.0884).astype(np.float32)
        v = rng.normal(0, 1, (n, 128)).astype(np.float32)
        g = rng.uniform(0.5, 1.0, n).astype(np.float32); beta = rng.uniform(0.0, 1.0, n).astype(np.float32)
        o_ref, S_ref = GC.loop(S0, q, k, v, g, beta)
        o, S = emulate(S0, q, k, v, g, beta)
        wo = max(wo, float(np.abs(o - o_ref).max() / np.abs(o_ref).max())); ws = max(ws, float(np.abs(S - S_ref).max()))
        flips += int((o.astype(np.float16) != o_ref.astype(np.float16)).sum()); tot += o.size
    print("VAR 1 / 2 vs the fp32 loop, 200 random heads, 1-16 rows: out max rel %.1e; state max abs difference %.1e;"
          " outputs rounding to a different 16-bit float: %d of %d (%.3f %%)" % (wo, ws, flips, tot, 100.0 * flips / tot))
    print("PASS" if wo < 1e-5 and ws == 0.0 else "FAIL")
