"""gdn_chunk.py (Fable B, DENSE_REBUILD v2, Fri 2026-11-06): DRAFT of a row-parallel gated-delta step. NOT RUN ON A GPU.
Why: gated_delta_step costs 25-27 us at 2 rows but 57.5 us at 16 [K Leap part B]. The 16-row time is not bytes (the 6.2 MB
of fp32 state moves at 239-250 GB/s at 2 rows): it is the row-by-row chain. Each row rewrites the whole state before the
next row can start, and each state row needs two simd_sum reductions per check row.
The recurrence per value head (state S [128 dv, 128 dk], rows t = 0..n-1; mlx_lm's loop):
    S <- g_t S;  d_t = beta_t (v_t - S k_t);  S <- S + d_t k_t^T;  o_t = S q_t
unrolls, with P(i, t) = g_{i+1} ... g_t (P(t, t) = 1) and G_t = g_0 ... g_t, into
    d_t = beta_t ( v_t - G_t (S0 k_t) - sum_{i<t}  P(i, t) (k_i . k_t) d_i )
    o_t =            G_t (S0 q_t) + sum_{i<=t} P(i, t) (k_i . q_t) d_i
    S_n =            G_{n-1} S0   + sum_{i<n}  P(i, n-1) d_i k_i^T
So the only work that touches the big state is S0 @ [k_0..k_{n-1}, q_0..q_{n-1}] (independent of the recursion) and one
rank-n update at the end. The recursion itself shrinks to a 16-step forward substitution on scalars per (head, dv), with
16 x 16 coefficient tables per head. The state is read once and written once whatever the row count.
This is the chunk-wise form of the delta rule (known from DeltaNet-style training); here it is used for the check.
Layout: one threadgroup per value head, NSG simdgroups (8 state rows each at NSG 16), lane = 4 state columns.
  1  coefficient tables kk[i][t], kq[i][t] (already times P) and G_t: one simdgroup task per (i, t) pair, simd_sum.
  2  state rows into registers; U_k[t][dv] = S0[dv] . k_t, U_q[t][dv] = S0[dv] . q_t: 2 n independent simd_sums per state
     row (no chain between them); written to a device scratch.
  3  thread dv < 128: the forward substitution; o -> out (bf16), w_i[dv] = P(i, n-1) d_i[dv] -> threadgroup memory.
  4  S_n rows from the registers: G_{n-1} S0 + sum_i w_i[dv] k_i (16 multiply-adds per element), written once.
Bits: fp32 with a different operation order than the loop, so FAST (see emulate() for the size of the difference).
Commit: S_m for any kept prefix m is the same phase 4 with n = m. A replay needs only d_i, k_i and g, not the loop.
Threadgroup memory: k, q 8 KB (bf16) + w 8 KB + tables 2.2 KB = 18.2 KB."""
import numpy as np

HDR = ""
SRC = r"""
    constexpr int HD = 128, NHV = 48, QK = 2048, VW = 6144;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint hv = threadgroup_position_in_grid.z, hk = hv / 3;
    threadgroup T ks[16 * HD];
    threadgroup T qs[16 * HD];
    threadgroup float wv[16 * HD];
    threadgroup float kk[16 * 16];
    threadgroup float kq[16 * 16];
    threadgroup float gc[16];

    for (uint i = tid; i < (uint)(S_C * HD); i += 32 * NSG) {      // this head's k, q rows into threadgroup memory
        ks[i] = k[(i / HD) * QK + hk * HD + i % HD];
        qs[i] = q[(i / HD) * QK + hk * HD + i % HD];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 1: tables. pair (i, t), i <= t: kk = P(i, t) (k_i . k_t), kq = P(i, t) (k_i . q_t); gc[t] = G_t ----
    for (uint p = sg; p < (uint)(S_C * S_C); p += NSG) {
        const uint i = p / S_C, t = p % S_C;
        if (i > t) continue;
        float a = 0.0f, b = 0.0f;
        for (int c = 0; c < 4; ++c) {
            const float ki = static_cast<float>(ks[i * HD + 4 * lane + c]);
            a += ki * static_cast<float>(ks[t * HD + 4 * lane + c]);
            b += ki * static_cast<float>(qs[t * HD + 4 * lane + c]);
        }
        a = simd_sum(a); b = simd_sum(b);
        if (lane == 0) {
            float pr = 1.0f;
            for (uint j = i + 1; j <= t; ++j) pr *= g[j * NHV + hv];
            kk[i * 16 + t] = pr * a; kq[i * 16 + t] = pr * b;
            if (i == 0) gc[t] = pr * g[hv];                         // G_t = g_0 P(0, t)
        }
    }

    // ---- 2: the state in registers; U_k, U_q per (row t, state row dv) ----
    constexpr int RPS = HD / NSG;
    float sr[RPS][4];
    for (int j = 0; j < RPS; ++j) {
        const uint dv = sg + NSG * j;
        const uint base = (hv * HD + dv) * HD + lane * 4;
        for (int c = 0; c < 4; ++c) sr[j][c] = dstate[base + c];
        for (int t = 0; t < S_C; ++t) {
            float a = 0.0f, b = 0.0f;
            for (int c = 0; c < 4; ++c) {
                a += sr[j][c] * static_cast<float>(ks[t * HD + 4 * lane + c]);
                b += sr[j][c] * static_cast<float>(qs[t * HD + 4 * lane + c]);
            }
            a = simd_sum(a); b = simd_sum(b);
            if (lane == 0) {
                u_scr[((size_t)hv * 32 + t) * HD + dv] = a;
                u_scr[((size_t)hv * 32 + 16 + t) * HD + dv] = b;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // ---- 3: forward substitution per state row dv (one thread each) ----
    if (tid < (uint)HD) {
        const uint dv = tid;
        float d[16];
        for (int t = 0; t < S_C; ++t) {
            float acc = static_cast<float>(v[t * VW + hv * HD + dv]) - gc[t] * u_scr[((size_t)hv * 32 + t) * HD + dv];
            for (int i = 0; i < t; ++i) acc -= kk[i * 16 + t] * d[i];
            d[t] = static_cast<float>(beta[t * NHV + hv]) * acc;
            float o = gc[t] * u_scr[((size_t)hv * 32 + 16 + t) * HD + dv];
            for (int i = 0; i <= t; ++i) o += kq[i * 16 + t] * d[i];
            out[t * VW + hv * HD + dv] = static_cast<T>(o);
        }
        for (int i = 0; i < S_C; ++i) {
            float pr = 1.0f;
            for (int j = i + 1; j < S_C; ++j) pr *= g[j * NHV + hv];
            wv[i * HD + dv] = pr * d[i];                            // P(i, n-1) d_i[dv]
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 4: the new state, one write ----
    const float gl = gc[S_C - 1];
    for (int j = 0; j < RPS; ++j) {
        const uint dv = sg + NSG * j;
        const uint base = (hv * HD + dv) * HD + lane * 4;
        for (int c = 0; c < 4; ++c) {
            float s = gl * sr[j][c];
            for (int i = 0; i < S_C; ++i) s += wv[i * HD + dv] * static_cast<float>(ks[i * HD + 4 * lane + c]);
            dstate_out[base + c] = s;
        }
    }
"""
IN = ["q", "k", "v", "g", "beta", "dstate"]
OUT = ["out", "dstate_out", "u_scr"]
TYPES = {"g": "float", "dstate": "float", "dstate_out": "float", "u_scr": "float"}


def loop(S0, q, k, v, g, beta):
    """mlx_lm's recurrence in fp32 (one head): S0 [dv, dk], q / k [n, dk], v [n, dv], g / beta [n]"""
    S = S0.astype(np.float32).copy(); o = np.zeros_like(v, dtype=np.float32)
    for t in range(len(g)):
        S = S * np.float32(g[t])
        d = (v[t] - S @ k[t]) * np.float32(beta[t])
        S = S + np.outer(d, k[t]).astype(np.float32)
        o[t] = S @ q[t]
    return o, S


def chunk(S0, q, k, v, g, beta, dt=np.float32):
    """the kernel's order of operations (dt = float32 as in the kernel, float64 for the algebra check)"""
    n = len(g); S0 = S0.astype(dt); q, k, v = q.astype(dt), k.astype(dt), v.astype(dt); g = g.astype(dt); beta = beta.astype(dt)
    P = np.ones((n, n), dt)
    for i in range(n):
        for t in range(i + 1, n):
            P[i, t] = P[i, t - 1] * g[t]
    G = g[0] * P[0]
    kk, kq = P * (k @ k.T), P * (k @ q.T)
    Uk, Uq = S0 @ k.T, S0 @ q.T                                # [dv, n]
    d = np.zeros((n, S0.shape[0]), dt); o = np.zeros((n, S0.shape[0]), dt)
    for t in range(n):
        acc = v[t] - G[t] * Uk[:, t]
        for i in range(t):
            acc = acc - kk[i, t] * d[i]
        d[t] = beta[t] * acc
        ot = G[t] * Uq[:, t]
        for i in range(t + 1):
            ot = ot + kq[i, t] * d[i]
        o[t] = ot
    Sn = G[n - 1] * S0
    for i in range(n):
        Sn = Sn + np.outer(P[i, n - 1] * d[i], k[i])
    return o, Sn


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    worst = [0.0, 0.0, 0.0, 0.0]; flips = tot = 0
    for trial in range(200):
        n = int(rng.integers(1, 17))
        S0 = (rng.normal(0, 0.3, (128, 128))).astype(np.float32)
        k = rng.normal(0, 1, (n, 128)); k = (k / np.linalg.norm(k, axis=1, keepdims=True)).astype(np.float32)
        q = rng.normal(0, 1, (n, 128)); q = (q / np.linalg.norm(q, axis=1, keepdims=True) * 0.0884).astype(np.float32)
        v = rng.normal(0, 1, (n, 128)).astype(np.float32)
        g = rng.uniform(0.5, 1.0, n).astype(np.float32); beta = rng.uniform(0.0, 1.0, n).astype(np.float32)
        o_ref, S_ref = loop(S0, q, k, v, g, beta)
        o64, S64 = chunk(S0, q, k, v, g, beta, np.float64)
        o32, S32 = chunk(S0, q, k, v, g, beta, np.float32)
        r = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
        worst = [max(worst[0], r(o64, o_ref)), max(worst[1], r(S64, S_ref)), max(worst[2], r(o32, o_ref)), max(worst[3], r(S32, S_ref))]
        h = lambda a: a.astype(np.float16)                     # stand-in for the bf16 rounding of the output
        flips += int((h(o32) != h(o_ref)).sum()); tot += o_ref.size
    print("chunk form vs the fp32 loop, 200 random heads, 1-16 rows (max relative difference):")
    print("  algebra (fp64 chunk vs fp32 loop): out %.1e, state %.1e" % (worst[0], worst[1]))
    print("  kernel order in fp32:              out %.1e, state %.1e" % (worst[2], worst[3]))
    print("  outputs that round to a different 16-bit float: %d of %d (%.2f %%)" % (flips, tot, 100.0 * flips / tot))
    ok = worst[2] < 1e-4 and worst[3] < 1e-4
    print("PASS" if ok else "FAIL")
