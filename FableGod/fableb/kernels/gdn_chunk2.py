"""gdn_chunk2.py (Fable B, Fri 2026-11-06, after Leap's KILL of gdn_chunk.py on time): the same algebra as gdn_chunk.py
(see its docstring and its CPU check), re-laid out for many small threadgroups. NOT RUN ON A GPU.
Leap's result on v1 [K]: bits fine, time KILL (S 16: 121.7-128.9 us against the reference's 64.5-86.3). v1 had 48
threadgroups against the reference's 1,536, and its substitution ran on 128 threads only.
v2 is two dispatches:
  TAB   (tiny) dots[key head][i][t] = (k_i . k_t, k_i . q_t) for i <= t. One simdgroup per (key head, pair).
        grid (32, 256, 16), threadgroup (32, 1, 1). 16 x 136 live simdgroups, one simd_sum pair each. Out 16 KB.
  MAIN  one simdgroup per (value head, block of RB state rows): grid (32, 1, 48 * 128 / RB), threadgroup (32, 1, 1).
        RB 8 -> 768 threadgroups, RB 4 -> 1,536 (the reference's count). Lane = 4 state columns.
        a  coefficient tables for this value head: P(i, t) from g, times TAB's dots; G_t. 8 entries per lane.
        b  the RB state rows into registers; U_k, U_q = 2 S independent simd_sums per row; lane 0 parks them in
           threadgroup memory.
        c  lane j < RB: the forward substitution of state row j (16 dependent steps, scalars only), o -> out,
           w_i = P(i, n-1) d_i -> threadgroup memory.
        d  every lane: S_n = G_{n-1} S0 + sum_i w_i k_i for its 4 columns of each row; one state write.
Registers per thread: 4 RB state floats + 16 d. Threadgroup memory: tables 2.2 KB + U 64 RB B + w 64 RB B (3.2 KB at RB 8).
What the time now depends on [G]: the reference does 2 S RB simd_sums per threadgroup in one dependent chain; MAIN does
the same number with no dependence between them, then 16 scalar steps on RB lanes. If independent simd_sums do not
pipeline on this GPU, MAIN will not beat the loop and the idea is dead for the check (it stays useful for the commit).
Same line as before: PASS <= 40 us chained at S 16, KILL > 50. Outputs within 1e-5 of the loop."""
HDR = ""
TAB_SRC = r"""
    constexpr int HD = 128, QK = 2048;
    const uint lane = thread_position_in_grid.x;
    const uint p = thread_position_in_grid.y, hk = thread_position_in_grid.z;
    const uint i = p / 16, t = p % 16;
    if (i > t || t >= (uint)S_C) return;
    float a = 0.0f, b = 0.0f;
    for (int c = 0; c < 4; ++c) {
        const float ki = static_cast<float>(k[i * QK + hk * HD + 4 * lane + c]);
        a += ki * static_cast<float>(k[t * QK + hk * HD + 4 * lane + c]);
        b += ki * static_cast<float>(q[t * QK + hk * HD + 4 * lane + c]);
    }
    a = simd_sum(a); b = simd_sum(b);
    if (lane == 0) { dots[(hk * 2 + 0) * 256 + p] = a; dots[(hk * 2 + 1) * 256 + p] = b; }
"""
TAB_IN, TAB_OUT = ["q", "k"], ["dots"]

SRC = r"""
    constexpr int HD = 128, NHV = 48, QK = 2048, VW = 6144, NB = HD / RB;
    const uint lane = thread_position_in_threadgroup.x;
    const uint hv = threadgroup_position_in_grid.z / NB, blk = threadgroup_position_in_grid.z % NB, hk = hv / 3;
    threadgroup float kk[256];
    threadgroup float kq[256];
    threadgroup float gc[16];
    threadgroup float ut[RB * 32];
    threadgroup float wv[RB * 16];

    // ---- a: this value head's coefficients ----
    for (uint e = lane; e < 256u; e += 32u) {
        const uint i = e / 16, t = e % 16;
        if (i > t || t >= (uint)S_C) continue;
        float pr = 1.0f;
        for (uint j = i + 1; j <= t; ++j) pr *= g[j * NHV + hv];
        kk[e] = pr * dots[(hk * 2 + 0) * 256 + e];
        kq[e] = pr * dots[(hk * 2 + 1) * 256 + e];
        if (i == 0) gc[t] = pr * g[hv];
    }

    // ---- b: state rows in registers; U_k, U_q (independent reductions) ----
    const device T* kd = k + hk * HD + 4 * lane;               // row t at + t * QK
    const device T* qd = q + hk * HD + 4 * lane;
    float sr[RB][4];
    for (int j = 0; j < RB; ++j) {
        const uint base = (hv * HD + blk * RB + j) * HD + lane * 4;
        for (int c = 0; c < 4; ++c) sr[j][c] = dstate[base + c];
    }
    for (int t = 0; t < S_C; ++t) {
        float kf[4], qf[4];
        for (int c = 0; c < 4; ++c) { kf[c] = static_cast<float>(kd[t * QK + c]); qf[c] = static_cast<float>(qd[t * QK + c]); }
        for (int j = 0; j < RB; ++j) {
            float a = 0.0f, b = 0.0f;
            for (int c = 0; c < 4; ++c) { a += sr[j][c] * kf[c]; b += sr[j][c] * qf[c]; }
            a = simd_sum(a); b = simd_sum(b);
            if (lane == 0) { ut[j * 32 + t] = a; ut[j * 32 + 16 + t] = b; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- c: forward substitution, one lane per state row ----
    if (lane < (uint)RB) {
        const uint j = lane, dv = blk * RB + j;
        float d[16];
        for (int t = 0; t < S_C; ++t) {
            float acc = static_cast<float>(v[t * VW + hv * HD + dv]) - gc[t] * ut[j * 32 + t];
            for (int i = 0; i < t; ++i) acc -= kk[i * 16 + t] * d[i];
            d[t] = static_cast<float>(beta[t * NHV + hv]) * acc;
            float o = gc[t] * ut[j * 32 + 16 + t];
            for (int i = 0; i <= t; ++i) o += kq[i * 16 + t] * d[i];
            out[t * VW + hv * HD + dv] = static_cast<T>(o);
        }
        float pr = 1.0f;
        for (int i = S_C - 1; i >= 0; --i) {                    // P(i, n-1) d_i, built from the last row down
            wv[j * 16 + i] = pr * d[i];
            pr *= g[i * NHV + hv];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- d: the new state, one write ----
    const float gl = gc[S_C - 1];
    float sn[RB][4];
    for (int j = 0; j < RB; ++j) for (int c = 0; c < 4; ++c) sn[j][c] = gl * sr[j][c];
    for (int i = 0; i < S_C; ++i) {
        float kf[4];
        for (int c = 0; c < 4; ++c) kf[c] = static_cast<float>(kd[i * QK + c]);
        for (int j = 0; j < RB; ++j) {
            const float w = wv[j * 16 + i];
            for (int c = 0; c < 4; ++c) sn[j][c] += w * kf[c];
        }
    }
    for (int j = 0; j < RB; ++j) {
        const uint base = (hv * HD + blk * RB + j) * HD + lane * 4;
        for (int c = 0; c < 4; ++c) dstate_out[base + c] = sn[j][c];
    }
"""
IN = ["q", "k", "v", "g", "beta", "dstate", "dots"]
OUT = ["out", "dstate_out"]
TYPES = {"g": "float", "dstate": "float", "dstate_out": "float", "dots": "float"}


def gdn_chunk2(q, k, v, g, beta, state, S, RB=8):
    """the two dispatches (needs MLX + a GPU; Leap runs it). q, k [S, 2048] bf16, v [S, 6144] bf16, g [S, 48] fp32,
    beta [S, 48] bf16, state [48, 128, 128] fp32"""
    import mlx.core as mx
    T = q.dtype
    kt = mx.fast.metal_kernel(name="fb_gdn_chunk_tab", input_names=TAB_IN, output_names=TAB_OUT, source=TAB_SRC)
    (dots,) = kt(inputs=[q.reshape(-1), k.reshape(-1)], template=[("T", T), ("S_C", S)], grid=(32, 256, 16),
                 threadgroup=(32, 1, 1), output_shapes=[(16 * 2 * 256,)], output_dtypes=[mx.float32], init_value=0)
    km = mx.fast.metal_kernel(name="fb_gdn_chunk_main", input_names=IN, output_names=OUT, source=SRC)
    return km(inputs=[q.reshape(-1), k.reshape(-1), v.reshape(-1), g.reshape(-1), beta.reshape(-1), state.reshape(-1), dots],
              template=[("T", T), ("S_C", S), ("RB", RB)], grid=(32, 1, 48 * 128 // RB), threadgroup=(32, 1, 1),
              output_shapes=[(S * 6144,), (48 * 128 * 128,)], output_dtypes=[T, mx.float32])
