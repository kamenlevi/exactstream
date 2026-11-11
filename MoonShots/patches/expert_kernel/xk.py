"""Bandwidth-oriented, deterministic, batch-invariant routed-expert kernel for 1..6 rows (Qwen3.8-Flash-Next 125B shapes:
512 experts, top-10, 4-bit affine g64, gate/up [640, 2560], down [2560, 640]). Reference-v2 candidate.

Three dispatches per MoE layer:
  A  gate/up : one simdgroup per (pair p = row*KT + rank, block of RJ output rows). A pair whose expert already appears at an
               earlier pair returns at once; the first occurrence ("owner") computes every pair routed to that expert from
               one load of its weight words (hoisted before decoding). h = bf(silu(bf(g))) * bf(u), as SwitchGLU rounds.
  B  down    : same ownership, per block of RO output rows of down; writes each pair's down output (bf16) to `part`.
  C  combine : y[row] = sum over the row's own ranks 0..KT-1 (fixed order) of part * score, in float, rounded to bf16.
Each pair's arithmetic (lanes, iteration order, simd_sum) is identical whichever simdgroup owns it, so a row's result does not
depend on the other rows: deterministic and batch-invariant by construction."""
import mlx.core as mx

HDR = """
inline float b2f(ushort u) { return as_type<float>((uint)u << 16); }
inline float bf(float v) { return (float)(bfloat16_t)v; }
inline void ld8(const device bfloat16_t* p, thread float v[8]) {
    const uint4 u = *(const device uint4*)p;
    v[0] = as_type<float>(u.x << 16); v[1] = as_type<float>(u.x & 0xFFFF0000u);
    v[2] = as_type<float>(u.y << 16); v[3] = as_type<float>(u.y & 0xFFFF0000u);
    v[4] = as_type<float>(u.z << 16); v[5] = as_type<float>(u.z & 0xFFFF0000u);
    v[6] = as_type<float>(u.w << 16); v[7] = as_type<float>(u.w & 0xFFFF0000u);
}
inline float dq8(uint w, thread const float x[8]) {      // sum_j q_j * x_j for 8 nibbles
    float d = 0.0f;
    for (uint j = 0; j < 8; ++j) d += (float)((w >> (4 * j)) & 15u) * x[j];
    return d;
}
"""

SRC_A = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint sg = t / 32, lane = t % 32;
    constexpr uint NB = F / RJ, NP = S * KT, KW = D / 8, NG = D / 64, NIT = D / 256;
    const uint gsid = tg * 8 + sg;
    const uint p = gsid / NB, jb = gsid % NB;
    if (p >= NP) return;
    const int e = ids[p];
    for (uint q = 0; q < p; ++q) if (ids[q] == e) return;          // not the owner of this expert
    uint srv[S]; uint ns = 0;                                       // pairs served: p + later pairs of the same expert
    for (uint q = p; q < NP; ++q) if (ids[q] == e) srv[ns++] = q;
    const uint j0 = jb * RJ;
    uint wg[RJ][NIT], wu[RJ][NIT];
    ushort sgv[RJ][NIT], bgv[RJ][NIT], suv[RJ][NIT], buv[RJ][NIT];
    for (uint r = 0; r < RJ; ++r) {                                 // every load issued before any decode
        const size_t base = ((size_t)e * F + j0 + r) * KW, gbase = ((size_t)e * F + j0 + r) * NG;
        for (uint it = 0; it < NIT; ++it) {
            const uint w = lane + it * 32;
            wg[r][it] = gw[base + w]; wu[r][it] = uw[base + w];
            sgv[r][it] = gs[gbase + w / 8]; bgv[r][it] = gb[gbase + w / 8];
            suv[r][it] = us[gbase + w / 8]; buv[r][it] = ub[gbase + w / 8];
        }
    }
    float ag[RJ][S], au[RJ][S];
    for (uint r = 0; r < RJ; ++r) for (uint i = 0; i < S; ++i) { ag[r][i] = 0.0f; au[r][i] = 0.0f; }
    for (uint it = 0; it < NIT; ++it) {
        const uint k = (lane + it * 32) * 8;
        for (uint i = 0; i < ns; ++i) {
            float xv[8]; ld8(x + (srv[i] / KT) * D + k, xv);
            float xs = 0.0f; for (uint j = 0; j < 8; ++j) xs += xv[j];
            for (uint r = 0; r < RJ; ++r) {
                ag[r][i] += b2f(sgv[r][it]) * dq8(wg[r][it], xv) + b2f(bgv[r][it]) * xs;
                au[r][i] += b2f(suv[r][it]) * dq8(wu[r][it], xv) + b2f(buv[r][it]) * xs;
            }
        }
    }
    for (uint i = 0; i < ns; ++i)
        for (uint r = 0; r < RJ; ++r) {
            const float gg = bf(simd_sum(ag[r][i])), uu = bf(simd_sum(au[r][i]));
            if (lane == 0) h[srv[i] * F + j0 + r] = (bfloat16_t)(bf(gg / (1.0f + metal::exp(-gg))) * uu);
        }
"""

SRC_B = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint sg = t / 32, lane = t % 32;
    constexpr uint NB = D / RO, NP = S * KT, KW = F / 8, NG = F / 64, NIT = (KW + 31) / 32;
    const uint gsid = tg * 8 + sg;
    const uint p = gsid / NB, ob = gsid % NB;
    if (p >= NP) return;
    const int e = ids[p];
    for (uint q = 0; q < p; ++q) if (ids[q] == e) return;
    uint srv[S]; uint ns = 0;
    for (uint q = p; q < NP; ++q) if (ids[q] == e) srv[ns++] = q;
    const uint o0 = ob * RO;
    uint wd[RO][NIT]; ushort sdv[RO][NIT], bdv[RO][NIT];
    for (uint r = 0; r < RO; ++r) {
        const size_t base = ((size_t)e * D + o0 + r) * KW, gbase = ((size_t)e * D + o0 + r) * NG;
        for (uint it = 0; it < NIT; ++it) {
            const uint w = lane + it * 32;
            const bool ok = w < KW;
            wd[r][it] = ok ? dw[base + w] : 0u;
            sdv[r][it] = ok ? ds[gbase + w / 8] : (ushort)0; bdv[r][it] = ok ? db[gbase + w / 8] : (ushort)0;
        }
    }
    float acc[RO][S];
    for (uint r = 0; r < RO; ++r) for (uint i = 0; i < S; ++i) acc[r][i] = 0.0f;
    for (uint it = 0; it < NIT; ++it) {
        const uint w = lane + it * 32;
        if (w >= KW) break;
        const uint k = w * 8;
        for (uint i = 0; i < ns; ++i) {
            float hv[8]; ld8(h + srv[i] * F + k, hv);
            float hs = 0.0f; for (uint j = 0; j < 8; ++j) hs += hv[j];
            for (uint r = 0; r < RO; ++r) acc[r][i] += b2f(sdv[r][it]) * dq8(wd[r][it], hv) + b2f(bdv[r][it]) * hs;
        }
    }
    for (uint i = 0; i < ns; ++i)
        for (uint r = 0; r < RO; ++r) {
            const float v = simd_sum(acc[r][i]);
            if (lane == 0) part[srv[i] * D + o0 + r] = (bfloat16_t)v;
        }
"""

SRC_C = """
    const uint i = thread_position_in_grid.x;
    if (i >= S * D) return;
    const uint row = i / D, o = i % D;
    float y = 0.0f;
    for (uint k = 0; k < KT; ++k) y += bf((float)part[(row * KT + k) * D + o] * sc[row * KT + k]);
    out[i] = (bfloat16_t)y;
"""

_KA = mx.fast.metal_kernel(name="xk_gateup", input_names=["x", "ids", "gw", "gs", "gb", "uw", "us", "ub"],
                           output_names=["h"], header=HDR, source=SRC_A)
_KB = mx.fast.metal_kernel(name="xk_down", input_names=["h", "ids", "dw", "ds", "db"], output_names=["part"],
                           header=HDR, source=SRC_B)
_KC = mx.fast.metal_kernel(name="xk_combine", input_names=["part", "sc"], output_names=["out"], header=HDR, source=SRC_C)

RJ, RO = int(__import__("os").environ.get("XK_RJ", "2")), int(__import__("os").environ.get("XK_RO", "4"))


def experts(x, ids, scores, gate, up, down, KT=10):
    """x [S, D] bf16; ids [S, KT] int32 (expert or slot ids); scores [S, KT] float32;
    gate/up/down: (weight, scales, biases) of [E, out, in] 4-bit g64 tensors (or slot pools). Returns [S, D] bf16."""
    S, D = x.shape
    F = gate[0].shape[1]
    idf = ids.reshape(-1).astype(mx.int32)
    tm = [("S", S), ("KT", KT), ("D", D), ("F", F), ("RJ", RJ), ("RO", RO)]
    h = _KA(inputs=[x, idf, gate[0], gate[1].view(mx.uint16), gate[2].view(mx.uint16),
                    up[0], up[1].view(mx.uint16), up[2].view(mx.uint16)], template=tm,
            grid=(-(-S * KT * (F // RJ) // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(S * KT, F)], output_dtypes=[mx.bfloat16])[0]
    part = _KB(inputs=[h, idf, down[0], down[1].view(mx.uint16), down[2].view(mx.uint16)], template=tm,
               grid=(-(-S * KT * (D // RO) // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(S * KT, D)], output_dtypes=[mx.bfloat16])[0]
    return _KC(inputs=[part, scores.reshape(-1).astype(mx.float32)], template=tm,
               grid=(-(-S * D // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(S, D)], output_dtypes=[mx.bfloat16])[0]
