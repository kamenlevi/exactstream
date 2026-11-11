"""Reference-v2 candidate: a packed, deterministic, batch-invariant hyper-connection (GatedResidual) read in 3 dispatches,
with its own fast schedule (not tied to MLX's GemvWide lanes). Weights: exponent-offset pack (hc_pack.pack_bf16), lossless.
  A  norm   : (optional pending write x + block_out*inject) and the grouped RMS norm with its weight; one threadgroup per (row, stream)
  B  down   : down (320 rows) and inject (4 rows) as one 324 x 10240 packed matrix; one threadgroup per row, 8 simdgroups each take
              1/8 of K, partials reduced in fixed order in threadgroup memory; then silu(/hc) or 2*sigmoid(/hc)
  C  up     : 10240 x 320 packed; one simdgroup per output column d owns the 4 stream rows that feed it: sigmoid gate x normed,
              summed over streams in fixed order, x 1/hc
Every reduction has a fixed order independent of the number of rows M, so results are deterministic and batch-invariant.
bf16 rounding points follow the module (Linear outputs, /hc, activations, the gate product).
"""
import os, sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_exp_pack"))
from hc_pack import pack_bf16  # noqa: E402

HDR = """
inline float bf(float v) { return (float)(bfloat16_t)v; }            // round to bf16 like an op boundary
inline float b2f(ushort u) { return as_type<float>((uint)u << 16); }
inline void ld8(const device bfloat16_t* p, thread float v[8]) {           // 8 bf16 in one 16-byte load
    const uint4 u = *(const device uint4*)p;
    v[0] = as_type<float>(u.x << 16); v[1] = as_type<float>(u.x & 0xFFFF0000u);
    v[2] = as_type<float>(u.y << 16); v[3] = as_type<float>(u.y & 0xFFFF0000u);
    v[4] = as_type<float>(u.z << 16); v[5] = as_type<float>(u.z & 0xFFFF0000u);
    v[6] = as_type<float>(u.w << 16); v[7] = as_type<float>(u.w & 0xFFFF0000u);
}
// 8 consecutive weights (i a multiple of 8, same 64-group) from the exponent-offset pack
inline void dec8(device const uchar* hi, device const uchar* off, device const uchar* base, device const ushort* esc,
                 device const ushort* raw, uint i, thread float w[8]) {
    const uint g = i >> 6;
    const ushort e = esc[g];
    if (e != 0xFFFF) {
        const uint4 r = *(device const uint4*)(raw + (uint)e * 64 + (i & 63));
        w[0] = b2f(r.x & 0xFFFF); w[1] = b2f(r.x >> 16); w[2] = b2f(r.y & 0xFFFF); w[3] = b2f(r.y >> 16);
        w[4] = b2f(r.z & 0xFFFF); w[5] = b2f(r.z >> 16); w[6] = b2f(r.w & 0xFFFF); w[7] = b2f(r.w >> 16);
        return;
    }
    const uint2 h = *(device const uint2*)(hi + i);
    const uint o = *(device const uint*)(off + (i >> 1));
    const uint b = base[g];
    for (uint j = 0; j < 8; ++j) {
        const uint hb = ((j < 4 ? h.x : h.y) >> (8 * (j & 3))) & 0xFF;
        const uint ob = (o >> (4 * j)) & 0xF;
        w[j] = as_type<float>((((hb & 0x80) << 8) | ((b + ob) << 7) | (hb & 0x7F)) << 16);
    }
}
"""

SRC_A = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint m = tg / HC, s = tg % HC, sg = t / 32, lane = t % 32;
    threadgroup float red[8];
    float v[D / 256];
    float ss = 0.0f;
    for (uint j = 0; j < D / 256; ++j) {
        const uint d = t + j * 256, idx = s * D + d;
        float xv = (float)x[m * HC * D + idx];
        if (PENDING) xv = bf(xv + bf((float)bo[m * D + d] * (float)inj[m * HC + s]));
        written[m * HC * D + idx] = (bfloat16_t)xv;
        v[j] = xv;
        ss += xv * xv;
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8; ++i) tot += red[i];
    const float r = metal::precise::rsqrt(tot / (float)D + epsv[0]);
    for (uint j = 0; j < D / 256; ++j) {
        const uint idx = s * D + t + j * 256;
        normed[m * HC * D + idx] = (bfloat16_t)(bf(v[j] * r) * (float)gamma[idx]);
    }
"""

SRC_B = """
    const uint row = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint sg = t / 32, lane = t % 32;
    constexpr uint KC = K / 8;
    threadgroup float part[8][M];
    float acc[M];
    for (uint q = 0; q < M; ++q) acc[q] = 0.0f;
    const uint k0 = sg * KC;
    constexpr uint NIT = KC / 256;                   // iterations per lane (5 for K = 10240)
    // issue every iteration's packed-weight loads first (memory-level parallelism), then decode + accumulate in order
    uint2 hh[NIT]; uint oo[NIT]; uint bb[NIT]; ushort ee[NIT];
    for (uint it = 0; it < NIT; ++it) {
        const uint i = row * K + k0 + lane * 8 + it * 256;
        hh[it] = *(device const uint2*)(p_hi + i);
        oo[it] = *(device const uint*)(p_off + (i >> 1));
        bb[it] = p_base[i >> 6];
        ee[it] = p_esc[i >> 6];
    }
    for (uint it = 0; it < NIT; ++it) {
        const uint k = k0 + lane * 8 + it * 256;
        float w[8];
        if (ee[it] != 0xFFFF) {
            dec8(p_hi, p_off, p_base, p_esc, p_raw, row * K + k, w);          // rare escape group
        } else {
            for (uint j = 0; j < 8; ++j) {
                const uint hb = ((j < 4 ? hh[it].x : hh[it].y) >> (8 * (j & 3))) & 0xFF;
                const uint ob = (oo[it] >> (4 * j)) & 0xF;
                w[j] = as_type<float>((((hb & 0x80) << 8) | ((bb[it] + ob) << 7) | (hb & 0x7F)) << 16);
            }
        }
        for (uint q = 0; q < M; ++q) {
            float xv[8];
            ld8(normed + q * K + k, xv);
            float d = 0.0f;
            for (uint j = 0; j < 8; ++j) d += w[j] * xv[j];
            acc[q] += d;
        }
    }
    for (uint q = 0; q < M; ++q) {
        const float a = simd_sum(acc[q]);
        if (lane == 0) part[sg][q] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < M) {
        float y = 0.0f;
        for (uint i = 0; i < 8; ++i) y += part[i][t];
        const float tq = bf(bf(y) * (1.0f / HC));
        if (row < R) mix[t * R + row] = (bfloat16_t)(tq / (1.0f + metal::exp(-tq)));
        else injo[t * HC + (row - R)] = (bfloat16_t)(2.0f / (1.0f + metal::exp(-tq)));
    }
"""

SRC_C = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint d = tg * 8 + t / 32, lane = t % 32;
    if (d >= D) return;
    float acc[HC][M];
    for (uint s = 0; s < HC; ++s) for (uint q = 0; q < M; ++q) acc[s][q] = 0.0f;
    for (uint s = 0; s < HC; ++s) {
        const uint row = s * D + d;
        for (uint c = lane; c < R / 8; c += 32) {
            float w[8];
            dec8(p_hi, p_off, p_base, p_esc, p_raw, row * R + c * 8, w);
            for (uint q = 0; q < M; ++q) {
                float mv[8];
                ld8(mix + q * R + c * 8, mv);
                float dd = 0.0f;
                for (uint j = 0; j < 8; ++j) dd += w[j] * mv[j];
                acc[s][q] += dd;
            }
        }
    }
    for (uint s = 0; s < HC; ++s) for (uint q = 0; q < M; ++q) acc[s][q] = simd_sum(acc[s][q]);
    if (lane == 0) {
        for (uint q = 0; q < M; ++q) {
            float tot = 0.0f;
            for (uint s = 0; s < HC; ++s) {
                const float a = bf(acc[s][q]);
                const float g = bf(1.0f / (1.0f + metal::exp(-a)));
                tot += bf(g * (float)normed[q * HC * D + s * D + d]);
            }
            mixed[q * D + d] = (bfloat16_t)(tot * (1.0f / HC));
        }
    }
"""
_KA = mx.fast.metal_kernel(name="hcv2_norm", input_names=["x", "gamma", "bo", "inj", "epsv"],
                           output_names=["normed", "written"], header=HDR, source=SRC_A)
_KB = mx.fast.metal_kernel(name="hcv2_down", input_names=["normed", "p_hi", "p_off", "p_base", "p_esc", "p_raw"],
                           output_names=["mix", "injo"], header=HDR, source=SRC_B)
_KC = mx.fast.metal_kernel(name="hcv2_up", input_names=["mix", "normed", "p_hi", "p_off", "p_base", "p_esc", "p_raw"],
                           output_names=["mixed"], header=HDR, source=SRC_C)


class HCReadV2:
    """Packed reference-v2 read for one MTPLX GatedResidual (with combine): __call__(h, pending=None) ->
    (mixed, written, inject), shapes (1, S, D), (1, S, HC*D), (1, S, HC), like the module."""

    def __init__(self, mod, eps):
        self.hc, self.D = mod.hc_count, mod.hidden_size
        self.R = mod.input_mix_weight_down.weight.shape[0]
        self.gamma = mod.hc_norm.weight
        wcat = mx.concatenate([mod.input_mix_weight_down.weight, mod.block_inject_weight.weight])
        self.pd = pack_bf16(wcat)
        self.pu = pack_bf16(mod.input_mix_weight_up.weight)
        self.epsv = mx.array([eps], mx.float32)
        self.dummy_bo = mx.zeros((1, self.D), mx.bfloat16)
        self.dummy_inj = mx.zeros((1, self.hc), mx.bfloat16)
        mx.eval(self.pd, self.pu, self.epsv, self.dummy_bo, self.dummy_inj)

    def __call__(self, h, pending=None):
        S = h.shape[1]
        HC, D, R = self.hc, self.D, self.R
        x = h.reshape(S, HC * D)
        if pending is None:
            bo, inj, pend = self.dummy_bo, self.dummy_inj, 0
        else:
            bo, inj, pend = pending[0].reshape(S, D), pending[1].reshape(S, HC), 1
        normed, written = _KA(inputs=[x, self.gamma, bo, inj, self.epsv],
                              template=[("M", S), ("HC", HC), ("D", D), ("PENDING", pend)],
                              grid=(S * HC * 256, 1, 1), threadgroup=(256, 1, 1),
                              output_shapes=[(S, HC * D), (S, HC * D)], output_dtypes=[mx.bfloat16, mx.bfloat16])
        mix, injo = _KB(inputs=[normed, *self.pd], template=[("M", S), ("K", HC * D), ("R", R), ("HC", HC)],
                        grid=((R + HC) * 256, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(S, R), (S, HC)], output_dtypes=[mx.bfloat16, mx.bfloat16])
        mixed = _KC(inputs=[mix, normed, *self.pu], template=[("M", S), ("D", D), ("R", R), ("HC", HC)],
                    grid=(-(-D // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(S, D)], output_dtypes=[mx.bfloat16])[0]
        return mixed.reshape(1, S, D), (written if pend else x).reshape(1, S, HC * D), injo.reshape(1, S, HC)
