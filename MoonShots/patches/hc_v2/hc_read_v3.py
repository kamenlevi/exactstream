"""Reference-v2 candidate, 2 dispatches: the packed HC read with the grouped RMS norm fused in.
  B' : every threadgroup (one per down/inject row, 324) first computes the 4 stream norms of its input rows in a fixed order
       (identical in every threadgroup), then the split-K down/inject dot products on normed = bf(bf(v*r)*gamma) computed
       on the fly; partials reduced in fixed order; silu(/hc) or 2*sigmoid(/hc). Threadgroup 0 also publishes r for C'.
  C' : one simdgroup per output column d: the 4 up rows (packed, loads hoisted), sigmoid gate x normed (recomputed from
       x, gamma, r), stream sum in fixed order, x 1/hc; writes the residual stream too when a pending write is applied.
Deterministic and batch-invariant (all reduction orders are independent of the row count M)."""
import os, sys
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc_exp_pack"))
from hc_pack import pack_bf16           # noqa: E402
from hc_read_v2 import HDR              # noqa: E402  (bf, b2f, ld8, dec8)

HDR3 = HDR + """
template <typename PX, typename PB, typename PI>          // small inputs (< 4 KB) arrive as `constant` pointers
inline float vin(PX x, PB bo, PI inj, uint m, uint s, uint d, uint HCD, uint D, uint HCn, bool pend) {
    float v = (float)x[m * HCD + s * D + d];
    if (pend) v = bf(v + bf((float)bo[m * D + d] * (float)inj[m * HCn + s]));
    return v;
}
"""

SRC_B = """
    const uint row = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint sg = t / 32, lane = t % 32;
    constexpr uint HCD = HC * D, KC = HCD / 8, NIT = KC / 256;
    threadgroup float red[8][M * HC];
    threadgroup float rr[M * HC];
    threadgroup float part[8][M];
    // 1) stream norms (fixed order: per-thread chunks, simd_sum, then simdgroups 0..7)
    for (uint q = 0; q < M; ++q) {
        float ss[HC];
        for (uint s = 0; s < HC; ++s) ss[s] = 0.0f;
        for (uint k = t * 8; k < HCD; k += 2048) {
            const uint s = k / D;
            for (uint j = 0; j < 8; ++j) { const float v = vin(x, bo, inj, q, s, k % D + j, HCD, D, HC, PENDING); ss[s] += v * v; }
        }
        for (uint s = 0; s < HC; ++s) { const float a = simd_sum(ss[s]); if (lane == 0) red[sg][q * HC + s] = a; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < M * HC) {
        float tot = 0.0f;
        for (uint i = 0; i < 8; ++i) tot += red[i][t];
        rr[t] = metal::precise::rsqrt(tot / (float)D + epsv[0]);
        if (row == 0) rout[t] = rr[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2) split-K dot products on the normed input, packed weights (loads hoisted)
    float acc[M];
    for (uint q = 0; q < M; ++q) acc[q] = 0.0f;
    const uint k0 = sg * KC;
    uint2 hh[NIT]; uint oo[NIT]; uint bb[NIT]; ushort ee[NIT];
    for (uint it = 0; it < NIT; ++it) {
        const uint i = row * HCD + k0 + lane * 8 + it * 256;
        hh[it] = *(device const uint2*)(p_hi + i); oo[it] = *(device const uint*)(p_off + (i >> 1));
        bb[it] = p_base[i >> 6]; ee[it] = p_esc[i >> 6];
    }
    for (uint it = 0; it < NIT; ++it) {
        const uint k = k0 + lane * 8 + it * 256;
        float w[8];
        if (ee[it] != 0xFFFF) dec8(p_hi, p_off, p_base, p_esc, p_raw, row * HCD + k, w);
        else for (uint j = 0; j < 8; ++j) {
            const uint hb = ((j < 4 ? hh[it].x : hh[it].y) >> (8 * (j & 3))) & 0xFF;
            const uint ob = (oo[it] >> (4 * j)) & 0xF;
            w[j] = as_type<float>((((hb & 0x80) << 8) | ((bb[it] + ob) << 7) | (hb & 0x7F)) << 16);
        }
        const uint s = k / D;
        float gm[8];
        ld8(gamma + k, gm);
        for (uint q = 0; q < M; ++q) {
            const float r = rr[q * HC + s];
            float d = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                const float n = bf(bf(vin(x, bo, inj, q, s, k % D + j, HCD, D, HC, PENDING) * r) * gm[j]);
                d += w[j] * n;
            }
            acc[q] += d;
        }
    }
    for (uint q = 0; q < M; ++q) { const float a = simd_sum(acc[q]); if (lane == 0) part[sg][q] = a; }
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
    constexpr uint HCD = HC * D, NC = R / 8, NCL = (NC + 31) / 32;    // chunks per row, per lane
    float acc[HC][M];
    for (uint s = 0; s < HC; ++s) for (uint q = 0; q < M; ++q) acc[s][q] = 0.0f;
    uint2 hh[HC][NCL]; uint oo[HC][NCL]; uint bb[HC][NCL]; ushort ee[HC][NCL];
    for (uint s = 0; s < HC; ++s) for (uint c = 0; c < NCL; ++c) {
        const uint cc = lane + c * 32;
        if (cc < NC) {
            const uint i = (s * D + d) * R + cc * 8;
            hh[s][c] = *(device const uint2*)(p_hi + i); oo[s][c] = *(device const uint*)(p_off + (i >> 1));
            bb[s][c] = p_base[i >> 6]; ee[s][c] = p_esc[i >> 6];
        }
    }
    for (uint s = 0; s < HC; ++s) for (uint c = 0; c < NCL; ++c) {
        const uint cc = lane + c * 32;
        if (cc >= NC) continue;
        const uint i = (s * D + d) * R + cc * 8;
        float w[8];
        if (ee[s][c] != 0xFFFF) dec8(p_hi, p_off, p_base, p_esc, p_raw, i, w);
        else for (uint j = 0; j < 8; ++j) {
            const uint hb = ((j < 4 ? hh[s][c].x : hh[s][c].y) >> (8 * (j & 3))) & 0xFF;
            const uint ob = (oo[s][c] >> (4 * j)) & 0xF;
            w[j] = as_type<float>((((hb & 0x80) << 8) | ((bb[s][c] + ob) << 7) | (hb & 0x7F)) << 16);
        }
        for (uint q = 0; q < M; ++q) {
            float mv[8];
            ld8(mix + q * R + cc * 8, mv);
            float dd = 0.0f;
            for (uint j = 0; j < 8; ++j) dd += w[j] * mv[j];
            acc[s][q] += dd;
        }
    }
    for (uint s = 0; s < HC; ++s) for (uint q = 0; q < M; ++q) acc[s][q] = simd_sum(acc[s][q]);
    if (lane == 0) {
        for (uint q = 0; q < M; ++q) {
            float tot = 0.0f;
            for (uint s = 0; s < HC; ++s) {
                const float v = vin(x, bo, inj, q, s, d, HCD, D, HC, PENDING);
                if (PENDING) written[q * HCD + s * D + d] = (bfloat16_t)v;
                const float n = bf(bf(v * rin[q * HC + s]) * (float)gamma[s * D + d]);
                const float a = bf(acc[s][q]);
                const float g = bf(1.0f / (1.0f + metal::exp(-a)));
                tot += bf(g * n);
            }
            mixed[q * D + d] = (bfloat16_t)(tot * (1.0f / HC));
        }
    }
"""
_KB = mx.fast.metal_kernel(name="hcv3_down", input_names=["x", "gamma", "bo", "inj", "epsv", "p_hi", "p_off", "p_base",
                           "p_esc", "p_raw"], output_names=["mix", "injo", "rout"], header=HDR3, source=SRC_B)
_KC = mx.fast.metal_kernel(name="hcv3_up", input_names=["mix", "x", "gamma", "bo", "inj", "rin", "p_hi", "p_off",
                           "p_base", "p_esc", "p_raw"], output_names=["mixed", "written"], header=HDR3, source=SRC_C)


class HCReadV3:
    def __init__(self, mod, eps):
        self.hc, self.D = mod.hc_count, mod.hidden_size
        self.R = mod.input_mix_weight_down.weight.shape[0]
        self.gamma = mod.hc_norm.weight
        self.pd = pack_bf16(mx.concatenate([mod.input_mix_weight_down.weight, mod.block_inject_weight.weight]))
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
        tmpl = [("M", S), ("HC", HC), ("D", D), ("R", R), ("PENDING", pend)]
        mix, injo, r = _KB(inputs=[x, self.gamma, bo, inj, self.epsv, *self.pd], template=tmpl,
                           grid=((R + HC) * 256, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(S, R), (S, HC), (S * HC,)],
                           output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32])
        mixed, written = _KC(inputs=[mix, x, self.gamma, bo, inj, r, *self.pu], template=tmpl,
                             grid=(-(-D // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
                             output_shapes=[(S, D), (S, HC * D) if pend else (1,)],
                             output_dtypes=[mx.bfloat16, mx.bfloat16])
        return mixed.reshape(1, S, D), (written if pend else x).reshape(1, S, HC * D), injo.reshape(1, S, HC)
