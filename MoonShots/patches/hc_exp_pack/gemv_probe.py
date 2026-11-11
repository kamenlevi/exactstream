"""Which MLX kernel (and k_lanes) computes x @ W.T for the 125B's HC shapes at M = 2..12 rows (BigMoE's dup-row reference)?
Clones MLX's GemvWide (mlx/include/.../kernels/gemv.h) as an mx.fast.metal_kernel with a pluggable weight load and
compares bitwise against MLX for k_lanes in {4, 8, 16, 32}. Per-row results of GemvWide depend only on k_lanes
(vecs_per_tg just chunks the vectors), so one matching k_lanes per shape means a clone is bit-identical at every M.
With PACKED=1 the clone reads the exponent-offset pack (hc_pack.pack_bf16) instead of bf16 weights.
Usage: gemv_probe.py   (tiny GPU job, quality only)"""
import os, sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))

HDR = """
inline float hcdec(device const uchar* hi, device const uchar* off, device const uchar* base,
                   device const ushort* esc, device const ushort* raw, uint i) {
    const uint g = i >> 6;
    const ushort e = esc[g];
    uint bits;
    if (e != 0xFFFF) {
        bits = raw[(uint)e * 64 + (i & 63)];
    } else {
        const uint h = hi[i];
        const uint o = (off[i >> 1] >> ((i & 1) * 4)) & 0xF;
        bits = ((h & 0x80) << 8) | (((uint)base[g] + o) << 7) | (h & 0x7F);
    }
    return as_type<float>(bits << 16);
}
// Four consecutive weights (i a multiple of 4: same 64-group) from one uchar4 + one ushort load.
inline float4 hcdec4(device const uchar* hi, device const uchar* off, device const uchar* base,
                     device const ushort* esc, device const ushort* raw, uint i) {
    const uint g = i >> 6;
    const ushort e = esc[g];
    uint4 bits;
    if (e != 0xFFFF) {
        bits = uint4(*(device const ushort4*)(raw + (uint)e * 64 + (i & 63)));
    } else {
        const uint4 h = uint4(*(device const uchar4*)(hi + i));
        const uint o2 = *(device const ushort*)(off + (i >> 1));
        const uint4 o = uint4(o2 & 0xF, (o2 >> 4) & 0xF, (o2 >> 8) & 0xF, (o2 >> 12) & 0xF);
        bits = ((h & 0x80) << 8) | ((uint4(base[g]) + o) << 7) | (h & 0x7F);
    }
    return as_type<float4>(bits << 16);
}
"""
# GemvWide::run, transcribed (no bias/axpby, no batch), weight load behind LOADW(idx4) -> float4
SRC = """
    constexpr int unroll = 8;
    constexpr int num_simdgroups = KL / 8;
    constexpr int rows_per_simdgroup = 32 / KL;
    constexpr int rows_per_tg = rows_per_simdgroup * num_simdgroups;
    const uint3 tid = threadgroup_position_in_grid;
    const uint3 tgpg = threadgroups_per_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const short k_lane = simd_lid % KL;
    const short sg_row = simd_lid / KL;
    const int out_row = tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;
    const int row = min(out_row, OUTN - 1);
    const int n_v4 = INN / 4;
    const int n_main = n_v4 - n_v4 % (KL * unroll);
    const device vec<bfloat16_t, 4>* x4 = (const device vec<bfloat16_t, 4>*)in_vec;
    const int x_ld4 = INN / 4;
    const int n_chunks = (M + VPT - 1) / VPT;
#if PACKED
#define LOADW(idx4) hcdec4(w_hi, w_off, w_base, w_esc, w_raw, (uint)(row * INN + (idx4) * 4))
#else
    const device vec<bfloat16_t, 4>* w4 = (const device vec<bfloat16_t, 4>*)(mat + (size_t)row * INN);
#define LOADW(idx4) float4(w4[(idx4)])
#endif
    for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
        const int vec0 = chunk * VPT;
        int x_off4[VPT];
        for (int v = 0; v < VPT; v++) x_off4[v] = min(vec0 + v, M - 1) * x_ld4;
        float result[VPT] = {0};
        for (int base = 0; base < n_main; base += KL * unroll) {
            float4 wf[unroll];
            for (int i = 0; i < unroll; i++) wf[i] = LOADW(base + i * KL + k_lane);
            for (int v = 0; v < VPT; v++) {
                float acc = 0;
                for (int i = 0; i < unroll; i++) acc += dot(wf[i], float4(x4[x_off4[v] + base + i * KL + k_lane]));
                result[v] += acc;
            }
        }
        for (int idx = n_main + k_lane; idx < n_v4; idx += KL) {
            const float4 wf = LOADW(idx);
            for (int v = 0; v < VPT; v++) result[v] += dot(wf, float4(x4[x_off4[v] + idx]));
        }
        for (int v = 0; v < VPT; v++)
            for (ushort off = KL / 2; off >= 1; off >>= 1) result[v] += simd_shuffle_down(result[v], off);
        if (k_lane == 0 && out_row < OUTN) {
            for (int v = 0; v < VPT; v++)
                if (vec0 + v < M) out_vec[(vec0 + v) * OUTN + out_row] = static_cast<bfloat16_t>(result[v]);
        }
    }
"""
_k = {}


def gemv_wide(W, x, kl, vpt=None, packed=None):
    """y = x @ W.T with MLX's GemvWide arithmetic. W: (OUTN, INN) bf16 (or its pack); x: (M, INN) bf16."""
    M, INN = x.shape
    vpt = vpt or min(M, 16)        # all rows in one pass: each weight is read and decoded once (per-row math unchanged)
    OUTN = W.shape[0] if packed is None else packed[1]
    key = packed is not None
    if key not in _k:
        names = ["in_vec"] + (["w_hi", "w_off", "w_base", "w_esc", "w_raw"] if key else ["mat"])
        # PACKED must be a real preprocessor define (template parameters are invisible to #if)
        _k[key] = mx.fast.metal_kernel(name=f"gemv_wide_clone_{int(key)}", input_names=names,
                                       output_names=["out_vec"], header=f"#define PACKED {int(key)}\n" + HDR,
                                       source=SRC)
    rows_per_tg = (32 // kl) * (kl // 8)
    ins = [x] + (list(packed[0]) if key else [W])
    return _k[key](inputs=ins, template=[("KL", kl), ("VPT", vpt), ("M", M), ("INN", INN), ("OUTN", OUTN)],
                   grid=(32 * (kl // 8), -(-OUTN // rows_per_tg), 1), threadgroup=(32 * (kl // 8), 1, 1),
                   output_shapes=[(M, OUTN)], output_dtypes=[mx.bfloat16])[0]


if __name__ == "__main__":
    from hc_pack import pack_bf16
    rng = np.random.default_rng(0)
    for (R, K) in ((320, 10240), (10240, 320), (4, 10240), (324, 10240)):
        W = (mx.random.normal((R, K)) * 0.02).astype(mx.bfloat16)
        for M in (1, 2, 3, 6, 12):
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            ref = x @ W.T
            mx.eval(ref)
            hits = []
            for kl in (4, 8, 16, 32):
                try:
                    y = gemv_wide(W, x, kl); mx.eval(y)
                    if mx.array_equal(y, ref).item():
                        hits.append(kl)
                except Exception as e:
                    print("compile/run error", kl, str(e)[:120])
            pk = ""
            if hits:
                yp = gemv_wide(None, x, hits[0], packed=(pack_bf16(W), R)); mx.eval(yp)
                pk = f"; packed clone == MLX: {mx.array_equal(yp, ref).item()}"
            print(f"W {R}x{K} M={M:>2}: bitwise match for k_lanes {hits if hits else 'NONE (MLX uses another kernel)'}{pk}",
                  flush=True)
