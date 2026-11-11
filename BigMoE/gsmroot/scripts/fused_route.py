"""Fused routing + residency lookup + cache-aware substitution, one Metal kernel per layer.

Replaces ~10 small MLX ops in slotmoe.slot_layer (softmax, argsort, take, second argsort, gathers,
where, normalise). One threadgroup of 256 threads per token:
  softmax over the 256 router logits -> bitonic sort (desc) of (prob, id) in threadgroup memory ->
  thread 0 walks the top-16, keeps the first 8 *resident* (table[id] >= 0), writes their slot ids
  and normalised weights (0 weight + slot 0 when fewer than 8 are resident), the wanted top-8 ids
  (for the CPU loader) and the miss count.
  fused_route.py test   -> compares against the reference MLX ops (summary only)"""
import sys
import mlx.core as mx

_src = r"""
    uint tok = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;          // 0..255 = expert id
    threadgroup float pv[256];
    threadgroup int   pi[256];
    threadgroup float red[256];
    float v = float(logits[tok * 256 + t]);
    // softmax: max
    red[t] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) { if (t < s) red[t] = max(red[t], red[t + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx_ = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = exp(v - mx_);
    red[t] = e; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float p = e / red[0];
    pv[t] = p; pi[t] = int(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // bitonic sort, descending by prob (ties: lower id first)
    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint ixj = t ^ j;
            if (ixj > t) {
                bool up = ((t & k) == 0);
                float a = pv[t], b = pv[ixj]; int ia = pi[t], ib = pi[ixj];
                bool a_first = (a > b) || (a == b && ia < ib);
                if (up ? !a_first : a_first) { pv[t] = b; pv[ixj] = a; pi[t] = ib; pi[ixj] = ia; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (t == 0) {
        int n = 0; float wsum = 0.0f; int miss = 0;
        int sl[8]; float w[8];
        for (int r = 0; r < 16 && n < 8; r++) {
            int id = pi[r]; int s = table[id];
            if (s >= 0) { sl[n] = s; w[n] = pv[r]; wsum += pv[r]; n++; }
        }
        for (int r = 0; r < 8; r++) { want[tok * 8 + r] = pi[r]; if (table[pi[r]] < 0) miss++; }
        for (int r = n; r < 8; r++) { sl[r] = 0; w[r] = 0.0f; }
        float inv = wsum > 0.0f ? 1.0f / wsum : 0.0f;
        for (int r = 0; r < 8; r++) { slots[tok * 8 + r] = uint(sl[r]); weights[tok * 8 + r] = w[r] * inv; }
        misses[tok] = miss;
    }
"""
_kernel = mx.fast.metal_kernel(name="moe_fused_route", input_names=["logits", "table"],
                               output_names=["slots", "weights", "want", "misses"], source=_src)


def fused_route(logits, table):
    """logits [..., 256] (router output), table int32[256]. Returns slots uint32 [..., 8],
    weights float32 [..., 8], want int32 [..., 8], misses int32 [...]"""
    lead = logits.shape[:-1]
    T = 1
    for d in lead:
        T *= d
    flat = logits.reshape(T, 256)
    slots, weights, want, misses = _kernel(
        inputs=[flat, table], grid=(T * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(T, 8), (T, 8), (T, 8), (T,)],
        output_dtypes=[mx.uint32, mx.float32, mx.int32, mx.int32])
    return (slots.reshape(*lead, 8), weights.reshape(*lead, 8), want.reshape(*lead, 8), misses.reshape(lead))


def reference(logits, table):
    p = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
    top16 = mx.argsort(-p, axis=-1)[..., :16]
    p16 = mx.take_along_axis(p, top16, axis=-1)
    s16 = mx.take(table, top16)
    res = s16 >= 0
    key = mx.arange(16, dtype=mx.int32) + 16 * (~res).astype(mx.int32)
    pos = mx.argsort(key, axis=-1)[..., :8]
    r8 = mx.take_along_axis(res, pos, axis=-1)
    sl = mx.where(r8, mx.take_along_axis(s16, pos, axis=-1), 0).astype(mx.uint32)
    w = mx.take_along_axis(p16, pos, axis=-1) * r8
    w = w / mx.maximum(w.sum(axis=-1, keepdims=True), 1e-6)
    return sl, w, top16[..., :8], (~res[..., :8]).sum(-1)


if __name__ == "__main__" and sys.argv[1:] == ["test"]:
    import time
    import numpy as np
    mx.random.seed(0)
    bad = 0
    for T in (1, 3, 64):
        lg = (mx.random.normal((T, 256)) * 2).astype(mx.bfloat16)
        tab = mx.where(mx.random.uniform(shape=(256,)) < 0.4, mx.arange(256, dtype=mx.int32) % 112, -1).astype(mx.int32)
        a = fused_route(lg, tab); b = reference(lg, tab)
        mx.eval(a, b)
        # compare as sets per token (order of equal-prob ties may differ)
        for i in range(T):
            sa = sorted(zip(np.array(a[0][i]).tolist(), np.round(np.array(a[1][i]), 4).tolist()))
            sb = sorted(zip(np.array(b[0][i]).tolist(), np.round(np.array(b[1][i]), 4).tolist()))
            bad += sa != sb
            bad += sorted(np.array(a[2][i]).tolist()) != sorted(np.array(b[2][i]).tolist())
            bad += int(a[3][i]) != int(b[3][i])
    print(f"fused vs reference: {bad} mismatching tokens (of {1 + 3 + 64})")
    lg = (mx.random.normal((1, 1, 256)) * 2).astype(mx.bfloat16)
    tab = mx.arange(256, dtype=mx.int32)
    for name, fn in (("reference ops", reference), ("fused kernel", fused_route)):
        for _ in range(20): mx.eval(fn(lg, tab))
        t0 = time.perf_counter()
        outs = [fn(lg, tab) for _ in range(2000)]
        mx.eval(outs)
        print(f"{name}: {(time.perf_counter()-t0)/2000*1e6:.1f} µs per call (lazy chain of 2000, single token)")


# ---- kernels for the selective-waiting gate path (scripts/waittop_mtp.py) ----
_topk_src = r"""
    uint tok = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    threadgroup float pv[NE]; threadgroup int pi[NE]; threadgroup float red[NE];
    float v = float(logits[tok * NE + t]);
    red[t] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = NE / 2; s > 0; s >>= 1) { if (t < s) red[t] = max(red[t], red[t + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx_ = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = exp(v - mx_);
    red[t] = e; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = NE / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    pv[t] = e / red[0]; pi[t] = int(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= NE; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint ixj = t ^ j;
            if (ixj > t) {
                bool up = ((t & k) == 0);
                float a = pv[t], b = pv[ixj]; int ia = pi[t], ib = pi[ixj];
                bool a_first = (a > b) || (a == b && ia < ib);
                if (up ? !a_first : a_first) { pv[t] = b; pv[ixj] = a; pi[t] = ib; pi[ixj] = ia; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (t < KC) { ids[tok * KC + t] = pi[t]; probs[tok * KC + t] = pv[t]; }
"""
_select_src = r"""
    uint tok = thread_position_in_grid.x;
    int n = 0; float wsum = 0.0f;
    for (int r = 0; r < KC && n < TOPK; r++) {
        int s = slots[tok * KC + r];
        if (s >= 0) { sl[tok * TOPK + n] = uint(s); w[tok * TOPK + n] = probs[tok * KC + r]; wsum += probs[tok * KC + r]; n++; }
    }
    for (int r = n; r < TOPK; r++) { sl[tok * TOPK + r] = 0u; w[tok * TOPK + r] = 0.0f; }
    float inv = wsum > 0.0f ? 1.0f / wsum : 0.0f;
    for (int r = 0; r < TOPK; r++) w[tok * TOPK + r] *= inv;
"""
_topk_k, _select_k = {}, {}


def topk_probs(logits, kc):
    """logits [T,NE] (NE = 256 or 512, ...) -> ids int32 [T,kc], probs float32 [T,kc] (softmax over NE, descending)"""
    T, ne = logits.shape
    key = (kc, ne)
    if key not in _topk_k:
        _topk_k[key] = mx.fast.metal_kernel(name=f"moe_topk_{kc}_{ne}", input_names=["logits"], output_names=["ids", "probs"],
                                            source=_topk_src, header=f"#define KC {kc}\n#define NE {ne}\n")
    return _topk_k[key](inputs=[logits], grid=(T * ne, 1, 1), threadgroup=(ne, 1, 1),
                       output_shapes=[(T, kc), (T, kc)], output_dtypes=[mx.int32, mx.float32])


def select_resident(slots, probs, topk=8):
    """slots int32 [T,kc] (-1 = absent), probs [T,kc] -> first topk resident: slots uint32 [T,topk], weights f32 [T,topk]"""
    T, kc = slots.shape
    key = (kc, topk)
    if key not in _select_k:
        _select_k[key] = mx.fast.metal_kernel(name=f"moe_select_{kc}_{topk}", input_names=["slots", "probs"], output_names=["sl", "w"],
                                              source=_select_src, header=f"#define KC {kc}\n#define TOPK {topk}\n")
    return _select_k[key](inputs=[slots, probs], grid=(T, 1, 1), threadgroup=(1, 1, 1),
                          output_shapes=[(T, topk), (T, topk)], output_dtypes=[mx.uint32, mx.float32])
