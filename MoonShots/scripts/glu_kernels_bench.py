"""Round 26 baseline: routed-expert GLU throughput on this Mac for the 125B's shapes (gate_up fused [1280, 2560] and down
[2560, 640], 4-bit g64, 128-slot pools as BigMoE's C=128), M = 1..4 rows x top-10, measured the way a forward pass runs
it: 48 GLUs per command buffer over 8 distinct layer pools (2.8 GB, far beyond the SLC), each layer's input depending on
the previous one ("chain", like the model) or all independent ("free", the scheduler may overlap). Variants:
  stock   MLX's chain as SwitchGLU runs it: gather_qmm(gu) -> split -> silu*mul -> gather_qmm(down) -> weighted sum
  mtplx   MTPLX's fused kernels (moe_glu_decode M=1 / moe_glu_verify M=2..4): two dispatches per layer
  stream  a pure read of the same expert bytes (uint4 loads, one dispatch per layer): the bandwidth ceiling for this
          access pattern, plus one big contiguous read for the machine's roofline
GB/s counts unique expert bytes per layer (2.77 MB each).   usage: glu_kernels_bench.py   (GPU ~3 GB; quiet lock)"""
import time
import mlx.core as mx
import mlx.nn as nn
from mtplx.kernels.moe_glu_decode import moe_glu_decode, moe_glu_verify
SLOTS, K, N, G, NLP, REPS = 128, 2560, 640, 64, 8, 6
mx.random.seed(0)


def qpool(rows, cols):
    w = (mx.random.normal((SLOTS, rows, cols)) * 0.02).astype(mx.bfloat16)
    q = mx.quantize(w, group_size=G, bits=4)
    mx.eval(q)
    del w
    return q


POOLS = [(qpool(2 * N, K), qpool(K, N)) for _ in range(NLP)]
BYTES = {}
for (gw, gs, gb), (dw, ds, db) in POOLS[:1]:
    per = (gw.nbytes + gs.nbytes + gb.nbytes + dw.nbytes + ds.nbytes + db.nbytes) / SLOTS
print(f"expert bytes {per / 1e6:.3f} MB; pools {NLP} x {SLOTS} slots = {NLP * SLOTS * per / 1e9:.1f} GB", flush=True)


def stock(x, li, idx, w, u=None):
    (gw, gs, gb), (dw, ds, db) = POOLS[li]
    xg = mx.expand_dims(x, (-2, -3))                                           # [M, 1, 1, K]
    gu = mx.gather_qmm(xg, gw, gs, gb, rhs_indices=idx, transpose=True, group_size=G, bits=4)   # [M, k, 1, 2N]
    g, u = mx.split(gu, 2, axis=-1)
    h = nn.silu(g) * u
    y = mx.gather_qmm(h, dw, ds, db, rhs_indices=idx, transpose=True, group_size=G, bits=4)    # [M, k, 1, K]
    return (y.squeeze(-2) * w[..., None].astype(y.dtype)).sum(axis=-2)


def mtplx(x, li, idx, w, u=None):
    (gw, gs, gb), (dw, ds, db) = POOLS[li]
    if x.shape[0] == 1:
        return moe_glu_decode(x[0], gw, gs, gb, dw, ds, db, idx[0].astype(mx.uint32), w[0].astype(mx.float32),
                              gu_group_size=G, dn_group_size=G)[None]
    return moe_glu_verify(x, gw, gs, gb, dw, ds, db, idx.reshape(-1).astype(mx.uint32), w.reshape(-1).astype(mx.float32),
                          gu_group_size=G, dn_group_size=G)


_rd = mx.fast.metal_kernel(
    name="stream_read",
    input_names=["a", "b", "ids", "n_ids"],
    output_names=["out"],
    source="""
    // each threadgroup reads one expert's gate_up (a) and down (b) slices with uint4 loads; xor-reduce, one word out
    const uint e = ids[threadgroup_position_in_grid.x];
    const uint t = thread_position_in_threadgroup.x;
    const uint nt = threads_per_threadgroup.x;
    const device uint4* pa = (const device uint4*)(a) + (size_t)e * A4;
    const device uint4* pb = (const device uint4*)(b) + (size_t)e * B4;
    uint4 acc = uint4(0);
    for (uint i = t; i < A4; i += nt) acc ^= pa[i];
    for (uint i = t; i < B4; i += nt) acc ^= pb[i];
    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    r = simd_xor(r) ;
    if ((t & 31) == 0) atomic_fetch_xor_explicit((device atomic_uint*)&out[0], r, memory_order_relaxed);
    """,
)


def stream(x, li, idx, w, u=None):
    (gw, gs, gb), (dw, ds, db) = POOLS[li]
    (o,) = _rd(inputs=[gw, dw, u, u.size], template=[("A4", gw.shape[1] * gw.shape[2] // 4), ("B4", dw.shape[1] * dw.shape[2] // 4)],
               grid=(u.size * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32],
               init_value=0)
    return x + o.astype(x.dtype) * 0


def run(fn, M, chain=True, iters=5):
    xs = mx.random.normal((M, K)).astype(mx.bfloat16) * 0.1
    idxs = [mx.stack([mx.random.permutation(SLOTS)[:10] for _ in range(M)]).astype(mx.uint32) for _ in range(NLP * REPS)]
    ws = [mx.softmax(mx.random.normal((M, 10)), axis=-1) for _ in range(NLP * REPS)]
    mx.eval(xs, *idxs, *ws)
    us = [mx.array(sorted(set(i.reshape(-1).tolist())), dtype=mx.uint32) for i in idxs]
    mx.eval(*us)
    uniq = sum(u.size for u in us)

    def once():
        x, acc = xs, []
        for l in range(NLP * REPS):
            y = fn(x if chain else xs, l % NLP, idxs[l], ws[l], us[l])
            if chain:
                x = (x + y * 0.01).astype(mx.bfloat16)
            else:
                acc.append(y)
        mx.eval(x if chain else mx.stack(acc).sum(axis=0))
    once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters / (NLP * REPS)
    return dt * 1e3, uniq / (NLP * REPS) * per / dt / 1e9


if __name__ == "__main__":
    big = mx.zeros((1 << 28,), mx.uint32)                          # 1 GiB contiguous
    mx.eval(big)
    t0 = time.perf_counter()
    for _ in range(10):
        mx.eval(big.sum())
    roof = 10 * big.nbytes / (time.perf_counter() - t0) / 1e9
    print(f"roofline (1 GiB contiguous sum): {roof:.0f} GB/s", flush=True)
    print("ms per layer GLU (GB/s of unique expert bytes):")
    for M in (1, 2, 3, 4):
        parts = []
        for name, fn in (("stock", stock), ("mtplx", mtplx), ("stream", stream)):
            for chain in (True, False):
                if name == "stream" and chain:
                    continue
                try:
                    ms, gbs = run(fn, M, chain)
                    parts.append(f"{name}{'/chain' if chain else '/free'} {ms:.3f} ({gbs:.0f})")
                except Exception as e:
                    parts.append(f"{name} ERR {type(e).__name__}: {str(e)[:60]}")
        print(f"M={M}: " + " | ".join(parts), flush=True)
