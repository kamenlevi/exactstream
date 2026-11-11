"""Split-K decode matvec vs MLX on Qwen3.8-Flash-Next 125B shapes (M = 1 and 4 input rows).
Kernel: grid = row tiles x SPLITS threadgroups; each simdgroup does RSG rows over one K-chunk and writes a partial;
the last threadgroup to finish a row tile (device atomic counter, reset by itself) sums the partials in fixed order
s = 0..SPLITS-1. Deterministic, and batch-invariant by construction (each input row uses the same order at any M).
Measured as a dependent chain of NC distinct matrices (output feeds the next input, same glue for both variants),
so per-op ramp/tail costs are included as in real decode. Prints µs per op, effective GB/s, and best SPLITS.
Usage: splitk.py [M]   (GPU, quiet lock)"""
import sys, time
import numpy as np
import mlx.core as mx

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NC = 16

SRC = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint sg = t / 32, lane = t % 32;
    const uint RT = 8 * RSG;
    const uint ntile = (R + RT - 1) / RT;
    const uint tile = tg % ntile, split = tg / ntile;
    const uint KC = K / SPLITS, k0 = split * KC;
    const uint NG = K / 64;
    for (uint rr = 0; rr < RSG; ++rr) {
        const uint row = tile * RT + sg * RSG + rr;
        if (row >= R) break;
        float acc[M];
        for (uint m = 0; m < M; ++m) acc[m] = 0.0f;
if (BITS == 16) {
        device const ushort4* wr = (device const ushort4*)(w + (size_t)row * K);
        for (uint k = k0 + lane * 4; k < k0 + KC; k += 128) {
            ushort4 q = wr[k / 4];
            float4 wf = float4(as_type<float>((uint)q.x << 16), as_type<float>((uint)q.y << 16),
                               as_type<float>((uint)q.z << 16), as_type<float>((uint)q.w << 16));
            for (uint m = 0; m < M; ++m) acc[m] += dot(wf, *(device const float4*)(x + m * K + k));
        }
        } else {
        constexpr uint PER = 32 / BITS;                 // values per uint32
        constexpr uint MASK = (1u << BITS) - 1u;
        device const uint2* wr = (device const uint2*)(w + (size_t)row * (K / PER));
        for (uint k = k0 + lane * 2 * PER; k < k0 + KC; k += 64 * PER) {
            uint2 q = wr[k / (2 * PER)];
            const uint g = k / 64;
            const float s = as_type<float>((uint)sc[row * NG + g] << 16), b = as_type<float>((uint)bi[row * NG + g] << 16);
            for (uint m = 0; m < M; ++m) {
                device const float* xm = x + m * K + k;
                float d = 0.0f, xs = 0.0f;
                for (uint j = 0; j < PER; ++j) { float xv = xm[j]; d += (float)((q.x >> (BITS * j)) & MASK) * xv; xs += xv; }
                for (uint j = 0; j < PER; ++j) { float xv = xm[PER + j]; d += (float)((q.y >> (BITS * j)) & MASK) * xv; xs += xv; }
                acc[m] += s * d + b * xs;
            }
        }
        }
        for (uint m = 0; m < M; ++m) {
            float a = simd_sum(acc[m]);
            if (lane == 0) part[((size_t)split * M + m) * R + row] = a;
        }
    }
    threadgroup bool last;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);   // every writer publishes
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        device atomic_uint* c = (device atomic_uint*)(const_cast<device uint*>(cnt)) + tile;
        uint prev = atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
        last = (prev == SPLITS - 1);
        if (last) atomic_store_explicit(c, 0u, memory_order_relaxed);   // ready for the next call
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);   // every reader acquires
    if (last) {
        for (uint i = t; i < RT * M; i += 256) {
            const uint row = tile * RT + i % RT, m = i / RT;
            if (row < R) {
                float a = 0.0f;
                for (uint s = 0; s < SPLITS; ++s) a += part[((size_t)s * M + m) * R + row];
                y[m * R + row] = a;
            }
        }
    }
"""
kern = mx.fast.metal_kernel(name="splitk_mv", input_names=["w", "sc", "bi", "x", "cnt"], output_names=["y", "part"],
                            source=SRC)


def make(R, K, bits):
    if bits == 16:
        W = (mx.random.normal((R, K)) * 0.02).astype(mx.bfloat16)
        return dict(w=W, wv=W.view(mx.uint16), sc=mx.zeros((1,), mx.uint16), bi=mx.zeros((1,), mx.uint16), bits=16)
    q, s, b = mx.quantize((mx.random.normal((R, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=bits)
    return dict(q=q, s=s, b=b, wv=q, sc=s.view(mx.uint16), bi=b.view(mx.uint16), bits=bits)


def mlx_op(P, x):                       # x: (M, K) bf16
    if P["bits"] == 16:
        return x @ P["w"].T
    return mx.quantized_matmul(x, P["q"], P["s"], P["b"], transpose=True, group_size=64, bits=P["bits"])


def sk_op(P, x, R, K, splits, rsg, cnt):
    ntile = -(-R // (8 * rsg))
    y, _ = kern(inputs=[P["wv"], P["sc"], P["bi"], x, cnt],
                template=[("R", R), ("K", K), ("SPLITS", splits), ("RSG", rsg), ("BITS", P["bits"]), ("M", M)],
                grid=(ntile * splits * 256, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(M, R), (splits * M * R,)], output_dtypes=[mx.float32, mx.float32])
    return y


def glue(y, K):
    R = y.shape[1]
    return y[:, :K] if R >= K else mx.tile(y, (1, -(-K // R)))[:, :K]


def chain(ops, x0, K):
    def run():
        x = x0
        for op in ops:
            x = glue(op(x), K)
        mx.eval(x)
        return x
    for _ in range(3):
        run()
    ts = []
    for _ in range(9):
        a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
    return sorted(ts)[4] / len(ops) * 1e6


SHAPES = [  # name, R, K, bits, calls per token on the 125B
    ("HC down 320x10240 bf16", 320, 10240, 16, 97),
    ("HC up 10240x320 bf16", 10240, 320, 16, 97),
    ("HC inject 4x10240 bf16", 4, 10240, 16, 96),
    ("router 512x2560 q8", 512, 2560, 8, 48),
    ("GDN in_proj_qkv 10240x2560 q4", 10240, 2560, 4, 36),
    ("GDN in_proj_z 6144x2560 q4", 6144, 2560, 4, 36),
    ("out/o_proj 2560x6144 q4", 2560, 6144, 4, 48),
    ("attn q_proj 12288x2560 q4", 12288, 2560, 4, 12),
    ("shared gate+up 1280x2560 q4", 1280, 2560, 4, 48),
    ("shared down 2560x640 q4", 2560, 640, 4, 48),
    ("PLE key_proj 10240x2560 bf16", 10240, 2560, 16, 1),
]
print(f"M = {M} input rows; dependent chain of {NC} distinct matrices per shape")
tot_save = 0.0
for name, R, K, bits, calls in SHAPES:
    Ps = [make(R, K, bits) for _ in range(NC)]
    mx.eval([v for P in Ps for v in P.values() if isinstance(v, mx.array)])
    wbytes = sum(a.nbytes for a in ((Ps[0]["w"],) if bits == 16 else (Ps[0]["q"], Ps[0]["s"], Ps[0]["b"])))
    xb = mx.random.normal((M, K)).astype(mx.bfloat16)
    g_b = chain([lambda x: x.astype(mx.bfloat16)[:, :R] if R <= K else mx.tile(x, (1, -(-R // K)))[:, :R]] * NC, xb, K)
    t_mlx = chain([lambda x, P=P: mlx_op(P, x) for P in Ps], xb, K) - g_b
    xf = xb.astype(mx.float32)
    best = None
    for rsg in (1, 2, 4):
        for splits in (1, 2, 4, 8, 16):
            kc = K // splits
            if K % splits or kc < 128 or (bits == 16 and kc % 4) or (bits != 16 and kc % 64):
                continue
            ntile = -(-R // (8 * rsg))
            cnt = mx.zeros((max(ntile, 4096),), mx.uint32); mx.eval(cnt)
            try:
                t = chain([lambda x, P=P: sk_op(P, x, R, K, splits, rsg, cnt) for P in Ps], xf, K) - g_b
            except Exception as e:
                s_ = str(e); print("  skip", name, splits, rsg, s_[s_.find("error:"):s_.find("error:")+160]); continue
            if best is None or t < best[0]:
                best = (t, splits, rsg)
    # correctness + determinism of the best config
    t_sk, splits, rsg = best
    ntile = -(-R // (8 * rsg)); cnt = mx.zeros((max(ntile, 4096),), mx.uint32); mx.eval(cnt)
    a1 = sk_op(Ps[0], xf, R, K, splits, rsg, cnt); a2 = sk_op(Ps[0], xf, R, K, splits, rsg, cnt)
    ref = mlx_op(Ps[0], xb).astype(mx.float32)
    err = (mx.abs(a1 - ref).max() / (mx.abs(ref).max() + 1e-9)).item()
    det = mx.array_equal(a1, a2).item()
    save = (t_mlx - t_sk) * calls
    tot_save += save
    print(f"{name:32s} MLX {t_mlx:6.1f} µs ({wbytes/t_mlx/1e3:5.0f} GB/s) | split-K {t_sk:6.1f} µs "
          f"({wbytes/t_sk/1e3:5.0f} GB/s, SPLITS={splits}, RSG={rsg}) | rel.err {err:.1e} det {det} | "
          f"x{calls}/token: {save/1e3:+.2f} ms", flush=True)
print(f"projected saving per token (all shapes, M={M}): {tot_save/1e3:+.2f} ms")
