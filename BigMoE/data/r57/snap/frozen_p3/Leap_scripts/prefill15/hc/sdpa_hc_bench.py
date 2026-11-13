"""Head-chunked prefill SDPA vs the stock call, standalone (Leap 11-13 01:2x; data/prefill15_1112.md section 9).

The engine's prefill attention at the 12 full-attention layers is mx.fast.scaled_dot_product_attention with head dim
256 and a boolean [1, 1, S, T] mask: MLX's unfused fallback, which builds [n_q_heads, S, T] planes for every 1,024-row
chunk. prof_sys measured 12.35 s of main-thread KERNEL time for them at 30K (fresh, growing buffers that MLX's 1 GB cache
cannot hold). MTPLX's _sdpa_head_chunked runs each group of query heads against its own single KV head (same per-head
arithmetic, no GQA-expanded K/V, no full plane), so its equal-size pieces could come from MLX's cache within a chunk.

This sweeps one layer's 29 chunks of the 30K prompt (S = 1,024 rows, T = 1,024 .. 28,966 keys) with the model's shapes
(24 query heads, 2 KV heads, head dim 256, bf16), random data, the engine's MLX cache limit (1 GB) and a wired limit,
for the stock call and heads_per_chunk 12 / 6 / 4 / 2 / 1. Per variant: is every chunk's output bitwise equal to the
stock call's (uint16 view), the sweep's wall time, and the main thread's kernel / user CPU (Mach thread_info).
GPU memory: K/V 60 MB + the stock call's planes (~1.5-3 GB at the last chunk).
Usage: sdpa_hc_bench.py [rounds] [wired GB (16; 0 = leave unset)] [MLX cache MB (1024)] [variants, comma-separated (all)]
"""
import ctypes as ct, json, sys, threading, time

import mlx.core as mx


class _TV(ct.Structure):
    _fields_ = [("s", ct.c_int), ("us", ct.c_int)]


class _TBI(ct.Structure):
    _fields_ = [("user", _TV), ("system", _TV), ("cpu_usage", ct.c_int), ("policy", ct.c_int), ("run_state", ct.c_int),
                ("flags", ct.c_int), ("suspend_count", ct.c_int), ("sleep_time", ct.c_int)]


_LS = ct.CDLL("/usr/lib/libSystem.B.dylib")
_LS.mach_thread_self.restype = ct.c_uint
_LS.thread_info.argtypes = [ct.c_uint, ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint)]
_PORT = _LS.mach_thread_self()


def cpu():
    i = _TBI(); n = ct.c_uint(10)
    assert _LS.thread_info(_PORT, 3, ct.byref(i), ct.byref(n)) == 0
    return i.user.s + i.user.us * 1e-6, i.system.s + i.system.us * 1e-6


def sdpa_head_chunked(q, k, v, *, scale, mask, heads_per_chunk):
    """MTPLX's _sdpa_head_chunked (mtplx/models/qwen4_exp.py), copied verbatim in logic."""
    B, n_q_heads, S, D = q.shape
    n_kv_heads = k.shape[1]
    gqa = n_q_heads // n_kv_heads
    qg = q.reshape(B, n_kv_heads, gqa, S, D)
    outs = []
    for j in range(n_kv_heads):
        kj = k[:, j: j + 1]
        vj = v[:, j: j + 1]
        for hs in range(0, gqa, heads_per_chunk):
            he = min(hs + heads_per_chunk, gqa)
            outs.append(mx.fast.scaled_dot_product_attention(qg[:, j, hs:he], kj, vj, scale=scale, mask=mask))
    return mx.concatenate(outs, axis=1)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    wired = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    cache_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    mx.set_cache_limit(cache_mb << 20)                               # the engine's BIGMOE_MLX_CACHE_MB default: 1024
    if wired > 0:
        mx.set_wired_limit(int(wired * 1e9))                         # the engine sets one (R57_WIRED); residency on
    print(f"setup: wired limit {wired} GB, MLX cache {cache_mb} MB, rounds {rounds}", flush=True)
    Hq, Hk, D, S, N = 24, 2, 256, 1024, 28966                        # the model; the 30K prompt's token count
    scale = D ** -0.5
    mx.random.seed(0)
    K = (mx.random.normal((1, Hk, N, D)) * 0.5).astype(mx.bfloat16)
    V = (mx.random.normal((1, Hk, N, D)) * 0.5).astype(mx.bfloat16)
    Q = (mx.random.normal((1, Hq, N, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(K, V, Q)
    ends = list(range(S, N, S)) + [N]                                 # 29 chunk ends, as the engine's 1,024-token chunks

    def chunk(T):
        s0 = T - S if T % S == 0 else T - (T % S)
        rows = T - s0
        q = Q[:, :, s0:T]
        pos_q = mx.arange(s0, T)[:, None]
        pos_k = mx.arange(0, T)[None, :]
        mask = (pos_k <= pos_q)[None, None]                           # bool [1, 1, rows, T], causal
        return q, K[:, :, :T], V[:, :, :T], mask, rows

    variants = [("stock", None), ("hc12", 12), ("hc6", 6), ("hc4", 4), ("hc2", 2), ("hc1", 1)]
    if only:
        variants = [v for v in variants if v[0] in only or v[0] == "stock"]
    ref = {}
    res = {}
    for r in range(rounds):
        for name, h in (variants if r % 2 == 0 else variants[::-1]):  # alternate the order between rounds
            mx.clear_cache()
            w0 = time.perf_counter(); u0, s0 = cpu()
            eq = 0; n = 0
            for T in ends:
                q, k, v, m, rows = chunk(T)
                y = (mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m) if h is None
                     else sdpa_head_chunked(q, k, v, scale=scale, mask=m, heads_per_chunk=h))
                mx.eval(y)
                if name == "stock" and T not in ref:
                    ref[T] = y                                       # evaluated; kept as the reference (12.6 MB each)
                if T in ref:
                    n += 1
                    eq += int(mx.array_equal(y.view(mx.uint16), ref[T].view(mx.uint16)).item())
                del y
            w1 = time.perf_counter(); u1, s1 = cpu()
            d = res.setdefault(name, dict(wall=[], user=[], sys=[], eq=None))
            d["wall"].append(round(w1 - w0, 3)); d["user"].append(round(u1 - u0, 3)); d["sys"].append(round(s1 - s0, 3))
            d["eq"] = f"{eq}/{n}"
            print(f"round {r} {name}: bitwise {eq}/{n} chunks equal to stock; wall {w1 - w0:.2f} s; main thread user "
                  f"{u1 - u0:.2f} s, kernel {s1 - s0:.2f} s; MLX peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
            mx.reset_peak_memory()
    print("SUMMARY " + json.dumps(res))


if __name__ == "__main__":
    main()
