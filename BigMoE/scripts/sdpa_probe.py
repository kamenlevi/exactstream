"""Is MLX's fused SDPA (decode/verify vector kernel) row-0 invariant between q_len 1 and q_len 4 at long key lengths?
qwen4 attention: 24 q heads, 2 kv heads, head_dim 256, bf16; boolean mask (QSA selection) or none. Also the per-kv-head
chunk (12 q heads x 1 kv head) MTPLX's head-chunked verify uses."""
import mlx.core as mx
for N in (2048, 4096, 8192, 16384, 24576, 32768, 40000, 65536):
    for masked in (False, True):
        k = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
        q = (mx.random.normal((1, 24, 4, 256)) * 0.06).astype(mx.bfloat16)
        m = (mx.random.uniform(shape=(1, 1, 4, N)) < 0.07) if masked else None
        if masked: m = m | (mx.arange(N)[None, None, None, :] == 0)
        o1 = mx.fast.scaled_dot_product_attention(q[:, :, :1], k, v, scale=1 / 16, mask=None if m is None else m[:, :, :1])
        o4 = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 16, mask=m)[:, :, :1]
        c4 = mx.fast.scaled_dot_product_attention(q[:, :12], k[:, :1], v[:, :1], scale=1 / 16, mask=m)[:, :, :1]
        c1 = mx.fast.scaled_dot_product_attention(q[:, :12, :1], k[:, :1], v[:, :1], scale=1 / 16, mask=None if m is None else m[:, :, :1])
        print(f"N={N:6d} mask={masked!s:5}: S1 vs S4 {'same' if mx.array_equal(o1, o4).item() else 'DIFF'}; "
              f"chunk S1 vs S4 {'same' if mx.array_equal(c1, c4).item() else 'DIFF'}; full vs chunk S1 "
              f"{'same' if mx.array_equal(o1[:, :12], c1).item() else 'DIFF'}", flush=True)
