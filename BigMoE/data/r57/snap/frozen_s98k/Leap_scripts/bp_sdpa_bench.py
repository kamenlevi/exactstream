"""BP-8a (Leap, Wed 2026-11-04 16:2x; TABLE_350 'BP-8'): DFlash2's attention core at its real shape, priced. BP-3: SDPA is
~1.8 ms of hidden_states (~0.36 ms per layer, ~10x its K / V bytes). Shape: 16 block queries x 32 heads (GQA: 8 kv heads)
x 128 over ctx_len context keys (the sliding window keeps up to 2,047) + the 16 block keys; mask = context keys inside
the window | causal block keys (DFlashAttention, verbatim).
Arms: MLX SDPA with the bool mask (today), with an additive bf16 mask (0 / -inf), mask None, mask "causal"; and the same
attention as plain matmuls + softmax (fp32 softmax). us per call (20 calls per eval, median of 9 rounds), and max |diff| vs
the bool-mask arm.
usage: [BigMoE venv] bp_sdpa_bench.py -> ~/Leap/data/bp_sdpa_bench.txt (GPU, small)"""
import os, time
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/bp_sdpa_bench.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nBP-8a {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
H, KVH, D, L, W = 32, 8, 128, 16, 2048
mx.random.seed(3)
for ctx_len in (1000, 2047):
    q = (mx.random.normal((1, H, L, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, KVH, ctx_len + L, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, KVH, ctx_len + L, D)) * 0.5).astype(mx.bfloat16)
    query = ctx_len + mx.arange(L)[:, None]
    key = mx.arange(ctx_len + L)[None]
    mask = ((key < ctx_len) & (query - key < W)) | ((key >= ctx_len) & (key <= query))
    addm = mx.where(mask, mx.array(0.0, mx.bfloat16), mx.array(-float("inf"), mx.bfloat16))
    sc = D ** -0.5
    mx.eval(q, k, v, mask, addm)
    def plain():
        kk = mx.repeat(k, H // KVH, axis=1); vv = mx.repeat(v, H // KVH, axis=1)
        s_ = (q.astype(mx.float32) @ kk.astype(mx.float32).transpose(0, 1, 3, 2)) * sc
        s_ = mx.where(mask, s_, -float("inf"))
        return (mx.softmax(s_, axis=-1) @ vv.astype(mx.float32)).astype(mx.bfloat16)
    arms = {"sdpa bool mask": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=mask),
            "sdpa additive": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=addm),
            "sdpa none": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc),
            "sdpa causal": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal"),
            "plain fp32": plain}
    ref = arms["sdpa bool mask"]()
    outs = {k_: f() for k_, f in arms.items()}
    mx.eval(ref, *outs.values())
    for f in arms.values():
        mx.eval([f() for _ in range(4)])
    T = {k_: [] for k_ in arms}
    for r in range(9):
        for k_ in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval([arms[k_]() for _ in range(20)]); mx.synchronize()
            T[k_].append((time.perf_counter() - t0) / 20)
    say(f"ctx {ctx_len}: " + "; ".join(
        f"{k_} {1e6 * np.median(v_):.1f} us (max|d| {float(mx.max(mx.abs(outs[k_].astype(mx.float32) - ref.astype(mx.float32))).item()):.2e})"
        for k_, v_ in T.items()))
