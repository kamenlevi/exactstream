"""DSDPA-0 (Leap, Wed 2026-11-04 17:4x; TABLE_350 'DSDPA-0'): the drafter's attention (5 sliding layers; q 32 heads x 16
rows, K / V 8 heads x (ctx_len + 16), head_dim 128; mask = context within the 2,048 window | the whole block) costs
~0.65 ms per propose in SDPA alone (BP-10), ~0.13 ms per call for ~4-8 MB of K / V. Probe: is that MLX's kernel choice
(16 rows -> the full-attention kernel, 32 threadgroups walking every key serially), and what do cheap alternatives give?
Arms, 5 layers' distinct q / K / V per pass, ctx_len 1,008 and 2,054 (the full window at S = 8):
  mlx16      as draft_fused runs it: one SDPA call, bool mask;
  mlx16add   the same with an additive bf16 mask (0 / -inf);
  mlx8x2     two 8-row calls (the vector kernels' range?), outputs concatenated;
  nomask     one call without a mask (a floor for MLX's kernel; not the drafter's semantics);
  gqa_mm     GQA-packed matmuls: per kv head 64 query rows (4 heads x 16) @ K^T, mask, fp32 softmax, @ V.
Each arm's output vs mlx16: max |diff| (fp32), bitwise flag. Timing only (garbage inputs, no model).
usage: [BigMoE venv] dsdpa_probe.py -> ~/Leap/data/dsdpa_probe.txt (GPU, MEM ~0.3 GB; share lock)"""
import os, time
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/dsdpa_probe.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nDSDPA-0 {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
HQ, HK, D, L, SW, NL = 32, 8, 128, 16, 2048, 5
scale = D ** -0.5


def mask_for(ctx_len):
    q = ctx_len + mx.arange(L)[:, None]
    k = mx.arange(ctx_len + L)[None]
    return ((k < ctx_len) & (q - k < SW)) | (k >= ctx_len)


def arms(ctx_len):
    mx.random.seed(ctx_len)
    N = ctx_len + L
    Q = [(mx.random.normal((1, HQ, L, D)) * 1.0).astype(mx.bfloat16) for _ in range(NL)]
    K = [(mx.random.normal((1, HK, N, D)) * 1.0).astype(mx.bfloat16) for _ in range(NL)]
    V = [(mx.random.normal((1, HK, N, D)) * 1.0).astype(mx.bfloat16) for _ in range(NL)]
    m = mask_for(ctx_len)
    madd = mx.where(m, mx.array(0.0, mx.bfloat16), mx.array(-float("inf"), mx.bfloat16))
    mx.eval(Q, K, V, m, madd)
    sd = mx.fast.scaled_dot_product_attention

    def mlx16():
        return [sd(q, k, v, scale=scale, mask=m) for q, k, v in zip(Q, K, V)]

    def mlx16add():
        return [sd(q, k, v, scale=scale, mask=madd) for q, k, v in zip(Q, K, V)]

    def mlx8x2():
        return [mx.concatenate([sd(q[:, :, :8], k, v, scale=scale, mask=m[:8]),
                                sd(q[:, :, 8:], k, v, scale=scale, mask=m[8:])], axis=2) for q, k, v in zip(Q, K, V)]

    def nomask():
        return [sd(q, k, v, scale=scale) for q, k, v in zip(Q, K, V)]

    def gqa_mm():
        out = []
        for q, k, v in zip(Q, K, V):
            qg = (q * scale).reshape(1, HK, (HQ // HK) * L, D)                      # q heads 4h..4h+3 -> kv head h
            s = (qg @ k.transpose(0, 1, 3, 2)).reshape(1, HK, HQ // HK, L, N)
            s = mx.where(m, s, mx.array(-float("inf"), s.dtype))
            p = mx.softmax(s.astype(mx.float32), axis=-1).astype(mx.bfloat16).reshape(1, HK, (HQ // HK) * L, N)
            out.append((p @ v).reshape(1, HQ, L, D))
        return out
    return dict(mlx16=mlx16, mlx16add=mlx16add, mlx8x2=mlx8x2, nomask=nomask, gqa_mm=gqa_mm)


for ctx_len in (1008, 2054):
    A = arms(ctx_len)
    ref = A["mlx16"](); mx.eval(ref)
    diff = {}
    for k, f in A.items():
        o = f(); mx.eval(o)
        d = max(float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) for a, b in zip(o, ref))
        bw = all(bool(mx.array_equal(a, b).item()) for a, b in zip(o, ref))
        diff[k] = (d, bw)
    T = {k: [] for k in A}
    for r in range(21):
        for k in (list(A) if r % 2 == 0 else list(A)[::-1]):
            mx.eval(A[k]())
            mx.synchronize(); t0 = time.perf_counter()
            for _ in range(4):
                mx.eval(A[k]())
            mx.synchronize()
            T[k].append((time.perf_counter() - t0) / 4)
    say(f"ctx_len {ctx_len} (+16 block keys): ms per 5 layers (median of 21) / max |diff| vs mlx16 / bitwise: " + "; ".join(
        f"{k} {1e3 * np.median(v):.3f} / {diff[k][0]:.2e} / {diff[k][1]}" for k, v in T.items()))
