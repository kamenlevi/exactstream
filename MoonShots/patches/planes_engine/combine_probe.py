"""Which accumulation does MLX's GPU sum use for the MoE block's combine (y * scores[..., None]).sum(-2), y [M, 10, D] bf16,
scores [M, 10] bf16? Candidates: sequential bf16 adds over ranks 0..9 (col_reduce with one row-thread: total = val + total,
init 0), the same split over LY row-threads (ranks j, j+LY, ... then partials combined j = 1..LY-1), and an fp32 sum.
Prints the mismatch count of each against MLX (0 = that is MLX's order). usage: combine_probe.py   (GPU, tiny)"""
import mlx.core as mx

mx.random.seed(0)
for M in (1, 2, 4, 8):
    y = (mx.random.normal((M, 10, 2560)) * 3).astype(mx.bfloat16)
    w = mx.softmax(mx.random.normal((M, 10)), axis=-1).astype(mx.bfloat16)
    p = y * w[..., None]
    ref = p.sum(axis=-2)
    cands = {}
    for ly in (1, 2, 4, 8):
        parts = []
        for j in range(ly):
            t = mx.zeros((M, 2560), mx.bfloat16)
            for r in range(j, 10, ly):
                t = p[:, r] + t
            parts.append(t)
        tot = parts[0]
        for j in range(1, ly):
            tot = parts[j] + tot
        cands[f"ly{ly}"] = tot
    cands["f32"] = p.astype(mx.float32).sum(axis=-2).astype(mx.bfloat16)
    print(f"M={M}: mismatches vs MLX", {k: int((v.view(mx.uint16) != ref.view(mx.uint16)).sum().item()) for k, v in cands.items()},
          flush=True)
