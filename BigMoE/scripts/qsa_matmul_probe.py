"""Is QSA's selector score matmul [1,S,H,D] @ [1,1,D,nb] (fp32) batch-invariant in S? Row 0 for S=1 vs S=2..8, over
head counts, head dims and block counts (nb ~ context / ratio). Prints the first non-identical configuration."""
import mlx.core as mx
bad = 0
for H in (16, 32, 64):
    for D in (64, 128):
        for nb in (64, 128, 256, 512, 1024, 2048):
            q = mx.random.normal((1, 8, H, D)).astype(mx.bfloat16).astype(mx.float32)
            p = mx.random.normal((1, 1, D, nb)).astype(mx.float32)
            r1 = mx.matmul(q[:, :1], p)
            for S in range(2, 9):
                rs = mx.matmul(q[:, :S], p)[:, :1]
                if not mx.array_equal(r1, rs).item():
                    d = mx.abs(r1 - rs).max().item()
                    print(f"DIFF H={H} D={D} nb={nb} S={S}: max|d| {d:.3g}"); bad += 1; break
print("all identical" if not bad else f"{bad} configs differ")
