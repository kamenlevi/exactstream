"""Round 26 PoC check: patches/glu_kernel/glu.py vs MLX's stock routed GLU (same pools, same routing): max / mean relative
error of y, then ms per layer and GB/s of unique expert bytes in the chained 48-layer harness of glu_kernels_bench.py.
usage: test_glu.py   (GPU ~3 GB; quiet lock)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import mlx.core as mx
import glu as G
import glu_kernels_bench as B                                   # pools, stock(), run()


def poc(x, li, idx, w, u=None):
    gu, dn = B.POOLS[li]
    return G.glu(x, gu, dn, idx.reshape(-1).astype(mx.uint32), w.reshape(-1).astype(mx.float32), group_size=B.G)


mx.random.seed(1)
for M in (1, 2, 4):
    x = (mx.random.normal((M, B.K)) * 0.5).astype(mx.bfloat16)
    idx = mx.stack([mx.random.permutation(B.SLOTS)[:10] for _ in range(M)]).astype(mx.uint32)
    w = mx.softmax(mx.random.normal((M, 10)), axis=-1)
    ys, yp = B.stock(x, 0, idx, w).astype(mx.float32), poc(x, 0, idx, w).astype(mx.float32)
    err = mx.abs(ys - yp)
    rel = (err.max() / mx.abs(ys).max()).item()
    print(f"M={M}: max |stock - poc| / max |stock| = {rel:.2e}, mean abs err {err.mean().item():.2e} (|y| mean {mx.abs(ys).mean().item():.2e})",
          flush=True)
for M in (1, 2, 3, 4):
    parts = []
    for name, fn in (("stock", B.stock), ("poc", poc)):
        ms, gbs = B.run(fn, M, True)
        parts.append(f"{name} {ms:.3f} ms ({gbs:.0f} GB/s)")
    print(f"M={M} chained: " + " | ".join(parts), flush=True)
