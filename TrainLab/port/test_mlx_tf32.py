"""MLX fp32 matmul precision on this Mac (MTP Head, Thu 11-05; root cause of the PORT steps FAIL): MLX 0.32.2 runs fp32
matmuls as TF32 unless MLX_ENABLE_TF32=0 (mlx/include/mlx/utils.h: get_var("MLX_ENABLE_TF32", 1)). Measures (1) the
relative error of an fp32 GPU matmul vs float64, (2) what a weight change 100x below TF32's resolution does to the output
(true fp32: the matching tiny change; TF32: nothing, or a jump of a whole TF32 step), each sign separately.
Run twice: as is and with MLX_ENABLE_TF32=0."""
import os
import numpy as np, mlx.core as mx
g = np.random.default_rng(0)
x = g.standard_normal((256, 2560)).astype(np.float32)
w = (g.standard_normal((2560, 640)) * 0.02).astype(np.float32)
w = (w.view(np.uint32) & 0xFFFF0000).view(np.float32)          # bf16-exact weights, as at the HF start
ref = x.astype(np.float64) @ w.astype(np.float64)
out = np.array(mx.matmul(mx.array(x), mx.array(w)), dtype=np.float64)
rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
d = (np.abs(w) * 1e-5).astype(np.float32)                      # ~100x below TF32's step (2^-10 relative)
res = []
for sgn in (+1, -1):
    w2 = (w + sgn * d).astype(np.float32)
    true = x.astype(np.float64) @ (w2.astype(np.float64) - w.astype(np.float64))
    got = np.array(mx.matmul(mx.array(x), mx.array(w2)), dtype=np.float64) - out
    res.append(np.linalg.norm(got) / np.linalg.norm(true))
print(f"MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32', 'unset (default 1)')}: fp32 matmul rel error vs float64 {rel:.2e}; "
      f"output change / true change for a +1e-5 relative weight change {res[0]:.2f}, for -1e-5 {res[1]:.2f}")
