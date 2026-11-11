"""Do the ops the 35B needs have VJPs w.r.t. their inputs in this MLX? (CPU, small shapes)"""
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU
mx.set_default_device(mx.cpu)
mx.random.seed(1)
def tryit(name, f, *a):
    try:
        g = mx.grad(f)(*a); mx.eval(g); print(f"{name}: OK |g| {mx.abs(g).max().item():.3e}")
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}: {str(e)[:200]}")
# quantized linear (8-bit), grad wrt x
ql = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=8)
x = mx.random.normal((1, 8, 256))
tryit("QuantizedLinear 8b dx", lambda x: ql(x).sum(), x)
# SwitchGLU quantized 4-bit with sorted path (indices.size >= 64)
sg = SwitchGLU(256, 128, 16); sg.set_dtype(mx.bfloat16)
nn.quantize(sg, group_size=64, bits=4)
xs = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
inds = mx.argpartition(mx.random.normal((1, 32, 16)), kth=-4, axis=-1)[..., -4:]
tryit("SwitchGLU q4 sorted dx", lambda x: sg(x, inds).astype(mx.float32).sum(), xs)
inds2 = inds[:, :8]
tryit("SwitchGLU q4 unsorted dx", lambda x: sg(x[:, :8], inds2).astype(mx.float32).sum(), xs)
# sdpa causal
qq = mx.random.normal((1, 4, 16, 32)); kk = mx.random.normal((1, 2, 16, 32)); vv = mx.random.normal((1, 2, 16, 32))
tryit("sdpa causal dq", lambda q: mx.fast.scaled_dot_product_attention(q, kk, vv, scale=0.17, mask="causal").sum(), qq)
# grouped depthwise conv1d
conv = nn.Conv1d(64, 64, kernel_size=4, groups=64, bias=False)
xc = mx.random.normal((1, 20, 64))
tryit("conv1d depthwise dx", lambda x: conv(x).sum(), xc)
tryit("rms_norm dx", lambda x: mx.fast.rms_norm(x, mx.ones(64), 1e-6).sum(), xc)
tryit("rope dx", lambda x: mx.fast.rope(x, 16, traditional=False, base=1e7, scale=1.0, offset=0).sum(), qq)
