"""Round 32l: pin MLX's arithmetic for the GDN verify prework (conv1d k=4 depthwise -> silu -> split -> l2norm -> scale),
so an exact fused kernel can replicate it (MTPLX's fused_gdn_conv_norm_rows keeps fp32 throughout and rounds once: not
bitwise, hence "exact only at short context"). Candidates are Metal kernels (the GPU's own arithmetic), compared bitwise
against MLX's eager ops on random bf16 data shaped like the 125B's GDN (conv_dim 10240, k=4, S rows):
  conv:  c_fma   acc = fma chain t = 0..3 in fp32 (w[c,t] * x[t,c]), round to bf16
         c_rev   the same, t = 3..0
         c_sep   separate multiply and add per step (no contraction), t = 0..3
         c_pair  (x0 w0 + x1 w1) + (x2 w2 + x3 w3)
  sum:   r_l4    lane l sums 4 contiguous values [4l, 4l+4) sequentially, then simd_sum over 32 lanes
         r_s32   lane l sums values l, l+32, l+64, l+96 sequentially, then simd_sum
         r_seq   one thread, sequential over 128
usage: gdn_probe.py   (GPU, seconds)"""
import mlx.core as mx
import mlx.nn as nn

C, KW, S = 10240, 4, 4
mx.random.seed(0)
xin = (mx.random.normal((1, S + KW - 1, C)) * 2.0).astype(mx.bfloat16)          # [conv_state(3) ; S rows]
conv = nn.Conv1d(C, C, kernel_size=KW, groups=C, bias=False)
conv.weight = (mx.random.normal(conv.weight.shape) * 0.3).astype(mx.bfloat16)  # [C, KW, 1]
ref_conv = conv(xin)                                                             # [1, S, C] bf16
mx.eval(ref_conv)

CONV_SRC = """
    const uint c = thread_position_in_grid.x, s = thread_position_in_grid.y;
    if (c >= (uint)C_) return;
    float a[4], w[4];
    for (int t = 0; t < 4; ++t) { a[t] = (float)x[(s + t) * C_ + c]; w[t] = (float)wt[c * 4 + t]; }
    float acc;
    if (MODE == 0) { acc = a[0] * w[0]; acc = fma(a[1], w[1], acc); acc = fma(a[2], w[2], acc); acc = fma(a[3], w[3], acc); }
    else if (MODE == 1) { acc = a[3] * w[3]; acc = fma(a[2], w[2], acc); acc = fma(a[1], w[1], acc); acc = fma(a[0], w[0], acc); }
    else if (MODE == 2) { float p; acc = a[0] * w[0];
                          p = a[1] * w[1]; acc = acc + p; p = a[2] * w[2]; acc = acc + p; p = a[3] * w[3]; acc = acc + p; }
    else { acc = (a[0] * w[0] + a[1] * w[1]) + (a[2] * w[2] + a[3] * w[3]); }
    y[s * C_ + c] = static_cast<T>(acc);
"""
ck = mx.fast.metal_kernel(name="gdn_probe_conv", input_names=["x", "wt"], output_names=["y"], source=CONV_SRC)
for mode, name in enumerate(("c_fma", "c_rev", "c_sep", "c_pair")):
    (y,) = ck(inputs=[xin.reshape(-1), conv.weight.reshape(-1)], template=[("T", mx.bfloat16), ("C_", C), ("MODE", mode)],
              grid=(C, S, 1), threadgroup=(256, 1, 1), output_shapes=[(S * C,)], output_dtypes=[mx.bfloat16])
    n = int((y.reshape(1, S, C).view(mx.uint16) != ref_conv.view(mx.uint16)).sum().item())
    print(f"conv {name}: mismatches vs MLX conv1d {n} of {S * C}", flush=True)

# l2norm's sum: x [rows, 128] fp32 (the model casts bf16 -> fp32 first), MLX (xf * xf).sum(-1)
rows = 4096
xf = (mx.random.normal((rows, 128)) * 1.5).astype(mx.bfloat16).astype(mx.float32)
ref_sum = (xf * xf).sum(-1)
mx.eval(ref_sum)
SUM_SRC = """
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x;
    const device float* row = xf + (size_t)r * 128;
    float p = 0.0f;
    if (MODE == 0) { for (int i = 0; i < 4; ++i) { const float v = row[lane * 4 + i]; p += v * v; } }
    else if (MODE == 1) { for (int i = 0; i < 4; ++i) { const float v = row[lane + 32 * i]; p += v * v; } }
    if (MODE == 2) { if (lane == 0) { float t = 0.0f; for (int i = 0; i < 128; ++i) { const float v = row[i]; t += v * v; } out[r] = t; } return; }
    const float tot = simd_sum(p);
    if (lane == 0) out[r] = tot;
"""
sk = mx.fast.metal_kernel(name="gdn_probe_sum", input_names=["xf"], output_names=["out"], source=SUM_SRC)
for mode, name in enumerate(("r_l4", "r_s32", "r_seq")):
    (o,) = sk(inputs=[xf.reshape(-1)], template=[("MODE", mode)], grid=(rows * 32, 1, 1), threadgroup=(32, 1, 1),
              output_shapes=[(rows,)], output_dtypes=[mx.float32])
    n = int((o.view(mx.uint32) != ref_sum.view(mx.uint32)).sum().item())
    print(f"sum {name}: mismatches vs MLX (xf*xf).sum(-1) {n} of {rows}", flush=True)
# the squares themselves: MLX computes xf * xf as its own kernel (fp32), then sums -> candidates square then sum: same
