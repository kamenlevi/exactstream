"""Payoff test for a between-gates megakernel on a Qwen3.8-Flash-Next GDN layer (timing skeleton, not the real math).
Phases with the real shapes and bytes (125B, hidden 2560, HC 4x2560, 4-bit projections, bf16 HC/router):
  P1 HC read down  bf16  320 x 10240     P2 HC read up   bf16 10240 x 320
  P3 in_proj qkv+z  4-bit 16384 x 2560   P4 gated-delta state r/w (48 x 128 x 128 fp32)
  P5 out_proj       4-bit 2560 x 6144    P6/P7 second HC read (down, up)
  P8 router         bf16  512 x 2560     P9 shared expert gate+up 4-bit 1280 x 2560   P10 shared down 4-bit 2560 x 640
(a) mega : one persistent dispatch per layer, 128 threadgroups x 256, grid barriers between phases
(b) eager: the same 9 matvecs + state update as separate MLX ops (mx.matmul / quantized_matmul), dependent chain
Weights for 8 distinct layers (~0.6 GB) so nothing is served from cache across layers.
Prints ms per layer for both and the bytes/280 GB/s floor. Needs a quiet GPU lock."""
import time
import numpy as np
import mlx.core as mx

NL, G, NSG = 8, 128, 128 * 8
D, HCD, R = 2560, 10240, 320
QZ, VO, NH, HD = 16384, 6144, 48, 128
SH = 640


def bf(shape):
    return (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)


def q4(o, i):
    return mx.quantize(bf((o, i)), group_size=64, bits=4)


layers = []
for _ in range(NL):
    L = dict(hcd=bf((R, HCD)), hcu=bf((HCD, R)), qz=q4(QZ, D), out=q4(D, VO), hcd2=bf((R, HCD)), hcu2=bf((HCD, R)),
             rt=bf((512, D)), shgu=q4(2 * SH, D), shd=q4(D, SH),
             st=mx.random.normal((NH * HD * HD,)) * 0.01)
    mx.eval(L)
    layers.append(L)
nbytes = sum(a.nbytes for k, v in layers[0].items() for a in (v if isinstance(v, (tuple, list)) else (v,)))
nbytes += layers[0]["st"].nbytes                      # state is written back as well
print(f"one layer skeleton: {nbytes/1e6:.1f} MB read+written -> floor {nbytes/280e9*1e3:.3f} ms at 280 GB/s")

HDR = """
inline float bf2f(ushort b) { return as_type<float>((uint)b << 16); }
"""
SRC = """
    const uint tg = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;
    const uint lane = t % 32, gsg = tg * 8 + t / 32;
    uint phase = 0;
    // scratch layout in `sc`: x(10240) | hc(320) | mix(10240) | qz(16384) | o(2560) | hc2(320) | mix2(10240) | rt(512) | sh(1280) | y(2560)
    device float* S = sc;
    device float* X = S; device float* HC = S + 10240; device float* MIX = HC + 320; device float* QZv = MIX + 10240;
    device float* O = QZv + 16384; device float* HC2 = O + 2560; device float* MIX2 = HC2 + 320; device float* RT = MIX2 + 10240;
    device float* SHv = RT + 512; device float* Y = SHv + 1280;

#define BARRIER() { phase++; threadgroup_barrier(mem_flags::mem_device); \\
    if (t == 0) { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \\
      atomic_fetch_add_explicit((device atomic_uint*)cnt, 1u, memory_order_relaxed); uint s = 0; \\
      while (atomic_load_explicit((device atomic_uint*)cnt, memory_order_relaxed) < phase * G && ++s < 20000000u) {} \\
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); } \\
    threadgroup_barrier(mem_flags::mem_device); }

    // bf16 matvec: y[r] = sum_k W[r,k] * f(x[k]); lanes read 4 bf16 (8 bytes) at a time
#define MV_BF16(W, ROWS, K, XIN, YOUT, ACT) \\
    for (uint r = gsg; r < ROWS; r += NSG) { float a = 0.0f; device const ushort4* w4 = (device const ushort4*)(W + (size_t)r * K); \\
      for (uint k4 = lane; k4 < K / 4; k4 += 32) { ushort4 w = w4[k4]; uint k = k4 * 4; \\
        a += bf2f(w.x) * ACT(XIN[k]) + bf2f(w.y) * ACT(XIN[k+1]) + bf2f(w.z) * ACT(XIN[k+2]) + bf2f(w.w) * ACT(XIN[k+3]); } \\
      a = simd_sum(a); if (lane == 0) YOUT[r] = a; }
    // 4-bit affine matvec (MLX layout: 8 nibbles per uint32, bf16 scale/bias per 64)
#define MV_Q4(W, SC, BI, ROWS, K, XIN, YOUT) \\
    for (uint r = gsg; r < ROWS; r += NSG) { float a = 0.0f; device const uint* wr = W + (size_t)r * (K / 8); \\
      for (uint w = lane; w < K / 8; w += 32) { uint q = wr[w]; uint k = w * 8; uint g = (size_t)r * (K / 64) + k / 64; \\
        float s = bf2f(SC[g]), b = bf2f(BI[g]); float d = 0.0f, xs = 0.0f; \\
        for (uint j = 0; j < 8; ++j) { float xv = XIN[k + j]; d += (float)((q >> (4 * j)) & 15u) * xv; xs += xv; } \\
        a += s * d + b * xs; } \\
      a = simd_sum(a); if (lane == 0) YOUT[r] = a; }
#define ID(v) (v)
#define SILU(v) ((v) / (1.0f + metal::exp(-(v))))

    if (tg == 0 && t < 32) { for (uint i = t; i < 10240; i += 32) X[i] = x0[i]; }
    BARRIER();
    MV_BF16(hcd, 320, 10240, X, HC, ID);  BARRIER();          // P1
    MV_BF16(hcu, 10240, 320, HC, MIX, SILU);  BARRIER();      // P2 (mean over HC folded into next reads)
    MV_Q4(qzw, qzs, qzb, 16384, 2560, MIX, QZv);  BARRIER();  // P3
    for (uint i = tg * 256 + t; i < 48 * 128 * 128; i += G * 256) {   // P4 gated-delta state update (r/w 3.1 MB)
        stout[i] = st[i] * 0.99f + QZv[i % 16384] * 0.001f; }
    BARRIER();
    MV_Q4(ow, osc, obi, 2560, 6144, QZv, O);  BARRIER();      // P5
    MV_BF16(hcd2, 320, 10240, MIX, HC2, ID);  BARRIER();      // P6 (stand-in input: the residual streams)
    MV_BF16(hcu2, 10240, 320, HC2, MIX2, SILU);  BARRIER();   // P7
    MV_BF16(rtw, 512, 2560, MIX2, RT, ID);
    MV_Q4(shw, shs, shb, 1280, 2560, MIX2, SHv);  BARRIER();  // P8 + P9 (independent: same phase)
    MV_Q4(sdw, sds, sdb, 2560, 640, SHv, Y);                  // P10
"""
names = ["x0", "hcd", "hcu", "qzw", "qzs", "qzb", "st", "ow", "osc", "obi", "hcd2", "hcu2", "rtw", "shw", "shs", "shb",
         "sdw", "sds", "sdb"]
mega = mx.fast.metal_kernel(name="mega_gdn_skeleton", input_names=names, output_names=["cnt", "sc", "stout"],
                            header=HDR, source=SRC, atomic_outputs=False)
SCN = 10240 + 320 + 10240 + 16384 + 2560 + 320 + 10240 + 512 + 1280 + 2560


def mega_layer(L, x):
    ins = [x, L["hcd"].view(mx.uint16), L["hcu"].view(mx.uint16), L["qz"][0], L["qz"][1].view(mx.uint16),
           L["qz"][2].view(mx.uint16), L["st"], L["out"][0], L["out"][1].view(mx.uint16), L["out"][2].view(mx.uint16),
           L["hcd2"].view(mx.uint16), L["hcu2"].view(mx.uint16), L["rt"].view(mx.uint16), L["shgu"][0],
           L["shgu"][1].view(mx.uint16), L["shgu"][2].view(mx.uint16), L["shd"][0], L["shd"][1].view(mx.uint16),
           L["shd"][2].view(mx.uint16)]
    cnt, sc, st = mega(inputs=ins, template=[("G", G), ("NSG", NSG)], grid=(G * 256, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(4,), (SCN,), (NH * HD * HD,)], output_dtypes=[mx.uint32, mx.float32, mx.float32],
                       init_value=0)
    # next layer's input: the 2560-float output broadcast to the 10240 streams (dependency between layers)
    return mx.tile(sc[SCN - 2560:], 4)


def eager_layer(L, x):
    q = lambda v, p: mx.quantized_matmul(v, *p, transpose=True, group_size=64, bits=4)
    xb = x.astype(mx.bfloat16)[None]
    hc = xb @ L["hcd"].T
    mix = (hc * mx.sigmoid(hc)) @ L["hcu"].T
    qz = q(mix[:, :D], L["qz"])
    st = L["st"] * 0.99 + mx.tile(qz[0].astype(mx.float32), NH * HD * HD // QZ) * 0.001
    o = q(qz[:, :VO], L["out"])
    hc2 = mix @ L["hcd2"].T
    mix2 = (hc2 * mx.sigmoid(hc2)) @ L["hcu2"].T
    rt = mix2[:, :D] @ L["rt"].T
    sh = q(mix2[:, :D], L["shgu"])
    y = q(sh[:, :SH], L["shd"]) + o + rt.sum() * 0 + st[:1].astype(mx.bfloat16) * 0
    return mx.tile(y[0].astype(mx.float32), 4)


def bench(fn, reps=15):
    x = mx.random.normal((HCD,)).astype(mx.float32)
    def run():
        v = x
        for L in layers:
            v = fn(L, v)
        mx.eval(v)
    for _ in range(3):
        run()
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
    return sorted(ts)[reps // 2] / NL * 1e3


m = bench(mega_layer)
e = bench(eager_layer)
print(f"megakernel (1 dispatch/layer, 9 grid barriers): {m:.3f} ms/layer = {nbytes/(m*1e-3)/1e9:.0f} GB/s")
print(f"eager MLX chain (~14 ops/layer):               {e:.3f} ms/layer = {nbytes/(e*1e-3)/1e9:.0f} GB/s")
print(f"floor at 280 GB/s:                              {nbytes/280e9*1e3:.3f} ms/layer")
