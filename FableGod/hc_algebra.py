#!/usr/bin/env python3 -I
"""hc_algebra.py (FableGod, mission 26). CPU-only numeric check (numpy; no GPU; no model weights: random weights at
the model's shapes) of the hyper-connection read's algebra as GatedResidual computes it (~/BigMoE/ref/qwen4_exp.py
lines 1294-1307, GroupedRMSNorm at 676, the write at 582):
  hn     = bf16(h + bf16(block_out * inject_s))           the pending write (folded into the next read's norm pass)
  normed = bf16(bf16(hn / rms_stream) * w)                grouped RMS norm: 4 streams of 2,560, one 10,240 weight
  c      = bf16(normed @ Wd^T)   [rows x 320]             down, K 10,240 (MLX: split-K, 3 partitions, fp32 partials)
  logits = bf16(normed @ Wi^T)   [rows x 4]               inject (merged with down as N 324 in X5 / S4)
  mix1   = bf16(silu(bf16(c / 4)))                        gate1
  g      = bf16(mix1 @ Wu^T)     [rows x 10,240]          up, K 320
  mixed  = bf16((1/4) sum_s bf16(bf16(sigmoid(g_s)) * normed_s))   tail: the 4 streams summed from +0 in order
  inject = bf16(2 * sigmoid(bf16(logits / 4)))            the write gate of this block's output
Checks: (1) bytes and FLOPs per read and per check; (2) design 1 (the read as two GEMM kernels: the norm in the
first kernel's prologue, W_up's columns interleaved by stream, accum + gate1 in the second kernel's prologue and
the tail in its epilogue) is the same arithmetic: bitwise; (3) design 2 (row-invariant: the norm weight folded into
Wd / Wi, the GEMM on the raw streams with the per-stream 1 / rms applied per partition in the epilogue,
stream-aligned split-K) against the reference in bf16 ulps, next to HC8 (8-bit g64 affine RTN weights) on the
same inputs, and its row invariance (1 row == 16 rows) by construction; (4) a different split-K partition count
alone; (5) the identity behind "merge the attention-side and MLP-side reads" (every weight slice is still needed).
Every GEMM here is an explicit fixed-order fp32 accumulation (32-wide chunks, partitions summed from +0 in order),
the same per output whatever the row count or the column position, so the emulation is row-invariant by
construction; MLX's own kernel orders are Leap's to screen on the GPU."""
import numpy as np, time
t0 = time.time()
np.random.seed(26)
R, HC, D, LR = 16, 4, 2560, 320
K = HC * D
EPS = 1e-6

def bf16(x):
    """round fp32 to bfloat16 (nearest even), returned as fp32."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = x.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)

def ulp(x):
    """the bf16 ulp at x."""
    x = np.abs(np.asarray(x, dtype=np.float32))
    e = np.floor(np.log2(np.maximum(x, np.float32(1e-30))))
    return (np.float32(2.0) ** (e - 7)).astype(np.float32)

def gemm_fixed(A, B, parts, scales=None):
    """fp32 A[R,K] @ B[N,K]^T; K split into parts [(k0, k1, stream)], each partition accumulated in order over
    32-wide chunks, then the partitions summed from +0 in order (MLX's gemm_splitk_accum), optionally each
    scaled by scales[:, stream] first (design 2's epilogue)."""
    A = A.astype(np.float32); B = B.astype(np.float32)
    total = np.zeros((A.shape[0], B.shape[0]), np.float32)
    for (k0, k1, s) in parts:
        acc = np.zeros_like(total)
        for k in range(k0, k1, 32):
            acc = acc + (A[:, None, k:k + 32] * B[None, :, k:k + 32]).sum(-1, dtype=np.float32)
        if scales is not None:
            acc = acc * scales[:, s][:, None]
        total = total + acc
    return total

def silu(x): return (x / (1 + np.exp(-x))).astype(np.float32)
def sigm(x): return (1 / (1 + np.exp(-x))).astype(np.float32)

REF_PARTS = [(0, 4096, 0), (4096, 8192, 1), (8192, K, 2)]                      # MLX's split for M 16 x N 324 x K 10,240
P16_EVEN = [(i * 640, (i + 1) * 640, i) for i in range(16)]                     # 16 even partitions (order change only)
P4_STREAM = [(s * D, (s + 1) * D, s) for s in range(HC)]                        # one partition per stream
P16_STREAM = [(s * D + m * 640, s * D + (m + 1) * 640, s) for s in range(HC) for m in range(4)]

def write(h, block_out, inj):
    return bf16(h + bf16((block_out[:, None, :] * inj[:, :, None]).reshape(h.shape[0], K)))

def norm_scales(hn):
    x = hn.reshape(-1, HC, D).astype(np.float32)
    ss = (x * x).sum(-1, dtype=np.float32)
    return (1.0 / np.sqrt(ss / D + EPS)).astype(np.float32)                    # [rows, 4]

def grouped_norm(hn, w):
    rs = norm_scales(hn)
    y = bf16((hn.reshape(-1, HC, D) * rs[:, :, None]).reshape(-1, K))
    return bf16(y * w[None, :]), rs

def tail(g, normed):
    p = bf16(bf16(sigm(g)) * normed).reshape(-1, HC, D)
    acc = np.zeros((p.shape[0], D), np.float32)
    for s in range(HC): acc = acc + p[:, s, :]
    return bf16(acc / HC)

def ref_read(h, block_out, inj_prev, w, Wd, Wu, Wi, parts=REF_PARTS):
    hn = write(h, block_out, inj_prev)
    normed, rs = grouped_norm(hn, w)
    out = gemm_fixed(normed, np.concatenate([Wd, Wi], 0), parts)
    c, logits = bf16(out[:, :LR]), bf16(out[:, LR:])
    mix1 = bf16(silu(bf16(c / HC)))
    inject = bf16(2.0 * bf16(sigm(bf16(logits / HC))))
    g = bf16(gemm_fixed(mix1, Wu, [(0, LR, 0)]))
    return dict(hn=hn, normed=normed, rs=rs, c=c, logits=logits, mix1=mix1, inject=inject, g=g, mixed=tail(g, normed))

def d1_read(h, block_out, inj_prev, w, Wd, Wu, Wi):
    """design 1: kernel A = write + norm (prologue) + the merged GEMM in MLX's order; kernel B = accum + gate1
    (prologue) + the up GEMM on W_up with stream-interleaved columns + the tail (epilogue)."""
    hn = write(h, block_out, inj_prev); normed, rs = grouped_norm(hn, w)
    out = gemm_fixed(normed, np.concatenate([Wd, Wi], 0), REF_PARTS)
    c, logits = bf16(out[:, :LR]), bf16(out[:, LR:])
    mix1 = bf16(silu(bf16(c / HC))); inject = bf16(2.0 * bf16(sigm(bf16(logits / HC))))
    perm = np.array([s * D + ch for ch in range(D) for s in range(HC)])         # new column ch * 4 + s <- old s * 2560 + ch
    g_perm = bf16(gemm_fixed(mix1, Wu[perm], [(0, LR, 0)]))                     # one 64-column tile = 16 channels x 4 streams
    g4 = g_perm.reshape(-1, D, HC)
    normed4 = normed.reshape(-1, HC, D).transpose(0, 2, 1)
    p = bf16(bf16(sigm(g4)) * normed4)                                          # the epilogue: the tail on its own tile
    acc = np.zeros((p.shape[0], D), np.float32)
    for s in range(HC): acc = acc + p[:, :, s]
    inv = np.argsort(perm)
    return dict(c=c, inject=inject, g=g_perm[:, inv], mixed=bf16(acc / HC))

def d2_read(h, block_out, inj_prev, w, Wd, Wu, Wi, parts, fold_round=True):
    """design 2 (row-invariant): the write kernel also emits the 64 (row, stream) rms scalars; the merged GEMM runs on
    the raw streams with the norm weight folded into the weights (one offline bf16 rounding when fold_round) and the
    per-stream 1 / rms applied per partition in the epilogue; the tail recomputes the reference's normed value."""
    hn = write(h, block_out, inj_prev)
    rs = norm_scales(hn)
    WdI = np.concatenate([Wd, Wi], 0) * w[None, :]
    if fold_round: WdI = bf16(WdI)
    out = gemm_fixed(hn, WdI, parts, scales=rs)
    c, logits = bf16(out[:, :LR]), bf16(out[:, LR:])
    mix1 = bf16(silu(bf16(c / HC))); inject = bf16(2.0 * bf16(sigm(bf16(logits / HC))))
    g = bf16(gemm_fixed(mix1, Wu, [(0, LR, 0)]))
    normed = bf16(bf16((hn.reshape(-1, HC, D) * rs[:, :, None]).reshape(-1, K)) * w[None, :])
    return dict(c=c, inject=inject, g=g, mixed=tail(g, normed))

def quant8_g64(W):
    """MLX-style affine 8-bit, group 64 along K: s = (max - min) / 255, beta = min (both bf16), q = round((w - beta) / s)."""
    Wg = W.reshape(W.shape[0], -1, 64).astype(np.float32)
    mn, mx = Wg.min(-1, keepdims=True), Wg.max(-1, keepdims=True)
    s = bf16((mx - mn) / 255); s = np.where(s == 0, np.float32(1), s); beta = bf16(mn)
    q = np.clip(np.round((Wg - beta) / s), 0, 255)
    return (q * s + beta).reshape(W.shape).astype(np.float32)

def diff(name, a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    nd = np.mean(a.view(np.uint32) != b.view(np.uint32))
    d = np.abs(a - b) / ulp(np.maximum(np.abs(a), np.abs(b)))      # ulp at the larger magnitude (0 where both are 0)
    rel = np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-30)
    print(f"  {name:<40} bits differ {100 * nd:6.2f} %   |delta| median {np.median(d):.2f} / mean {d.mean():.3f} / max {d.max():7.1f} bf16 ulp   within 1 ulp {100 * np.mean(d <= 1):6.2f} %   rel rms {rel:.2e}")
    return rel

# ---- data at the model's shapes (random; the streams at four magnitudes, as residual streams grow with depth)
scale = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
h = bf16((np.random.randn(R, HC, D).astype(np.float32) * scale[None, :, None]).reshape(R, K))
block_out = bf16(np.random.randn(R, D) * 0.5)
inj_prev = bf16(2 * sigm(np.random.randn(R, HC).astype(np.float32)))
w = bf16(1 + 0.1 * np.random.randn(K))
Wd = bf16(np.random.randn(LR, K) / np.sqrt(K))
Wi = bf16(np.random.randn(HC, K) / np.sqrt(K))
Wu = bf16(3 * np.random.randn(K, LR) / np.sqrt(LR))

print("## 1. bytes and FLOPs per read (16 rows) and per check (97 reads: 48 layers x 2 + the final mixer)")
wb = (Wd.size + Wu.size + Wi.size + w.size) * 2
print(f"  weights per read: Wd {Wd.size * 2 / 1e6:.2f} MB + Wu {Wu.size * 2 / 1e6:.2f} + Wi {Wi.size * 2 / 1e6:.3f} + w {w.size * 2 / 1e6:.3f} = {wb / 1e6:.2f} MB bf16; per check {(96 * wb + wb - Wi.size * 2) / 1e9:.3f} GB")
act = dict(h_in=R * K * 2, block_out=R * D * 2, hn_out=R * K * 2, normed_out=R * K * 2, normed_in_gemm_tiles=11 * R * K * 2, partials=3 * R * 324 * 4 * 2, mix1=R * LR * 2 * 2, g_out=R * K * 2, g_in=R * K * 2, normed_in_tail=R * K * 2, mixed_out=R * D * 2)
print("  activation traffic per read (S4 chain), KB: " + ", ".join(f"{k} {v / 1e3:.0f}" for k, v in act.items()) + f"; sum {sum(act.values()) / 1e6:.2f} MB (of it {act['normed_in_gemm_tiles'] / 1e6:.2f} MB are the 11 column tiles' re-reads of normed, cache-level)")
fl = 2 * R * K * (LR + HC) + 2 * R * LR * K
print(f"  FLOPs per read {fl / 1e6:.1f} M (down + inject {2 * R * K * (LR + HC) / 1e6:.1f}, up {2 * R * LR * K / 1e6:.1f}); per check {97 * fl / 1e9:.1f} GFLOP")
for bw in (258, 284, 294): print(f"  byte floor per check at {bw} GB/s: weights {1.281e9 / bw / 1e6:.2f} ms; at 8 bits {0.641e9 / bw / 1e6:.2f} ms")

print("\n## 2. design 1 (two kernels, the same arithmetic) vs the reference chain: must be bitwise")
REF = ref_read(h, block_out, inj_prev, w, Wd, Wu, Wi)
D1 = d1_read(h, block_out, inj_prev, w, Wd, Wu, Wi)
for k in ("c", "inject", "g", "mixed"): diff(f"D1 {k}", REF[k], D1[k])

print("\n## 3. a split-K order change alone (16 even partitions instead of MLX's 3; fp32 partials; same inputs)")
REF16 = ref_read(h, block_out, inj_prev, w, Wd, Wu, Wi, parts=P16_EVEN)
for k in ("c", "inject", "mixed"): diff(f"order-only {k}", REF[k], REF16[k])

print("\n## 4. design 2 (row-invariant read) vs the reference, next to HC8 on the same inputs")
out = {}
for name, parts, fr in (("D2 4 stream partitions, fold rounded", P4_STREAM, True), ("D2 16 partitions, fold rounded", P16_STREAM, True), ("D2 16 partitions, fold in fp32", P16_STREAM, False)):
    X = d2_read(h, block_out, inj_prev, w, Wd, Wu, Wi, parts, fr)
    print(f"  {name}:")
    out[name] = {k: diff(f"   {k}", REF[k], X[k]) for k in ("c", "inject", "g", "mixed")}
H8 = ref_read(h, block_out, inj_prev, w, quant8_g64(Wd), quant8_g64(Wu), quant8_g64(Wi))
print("  HC8 (8-bit g64 affine RTN on Wd, Wu, Wi; the reference arithmetic):")
r8 = {k: diff(f"   {k}", REF[k], H8[k]) for k in ("c", "inject", "g", "mixed")}
r2 = out["D2 16 partitions, fold rounded"]
print(f"  ratio HC8 / D2(16, rounded) of the relative rms deviation: c {r8['c'] / r2['c']:.1f}x, g {r8['g'] / r2['g']:.1f}x, mixed {r8['mixed'] / r2['mixed']:.1f}x, inject {r8['inject'] / max(r2['inject'], 1e-12):.1f}x")

print("\n## 5. row invariance of design 2: each row alone vs inside the 16-row batch (the same fixed-order code)")
X16 = d2_read(h, block_out, inj_prev, w, Wd, Wu, Wi, P16_STREAM, True)
same = True
for r in (0, 7, 15):
    X1 = d2_read(h[r:r + 1], block_out[r:r + 1], inj_prev[r:r + 1], w, Wd, Wu, Wi, P16_STREAM, True)
    same &= all(np.array_equal(X1[k].view(np.uint32), X16[k][r:r + 1].view(np.uint32)) for k in ("c", "inject", "g", "mixed"))
print(f"  rows 0 / 7 / 15 alone == the batch's rows, all outputs bitwise: {same}")

print("\n## 6. 'merge the two reads': (hn + inject (x) o) @ W^T = hn @ W^T + sum_s inject_s (o @ W_s^T) in fp64")
o = np.random.randn(R, D); inj = 2 * sigm(np.random.randn(R, HC).astype(np.float32)).astype(np.float64)
Wd64 = Wd.astype(np.float64); hn64 = REF["hn"].astype(np.float64)
delta = (o[:, None, :] * inj[:, :, None]).reshape(R, K)
lhs = (hn64 + delta) @ Wd64.T - hn64 @ Wd64.T
rhs = sum(inj[:, s][:, None] * (o @ Wd64[:, s * D:(s + 1) * D].T) for s in range(HC))
print(f"  max |lhs - rhs| / max |lhs| = {np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)):.1e}: the identity holds, and the right side needs every one of the 4 weight slices (all 6.6 MB) plus the new rms; nothing is saved")
print(f"\n({time.time() - t0:.1f} s, numpy {np.__version__}, CPU only)")
