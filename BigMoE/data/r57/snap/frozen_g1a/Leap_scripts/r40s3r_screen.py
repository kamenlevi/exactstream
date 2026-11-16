"""R-40 STEP 3r (Leap, pre-registered Thu 2026-10-29 22:25 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3r'): which reduction in
the PLE layer depends on the number of rows in the call? STEP 3t: the commit's lone PLE-layer replay (S = keep rows)
leaves a different PLE conv state than the AR's 1-row steps; 3s: conv1d is row-invariant. The PLE output's other
row-local steps that REF2 does not pin to 16 rows are reductions: the gate's (key * query).sum(-1) over 2560 per
(row, hc), GroupedRMSNorm (mx.fast.rms_norm per 2560-group), and the HC read's mean over the 4 streams.
Synthetic bf16 tensors on the model's shapes, MLX 0.32.2 GPU: for S = 1..16 each op on S rows at once vs the same rows
one at a time; rows differing bitwise per S. A measurement; no lines.
Usage: [BigMoE venv] r40s3r_screen.py -> ~/Leap/data/r40s3r_screen.txt"""
import math, os, sys, time
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
t0 = time.time()
mx.random.seed(41)
H, HC, T = 2560, 4, 48

def per_row(f, xs):
    return mx.concatenate([f(*[x[i:i + 1] for x in xs]) for i in range(xs[0].shape[0])], axis=0)

def batched(f, xs, S):
    return mx.concatenate([f(*[x[a:a + S] for x in xs]) for a in range(0, xs[0].shape[0], S)], axis=0)

def bad_rows(a, b):
    eq = mx.view(a.reshape(a.shape[0], -1), mx.uint16) == mx.view(b.reshape(b.shape[0], -1), mx.uint16) \
        if a.dtype in (mx.bfloat16, mx.float16) else (a.reshape(a.shape[0], -1) == b.reshape(b.shape[0], -1))
    return int(a.shape[0] - mx.sum(mx.all(eq, axis=-1)).item())

key = (mx.random.normal((T, HC, H)) * 0.7).astype(mx.bfloat16)
query = (mx.random.normal((T, HC, H)) * 0.7).astype(mx.bfloat16)
x10 = (mx.random.normal((T, HC * H)) * 1.3).astype(mx.bfloat16)
mix = mx.sigmoid(mx.random.normal((T, HC, H))).astype(mx.bfloat16)
mx.eval(key, query, x10, mix)
ops = {
    "PLE gate (key*query).sum(-1)/sqrt(H)": (lambda k, q: (k * q).sum(axis=-1, keepdims=True) / math.sqrt(H), [key, query]),
    "PLE gate full (sqrt|g| * sign)": (lambda k, q: (lambda g: mx.sqrt(mx.maximum(mx.abs(g), 1e-6)) * mx.sign(g))(
        (k * q).sum(axis=-1, keepdims=True) / math.sqrt(H)), [key, query]),
    "GroupedRMSNorm (rms_norm per 2560)": (lambda x: mx.fast.rms_norm(x.reshape(x.shape[0], HC, H), None, 1e-6)
                                           .reshape(x.shape[0], HC * H), [x10]),
    "HC mean over 4 streams": (lambda m, g: mx.mean(m * g, axis=-2), [mix, key]),
    "sum over 2560 in fp32": (lambda k, q: (k.astype(mx.float32) * q.astype(mx.float32)).sum(axis=-1), [key, query]),
}
say(f"R-40 STEP 3r: MLX {mx.__version__}, {mx.default_device()}; {T} rows, hc {HC}, H {H}")
for name, (f, xs) in ops.items():
    ref = per_row(f, xs)
    mx.eval(ref)
    res = []
    for S in (2, 3, 4, 5, 6, 8, 10, 12, 15, 16):
        b = batched(f, xs, S)
        mx.eval(b)
        res.append(f"S{S}:{bad_rows(b, ref)}")
    say(f"  {name}: rows differing from 1-row calls  " + "  ".join(res))
say(f"R-40 STEP 3r done in {time.time() - t0:.1f} s")
open(os.path.join(D, "r40s3r_screen.txt"), "w").write("\n".join(lines) + "\n")
