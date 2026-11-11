"""R-40 STEP 3o (Leap, pre-registered Thu 2026-10-29 22:30 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3o'): does MLX's fused vector
SDPA give a row different bits when the call's total key length N changes? STEP 3p: on algebra/694 a kept row at
position p >= 112 inside a check that started before 112 left REF2's AR at a QSA layer's attention output (layer ~35),
everything before it equal. REF2's AR pads attention to 16 rows, so its key length is p + 16; the check's is start + 16
for all of its rows: they straddle 128 exactly there. MTPLX's _verify_sdpa splits heads so every 16-row call stays on
mx.fast.scaled_dot_product_attention's vector kernel (q_len x heads per call <= 32).
Synthetic bf16 test on the model's attention shape (head_dim 256, GQA: 2 query heads per call on 1 KV head, as the
head-chunked verify path; 16 query rows), MLX 0.32.2 GPU: one query block at fixed positions with fixed keys; the
same rows computed in calls whose key length N runs from (last row + 1) to (last row + 40), the extra keys masked
(boolean causal mask, as MTPLX passes). Per N: rows whose output differs bitwise from the smallest-N call's. Then
the same with the keys padded to a multiple of 128 / 256 (the candidate fix: one N per bucket).
A measurement; no lines.
Usage: [BigMoE venv] r40s3o_screen.py -> ~/Leap/data/r40s3o_screen.txt"""
import os, time
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
t0 = time.time()
mx.random.seed(42)
HD, HQ, S = 256, 2, 16
scale = HD ** -0.5

def attend(q, kf, vf, start, N):
    """the S query rows at positions start..start+S-1 against the first N keys (causal by position)"""
    k, v = kf[:, :, :N, :], vf[:, :, :N, :]
    qpos = mx.arange(start, start + S)
    tpos = mx.arange(N)
    mask = (tpos[None, :] <= qpos[:, None])[None, None]
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(y)
    return y

def rows_bad(a, b):
    eq = mx.view(a, mx.uint16) == mx.view(b, mx.uint16)            # [1, HQ, S, HD]
    return [int(j) for j in range(S) if not bool(mx.all(eq[:, :, j, :]).item())]

say(f"R-40 STEP 3o: MLX {mx.__version__}, {mx.default_device()}; head_dim {HD}, {HQ} q heads / 1 kv head, {S} query rows")
for start in (90, 104, 112, 120, 240, 500):
    Tmax = start + S + 64
    kf = (mx.random.normal((1, 1, Tmax, HD)) * 0.8).astype(mx.bfloat16)
    vf = mx.random.normal((1, 1, Tmax, HD)).astype(mx.bfloat16)
    q = (mx.random.normal((1, HQ, S, HD)) * 0.8).astype(mx.bfloat16)
    mx.eval(kf, vf, q)
    base_N = start + S
    base = attend(q, kf, vf, start, base_N)
    changes = []
    for N in range(base_N + 1, base_N + 41):
        bad = rows_bad(attend(q, kf, vf, start, N), base)
        if bad:
            changes.append((N, len(bad)))
    say(f"  rows at {start}..{start + S - 1}: key length N from {base_N} to {base_N + 40}: N values where any row differs "
        f"from N = {base_N}: {[n for n, _ in changes][:12]}{' ...' if len(changes) > 12 else ''} ({len(changes)} of 40)")
    for bucket in (128, 256):
        pad = lambda n: -(-n // bucket) * bucket
        if pad(base_N + 40) > Tmax:
            continue
        b0 = attend(q, kf, vf, start, pad(base_N))
        diff_b = [N for N in range(base_N + 1, base_N + 41) if rows_bad(attend(q, kf, vf, start, pad(N)), b0)]
        say(f"    keys padded to a multiple of {bucket}: N values that differ from N = {base_N}'s: {diff_b[:12]} "
            f"({len(diff_b)} of 40; bucket changes at {[N for N in range(base_N + 1, base_N + 41) if pad(N) != pad(N - 1)]})")
say(f"R-40 STEP 3o done in {time.time() - t0:.1f} s")
open(os.path.join(D, "r40s3o_screen.txt"), "w").write("\n".join(lines) + "\n")
