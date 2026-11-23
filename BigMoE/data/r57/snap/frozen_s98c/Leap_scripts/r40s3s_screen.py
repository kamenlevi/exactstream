"""R-40 STEP 3s (Leap, pre-registered Thu 2026-10-29 22:20 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3s'): is MLX's conv1d
row-invariant? STEP 3t: REF2's live loop leaves its AR exactly when a commit's lone PLE-layer replay (rows = keep)
leaves layer 1's state (GDN conv / recurrent, PLE conv) different from the AR's 1-row steps; the n-gram ids (slot 3)
stay equal. REF2 pins every matmul / HC read / attention / expert call to 16 rows, but not mx.conv1d: the PLE layer's
depthwise dilated short conv (10240 channels, k 4, dilation 3, 9 state rows) and the GDN layers' short conv (10240, k 4,
dilation 1, 3 state rows) run with 1 new row in the AR, 16 in a check, keep rows in a replay.
Synthetic test on the GPU, MLX 0.32.2 (the BigMoE venv), bf16 inputs at the model's scale: for each conv shape, one
random stream of T rows. AR = T 1-row calls, each with the previous rows as its state window; batched = the same rows
in calls of S rows (S = 2..16). Every output row is compared bitwise with the AR's; counts of differing rows /
elements per S. Plus the explicit form (sum of k shifted multiplies, fixed order) vs the AR, per S.
A measurement; no lines.
Usage: [BigMoE venv] r40s3s_screen.py -> ~/Leap/data/r40s3s_screen.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

D = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
t0 = time.time()
mx.random.seed(40)

def conv_calls(x, w, state_len, dil, S):
    """outputs for rows [state_len:] of x, computed in calls of S new rows, each with its state window"""
    T = x.shape[1] - state_len
    outs = []
    for a in range(0, T, S):
        n = min(S, T - a)
        window = x[:, a:a + state_len + n, :]
        y = mx.conv1d(window, w, stride=1, padding=0, dilation=dil, groups=x.shape[-1])
        outs.append(y[:, -n:, :])
    y = mx.concatenate(outs, axis=1)
    mx.eval(y)
    return y

def explicit(x, w, state_len, dil, S):
    """sum_k w[:, k] * x[t - (K-1-k) * dil], fixed order k = 0..K-1, in fp32, in calls of S rows"""
    K = w.shape[1]
    T = x.shape[1] - state_len
    wk = w[:, :, 0].astype(mx.float32)                           # [C, K]
    outs = []
    for a in range(0, T, S):
        n = min(S, T - a)
        window = x[:, a:a + state_len + n, :].astype(mx.float32)
        acc = None
        for k in range(K):
            off = k * dil
            term = window[:, off:off + n, :] * wk[:, k]
            acc = term if acc is None else acc + term
        outs.append(acc.astype(x.dtype))
    y = mx.concatenate(outs, axis=1)
    mx.eval(y)
    return y

say(f"R-40 STEP 3s: MLX {mx.__version__}, device {mx.default_device()}")
for name, C, K, dil in (("PLE short conv", 10240, 4, 3), ("GDN short conv", 10240, 4, 1)):
    state_len = (K - 1) * dil
    T = 64
    x = (mx.random.normal((1, state_len + T, C)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((C, K, 1)) * 0.3).astype(mx.bfloat16)
    mx.eval(x, w)
    ar = conv_calls(x, w, state_len, dil, 1)
    ex1 = explicit(x, w, state_len, dil, 1)
    say(f"  {name} (C {C}, k {K}, dilation {dil}, state {state_len} rows, {T} rows): explicit 1-row == conv1d 1-row "
        f"rows {int(mx.sum(mx.all(ex1 == ar, axis=-1)).item())} / {T}")
    for S in (2, 3, 4, 5, 6, 8, 10, 12, 15, 16):
        b = conv_calls(x, w, state_len, dil, S)
        eq_el = (mx.view(b, mx.uint16) == mx.view(ar, mx.uint16))
        rows_bad = int(T - mx.sum(mx.all(eq_el, axis=-1)).item())
        el_bad = int(eq_el.size - mx.sum(eq_el).item())
        e = explicit(x, w, state_len, dil, S)
        ex_bad = int(T - mx.sum(mx.all(mx.view(e, mx.uint16) == mx.view(ex1, mx.uint16), axis=-1)).item())
        say(f"    S {S:2d}: conv1d in {S}-row calls vs 1-row calls: {rows_bad} / {T} rows differ ({el_bad} of "
            f"{eq_el.size} elements); explicit form S rows vs 1 row: {ex_bad} rows differ")
say(f"R-40 STEP 3s done in {time.time() - t0:.1f} s")
open(os.path.join(D, "r40s3s_screen.txt"), "w").write("\n".join(lines) + "\n")
