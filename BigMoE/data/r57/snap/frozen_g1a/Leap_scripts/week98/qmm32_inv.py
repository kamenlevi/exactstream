"""QMM32-INV (CHECK55 step 1; Leap, Sun 15 Nov 2026 11:1x): for which of the exact-45 engine's quantized matmul shapes
is MLX's own call at 17-32 rows bitwise, per row, with v32's two calls (rows 0-15, rows 16-n)? Where it is, the
one-pass 32-row check calls MLX once (one dispatch instead of two, no new kernel); where not, it keeps the split or
needs a replica. C2 (d32_1115.txt) answered this for the model's separate projections. This adds the engine's
concatenated packs (GDN qkv|z 16384 and b|a 96 (ref2 R-83), K-73's shared gate|up|gate 1281) and the 8-bit router,
at row counts 17 / 20 / 24 / 31 / 32, 3 seeds each, random weights quantized by MLX (affine, group 64).
Line: a shape is ONE-CALL-OK iff every row is bitwise at every row count and seed.
usage: [BigMoE venv] qmm32_inv.py   (MEM ~3 GB with the head; share lock)"""
import time
import numpy as np
import mlx.core as mx

R = 16
SHAPES = [  # name, N, K, bits
    ("gdn qkv|z (R-83 qz)", 16384, 2560, 4), ("gdn b|a (R-83 ba)", 96, 2560, 4), ("gdn out_proj", 2560, 6144, 4),
    ("att q_proj", 12288, 2560, 4), ("att k / v_proj", 512, 2560, 4), ("att o_proj", 2560, 6144, 4),
    ("indexer qk", 640, 2560, 4), ("router (8-bit)", 512, 2560, 8), ("shared gate|up|gate (K-73)", 1281, 2560, 4),
    ("shared down", 2560, 640, 4), ("ple key", 10240, 2560, 4), ("ple value", 2560, 2560, 4),
    ("lm_head", 248320, 2560, 4)]
ROWS = (17, 20, 24, 31, 32)
SEEDS = (1, 2, 3)


def say(s):
    print(s, flush=True)


t0 = time.time()
say(f"QMM32-INV {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}")
ok_list = []
for name, N, K, bits in SHAPES:
    mx.random.seed(N + K + bits)
    w = (mx.random.normal((N, K), dtype=mx.bfloat16) * 0.02)
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    del w
    mx.eval(wq, s, b)
    q = lambda x: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
    per = {}                                               # (form, M) -> rows bitwise over the seeds
    for sd in SEEDS:
        x = (mx.random.normal((32, K), dtype=mx.bfloat16, key=mx.random.key(sd)) * 0.7)
        for M in ROWS:
            ref = mx.concatenate([q(x[:R]), q(x[R:M])], axis=0)
            xp = x[:M] if M == 32 else mx.concatenate([x[:M], mx.zeros((32 - M, K), x.dtype)], axis=0)
            for form, one in (("raw", q(x[:M])), ("pad32", q(xp)[:M])):
                eq = mx.all(one == ref, axis=-1)
                mx.eval(eq)
                e = np.array(eq)
                g = per.setdefault((form, M), [0, 0, set()])
                g[0] += int(e.sum()); g[1] += M; g[2].update(int(i) for i in np.nonzero(~e)[0])
    for form in ("raw", "pad32"):
        good = sum(per[(form, M)][0] for M in ROWS); tot = sum(per[(form, M)][1] for M in ROWS)
        detail = ", ".join(f"M{M} {per[(form, M)][0]}/{per[(form, M)][1]}"
                           + (f" bad rows {sorted(per[(form, M)][2])[:6]}" if per[(form, M)][2] else "") for M in ROWS)
        ok = good == tot
        if ok:
            ok_list.append((form, N, K, bits))
        say(f"  {name:28s} {form:5s} N {N:6d} K {K:5d} {bits}-bit: {good:4d} / {tot:4d} -> {'ONE-CALL-OK' if ok else 'no'} ({detail})")
    del wq, s, b
say(f"QMM32-INV one-call whitelist (N, K, bits): {ok_list} ({time.time() - t0:.0f} s)")
