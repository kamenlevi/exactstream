"""DSDPA-1 check (Leap, Wed 2026-11-04 17:5x; TABLE_350 'DSDPA'): scripts/dsdpa.py vs the drafter's MLX path (concat K / V,
bool mask, SDPA, transpose + reshape -> [1, 16, 4096]).
 (a) accuracy at ctx_len 1 .. 2,062 (q / K / V ~ N(0, s^2), s 1 and 3), plus an adversarial case (the cache's oldest
     rows aligned with q at 10x: they win the softmax if the window mask leaks): max |diff| in output ulps (bf16 ulp at
     max |ref|), NaN check;
 (b) ms per 5 layers (distinct q / K / V), interleaved, median of 21: mlx16 (as draft_fused), mlx16add, dsdpa nc 4 / 8 / 16,
     at ctx_len 1,008 and 2,054.
Pre-registered: (a) every case <= 4 ulps of max |ref| and no NaN, adversarial case the same; (b) dsdpa (best nc) <= 1/2 of
mlx16 at ctx 2,054 -> go to the walk (acceptance) + propose timing; else KILL.
usage: [BigMoE venv] dsdpa_check.py -> ~/Leap/data/dsdpa_check.txt (GPU, MEM ~0.3 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
import dsdpa as DS

OUT = os.path.expanduser("~/Leap/data/dsdpa_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nDSDPA-1 check {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; (a) checks attend's default (v2 since 17:4x)")
HQ, HK, D, L, SW, NL = 32, 8, 128, 16, 2048, 5
scale = D ** -0.5


def mask_for(ctx_len):
    q = ctx_len + mx.arange(L)[:, None]
    k = mx.arange(ctx_len + L)[None]
    return ((k < ctx_len) & (q - k < SW)) | (k >= ctx_len)


def ref(q, kc, vc, kp, vp, ctx_len, m=None):
    k = mx.concatenate([kc[:, :, :ctx_len], kp], axis=2)
    v = mx.concatenate([vc[:, :, :ctx_len], vp], axis=2)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_for(ctx_len) if m is None else m)
    return o.transpose(0, 2, 1, 3).reshape(1, L, HQ * D)


def inputs(ctx_len, s, seed):
    mx.random.seed(seed)
    q = (mx.random.normal((1, HQ, L, D)) * s).astype(mx.bfloat16)
    kc = (mx.random.normal((1, HK, ctx_len, D)) * s).astype(mx.bfloat16)
    vc = mx.random.normal((1, HK, ctx_len, D)).astype(mx.bfloat16)
    kp = (mx.random.normal((1, HK, L, D)) * s).astype(mx.bfloat16)
    vp = mx.random.normal((1, HK, L, D)).astype(mx.bfloat16)
    return q, kc, vc, kp, vp


def ulps(o, r):
    o32, r32 = np.array(o.astype(mx.float32)), np.array(r.astype(mx.float32))
    if not np.all(np.isfinite(o32)):
        return float("inf")
    top = float(np.abs(r32).max())
    ulp = 2.0 ** (np.floor(np.log2(top)) - 7)
    return float(np.abs(o32 - r32).max() / ulp)


worst, rows = 0.0, []
for ctx_len in (1, 5, 31, 32, 33, 100, 1008, 2031, 2046, 2047, 2054, 2062):
    for s in (1.0, 3.0):
        a = inputs(ctx_len, s, ctx_len * 7 + int(s))
        o, r = DS.attend(*a, ctx_len), ref(*a, ctx_len)
        mx.eval(o, r)
        u = ulps(o, r)
        worst = max(worst, u)
        rows.append(f"{ctx_len}/{s:g}: {u:.2f}")
say("(a) max |diff| in ulps of max |ref| (ctx_len / s): " + ", ".join(rows))
# adversarial: the cache's oldest rows (outside every query's window at ctx_len 2,062) aligned with q at 10x
ctx_len = 2062
q, kc, vc, kp, vp = inputs(ctx_len, 1.0, 99)
qm = q.reshape(1, HK, 4, L, D).mean(axis=(2, 3))                    # per kv head
kc = mx.concatenate([mx.broadcast_to((qm * 10)[:, :, None], (1, HK, 30, D)).astype(mx.bfloat16), kc[:, :, 30:]], axis=2)
vc = mx.concatenate([mx.full((1, HK, 30, D), 50.0, mx.bfloat16), vc[:, :, 30:]], axis=2)
o, r = DS.attend(q, kc, vc, kp, vp, ctx_len), ref(q, kc, vc, kp, vp, ctx_len)
leak = ref(q, kc, vc, kp, vp, ctx_len, m=mx.ones((L, ctx_len + L), mx.bool_))
mx.eval(o, r, leak)
ua, ul = ulps(o, r), ulps(leak, r)
ok_a = worst <= 4.0 and ua <= 4.0
say(f"(a') adversarial window rows (10x-aligned keys, V = 50 in rows 0-29 at ctx_len 2,062): dsdpa {ua:.2f} ulps; a leaking "
    f"mask would be {ul:.0f} ulps off -> (a) {'PASS' if ok_a else 'FAIL'} (worst {max(worst, ua):.2f} ulps)")

for ctx_len in (1008, 2054):
    mx.random.seed(ctx_len)
    X = [inputs(ctx_len, 1.0, 1000 + i) for i in range(NL)]
    m = mask_for(ctx_len)
    madd = mx.where(m, mx.array(0.0, mx.bfloat16), mx.array(-float("inf"), mx.bfloat16))
    mx.eval(X, m, madd)
    sd = mx.fast.scaled_dot_product_attention

    def mlx(mask):
        return lambda: [sd(q, mx.concatenate([kc, kp], axis=2), mx.concatenate([vc, vp], axis=2), scale=scale,
                           mask=mask).transpose(0, 2, 1, 3).reshape(1, L, HQ * D) for q, kc, vc, kp, vp in X]

    A = {"mlx16": mlx(m), "mlx16add": mlx(madd)}
    for v, nc in ((1, 4), (2, 2), (2, 4), (2, 8), (2, 16)):
        A[f"dsdpa{'v1' if v == 1 else ''}{nc}"] = (lambda nc_, v_: lambda: [DS.attend(q, kc, vc, kp, vp, ctx_len, nc=nc_, v=v_)
                                                                        for q, kc, vc, kp, vp in X])(nc, v)
    for f in A.values():
        mx.eval(f())
    T = {k: [] for k in A}
    for rnd in range(21):
        for k in (list(A) if rnd % 2 == 0 else list(A)[::-1]):
            mx.synchronize(); t0 = time.perf_counter()
            for _ in range(4):
                mx.eval(A[k]())
            mx.synchronize()
            T[k].append((time.perf_counter() - t0) / 4)
    med = {k: float(np.median(v)) for k, v in T.items()}
    best = min((k for k in med if k.startswith("dsdpa") and "v1" not in k), key=med.get)
    say(f"(b) ctx_len {ctx_len}: ms per 5 layers (median of 21): " + ", ".join(f"{k} {1e3 * v:.3f}" for k, v in med.items())
        + f"; best {best} = x{med[best] / med['mlx16']:.2f} of mlx16")
