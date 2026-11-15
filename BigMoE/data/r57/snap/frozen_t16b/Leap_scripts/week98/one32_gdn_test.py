"""one32_gdn_test.py (Leap, Sun 15 Nov 2026 11:2x; CHECK55 step 1): one32.gdn_body32 (17-32 GDN rows in one dispatch)
vs gdnbody.gdn_body on rows 0-15 then on the rest (fed chunk 1's conv state and recurrent state), bitwise on every
output: y, q, k, v, the conv state and the fp32 state. Also 9 and 16 rows vs one gdn_body call (one tile).
Random inputs at the 125B's GDN shapes, 3 seeds.
usage: [BigMoE venv] one32_gdn_test.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16b/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_t16b/Leap_scripts/week98"))
import gdnbody as GB
import one32

BF = mx.bfloat16


def say(s):
    print(s, flush=True)


def inputs(S, seed):
    k = mx.random.key(seed)
    ks = mx.random.split(k, 12)
    n = lambda i, shape, sc: (mx.random.normal(shape, key=ks[i]) * sc)
    qkv = n(0, (1, S, 10240), 1.0).astype(BF)
    z = n(1, (1, S, 48, 128), 1.0).astype(BF)
    a = n(2, (1, S, 48), 1.0).astype(BF)
    b = n(3, (1, S, 48), 1.5).astype(BF)
    conv_state = n(4, (1, 3, 10240), 1.0).astype(BF)
    conv_w = n(5, (10240, 4, 1), 0.3).astype(BF)
    A_log = mx.log(mx.random.uniform(1.0, 16.0, (48,), key=ks[6])).astype(BF)
    dt_bias = n(7, (48,), 0.5).astype(BF)
    state = n(8, (1, 48, 128, 128), 0.05).astype(mx.float32)
    norm_w = (1.0 + n(9, (128,), 0.1)).astype(BF)
    scale = mx.array([128 ** -0.5], BF)
    return qkv, z, a, b, conv_state, conv_w, A_log, dt_bias, state, norm_w, 1e-6, scale


def eqall(xs, ys):
    return all(bool(mx.all(x == y).item()) for x, y in zip(xs, ys))


t0 = time.time()
say(f"ONE32-GDN {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; gdnbody NSG {GB.NSG_DEFAULT[0]}, G3 {GB.G3_DEFAULT[0]}")
assert GB.NSG_DEFAULT[0] == 8 and not GB.G3_DEFAULT[0]
ok_all = True
for S in (9, 16, 17, 20, 24, 31, 32):
    res = []
    for seed in (1, 2, 3):
        qkv, z, a, b, cst, cw, alog, dtb, st, nw, eps, sc = inputs(S, seed)
        if S <= 16:
            ref = GB.gdn_body(qkv, z, a, b, cst, cw, alog, dtb, st, nw, eps, sc)
        else:
            r1 = GB.gdn_body(qkv[:, :16], z[:, :16], a[:, :16], b[:, :16], cst, cw, alog, dtb, st, nw, eps, sc)
            r2 = GB.gdn_body(qkv[:, 16:], z[:, 16:], a[:, 16:], b[:, 16:], r1[4], cw, alog, dtb, r1[5], nw, eps, sc)
            ref = tuple(mx.concatenate([u, v], axis=1) for u, v in zip(r1[:4], r2[:4])) + (r2[4], r2[5])
        got = one32.gdn_body32(qkv, z, a, b, cst, cw, alog, dtb, st, nw, eps, sc)
        mx.eval(ref, got)
        names = ("y", "q", "k", "v", "conv", "state")
        bad = [nm for nm, u, v in zip(names, ref, got) if not bool(mx.all(u == v).item())]
        res.append("ok" if not bad else "DIFF " + "/".join(bad))
        ok_all &= not bad
    say(f"  S {S:2d}: " + ", ".join(res))
say(f"ONE32-GDN verdict: {'ALL BITWISE (y, q, k, v, conv state, state)' if ok_all else 'NOT BITWISE'} ({time.time() - t0:.0f} s)")
