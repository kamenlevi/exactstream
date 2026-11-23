"""one32_hc_test.py (Leap, Sun 15 Nov 2026 11:3x; CHECK55 step 1): one32.hc_read32b (the HC read at 17-32 rows, one
dispatch per kernel) vs leapk.hc_read (fuse_gate, the exact path's 16-row read) on rows 0-15 and on rows 16-n
zero-padded to 16, bitwise on mixed / written / inject, with and without the pending carry. Random weights at the
125B's HC shapes (4 streams x 2560, low rank 320, block inject 4), rows 17 / 20 / 24 / 31 / 32, 3 seeds.
usage: [BigMoE venv] one32_hc_test.py   (MEM < 1 GB; share lock)"""
import os, sys, time
from types import SimpleNamespace as NS
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts/week98"))
import leapk as LK
import one32

BF = mx.bfloat16
HC, D, LR = 4, 2560, 320


def say(s):
    print(s, flush=True)


def pad(x, n):
    m = x.shape[1]
    return x if m == n else mx.concatenate([x, mx.zeros((1, n - m) + x.shape[2:], x.dtype)], axis=1)


t0 = time.time()
say(f"ONE32-HC {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}")
mx.random.seed(7)
g = NS(hc_count=HC, hidden_size=D, hc_norm=NS(weight=(1.0 + mx.random.normal((HC * D,)) * 0.1).astype(BF), eps=1e-6),
       input_mix_weight_down=NS(weight=(mx.random.normal((LR, HC * D)) * 0.01).astype(BF)),
       input_mix_weight_up=NS(weight=(mx.random.normal((HC * D, LR)) * 0.05).astype(BF)))
binj = (mx.random.normal((HC, HC * D)) * 0.01).astype(BF)
wcat = mx.concatenate([g.input_mix_weight_down.weight, binj], axis=0)
mx.eval(wcat)
ok_all = True
for M in (17, 20, 24, 31, 32):
    res = []
    for seed in (1, 2, 3):
        for with_p in (False, True):
            ks = mx.random.split(mx.random.key(100 * M + seed), 3)
            h = (mx.random.normal((1, M, HC * D), key=ks[0]) * 1.0).astype(BF)
            p = ((mx.random.normal((1, M, D), key=ks[1]) * 0.5).astype(BF),
                 (mx.random.normal((1, M, HC), key=ks[2]) * 0.5).astype(BF)) if with_p else None
            r1 = LK.hc_read(g, wcat, h[:, :16], None if p is None else tuple(t[:, :16] for t in p), rows=16, fuse_gate=True)
            r2 = LK.hc_read(g, wcat, pad(h[:, 16:], 16), None if p is None else tuple(pad(t[:, 16:], 16) for t in p),
                            rows=16, fuse_gate=True)
            ref = [mx.concatenate([u, v[:, :M - 16]], axis=1) for u, v in zip(r1, r2)]
            bad = []
            for form in ("2", "2i"):
                one32.S4_FORM[0] = form
                got = one32.hc_read32b(g, wcat, pad(h, 32), None if p is None else tuple(pad(t, 32) for t in p))
                got = [x[:, :M] for x in got]
                mx.eval(ref, got)
                bad += [f"{nm}({form})" for nm, u, v in zip(("mixed", "written", "inject"), ref, got)
                        if not bool(mx.all(u == v).item())]
            one32.S4_FORM[0] = "2"
            res.append(("p" if with_p else "-") + ("ok" if not bad else "DIFF " + "/".join(bad)))
            ok_all &= not bad
    say(f"  M {M:2d}: " + ", ".join(res))
say(f"ONE32-HC verdict: {'ALL BITWISE (mixed, written, inject)' if ok_all else 'NOT BITWISE'} ({time.time() - t0:.0f} s)")
