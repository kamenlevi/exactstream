"""BP-11b (Leap, Wed 2026-11-04 17:2x): qmm16c8's in-kernel row bound (M_C) vs the padded call it replaces: rows < M
bitwise equal on the drafter's shapes at M 8 / 1 / 16. -> ~/Leap/data/qmm16c8_check.txt (appends)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from qmm16c8 import qmm16c8
mx.random.seed(6)
ok = True
for N, K in ((4096, 2560), (1024, 2560), (2560, 4096), (7680, 2560), (2560, 7680), (2560, 12800)):
    W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    w8, s8, b8 = mx.quantize(W, group_size=64, bits=8)
    for M in (8, 1, 16):
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
        x16 = x if M == 16 else mx.concatenate([x, mx.zeros((16 - M, K), x.dtype)])
        for ks in (1, 2):
            if K % (64 * ks):
                continue
            a = qmm16c8(x, w8, s8, b8, ks=ks, row_bound=True)
            b = qmm16c8(x16, w8, s8, b8, ks=ks)[:M]
            c_ = qmm16c8(x, w8, s8, b8, ks=ks)
            mx.eval(a, b, c_)
            ok &= bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item()) and bool(mx.all(c_.view(mx.uint16) == b.view(mx.uint16)).item())
line = f"BP-11b {time.strftime('%a %H:%M')}: qmm16c8 default (padded, M_C 16) and row_bound=True == the explicitly padded call's rows, bitwise, 6 shapes x M 8 / 1 / 16 x ks 1 / 2: {ok}"
print(line); open(os.path.expanduser("~/Leap/data/qmm16c8_check.txt"), "a").write(line + "\n")
