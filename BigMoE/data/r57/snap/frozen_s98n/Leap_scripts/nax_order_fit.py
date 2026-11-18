"""N7-7b (Leap, Mon 2026-11-02 21:5x; CPU only): fit the matrix unit's 16 x 32 x 32 accumulation to block floating-point
adder models on the order probe's saved operands / outputs (nax_order_probe.py NOP_SAVE). Model: the K products are taken
in chunks of Kc; in each chunk every addend is aligned to the chunk's largest exponent and kept to F fraction bits below
it (truncated toward zero, or rounded to nearest), the aligned values summed exactly, and the chunk sum rounded to fp32;
C enters either (a) as a separate fp32 add before the chunks, (b) after them, or (c) inside the first chunk's block
(aligned with the products). Reports the bitwise match rate per model, best first.
Usage: [BigMoE venv] nax_order_fit.py <npz> -> ~/Leap/data/nax_order_fit.txt"""
import os, sys, time
import numpy as np

OUT = os.path.expanduser("~/Leap/data/nax_order_fit.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
z = np.load(sys.argv[1])
A32, Bn, C32, D = z["A32"], z["Bn"].astype(np.float64), z["C32"], z["D"]
NT = A32.shape[0]
P = A32.astype(np.float64)[:, :, :, None] * Bn.transpose(0, 2, 1)[:, None, :, :]    # [t][m][k][n], exact
C = C32.astype(np.float64)
Du = D.view(np.uint32)
say(f"N7-7b block-fp fit {time.strftime('%a %H:%M')}; {NT} tests, {D.size} outputs")
def expo(v):
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.abs(v)))
    return np.where(v == 0, -10 ** 6, e)
def align(v, emax, F, mode):
    g = np.exp2(emax - F)
    r = v / g
    r = np.trunc(r) if mode == "trunc" else np.rint(r) if mode == "rne" else np.floor(r)
    return r * g
res = []
f32 = np.float32
for Kc in (4, 8, 16, 32):
    nc = 32 // Kc
    Pc = P.reshape(NT, 16, nc, Kc, 32)
    Ec = expo(Pc).max(axis=3)                                        # [t][m][nc][n]
    for F in range(20, 33):
        for mode in ("trunc", "rne", "floor"):
            S = align(Pc, Ec[:, :, :, None, :], F, mode).sum(axis=3).astype(f32)   # chunk sums, rounded to fp32
            acc = C32.copy()
            for c in range(nc):                                      # (a) C first, then the chunks in order
                acc = (acc + S[:, :, c, :]).astype(f32)
            res.append(((acc.view(np.uint32) == Du).mean(), f"Kc {Kc:2d} F {F} {mode:5s} C first"))
            acc = S[:, :, 0, :].astype(f32)                          # (b) the chunks, then C
            for c in range(1, nc):
                acc = (acc + S[:, :, c, :]).astype(f32)
            acc = (acc + C32).astype(f32)
            res.append(((acc.view(np.uint32) == Du).mean(), f"Kc {Kc:2d} F {F} {mode:5s} C last"))
            # (c) C inside the first chunk's block
            E0 = np.maximum(Ec[:, :, 0, :], expo(C))
            b0 = align(Pc[:, :, 0], E0[:, :, None, :], F, mode).sum(axis=2) + align(C, E0, F, mode)
            acc = b0.astype(f32)
            for c in range(1, nc):
                acc = (acc + S[:, :, c, :]).astype(f32)
            res.append(((acc.view(np.uint32) == Du).mean(), f"Kc {Kc:2d} F {F} {mode:5s} C in block 0"))
# one block of all 32 products + C
Pall = P
Eall = np.maximum(expo(Pall).max(axis=2), expo(C))
for F in range(20, 33):
    for mode in ("trunc", "rne", "floor"):
        s_ = align(Pall, Eall[:, :, None, :], F, mode).sum(axis=2) + align(C, Eall, F, mode)
        res.append(((s_.astype(f32).view(np.uint32) == Du).mean(), f"Kc 32 F {F} {mode:5s} C + all in one block"))
res.sort(reverse=True)
for r, name in res[:15]:
    say(f"  {100 * r:7.3f} %  {name}")
