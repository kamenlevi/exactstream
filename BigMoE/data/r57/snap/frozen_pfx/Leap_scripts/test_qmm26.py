"""L1-26 test (Leap, Sat 2026-10-24): fragment-native 4-bit layout (qmm26.py) vs qmm16c / experts_nax.
1. Dense shapes (random weights, per-check counts, cold): bitwise vs qmm16c at 16 rows; ms per check over (ks, nt).
2. Routed experts (real 4-bit weights of layers 12 / 20 / 28 / 36, repacked): bitwise vs experts_nax at 4 / 8 / 16 rows;
   ms per layer vs experts_nax and the floor.
Usage: [BigMoE venv] test_qmm26.py -> ~/Leap/data/qmm26.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
from qmm16 import qmm16c
from qmmg25 import experts_nax
from qmm26 import repack, qmm16n, experts_native

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

mx.random.seed(26)
say("L1-26: fragment-native 4-bit layout vs qmm16c (dense) and experts_nax (routed experts)")
CFG = [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)]
for name, N, K, cnt, best_c in (("in_proj_qkv", 10240, 2560, 36, (2, 2)), ("o_proj+out_proj", 2560, 6144, 48, (2, 2)),
                                ("in_proj_z", 6144, 2560, 36, (1, 1))):
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(cnt)]
    mx.eval(Ws)
    Wn = [repack(*W) for W in Ws]
    mx.eval(Wn)
    X = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    floor = cnt * sum(a.nbytes for a in Ws[0]) / 280e9 * 1e3
    eq = np.array_equal(bits(qmm16c(X, *Ws[0], ks=best_c[0], nt=best_c[1])), bits(qmm16n(X, *Wn[0], N, *best_c)))
    t_c = timeit(lambda: [qmm16c(X, *W, ks=best_c[0], nt=best_c[1]) for W in Ws])
    cells = []
    for ks, nt in CFG:
        if N % (32 * nt) or K % (64 * ks):
            continue
        cells.append((timeit(lambda: [qmm16n(X, *W, N, ks, nt) for W in Wn]), (ks, nt)))
    cells.sort()
    say(f"  {name:16s} {N}x{K} x{cnt}: floor {floor:.2f} ms; qmm16c {best_c} {t_c:.2f} ({floor / t_c:.0%}); native best "
        f"{cells[0][0]:.2f} ({floor / cells[0][0]:.0%}) at {cells[0][1]}; bitwise = qmm16c at {best_c}: {eq}")
    del Ws, Wn

MD = os.path.expanduser("~/BigMoE/models/bare")
WMAP = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def load_layer(L):
    p = f"language_model.model.layers.{L}.mlp.switch_mlp."
    keys = [p + f"{n}.{s}" for n in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "scales", "biases")]
    got = {}
    for f in sorted({WMAP[k] for k in keys}):
        d = mx.load(os.path.join(MD, f))
        got.update({k: d[k] for k in keys if k in d})
        del d
    gu = [mx.concatenate([got[p + f"gate_proj.{s}"], got[p + f"up_proj.{s}"]], axis=1) for s in ("weight", "scales", "biases")]
    dn = [got[p + f"down_proj.{s}"] for s in ("weight", "scales", "biases")]
    mx.eval(gu, dn)
    return gu, dn
LAYERS = (12, 20, 28, 36)
t0 = time.time()
W = [load_layer(L) for L in LAYERS]
WN = [(repack(*gu), repack(*dn)) for gu, dn in W]
mx.eval(WN)
say(f"  experts: loaded + repacked {len(LAYERS)} layers in {time.time() - t0:.0f} s")
E = 512
EB = sum(a[0].nbytes for a in W[0][0]) + sum(a[0].nbytes for a in W[0][1])
rng = np.random.default_rng(26)
def routing(rows, k=10):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))
X = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
for rows in (4, 8, 16):
    R = [routing(rows) for _ in LAYERS]
    eq = all(np.array_equal(bits(experts_nax(X[:rows], *W[i], R[i], (4, 1), (1, 2))),
                            bits(experts_native(X[:rows], *WN[i], R[i], (4, 1), (1, 2)))) for i in range(len(LAYERS)))
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    floor = U * EB / 280e9 * 1e3
    t_n = timeit(lambda: [experts_nax(X[:rows], *W[i], R[i], (4, 1), (1, 2)) for i in range(len(LAYERS))]) / len(LAYERS)
    cells = []
    for ca in ((4, 1), (2, 1), (1, 1), (2, 2)):
        for cb in ((1, 2), (1, 1), (2, 1)):
            cells.append((timeit(lambda: [experts_native(X[:rows], *WN[i], R[i], ca, cb) for i in range(len(LAYERS))])
                          / len(LAYERS), ca, cb))
    cells.sort()
    say(f"  {rows:2d} rows ({U:.0f} distinct experts): floor {floor:.3f} ms per layer; experts_nax {t_n:.3f} ({floor / t_n:.0%}); "
        f"native best {cells[0][0]:.3f} ({floor / cells[0][0]:.0%}) at {cells[0][1]} {cells[0][2]}; bitwise = experts_nax "
        f"(4,1)(1,2): {eq}; per 48-layer check: nax {48 * t_n:.1f}, native {48 * cells[0][0]:.1f} ms")
open(os.path.expanduser("~/Leap/data/qmm26.txt"), "w").write("\n".join(lines) + "\n")
