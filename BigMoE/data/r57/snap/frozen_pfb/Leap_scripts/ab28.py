"""Fair A/B (Leap, Sat 2026-10-24): L1-26's dense gain came from native's best-of-5 configurations vs qmm16c's one, and
did not reproduce at equal configurations in L1-27's run. Here every kernel runs at the SAME (ks, nt), interleaved in a
shuffled order, 12 rounds x median of 7; report each kernel's median over rounds and the per-round paired ratio
(median and IQR). Dense: qmm16c vs qmm16n (native layout) vs qmm16i (int4 operands) on 3 shapes; experts: experts_nax
vs experts_native vs experts_int4 at 16 rows on real weights of 4 layers.
Usage: [BigMoE venv] ab28.py -> ~/Leap/data/ab28.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
from qmm16 import qmm16c
from qmm26 import repack, qmm16n, experts_native
from qmmg25 import experts_nax
from qmm27 import qmm16i, experts_int4

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def t_once(fn, inner=7):
    mx.eval(fn())
    ts = []
    for _ in range(inner):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[inner // 2] * 1e3

def ab(fns, rounds=12, seed=0):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in fns}
    for _ in range(rounds):
        for k in rng.permutation(list(fns)):
            res[k].append(t_once(fns[k]))
    return {k: np.array(v) for k, v in res.items()}

def report(name, res, floor, base):
    cells = []
    for k, v in res.items():
        r = res[base] / v                                           # >1: faster than the base
        cells.append(f"{k} {np.median(v):.3f} ms ({floor / np.median(v):.0%} of floor; x{np.median(r):.2f} vs {base}, "
                     f"IQR {np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f})")
    say(f"  {name}: " + "; ".join(cells))

mx.random.seed(28)
say("Fair A/B at equal (ks, nt), interleaved, 12 rounds x median of 7")
for name, N, K, cnt, cfg in (("in_proj_qkv", 10240, 2560, 36, (2, 2)), ("o_proj+out_proj", 2560, 6144, 48, (2, 2)),
                             ("in_proj_z", 6144, 2560, 36, (1, 1))):
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(cnt)]
    Wn = [repack(*W) for W in Ws]
    mx.eval(Ws, Wn)
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    floor = cnt * sum(a.nbytes for a in Ws[0]) / 280e9 * 1e3
    res = ab({"qmm16c": lambda: [qmm16c(x, *W, ks=cfg[0], nt=cfg[1]) for W in Ws],
              "native": lambda: [qmm16n(x, *W, N, *cfg) for W in Wn],
              "int4": lambda: [qmm16i(x, *W) for W in Ws]})
    report(f"{name} {N}x{K} x{cnt} at {cfg}", res, floor, "qmm16c")
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
W = [load_layer(L) for L in LAYERS]
WN = [(repack(*gu), repack(*dn)) for gu, dn in W]
mx.eval(WN)
EB = sum(a[0].nbytes for a in W[0][0]) + sum(a[0].nbytes for a in W[0][1])
rng = np.random.default_rng(28)
def routing(rows, k=10):
    U = min(512, round(10 * rows ** 0.72))
    pool = rng.choice(512, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))
x = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
for rows in (8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    floor = U * EB / 280e9 * 1e3 * len(LAYERS)
    res = ab({"nax": lambda: [experts_nax(x[:rows], *W[i], R[i], (4, 1), (1, 2)) for i in range(len(LAYERS))],
              "native": lambda: [experts_native(x[:rows], *WN[i], R[i], (4, 1), (1, 2)) for i in range(len(LAYERS))],
              "int4": lambda: [experts_int4(x[:rows], *W[i], R[i]) for i in range(len(LAYERS))]})
    report(f"experts {rows} rows ({U:.0f} distinct, 4 layers)", res, floor, "nax")
open(os.path.expanduser("~/Leap/data/ab28.txt"), "w").write("\n".join(lines) + "\n")
