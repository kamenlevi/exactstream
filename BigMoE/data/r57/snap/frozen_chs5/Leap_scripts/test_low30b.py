"""L1-30b test (Leap, Sun 2026-10-25): (1) bitwise: low30b MODE 0 PAIRK 0 == low30, PAIRK 1 == PAIRK 0, at 4 / 3 / 2
bits; (2) profile at 16 rows, 2-bit and 4-bit, MODE 0-4 interleaved (12 rounds x median of 7); (3) fair A/B PAIRK 0 vs
1 at 4 / 3 / 2 bits, 8 and 16 rows. Real codes of 4 layers (first 128 experts), routing as L1-30, L1-30's configs.
Usage: [BigMoE venv] test_low30b.py [smoke] -> ~/Leap/data/low30b.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts"))
from low30 import experts_low
from low30b import repack_k, experts_prof

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
E = 32 if SMOKE else 128
LAYERS = (12,) if SMOKE else (12, 20, 28, 36)
CFG = {4: ((4, 1), (2, 1)), 3: ((5, 1), (1, 2)), 2: ((4, 1), (1, 2))}      # L1-30's sweep winners
BITS = (4, 3, 2)
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

MD = os.path.expanduser("~/BigMoE/models/bare")
WMAP = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def load_layer(L):
    p = f"language_model.model.layers.{L}.mlp.switch_mlp."
    keys = [p + f"{n}.{s}" for n in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "scales", "biases")]
    got = {}
    for f in sorted({WMAP[k] for k in keys}):
        d = mx.load(os.path.join(MD, f))
        got.update({k: mx.contiguous(d[k][:E]) for k in keys if k in d})
        del d
    gu = [mx.concatenate([got[p + f"gate_proj.{s}"], got[p + f"up_proj.{s}"]], axis=1) for s in ("weight", "scales", "biases")]
    dn = [got[p + f"down_proj.{s}"] for s in ("weight", "scales", "biases")]
    mx.eval(gu, dn)
    return gu, dn

def codes4(w):
    W = np.array(w).view(np.uint32)
    return np.stack([(W >> np.uint32(4 * j)) & np.uint32(15) for j in range(8)], -1).reshape(W.shape[0], W.shape[1], -1).astype(np.uint8)

def derive(Q4, s, b, bits):
    d = 4 - bits
    sf, bf = s.astype(mx.float32), b.astype(mx.float32)
    return Q4 >> np.uint8(d), (sf * (1 << d)).astype(s.dtype), (bf + sf * (((1 << d) - 1) / 2)).astype(b.dtype)

t0 = time.time()
G = []                                                        # G[layer][(bits, pairk)] = (gu, dn) native arrays
for L in LAYERS:
    gu, dn = load_layer(L)
    q_gu, q_dn = codes4(gu[0]), codes4(dn[0])
    g_l = {}
    for b in BITS:
        dg, dd = derive(q_gu, gu[1], gu[2], b), derive(q_dn, dn[1], dn[2], b)
        for pk in (0, 1):
            g_l[(b, pk)] = (repack_k(*dg, b, pk), repack_k(*dd, b, pk))
            mx.eval(g_l[(b, pk)])
    G.append(g_l)
    del gu, dn, q_gu, q_dn
say(f"L1-30b: profile + K-paired layout; {E} experts x {len(LAYERS)} layers; prep {time.time() - t0:.0f} s")

rng = np.random.default_rng(30)
def routing(rows, k=10):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))
def bits_of(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

mx.random.seed(30)
x = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
ids16 = routing(16)
for b in BITS:
    y_old = bits_of(experts_low(x, *G[0][(b, 0)], ids16, b, *CFG[b]))
    y0 = bits_of(experts_prof(x, *G[0][(b, 0)], ids16, b, *CFG[b], 0, 0))
    y1 = bits_of(experts_prof(x, *G[0][(b, 1)], ids16, b, *CFG[b], 0, 1))
    say(f"  {b}-bit: low30b == low30 bitwise {np.array_equal(y_old, y0)}; K-paired == row-paired bitwise {np.array_equal(y0, y1)}")
if SMOKE:
    for b in BITS:
        for m in (1, 2, 3, 4):
            mx.eval(experts_prof(x, *G[0][(b, 0)], ids16, b, *CFG[b], m, 0))
    say("  smoke: every MODE compiles and runs")
    open(os.path.expanduser("~/Leap/data/low30b_smoke.txt"), "w").write("\n".join(lines) + "\n")
    sys.exit(0)

NW = 1280 * 2560 + 2560 * 640
EB = {b: NW * (b / 8 + 4 / 64) for b in BITS}
NAMES = {0: "full", 1: "no multiply", 2: "no unpack", 3: "no x work", 4: "reads only"}
R16 = [routing(16) for _ in LAYERS]
U16 = np.mean([len(set(np.array(r).flatten().tolist())) for r in R16])
for b in (2, 4):
    fns = {m: (lambda m=m: [experts_prof(x, *G[i][(b, 0)], R16[i], b, *CFG[b], m, 0) for i in range(len(LAYERS))])
           for m in NAMES}
    res = ab(fns)
    floor = U16 * EB[b] / 280e9 * 1e3 * len(LAYERS)
    say(f"  profile {b}-bit, 16 rows ({U16:.0f} distinct, {len(LAYERS)} layers; floor {floor:.3f} ms):")
    for m, v in res.items():
        say(f"    MODE {m} {NAMES[m]:11s} {np.median(v):.3f} ms ({1e3 * np.median(v) / (U16 * len(LAYERS)):.1f} us per expert; "
            f"{floor / np.median(v):.0%} of floor)")
for rows in (8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    fns = {(b, pk): (lambda b=b, pk=pk: [experts_prof(x[:rows], *G[i][(b, pk)], R[i], b, *CFG[b], 0, pk)
                                         for i in range(len(LAYERS))]) for b in BITS for pk in (0, 1)}
    res = ab(fns)
    say(f"  K-paired vs row-paired, {rows} rows ({U:.0f} distinct):")
    for b in BITS:
        floor = U * EB[b] / 280e9 * 1e3 * len(LAYERS)
        r = res[(b, 0)] / res[(b, 1)]
        say(f"    {b}-bit: row-paired {np.median(res[(b, 0)]):.3f} ms ({floor / np.median(res[(b, 0)]):.0%}), K-paired "
            f"{np.median(res[(b, 1)]):.3f} ms ({floor / np.median(res[(b, 1)]):.0%}); paired x{np.median(r):.3f} "
            f"(IQR {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f})")
open(os.path.expanduser("~/Leap/data/low30b.txt"), "w").write("\n".join(lines) + "\n")
