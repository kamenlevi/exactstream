"""L1-30c test (Leap, Sun 2026-10-25): read-side fixes of the native expert kernel, bitwise-equal outputs.
Variants per bits (4 / 3 / 2): base (low30b MODE 0), kpair (PAIRK 1), grid (launch exactly the distinct-expert count),
pref (prefetch the next group), all (kpair + grid + pref). (1) bitwise vs base; (2) fair interleaved A/B (12 rounds x
median of 7) at 8 and 16 rows, % of each one's byte floor, paired x vs base and vs 4-bit base.
Usage: [BigMoE venv] test_low30c.py [smoke] -> ~/Leap/data/low30c.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
from low30b import repack_k
from low30c import experts_prof

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
G = []
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
say(f"L1-30c: read-side fixes; {E} experts x {len(LAYERS)} layers; prep {time.time() - t0:.0f} s")

rng = np.random.default_rng(30)
def routing(rows, k=10):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))
def bits_of(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)
def ndist(ids):
    return len(set(np.array(ids).flatten().tolist()))
VAR = {"base": (0, 0, False), "kpair": (1, 0, False), "grid": (0, 0, True), "pref": (0, 1, False), "all": (1, 1, True)}
def call(xx, i, b, ids, v, nd):
    pk, pf, gr = VAR[v]
    return experts_prof(xx, *G[i][(b, pk)], ids, b, *CFG[b], 0, pk, pf, nd if gr else None)

mx.random.seed(30)
x = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
ids16 = routing(16)
for b in BITS:
    y0 = bits_of(call(x, 0, b, ids16, "base", ndist(ids16)))
    eq = {v: np.array_equal(y0, bits_of(call(x, 0, b, ids16, v, ndist(ids16)))) for v in VAR if v != "base"}
    say(f"  {b}-bit bitwise vs base: " + ", ".join(f"{v} {e}" for v, e in eq.items()))
if SMOKE:
    open(os.path.expanduser("~/Leap/data/low30c_smoke.txt"), "w").write("\n".join(lines) + "\n")
    sys.exit(0)

NW = 1280 * 2560 + 2560 * 640
EB = {b: NW * (b / 8 + 4 / 64) for b in BITS}
for rows in (8, 16):
    R = [routing(rows) for _ in LAYERS]
    ND = [ndist(r) for r in R]
    U = np.mean(ND)
    fns = {f"{b}:{v}": (lambda b=b, v=v: [call(x[:rows], i, b, R[i], v, ND[i]) for i in range(len(LAYERS))])
           for b in BITS for v in VAR}
    res = ab(fns)
    say(f"  {rows} rows ({U:.0f} distinct, {len(LAYERS)} layers), ms over 4 layers (% of own floor; paired x vs own base; x vs 4-bit base):")
    for b in BITS:
        floor = U * EB[b] / 280e9 * 1e3 * len(LAYERS)
        cells = []
        for v in VAR:
            t = res[f"{b}:{v}"]
            r = res[f"{b}:base"] / t
            r4 = res["4:base"] / t
            cells.append(f"{v} {np.median(t):.3f} ({floor / np.median(t):.0%}; x{np.median(r):.3f} "
                         f"[{np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f}]; x{np.median(r4):.2f})")
        say(f"    {b}-bit: " + "; ".join(cells))
open(os.path.expanduser("~/Leap/data/low30c.txt"), "w").write("\n".join(lines) + "\n")
