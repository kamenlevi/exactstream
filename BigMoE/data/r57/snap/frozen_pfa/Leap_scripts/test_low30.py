"""L1-30 test (Leap, Sun 2026-10-25): native-layout expert kernel at 4 / 3 / 2 bits vs the 4-bit native kernel (qmm26)
and MLX's sorted gather_qmm at the same bits, real codes of 4 layers (first E experts each; 3-bit = q >> 1 as the v3
tier stores it, 2-bit = q >> 2; scale x 2^d, bias + scale x (2^d - 1) / 2), synthetic routing as ab28.
1. correctness: generic 4-bit == experts_native bitwise; error vs fp32 at each bits; rows alone == rows in 16, bitwise
2. config sweep (cfg_a with cfg_b (1, 2), then cfg_b) for every generic bits, on its own timings
3. fair interleaved A/B (12 rounds x median of 7) at 8 / 16 rows: % of each one's byte floor, paired x vs native 4-bit
Usage: [BigMoE venv] test_low30.py [smoke] -> ~/Leap/data/low30.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
from qmm26 import repack, experts_native
from low30 import repack_codes, experts_low

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
E = 32 if SMOKE else 128
LAYERS = (12,) if SMOKE else (12, 20, 28, 36)
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

def dequant(Q, s, b):
    """fp32 [E, N, K] from codes and per-64 scale / bias"""
    E_, N, K = Q.shape
    sf = np.array(s.astype(mx.float32)).reshape(E_, N, K // 64, 1)
    bf = np.array(b.astype(mx.float32)).reshape(E_, N, K // 64, 1)
    return (Q.reshape(E_, N, K // 64, 64).astype(np.float32) * sf + bf).reshape(E_, N, K)

BITS = (4, 3, 2)
NW = 1280 * 2560 + 2560 * 640
EB = {b: NW * (b / 8 + 4 / 64) for b in BITS}                     # bytes per expert: codes + bf16 scale and bias per 64
t0 = time.time()
W, WN, G, M, DQ = [], [], [], [], {}
for li, L in enumerate(LAYERS):
    gu, dn = load_layer(L)
    W.append((gu, dn)); WN.append((repack(*gu), repack(*dn)))
    q_gu, q_dn = codes4(gu[0]), codes4(dn[0])
    g_l, m_l = {}, {}
    for b in BITS:
        dg, dd = derive(q_gu, gu[1], gu[2], b), derive(q_dn, dn[1], dn[2], b)
        g_l[b] = (repack_codes(*dg, b), repack_codes(*dd, b))
        if b == 4:
            m_l[b] = (gu, dn)
        else:
            m_l[b] = tuple(mx.quantize(mx.dequantize(*mat, group_size=64, bits=4), group_size=64, bits=b) for mat in (gu, dn))
        if li == 0:
            DQ[b] = (dg, dd)
        mx.eval(g_l[b], m_l[b])
    G.append(g_l); M.append(m_l)
    del q_gu, q_dn
say(f"L1-30: native-layout tensor-unit experts at 4 / 3 / 2 bits; {E} experts x {len(LAYERS)} layers; prep {time.time() - t0:.0f} s")

def p_mlx(x, gu, dn, ids, bits):
    k = ids.shape[-1]; flat = ids.flatten(); order = mx.argsort(flat); inv = mx.argsort(order)
    xs = x.reshape(-1, 1, x.shape[-1])[order // k]
    g = mx.gather_qmm(xs, *gu, rhs_indices=flat[order], transpose=True, group_size=64, bits=bits, sorted_indices=True)
    gate, up = mx.split(g, 2, axis=-1)
    y = mx.gather_qmm(nn.silu(gate) * up, *dn, rhs_indices=flat[order], transpose=True, group_size=64, bits=bits,
                      sorted_indices=True)
    return y[inv].reshape(ids.shape[0], k, -1)

rng = np.random.default_rng(30)
def routing(rows, k=10):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))

mx.random.seed(30)
x = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
ids16 = routing(16)
def bits_of(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

# 1. correctness
y_nat = experts_native(x, *WN[0], ids16)
y_g4 = experts_low(x, *G[0][4], ids16, 4)
say(f"  generic 4-bit == experts_native bitwise: {np.array_equal(bits_of(y_nat), bits_of(y_g4))}")
pairs = [(r, int(e)) for r in range(4) for e in np.array(ids16[r]).tolist()]
for b in BITS:
    (Qg, sg, bg), (Qd, sd, bd) = DQ[b]
    yk = np.array(experts_low(x, *G[0][b], ids16, b).astype(mx.float32))
    xf = np.array(x.astype(mx.float32))
    err, top = 0.0, 0.0
    for r, e in pairs:
        wg = dequant(Qg[e:e + 1], sg[e:e + 1], bg[e:e + 1])[0]
        wd = dequant(Qd[e:e + 1], sd[e:e + 1], bd[e:e + 1])[0]
        gg = wg @ xf[r]
        gate, up = gg[:640], gg[640:]
        h = np.array(mx.array(gate / (1 + np.exp(-gate)) * up).astype(mx.bfloat16).astype(mx.float32))
        yr = wd @ h
        j = np.array(ids16[r]).tolist().index(e)
        err = max(err, float(np.abs(yk[r, j] - yr).max())); top = max(top, float(np.abs(yr).max()))
    full = bits_of(experts_low(x, *G[0][b], ids16, b))
    inv = all(np.array_equal(bits_of(experts_low(x[:m], *G[0][b], ids16[:m], b)), full[:m]) for m in (1, 5, 12))
    say(f"  {b}-bit: max |err| vs fp32 {err:.3g} (max |y| {top:.3g}, rel {err / top:.2g}); rows alone == in 16: {inv}")
if SMOKE:
    open(os.path.expanduser("~/Leap/data/low30_smoke.txt"), "w").write("\n".join(lines) + "\n")
    sys.exit(0)

# 2. config sweep at 16 rows
R16 = [routing(16) for _ in LAYERS]
def run_g(b, rows, R, ca, cb):
    return lambda: [experts_low(x[:rows], *G[i][b], R[i], b, ca, cb) for i in range(len(LAYERS))]
CFG = {}
for b in BITS:
    ta = sorted((t_once(run_g(b, 16, R16, ca, (1, 2))), ca) for ca in ((2, 1), (4, 1), (5, 1), (8, 1), (4, 2), (2, 2)))
    tb = sorted((t_once(run_g(b, 16, R16, ta[0][1], cb)), cb) for cb in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (5, 1)))
    CFG[b] = (ta[0][1], tb[0][1])
    say(f"  {b}-bit sweep (16 rows, 4 layers): gate/up {', '.join(f'{t:.2f}{c}' for t, c in ta[:3])}; down "
        f"{', '.join(f'{t:.2f}{c}' for t, c in tb[:3])} -> {CFG[b]}")

# 3. fair A/B
for rows in (8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    fns = {"native4": lambda: [experts_native(x[:rows], *WN[i], R[i]) for i in range(len(LAYERS))]}
    for b in BITS:
        fns[f"low{b}"] = run_g(b, rows, R, *CFG[b])
        fns[f"mlx{b}"] = (lambda b=b: [p_mlx(x[:rows], *M[i][b], R[i], b) for i in range(len(LAYERS))])
    res = ab(fns)
    say(f"  experts {rows} rows ({U:.0f} distinct, {len(LAYERS)} layers):")
    for k, v in res.items():
        b = int(k[-1])
        floor = U * EB[b] / 280e9 * 1e3 * len(LAYERS)
        r = res["native4"] / v
        say(f"    {k:7s} {np.median(v):.3f} ms ({floor / np.median(v):.0%} of its {b}-bit floor {floor:.3f}); "
            f"x{np.median(r):.2f} vs native4 (IQR {np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f})")
say(f"  ideal from bytes vs 4-bit: 3-bit x{EB[4] / EB[3]:.2f}, 2-bit x{EB[4] / EB[2]:.2f}")
open(os.path.expanduser("~/Leap/data/low30.txt"), "w").write("\n".join(lines) + "\n")
