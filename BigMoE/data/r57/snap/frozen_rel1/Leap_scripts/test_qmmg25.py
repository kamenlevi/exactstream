"""L1-25 test (Leap, Sat 2026-10-24): experts_nax (qmmg25.py) on real 4-bit weights of 4 layers.
1. Batch invariance: rows 0..M-1 run alone (M = 1, 2, 4, 7, 8, 12) equal the same rows inside the 16-row call, bitwise.
2. Accuracy vs an fp32 reference (fp32 x, dequantized experts, fp32 SwiGLU / down): max |err| of the engine's MLX path vs
   the NAX path (both round h to bf16 between the projections, as the engine does).
3. ms per layer at 4 / 8 / 16 rows for (ks, nt) configurations vs MLX's sorted path and the floor.
Usage: [BigMoE venv] test_qmmg25.py -> ~/Leap/data/qmmg25.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
from qmmg25 import experts_nax

MD = os.path.expanduser("~/BigMoE/models/bare")
WMAP = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
LAYERS = (12, 20, 28, 36)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

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

W = [load_layer(L) for L in LAYERS]
E = W[0][0][0].shape[0]
EB = sum(a[0].nbytes for a in W[0][0]) + sum(a[0].nbytes for a in W[0][1])
rng = np.random.default_rng(25)

def routing(rows, k=10):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    return mx.array(np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)]).astype(np.uint32))

def p_sorted(x, gu, dn, ids):
    k = ids.shape[-1]; flat = ids.flatten(); order = mx.argsort(flat); inv = mx.argsort(order)
    xs = x.reshape(-1, 1, x.shape[-1])[order // k]
    g = mx.gather_qmm(xs, *gu, rhs_indices=flat[order], transpose=True, group_size=64, bits=4, sorted_indices=True)
    gate, up = mx.split(g, 2, axis=-1)
    y = mx.gather_qmm(nn.silu(gate) * up, *dn, rhs_indices=flat[order], transpose=True, group_size=64, bits=4,
                      sorted_indices=True)
    return y[inv].reshape(ids.shape[0], k, -1)

def ref32(x, gu, dn, ids):
    """fp32 reference with the engine's bf16 rounding of h"""
    out = []
    xf = x.astype(mx.float32)
    for r in range(ids.shape[0]):
        row = []
        for e in np.array(ids[r]).tolist():
            Wg = mx.dequantize(gu[0][e], gu[1][e], gu[2][e], group_size=64, bits=4).astype(mx.float32)
            Wd = mx.dequantize(dn[0][e], dn[1][e], dn[2][e], group_size=64, bits=4).astype(mx.float32)
            g = xf[r] @ Wg.T
            gate, up = g[:g.shape[0] // 2], g[g.shape[0] // 2:]
            h = (gate * mx.sigmoid(gate) * up).astype(mx.bfloat16).astype(mx.float32)
            row.append(h @ Wd.T)
        out.append(mx.stack(row))
    return mx.stack(out)

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

X = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
say(f"L1-25: experts on the tensor units (16-row NAX tile per expert), real 4-bit weights of layers {LAYERS}")
CA, CB = (1, 1), (1, 1)
ids16 = routing(16)
full = bits(experts_nax(X, *W[1], ids16, CA, CB))
inv_ok = []
for M in (1, 2, 4, 7, 8, 12):
    y = bits(experts_nax(X[:M], *W[1], ids16[:M], CA, CB))
    inv_ok.append(f"{M}:{'=' if np.array_equal(y, full[:M]) else 'DIFF'}")
say("  batch invariance (rows alone vs inside the 16-row call, bitwise): " + " ".join(inv_ok))
ids4 = routing(4)
ref = ref32(X[:4], *W[2], ids4)
e_m = float(mx.abs(p_sorted(X[:4], *W[2], ids4).astype(mx.float32) - ref).max())
e_n = float(mx.abs(experts_nax(X[:4], *W[2], ids4, CA, CB).astype(mx.float32) - ref).max())
say(f"  accuracy vs fp32 (4 rows x 10 experts, layer {LAYERS[2]}): max |err| MLX path {e_m:.4g}, NAX path {e_n:.4g}; "
    f"|ref| max {float(mx.abs(ref).max()):.3g}")
for rows in (4, 8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    floor = U * EB / 280e9 * 1e3
    t_s = timeit(lambda: [p_sorted(X[:rows], *W[i], R[i]) for i in range(len(LAYERS))]) / len(LAYERS)
    best = None
    cells = []
    for ca in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1)):
        for cb in ((1, 1), (2, 1), (1, 2), (5, 1)):
            t = timeit(lambda: [experts_nax(X[:rows], *W[i], R[i], ca, cb) for i in range(len(LAYERS))]) / len(LAYERS)
            cells.append((t, ca, cb))
    cells.sort()
    t_n, ca, cb = cells[0]
    say(f"  {rows:2d} rows ({U:.0f} distinct experts): floor {floor:.3f} ms per layer; MLX sorted {t_s:.3f} ({floor / t_s:.0%}); "
        f"NAX best {t_n:.3f} ({floor / t_n:.0%}) at gate/up {ca}, down {cb} (next: "
        + ", ".join(f"{t:.3f} {a}{b}" for t, a, b in cells[1:4]) + f"); per 48-layer check: floor {48 * floor:.1f}, "
        f"sorted {48 * t_s:.1f}, NAX {48 * t_n:.1f} ms")
open(os.path.expanduser("~/Leap/data/qmmg25.txt"), "w").write("\n".join(lines) + "\n")
