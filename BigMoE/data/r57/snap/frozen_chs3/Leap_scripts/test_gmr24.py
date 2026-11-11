"""L1-24 test (Leap, Sat 2026-10-24): experts_gmr (gmr24.py) vs the engine's expert path on real 4-bit weights of 4 layers.
Bitwise per (row, expert) output vs MLX's per-pair gather_qmv chain (exp24's 'unsorted', = sorted = BigMoE's) at 1 / 4 /
8 / 16 rows, plus a case where one expert serves all 16 rows; then ms per layer at 4 / 8 / 16 rows vs the sorted path
and the floor (distinct experts x 2.765 MB / 280 GB/s).
Usage: [BigMoE venv] test_gmr24.py -> ~/Leap/data/gmr24.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
from gmr24 import experts_gmr

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
rng = np.random.default_rng(24)

def routing(rows, k=10, shared=False):
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    out = np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)])
    if shared:                                               # one expert in every row (a segment of 16 rows)
        for r in range(rows):
            if pool[0] not in out[r]:
                out[r, 0] = pool[0]
    return mx.array(out.astype(np.uint32))

def p_unsorted(x, gu, dn, ids):
    xx = mx.expand_dims(x, (-2, -3))
    g = mx.gather_qmm(xx, *gu, rhs_indices=ids, transpose=True, group_size=64, bits=4)
    gate, up = mx.split(g, 2, axis=-1)
    y = mx.gather_qmm(nn.silu(gate) * up, *dn, rhs_indices=ids, transpose=True, group_size=64, bits=4)
    return y.squeeze(-2)

def p_sorted(x, gu, dn, ids):
    k = ids.shape[-1]; flat = ids.flatten(); order = mx.argsort(flat); inv = mx.argsort(order)
    xs = x.reshape(-1, 1, x.shape[-1])[order // k]
    g = mx.gather_qmm(xs, *gu, rhs_indices=flat[order], transpose=True, group_size=64, bits=4, sorted_indices=True)
    gate, up = mx.split(g, 2, axis=-1)
    y = mx.gather_qmm(nn.silu(gate) * up, *dn, rhs_indices=flat[order], transpose=True, group_size=64, bits=4,
                      sorted_indices=True)
    return y[inv].reshape(ids.shape[0], k, -1)

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

X = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
say(f"L1-24: experts_gmr vs the engine's expert path, real 4-bit weights of layers {LAYERS} ({EB / 1e6:.3f} MB per expert)")
for rows, shared in ((1, False), (4, False), (8, False), (16, False), (16, True)):
    ok = True
    for i in range(len(LAYERS)):
        ids = routing(rows, shared=shared)
        a = bits(p_unsorted(X[:rows], *W[i], ids)); b = bits(experts_gmr(X[:rows], *W[i], ids))
        if not np.array_equal(a, b):
            ok = False
            d = np.nonzero(a != b)
            say(f"    mismatch rows {rows} layer {LAYERS[i]}: {len(d[0])} of {a.size} values differ, first at {tuple(int(t[0]) for t in d)}")
    say(f"  bitwise vs the engine path, {rows:2d} rows{' (one expert in all 16 rows)' if shared else ''}: {'EQUAL' if ok else 'DIFF'}")
for rows in (4, 8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    floor = U * EB / 280e9 * 1e3
    t_s = timeit(lambda: [p_sorted(X[:rows], *W[i], R[i]) for i in range(len(LAYERS))]) / len(LAYERS)
    t_u = timeit(lambda: [p_unsorted(X[:rows], *W[i], R[i]) for i in range(len(LAYERS))]) / len(LAYERS)
    t_g = timeit(lambda: [experts_gmr(X[:rows], *W[i], R[i]) for i in range(len(LAYERS))]) / len(LAYERS)
    say(f"  {rows:2d} rows ({U:.0f} distinct experts): floor {floor:.3f} ms per layer; sorted {t_s:.3f} ({floor / t_s:.0%}), "
        f"unsorted {t_u:.3f} ({floor / t_u:.0%}), gmr {t_g:.3f} ({floor / t_g:.0%}); per 48-layer check: floor "
        f"{48 * floor:.1f}, sorted {48 * t_s:.1f}, gmr {48 * t_g:.1f} ms")
open(os.path.expanduser("~/Leap/data/gmr24.txt"), "w").write("\n".join(lines) + "\n")
