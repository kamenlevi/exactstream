"""L1-24 screen (Leap, Sat 2026-10-24): the routed experts of a verify check, the biggest byte block (~73 distinct experts
per layer at 16 rows, top-10). How close to the bandwidth floor do the engine's expert paths run at 4 / 8 / 16 rows, and
is each one batch-invariant (a row's expert outputs independent of how many rows run)?
Real 4-bit expert weights of 4 layers (12 / 20 / 28 / 36; gate + up fused as BigMoE's _FusedGateUpSwitchGLU does), a
synthetic routing per layer with the measured union (10 x rows^0.72 distinct experts; Zipf-weighted picks), x random.
Paths, as BigMoE's switch MLP runs them (read-only import of mtplx.moe_sorted_gather):
  unsorted  mx.gather_qmm per (row, expert) pair (the engine's path below 64 routed rows, i.e. <= 6 rows)
  sorted    mlx-lm's gather-sort: rows copied in expert order, gather_qmm(sorted_indices=True) (stock MLX NAX kernel)
  bigmoe    mtplx sort_rows + swiglu_rows (its NAX kernel reading rows through a map) + sorted down gather
Floor: distinct experts x 2.765 MB / 280 GB/s. Per check = per layer x 48.
Usage: [BigMoE venv] exp24.py -> ~/Leap/data/exp24.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mtplx import moe_sorted_gather as G

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

t0 = time.time()
W = [load_layer(L) for L in LAYERS]
E = W[0][0][0].shape[0]
ebytes = sum(a[0].nbytes for a in W[0][0]) + sum(a[0].nbytes for a in W[0][1])   # a[0] = one expert
say(f"L1-24 screen: routed experts, real 4-bit weights of layers {LAYERS} ({E} experts, {ebytes / 1e6:.3f} MB each incl. "
    f"scales / biases), loaded in {time.time() - t0:.0f} s")
rng = np.random.default_rng(24)

def routing(rows, k=10):
    """rows x k distinct expert ids drawn from a pool of round(10 * rows^0.72) experts (Zipf-weighted)"""
    U = min(E, round(10 * rows ** 0.72))
    pool = rng.choice(E, U, replace=False)
    pw = 1.0 / np.arange(1, U + 1) ** 0.8; pw /= pw.sum()
    out = np.stack([rng.choice(pool, k, replace=False, p=pw) for _ in range(rows)])
    # make sure every pool expert appears at least once (union = U), by overwriting rare picks
    missing = [e for e in pool if e not in set(out.flatten())]
    for i, e in enumerate(missing):
        r, c = i % rows, k - 1 - (i // rows) % k
        if e not in out[r]:
            out[r, c] = e
    return mx.array(out.astype(np.uint32))

def p_unsorted(x, gu, dn, ids):
    xx = mx.expand_dims(x, (-2, -3))
    g = mx.gather_qmm(xx, *gu, rhs_indices=ids, transpose=True, group_size=64, bits=4)
    gate, up = mx.split(g, 2, axis=-1)
    y = mx.gather_qmm(nn.silu(gate) * up, *dn, rhs_indices=ids, transpose=True, group_size=64, bits=4)
    return y.squeeze(-2)                                                   # [rows, k, hidden]

def p_sorted(x, gu, dn, ids):
    k = ids.shape[-1]; flat = ids.flatten(); order = mx.argsort(flat); inv = mx.argsort(order)
    xs = x.reshape(-1, 1, x.shape[-1])[order // k]
    g = mx.gather_qmm(xs, *gu, rhs_indices=flat[order], transpose=True, group_size=64, bits=4, sorted_indices=True)
    gate, up = mx.split(g, 2, axis=-1)
    y = mx.gather_qmm(nn.silu(gate) * up, *dn, rhs_indices=flat[order], transpose=True, group_size=64, bits=4,
                      sorted_indices=True)
    return y[inv].reshape(ids.shape[0], k, -1)

def p_bigmoe(x, gu, dn, ids):
    k = ids.shape[-1]
    tokens, row_map, sidx, inv = G.sort_rows(x, ids)
    h = G.swiglu_rows(tokens, row_map, *gu, sidx, group_size=64, bits=4)
    y = G.gather_qmm(h, *dn, rhs_indices=sidx, transpose=True, group_size=64, bits=4, sorted_indices=True)
    return y[inv].reshape(ids.shape[0], k, -1)

PATHS = {"unsorted": p_unsorted, "sorted": p_sorted, "bigmoe": p_bigmoe}

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

X = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
for rows in (4, 8, 16):
    R = [routing(rows) for _ in LAYERS]
    U = np.mean([len(set(np.array(r).flatten().tolist())) for r in R])
    floor = U * ebytes / 280e9 * 1e3
    res = {}
    for name, fn in PATHS.items():
        t = timeit(lambda: [fn(X[:rows], *W[i], R[i]) for i in range(len(LAYERS))]) / len(LAYERS)
        res[name] = t
    say(f"  {rows:2d} rows ({rows * 10} routed, {U:.0f} distinct experts per layer): floor {floor:.3f} ms per layer; "
        + "; ".join(f"{n} {t:.3f} ms ({floor / t:.0%} of floor)" for n, t in res.items())
        + f"  -> per 48-layer check: floor {48 * floor:.1f} ms, best {48 * min(res.values()):.1f} ms")

# batch invariance: rows 0..M-1 alone vs inside the 16-row call (same routing rows), per path; and across paths
R16 = routing(16)
full = {n: np.array(fn(X, *W[1], R16).astype(mx.float32)).view(np.uint32) for n, fn in PATHS.items()}
for n, fn in PATHS.items():
    same = []
    for M in (1, 2, 4, 7, 8, 12):
        y = np.array(fn(X[:M], *W[1], R16[:M]).astype(mx.float32)).view(np.uint32)
        same.append(f"{M}:{'=' if np.array_equal(y, full[n][:M]) else 'DIFF'}")
    say(f"  {n:8s} rows alone vs inside the 16-row call (bitwise): " + " ".join(same))
say("  across paths at 16 rows (bitwise): unsorted = sorted " + str(np.array_equal(full["unsorted"], full["sorted"]))
    + ", sorted = bigmoe " + str(np.array_equal(full["sorted"], full["bigmoe"]))
    + ", unsorted = bigmoe " + str(np.array_equal(full["unsorted"], full["bigmoe"])))
open(os.path.expanduser("~/Leap/data/exp24.txt"), "w").write("\n".join(lines) + "\n")
