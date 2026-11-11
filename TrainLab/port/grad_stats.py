"""Per-parameter gradient statistics for the MLX-vs-port gradient parity (MTP Head, Thu 11-05; the PORT steps check
FAILed at steps 2-5 while step 1's loss matched, so the update or its gradient differs).
For each parameter (MLX names and row-major MLX shapes on both sides): norm, sum, L1 and K fixed +-1 projections
<g, s_k>. The signs come from an integer hash of (element index, parameter name, k), so both sides (different venvs,
maybe different numpy versions) use identical vectors. Projections are linear, so for two gradients a, b:
mean_k (p_k(a) - p_k(b))^2 estimates ||a - b||^2; rel_diff = that root / ||a|| (rough at K = 8, ~+-25 %).
Gradients with <= SMALL elements are also kept whole."""
import json, zlib
import numpy as np

K, CHUNK, SMALL = 8, 1 << 22, 20000
MUL = np.uint64(0x9E3779B97F4A7C15)


def _signs(c0, n, seed):
    with np.errstate(over="ignore"):
        h = (np.arange(c0, c0 + n, dtype=np.uint64) + np.uint64(seed)) * MUL
        h ^= h >> np.uint64(29)
        h *= MUL
    return ((h >> np.uint64(40)) & np.uint64(1)).astype(np.float64) * 2.0 - 1.0


def stats(name, g):
    """g: the gradient as a float32 numpy array (any shape, row-major)"""
    g = np.ascontiguousarray(g, dtype=np.float32).reshape(-1)
    base = zlib.crc32(name.encode()) & 0xFFFFFF
    sq = s1 = l1 = 0.0
    proj = np.zeros(K)
    for c0 in range(0, g.size, CHUNK):
        c = g[c0:c0 + CHUNK].astype(np.float64)
        sq += float(c @ c); s1 += float(c.sum()); l1 += float(np.abs(c).sum())
        for k in range(K):
            proj[k] += float(c @ _signs(c0, c.size, base * 64 + k + 1))
    return dict(n=int(g.size), norm=sq ** 0.5, sum=s1, l1=l1, proj=proj.tolist(),
                full=(g.astype(np.float64).tolist() if g.size <= SMALL else None))


def save(path, loss, per_param, extra=None):
    json.dump(dict(loss=float(loss), params=per_param, **(extra or {})), open(path, "w"))


def compare(a_path, b_path, label_a="mlx", label_b="port"):
    """per-parameter norm ratio and estimated relative difference ||a - b|| / ||a||; returns lines"""
    A, B = json.load(open(a_path)), json.load(open(b_path))
    out = [f"loss: {label_a} {A['loss']:.6f}  {label_b} {B['loss']:.6f}  rel {abs(A['loss'] - B['loss']) / abs(A['loss']):.2e}"]
    tot_a = sum(v["norm"] ** 2 for v in A["params"].values()) ** 0.5
    tot_b = sum(v["norm"] ** 2 for v in B["params"].values()) ** 0.5
    out.append(f"total grad norm: {label_a} {tot_a:.6f}  {label_b} {tot_b:.6f}  ratio {tot_b / tot_a:.5f}")
    miss = sorted(set(A["params"]) ^ set(B["params"]))
    if miss:
        out.append(f"names only on one side: {miss}")
    rows = []
    for name in sorted(set(A["params"]) & set(B["params"])):
        a, b = A["params"][name], B["params"][name]
        pa, pb = np.array(a["proj"]), np.array(b["proj"])
        est = float(np.sqrt(np.mean((pa - pb) ** 2))) / max(a["norm"], 1e-30)
        exact = None
        if a["full"] is not None and b["full"] is not None:
            fa, fb = np.array(a["full"]), np.array(b["full"])
            exact = float(np.linalg.norm(fa - fb) / max(np.linalg.norm(fa), 1e-30))
        rows.append((est if exact is None else exact, name, a["norm"], b["norm"], est, exact))
    for r_, name, na, nb, est, exact in sorted(rows, reverse=True):
        out.append(f"{name:60s} norm {label_a} {na:.4e} {label_b} {nb:.4e} ratio {nb / max(na, 1e-30):.4f}  "
                   f"rel diff ~{est:.2e}" + (f" (exact {exact:.2e})" if exact is not None else ""))
    return out
