"""1c expert tiers for a lossy check (Breakthrough's ~/FableGod/deepdraft/LOSSY_CHECK_SPEC.md, Sun 15 Nov 2026).

Per check and per layer, after the router: for every expert e in the check's union,
    I_e = sum over the check's rows d of s^d * w(d, e)^2      (d = 0 for the first verified row; w = the row's top-10
                                                              weights, normalised as the block does)
Tier: 4-bit if I_e > 2.775 lam; MSB-3 (read only the records3 MSB plane) if 0.8457 lam < I_e <= 2.775 lam; else drop.
At lam 0.016: 4-bit if I > 0.0444, MSB-3 if 0.0135 < I <= 0.0444. The model's own weights, no renorm.

tiers_np: the float64 reference (the rule as the decision map priced it).
tiers_mx: the engine's form (MLX, one scatter-add over the expert axis; works for any check size).
Both return per (row, rank) slot: 0 = 4-bit, 1 = MSB-3, 2 = drop.
"""
import numpy as np

T4_K, T3_K = 2.775, 0.8457
N_EXPERTS = 512


def tiers_np(ids, w, s=0.98, lam=0.016):
    """ids, w: [rows, 10] for ONE layer of one check. -> tier [rows, 10] int8."""
    ids = np.asarray(ids, np.int64); w = np.asarray(w, np.float64)
    rw = s ** np.arange(ids.shape[0])
    u, inv = np.unique(ids.ravel(), return_inverse=True)
    imp = np.bincount(inv, weights=(w ** 2 * rw[:, None]).ravel(), minlength=len(u))
    tier = np.where(imp > T4_K * lam, 0, np.where(imp > T3_K * lam, 1, 2)).astype(np.int8)
    return tier[inv].reshape(ids.shape)


def tiers_mx(inds, w, s=0.98, lam=0.016, n_experts=N_EXPERTS):
    """inds [rows, k] int, w [rows, k] float (any MLX dtype; computed in float32). -> tier [rows, k] uint8."""
    import mlx.core as mx
    n = inds.shape[0]
    rw = mx.power(mx.array(s, mx.float32), mx.arange(n, dtype=mx.float32))[:, None]
    wf = w.astype(mx.float32)
    imp = mx.zeros((n_experts,), mx.float32).at[inds.reshape(-1)].add((wf * wf * rw).reshape(-1))
    Ii = imp[inds]
    return mx.where(Ii > T4_K * lam, 0, mx.where(Ii > T3_K * lam, 1, 2)).astype(mx.uint8)
