"""Chunked, differentiable gated delta rule for MLX training (TrainLab, 2026-10-21).

mlx_lm's training path for GatedDeltaNet (gated_delta_ops) is a Python loop over time steps: autodiff then keeps the
[Hv, Dv, Dk] fp32 state of every step (2 MB per layer per token on the 35B: ~60 MB per token over 30 GDN layers), so it
cannot train on sequences of useful length. This is the chunk-parallel form (Yang et al., Gated Delta Networks; FLA's
naive chunk reference), written in plain MLX ops so autodiff handles the backward; only one state per chunk is kept.

Recurrence (per head), as in mlx_lm: S <- g_t S; S <- S + beta_t (v_t - S k_t) k_t^T; y_t = S q_t, S = [Dv, Dk].
Inside a chunk with cumulative decay G_t = prod_{s<=t} g_s and start state S0:
  S_t = G_t S0 + sum_{s<=t} (G_t / G_s) u_s k_s^T
  (I + A) U = diag(beta) V - diag(beta G) K S0^T,  A[t, s] = beta_t (G_t / G_s) k_t.k_s for s < t
  Y = diag(G) Q S0^T + tril(Q K^T * G_t / G_s) U;   S_C = G_C S0 + U^T diag(G_C / G) K
(I + A)^-1 for strictly lower A: recursive block inversion (see _tri_inverse; the power series is unstable in f32).
"""
import mlx.core as mx


def _tri_inverse(A):
    """(I + A)^-1 for strictly lower-triangular A [..., C, C] (C a power of two), by recursive block inversion of the
    unit lower-triangular M = I + A: inv([[P, 0], [Q, R]]) = [[P^-1, 0], [-R^-1 Q P^-1, R^-1]], doubling the block size
    from 1 to C (6 levels at C = 64). Stable like forward substitution. The power form prod (I + N^(2^i)) is not: with
    near-identical keys and beta ~ 1 its intermediate powers reach ~C(63, 31) ~ 1e17 before cancelling (NaN in f32,
    the first 35B training run on 10-21)."""
    C = A.shape[-1]
    lead = A.shape[:-2]
    M = A + mx.eye(C, dtype=A.dtype)
    inv = mx.ones(lead + (C, 1, 1), dtype=A.dtype)           # inverses of the 1 x 1 diagonal blocks
    b = 1
    while b < C:
        n = C // (2 * b)
        D = M.reshape(lead + (n, 2 * b, n, 2 * b))
        D = mx.moveaxis(mx.diagonal(D, axis1=-4, axis2=-2), -1, -3)  # diagonal 2b x 2b blocks [..., n, 2b, 2b]
        Q = D[..., b:, :b]
        # pair the blocks by reshape, not by strided slices: in MLX 0.32 the VJP of inv[..., 0::2] taken from this
        # concatenated array is wrong (forward right, gradient off by up to ~16x on some entries; found 10-21)
        ir = inv.reshape(lead + (n, 2, b, b))
        X, Y = ir[..., :, 0, :, :], ir[..., :, 1, :, :]
        Z = -(Y @ Q @ X)
        zero = mx.zeros(Z.shape, dtype=A.dtype)
        inv = mx.concatenate([mx.concatenate([X, zero], axis=-1), mx.concatenate([Z, Y], axis=-1)], axis=-2)
        b *= 2
    return inv[..., 0, :, :]


def gated_delta_chunked(q, k, v, log_g, beta, state=None, chunk=64):
    """q, k: [B, T, Hk, Dk]; v: [B, T, Hv, Dv]; log_g: [B, T, Hv] (log of the per-step decay); beta: [B, T, Hv];
    state: [B, Hv, Dv, Dk] or None. Returns y [B, T, Hv, Dv] (dtype of q) and the final state (fp32)."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    dt = q.dtype
    if Hv // Hk > 1:
        q = mx.repeat(q, Hv // Hk, -2)
        k = mx.repeat(k, Hv // Hk, -2)
    C = chunk
    pad = (-T) % C
    f32 = mx.float32
    # [B, H, T, D] in fp32
    q, k, v = (x.astype(f32).transpose(0, 2, 1, 3) for x in (q, k, v))
    log_g = log_g.astype(f32).transpose(0, 2, 1)
    beta = beta.astype(f32).transpose(0, 2, 1)
    if pad:                                           # padded steps: no decay, no write (beta 0), outputs dropped
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)])
        log_g = mx.pad(log_g, [(0, 0), (0, 0), (0, pad)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
    n = (T + pad) // C
    q = q.reshape(B, Hv, n, C, Dk)
    k = k.reshape(B, Hv, n, C, Dk)
    v = v.reshape(B, Hv, n, C, Dv)
    lg = mx.cumsum(log_g.reshape(B, Hv, n, C), axis=-1)             # log G_t within each chunk
    beta = beta.reshape(B, Hv, n, C)
    diff = lg[..., :, None] - lg[..., None, :]                      # log G_t - log G_s
    tri_strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    tri_incl = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=0)
    Gam = mx.exp(mx.where(tri_incl, diff, -mx.inf))                 # G_t / G_s for s <= t, else 0
    kk = k @ k.swapaxes(-1, -2)
    A = mx.where(tri_strict, beta[..., :, None] * Gam * kk, 0.0)
    Tinv = _tri_inverse(A)
    Gt = mx.exp(lg)                                                  # G_t
    rhs = mx.concatenate([beta[..., None] * v, (beta * Gt)[..., None] * k], axis=-1)
    sol = Tinv @ rhs
    U_raw, W = sol[..., :Dv], sol[..., Dv:]                          # U = U_raw - W S0^T
    Lqk = mx.where(tri_incl, (q @ k.swapaxes(-1, -2)) * Gam, 0.0)
    qG = Gt[..., None] * q
    to_end = mx.exp(lg[..., -1:] - lg)                               # G_C / G_s
    kE = to_end[..., None] * k
    gC = mx.exp(lg[..., -1])                                         # [B, H, n]
    S = mx.zeros((B, Hv, Dv, Dk), dtype=f32) if state is None else state.astype(f32)
    ys = []
    for i in range(n):
        St = S.swapaxes(-1, -2)                                      # [B, H, Dk, Dv]
        U = U_raw[:, :, i] - W[:, :, i] @ St
        ys.append(qG[:, :, i] @ St + Lqk[:, :, i] @ U)
        S = gC[:, :, i][..., None, None] * S + U.swapaxes(-1, -2) @ kE[:, :, i]
    y = mx.stack(ys, axis=2).reshape(B, Hv, n * C, Dv)[:, :, :T]
    return y.transpose(0, 2, 1, 3).astype(dt), S
