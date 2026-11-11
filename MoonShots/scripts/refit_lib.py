"""Helpers for the activation-aware refit of truncated expert tiers (scripts/refit35.py, scripts/test_refit_lib.py)."""
import numpy as np
import mlx.core as mx


def fit_proj(inp, qe, se, be, n, minrows=3, wrow=None, lam0=1e-3):
    """Activation-aware refit of tier n for one projection of one expert. inp [S, K] f32 routed inputs (or None), qe [out, K]
    codes 0..15, se/be [out, G] f32 4-bit scales/biases. Returns (a0, b0, None) to keep the truncation, or
    (A [out, 2G, 2G], rhs [out, 2G], G) for the ridge system whose solution is (alpha | beta). wrow [S]: per-row weights
    (e.g. gate score^2: the expert's share of the block output); lam0: ridge strength relative to the mean diagonal."""
    out, G = se.shape
    sh = 4 - n
    q = qe.reshape(out, G, 64)
    m = (q >> sh).astype(mx.float32)
    low = (q & ((1 << sh) - 1)).astype(mx.float32)
    a0 = se * (1 << sh)
    b0 = be + se * low.mean(-1)
    if inp is None or inp.shape[0] < minrows * 2 * G:
        return a0, b0, None
    ns = inp.shape[0]
    if wrow is not None:                                                   # weighted LS: scale rows by sqrt(weight)
        inp = inp * mx.sqrt(wrow / wrow.mean())[:, None]
    w = se[..., None] * q.astype(mx.float32) + be[..., None]
    xg = inp.reshape(ns, G, 64)
    pa = mx.einsum("sgk,ngk->nsg", xg, m)                                 # [out, S, G]
    pb = xg.sum(-1)                                                        # [S, G]
    t = inp @ w.reshape(out, -1).T                                         # [S, out]
    Aaa = mx.einsum("nsg,nsh->ngh", pa, pa)
    Aab = mx.einsum("nsg,sh->ngh", pa, pb)
    Abb = mx.broadcast_to((pb.T @ pb)[None], (out, G, G))
    A = mx.concatenate([mx.concatenate([Aaa, Aab], -1), mx.concatenate([Aab.transpose(0, 2, 1), Abb], -1)], 1)
    ra, rb = mx.einsum("nsg,sn->ng", pa, t), (pb.T @ t).T
    th0 = mx.concatenate([a0, b0], -1)
    lam = (lam0 * max(1.0, 512.0 / ns)) * mx.trace(A, axis1=1, axis2=2) / (2 * G) + 1e-6
    rhs = mx.concatenate([ra, rb], -1) + lam[:, None] * th0
    return A + lam[:, None, None] * mx.eye(2 * G)[None], rhs, G


def codes(w):
    return ((w[..., None] >> mx.arange(0, 32, 4, dtype=mx.uint32)) & 15).reshape(w.shape[0], w.shape[1], -1)


def solve(A, rhs):
    return np.linalg.solve(A, rhs[..., None])[..., 0]
