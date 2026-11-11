"""Fit12 Q2 (2026-10-24): OUTPUT error of fewer-bit expert codes on real MoE inputs (C1/Q1 judged weight-space error and
scalar rounding only). Same experts as Q1 (layers 2 / 12 / 24 / 36 / 46, 8 random experts each, seed 0). X = the layer's
MoE inputs captured in a ~4K-token B12 prefill (fit_wv.capture -> ~/Fit12/data/q2/x_L*.npy; all tokens, not only the routed
ones: a first proxy for each expert's input distribution). gate / up see X; down sees h = silu(X Wg4^T) * (X Wu4^T).
err(W) = tr((W - W4) H (W - W4)^T) / tr(W4 H W4^T), H = X^T X (or h^T h). Formats (bits incl. bf16 scale + bias):
MSB-3 (B12's base, 3.5 b), RTN-2 g64 (2.5), GPTQ-2 g64 (2.5), GPTQ-2 g128 (2.25), GPTQ-3 g64 (3.5). Pre-registered (to the
orchestrator, 13:12): GPTQ-2's error <= 1.5x MSB-3's -> go on; > 3x -> kill. Usage: python3 fit_q2.py (system python, CPU)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_q as F

D = os.path.expanduser("~/Fit12/data/q2")
LAYERS, NE = (2, 12, 24, 36, 46), 8


def rtn(W, bits, g):
    return F.rtn(W.astype(np.float32), bits, g)


def prep(H, damp=0.01):
    """upper Cholesky factor of (H + damp)^-1 and the dead-input mask (computed once per Hessian)"""
    H = H.astype(np.float64).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    H += damp * np.mean(np.diag(H)) * np.eye(H.shape[0])
    return np.linalg.cholesky(np.linalg.inv(H)).T, dead


def gptq(W, UD, bits, g, blk=128):
    """GPTQ (no act-order), per-group affine min-max levels set at each group's start from the updated weights, bf16
    scale / bias like MLX; lazy-batch updates in blocks of 128 columns. UD = prep(H)."""
    U, dead = UD
    W = W.astype(np.float64).copy()
    o, K = W.shape
    W[:, dead] = 0.0
    Qw = np.zeros_like(W)
    n = (1 << bits) - 1
    s = b = None
    for i1 in range(0, K, blk):
        i2 = min(i1 + blk, K)
        W1, E1 = W[:, i1:i2].copy(), np.zeros((o, i2 - i1))
        for j in range(i2 - i1):
            c = i1 + j
            if c % g == 0:
                G = W1[:, j:j + g] if j + g <= i2 - i1 else np.concatenate([W1[:, j:], W[:, i2:c + g]], 1)
                lo, hi = G.min(1), G.max(1)
                s = F.rbf16(np.maximum((hi - lo) / n, 1e-12).astype(np.float32)).astype(np.float64)
                b = F.rbf16(lo.astype(np.float32)).astype(np.float64)
            w = W1[:, j]
            q = np.clip(np.round((w - b) / s), 0, n) * s + b
            Qw[:, c] = q
            e = (w - q) / U[c, c]
            W1[:, j + 1:] -= np.outer(e, U[c, c + 1:i2])
            E1[:, j] = e
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    return Qw


def err(W, W4, H):
    d = (W - W4).astype(np.float64)
    w = W4.astype(np.float64)
    return float(((d @ H) * d).sum() / ((w @ H) * w).sum())     # BLAS, not einsum


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    names = ("MSB-3 (3.5 b)", "RTN-2 g64 (2.5 b)", "GPTQ-2 g64 (2.5 b)", "GPTQ-2 g128 (2.25 b)", "GPTQ-3 g64 (3.5 b)")
    res = {nm: {p: [] for p in F.PROJ} for nm in names}
    for L in LAYERS:
        X = np.load(os.path.join(D, f"x_L{L}.npy")).astype(np.float64)
        Hx = X.T @ X
        UDx = prep(Hx)                                  # shared by every gate / up matrix of this layer
        es = sorted(rng.choice(512, NE, replace=False).tolist())
        for e in es:
            rec = F.read(L, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                o, K = F.SH[p]
                m, l = r["m"], r["l"]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = np.repeat(s3 / 2, 64, 1) * (2 * m + l) + np.repeat(bb, 64, 1)
                W3[p] = np.repeat(s3, 64, 1) * m + np.repeat(b3, 64, 1)
            a = X @ W4["gate"].T
            h = (a / (1 + np.exp(-a))) * (X @ W4["up"].T)
            Hh = h.T @ h
            UDh = prep(Hh)
            for p in F.PROJ:
                H, UD = (Hh, UDh) if p == "down" else (Hx, UDx)
                for nm, Wq in zip(names, (W3[p], rtn(W4[p], 2, 64), gptq(W4[p], UD, 2, 64), gptq(W4[p], UD, 2, 128),
                                          gptq(W4[p], UD, 3, 64))):
                    res[nm][p].append(err(Wq, W4[p], H))
        print(f"[q2] layer {L} done ({X.shape[0]} tokens, {time.time() - t0:.0f} s)", flush=True)
    base = {p: np.mean(res[names[0]][p]) for p in F.PROJ}
    print("format | gate | up | down | mean vs MSB-3", flush=True)
    for nm in names:
        v = {p: np.mean(res[nm][p]) for p in F.PROJ}
        rel = np.mean([v[p] / base[p] for p in F.PROJ])
        print(f"{nm} | {v['gate']:.4f} | {v['up']:.4f} | {v['down']:.4f} | {rel:.2f}x", flush=True)


if __name__ == "__main__":
    main()
