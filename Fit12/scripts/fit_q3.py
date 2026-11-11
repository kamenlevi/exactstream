"""Fit12 Q3 (2026-10-24): transform coding of gate / up on the layer's activation eigenbasis (Leap's L1-16 finding + reverse
water-filling). H = X^T X = V diag(lam) V^T from the Q2 captures (shared by all 512 experts of a layer). C = W4 V; the output
error tr(dW H dW^T) = sum_j lam_j |dC_j|^2 (H is diagonal in this basis, so GPTQ's error feedback vanishes: RTN is the
right quantizer). Leap's kernel layout (15:10): eigen-columns sorted by eigenvalue, groups of 64 columns, ONE bit width per
group, MLX-style bf16 scale + bias per (row, group); 0-bit tail groups are not stored. Group g gets b_g = clip(round(
log2(imp_g / theta) / 2), 0, 6), imp_g = mean over its columns of lam_j s_j^2; theta by bisection for an average budget B
(bits incl. the 32-bit scale + bias per 64 values of every coded group). Budgets
1.5 / 2.0 / 2.5 / 3.0 b vs MSB-3 (Q2: gate 0.0165, up 0.0219 at 3.5 b) and GPTQ-2 g64 (0.045 / 0.060 at 2.5 b). Same 40
experts as Q1 / Q2. Decision (sent to Leap 15:10): gate / up at MSB-3's error for <= 2 b -> continue jointly."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_q as F

D = os.path.expanduser("~/Fit12/data/q2")
LAYERS, NE, G = (2, 12, 24, 36, 46), 8, 128
BUDGETS = (1.5, 2.0, 2.5, 3.0)


def code(C, lam, B):
    """C [o, K] coefficients (columns sorted by lam, descending), lam [K]; per-64-column-group bit widths; returns
    (relative weighted error, achieved bits / weight, coded columns)"""
    o, K = C.shape
    ng = K // 64
    imp = (lam * ((C * C).mean(0) + 1e-30)).reshape(ng, 64).mean(1)

    def alloc(th):
        return np.clip(np.round(0.5 * np.log2(np.maximum(imp / th, 1e-30))), 0, 6).astype(int)

    def bits(b):
        return ((b * 64 * o).sum() + (b > 0).sum() * o * 32) / (o * K)
    lo, hi = np.log(imp.min() * 1e-6 + 1e-30), np.log(imp.max() * 1e6)
    for _ in range(60):                               # bisection on log theta: more theta -> fewer bits
        mid = 0.5 * (lo + hi)
        if bits(alloc(np.exp(mid))) > B:
            lo = mid
        else:
            hi = mid
    b = alloc(np.exp(hi))
    Cq = np.zeros_like(C)
    for g in range(ng):
        if b[g]:
            sl = slice(64 * g, 64 * g + 64)
            Cq[:, sl] = F.rtn(np.ascontiguousarray(C[:, sl]).astype(np.float32), int(b[g]), 64)
    dC = Cq - C
    return float((lam * (dC * dC).sum(0)).sum() / (lam * (C * C).sum(0)).sum()), bits(b), int((b > 0).sum()) * 64


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    res = {(p, B): [] for p in ("gate", "up") for B in BUDGETS}
    cols = {(p, B): [] for p in ("gate", "up") for B in BUDGETS}
    for L in LAYERS:
        X = np.load(os.path.join(D, f"x_L{L}.npy")).astype(np.float64)
        lam, V = np.linalg.eigh(X.T @ X)
        lam, V = np.maximum(lam[::-1], 0.0), V[:, ::-1]      # descending eigenvalues
        es = sorted(rng.choice(512, NE, replace=False).tolist())
        for e in es:
            rec = F.read(L, e)
            for p in ("gate", "up"):
                r = rec[p]
                s3, bb = F.bf16(r["s3u"]), F.bf16(r["bu"])
                W4 = np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)
                C = W4.astype(np.float64) @ V
                for B in BUDGETS:
                    er, bt, nc = code(C, lam, B)
                    res[(p, B)].append(er); cols[(p, B)].append(nc)
        print(f"[q3] layer {L} done ({time.time() - t0:.0f} s)", flush=True)
    ref = {"gate": 0.0165, "up": 0.0219}
    print("budget (b/w) | gate err | up err | mean vs MSB-3 (3.5 b) | coded columns of 2560 (gate / up)", flush=True)
    for B in BUDGETS:
        g, u = np.mean(res[("gate", B)]), np.mean(res[("up", B)])
        print(f"{B} | {g:.4f} | {u:.4f} | {0.5 * (g / ref['gate'] + u / ref['up']):.2f}x | "
              f"{np.mean(cols[('gate', B)]):.0f} / {np.mean(cols[('up', B)]):.0f}", flush=True)


if __name__ == "__main__":
    main()
