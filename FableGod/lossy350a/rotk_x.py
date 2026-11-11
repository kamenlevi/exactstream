"""LOSSY350-A replay 1 (Fable A, Thu 5 Nov 2026): K-prefix in the layer's rotated input basis, on REAL MoE inputs.
Inputs: Fit12's Q2 capture ~/Fit12/data/q2/x_L{2,12,24,36,46}.npy (8,336 tokens x 2560, the MoE block inputs; read only).
No weights, no GPU. numpy, one process, 4 threads, nice 15.
For each layer: H = X^T X on a TRAIN split (first 5,836 tokens), eigenbasis V (descending); held-out rows (last 2,500).
x' = V^T x. Per row: tail energy share t(r) = ||x'_{>r}||^2 / ||x||^2 at r = 64, 128, ..., 2560 (the kernel's K-groups).
Reports, held out:
 (a) fixed r: mean tail share at r = 512, 768, 1024, 1280, 1536, 1792, 2048, 2304 (compare Leap act16: top-512 0.78-0.88,
     top-1024 0.90-0.95 in sample);
 (b) per-row adaptive prefix: smallest r (multiple of 64) with t(r) <= eps, for eps in a list; mean kept fraction, p10/p50/p90;
     and the mean achieved tail (<= eps by construction);
 (c) 2-row groups: mean of max(r_a, r_b) over random held-out pairs;
 (d) the unrotated basis for reference: per-row energy share of the top-k |x_j| coordinates (dynamic top-k, TEAL-x style)
     and of a STATIC coordinate order (by train energy);
 (e) in-sample vs held-out gap at the same r (basis stability).
"""
import os, sys, time, numpy as np
XD = os.path.expanduser("~/Fit12/data/q2")
LAYERS = (2, 12, 24, 36, 46)
EPS = (0.002, 0.005, 0.01, 0.017, 0.03, 0.058, 0.08, 0.13)
RS = (512, 768, 1024, 1280, 1536, 1792, 2048, 2304)
t0 = time.time()
rng = np.random.default_rng(350)
out = []
def say(s):
    print(s, flush=True); out.append(s)
say("ROT-K replay 1: real MoE inputs (Fit12 Q2), train 5,836 / held-out 2,500 tokens per layer; K-groups of 64")
summary = {}
for L in LAYERS:
    X = np.load(os.path.join(XD, f"x_L{L}.npy")).astype(np.float64).reshape(-1, 2560)
    n = X.shape[0]; ntr = n - 2500
    Xtr, Xte = X[:ntr], X[ntr:]
    H = Xtr.T @ Xtr
    lam, V = np.linalg.eigh(H); V = V[:, ::-1]; lam = lam[::-1]
    def tails(Xs):
        Xp = Xs @ V                                   # rotated rows
        e = Xp * Xp
        g = e.reshape(len(Xs), 40, 64).sum(-1)        # energy per K-group of 64
        tot = g.sum(-1, keepdims=True)
        cum = np.cumsum(g, -1) / tot                  # cum[:, i] = share in the first (i+1) groups
        return 1.0 - cum, g / tot                     # tail after (i+1) groups ; per-group share
    Tte, Gte = tails(Xte); Ttr, _ = tails(Xtr)
    say(f"\nlayer {L}: n {n}; rms row norm {np.sqrt((X*X).sum(-1)).mean():.2f}; in-sample top-512 / top-1024 energy "
        f"{lam[:512].sum()/lam.sum():.4f} / {lam[:1024].sum()/lam.sum():.4f}")
    say("  (a) fixed r, mean tail share held out (in sample): " +
        "  ".join(f"r{r} {Tte[:, r//64-1].mean():.4f} ({Ttr[:, r//64-1].mean():.4f})" for r in RS))
    say("      held-out p90 tail at the same r: " + "  ".join(f"r{r} {np.quantile(Tte[:, r//64-1], 0.9):.4f}" for r in RS))
    say("  (b) per-row adaptive prefix (held out): eps | mean kept K | p10 / p50 / p90 kept | mean achieved tail | fixed-r kept for the same MEAN tail")
    for eps in EPS:
        ok = Tte <= eps                                # [rows, 40]
        first = np.where(ok.any(-1), ok.argmax(-1) + 1, 40)   # groups kept
        kept = first / 40.0
        ach = Tte[np.arange(len(Tte)), first - 1]
        mt = Tte.mean(0)                               # mean tail vs groups
        fx = int(np.argmax(mt <= ach.mean()) + 1) if (mt <= ach.mean()).any() else 40
        say(f"      {eps:5.3f} | {kept.mean():.3f} | {np.quantile(kept,0.1):.3f} / {np.quantile(kept,0.5):.3f} / {np.quantile(kept,0.9):.3f}"
            f" | {ach.mean():.4f} | {fx/40:.3f}")
        summary.setdefault(eps, []).append((kept.mean(), fx / 40.0))
        if eps in (0.017, 0.058):
            a = rng.integers(0, len(first), 20000); b = rng.integers(0, len(first), 20000)
            say(f"            2-row groups at eps {eps}: mean max kept {np.maximum(first[a], first[b]).mean()/40:.3f}")
    # (d) unrotated basis
    E = Xte * Xte; tot = E.sum(-1, keepdims=True)
    Es = -np.sort(-E, axis=-1); cs = np.cumsum(Es, -1) / tot
    order = np.argsort(-(Xtr * Xtr).sum(0)); cst = np.cumsum(E[:, order], -1) / tot
    say("  (d) unrotated: dynamic top-k |x_j| tail share: " + "  ".join(f"k{k} {1-cs[:, k-1].mean():.4f}" for k in (640, 1280, 1600, 1920)) +
        " ; static order: " + "  ".join(f"k{k} {1-cst[:, k-1].mean():.4f}" for k in (640, 1280, 1920)))
    # 64-blocks in the unrotated basis, dynamic choice of blocks (what a block-skip kernel could do without rotation)
    Gb = E.reshape(len(E), 40, 64).sum(-1) / tot; Gs = -np.sort(-Gb, -1); cb = np.cumsum(Gs, -1)
    say("      unrotated 64-blocks, dynamic best blocks tail share: " + "  ".join(f"{k} blocks {1-cb[:, k-1].mean():.4f}" for k in (10, 20, 30)))
say("\nSUMMARY over layers (mean kept K fraction, adaptive | fixed r at the same mean tail):")
for eps in EPS:
    v = np.array(summary[eps]); say(f"  eps {eps:5.3f}: adaptive {v[:,0].mean():.3f} | fixed {v[:,1].mean():.3f}")
say(f"done in {time.time()-t0:.0f} s")
open(os.path.expanduser("~/FableGod/lossy350a/rotk_x.out"), "w").write("\n".join(out) + "\n")
