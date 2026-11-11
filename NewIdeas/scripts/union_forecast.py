"""Prefill read-ahead by a predicted UNION (NewIdeas 2026-10-22, CPU). On the 125B capture (tf_samelayer125.py: per decode
call and layer the true top-10 T, same-layer pre-attention forecast S, d=1 tap D and corrected DC, top-16 each), for chunks
of N consecutive tokens: the true union of experts per layer vs the union of each forecast's top-k per token. Reports the
share of the true union covered early (reads hidden) and the extra reads (forecast union minus true union), per layer."""
import glob, os
import numpy as np
NL = 48
seqs = []
for f in sorted(glob.glob(os.path.expanduser("~/NewIdeas/data/samelayer/S*_e16.npz"))):
    z = np.load(f)
    T = [z[f"T{l}"] for l in range(NL)]
    S = [z[f"S{l}"] for l in range(NL)]
    DC = [z[f"DC{l-1}"] if l > 0 else None for l in range(NL)]       # DC[l-1] forecasts layer l
    D = [z[f"D{l-1}"] if l > 0 else None for l in range(NL)]
    seqs.append((T, S, D, DC))
print("chunk N | forecast top-k | covered share of the true union | extra reads (x true union) | true union size")
for N in (16, 64, 150, 326):
    for name, idx, k in (("same-layer", 1, 10), ("same-layer", 1, 16), ("d=1 corrected", 3, 10), ("d=1 corrected", 3, 16), ("d=1 tap", 2, 16)):
        cov = []; extra = []; size = []
        for seq in seqs:
            T = seq[0]; F = seq[idx]
            n = T[0].shape[0]
            for s0 in range(0, n - N + 1, N):
                for l in range(1, NL):
                    tu = set(T[l][s0:s0 + N].ravel().tolist())
                    fu = set(F[l][s0:s0 + N, :k].ravel().tolist())
                    cov.append(len(tu & fu) / len(tu)); extra.append(len(fu - tu) / len(tu)); size.append(len(tu))
        print(f"  {N:4d} | {name:13s} top-{k:2d} | {np.mean(cov):.3f} | {np.mean(extra):.3f} | {np.mean(size):.0f}", flush=True)
