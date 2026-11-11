"""LEAD1CORR (MTP Head, Tue 11-03 evening; PREP for the WALL1 sprint's 'learned L+1 predictor, only if the pre-route misses
its line', CPU, exploratory: no line): how much a learned correction adds to the lead-1 forecast on BigMoE's 45 GB capture
(~/BigMoE/data/cap45: 6 prompts, every row's MoE input x at every layer and its true top-10). Plain = layer L+1's router on
layer L's MoE input (R-57's forecast; per-row recall@10 0.611 in lead1_recall.py). Corrected (SlotStream / CORR style): a
per-layer ridge map x_L -> x_{L+1} (2560 x 2560, residual form: predict x_{L+1} - x_L), then L+1's router on the forecast.
Leave-one-prompt-out (fit on 5 prompts, score the 6th); ridge lambda chosen on an inner prompt of the 5. Metric: per-row
recall@10 / @16 of L+1's true top-10, layers 1-46 (L+1 = 2..47), pooled. Output: data/lead1_corr.txt (appended)."""
import os, glob
import numpy as np
D = os.path.expanduser("~/BigMoE/data/cap45")
TD = os.path.expanduser("~/TrainLab/data")
NL, K = 48, 10
LAMS = (1e2, 1e3, 1e4)
RT = {int(k[1:]): v.astype(np.float32) for k, v in np.load(os.path.join(D, "routers.npz")).items()}   # L -> [E, 2560]

def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

# per prompt and layer: the MoE inputs of every row of every full check (first call per (check, layer)), row-aligned
P = []
for f in sorted(glob.glob(os.path.join(D, "*_gate_0.npz"))):
    z = np.load(f)
    by = {}
    for j, (s, L) in enumerate(zip(z["seq"].tolist(), z["L"].tolist())):
        by.setdefault(s, {}).setdefault(L, j)
    seqs = [s for s in sorted(by) if all(L in by[s] for L in range(NL))
            and len({int(z["rows"][by[s][L]]) for L in range(NL)}) == 1]
    X = {L: [] for L in range(NL)}; I = {L: [] for L in range(NL)}
    xr, ids = z["x"], z["ids"]
    for s in seqs:
        for L in range(NL):
            j = by[s][L]; o, r, xo = int(z["row_off"][j]), int(z["rows"][j]), int(z["x_off"][j])
            X[L].append(xr[xo:xo + r]); I[L].append(ids[o:o + r])
    P.append((os.path.basename(f), {L: np.concatenate(X[L]) for L in range(NL)}, {L: np.concatenate(I[L]).astype(np.int64) for L in range(NL)}))
print("loaded", [(n, len(I[1])) for n, _, I in P], flush=True)

def recall(scores, ids, k):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top, ids)) / (K * len(ids))

def fit(Xs, Ys, lam):
    X = np.concatenate(Xs); Y = np.concatenate(Ys) - X
    G = X.T @ X; G[np.diag_indices_from(G)] += lam
    return np.linalg.solve(G, X.T @ Y)

lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
tot = {"plain10": [0.0, 0], "corr10": [0.0, 0], "plain16": [0.0, 0], "corr16": [0.0, 0]}
for h in range(len(P)):
    tr = [p for i, p in enumerate(P) if i != h]; inner, rest = tr[0], tr[1:]
    hn, hX, hI = P[h]
    for L in range(1, NL - 1):
        W1 = RT[L + 1]
        xs = [bf16(p[1][L]) for p in rest]; ys = [bf16(p[1][L + 1]) for p in rest]
        xi, ii = bf16(inner[1][L]), inner[2][L + 1]
        best = max(LAMS, key=lambda lam: recall((xi + xi @ fit(xs, ys, lam)) @ W1.T, ii, K))
        Wc = fit(xs + [xi], ys + [bf16(inner[1][L + 1])], best)
        xh = bf16(hX[L]); ih = hI[L + 1]
        sp, sc = xh @ W1.T, (xh + xh @ Wc) @ W1.T
        for nm, s_, k in (("plain10", sp, 10), ("corr10", sc, 10), ("plain16", sp, 16), ("corr16", sc, 16)):
            tot[nm][0] += recall(s_, ih, k) * len(ih); tot[nm][1] += len(ih)
    say(f"  held out {hn}: running per-row recall@10 plain {tot['plain10'][0] / tot['plain10'][1]:.3f} / corrected "
        f"{tot['corr10'][0] / tot['corr10'][1]:.3f}")
say(f"LEAD1CORR (cap45, leave-one-prompt-out, layers L+1 = 2..47): per-row recall of L+1's true top-10: @10 plain "
    f"{tot['plain10'][0] / tot['plain10'][1]:.3f} -> corrected {tot['corr10'][0] / tot['corr10'][1]:.3f}; @16 plain "
    f"{tot['plain16'][0] / tot['plain16'][1]:.3f} -> corrected {tot['corr16'][0] / tot['corr16'][1]:.3f} (the pre-route's line: @10 >= 0.85)")
open(os.path.join(TD, "lead1_corr.txt"), "a").write("\n".join(lines) + "\n\n")
