"""TrainLab (CPU): how much can ROUTING HISTORY add to the 125B's training-free look-ahead for the top-2 experts of the
next layer? On MoEPrefetch's decode traces (data/traces125/{code,chat,bulgarian,math}_400_pf.npz: per layer L the true
top-20 ids, and BigMoE's look-ahead top-32 for L+1 (P1_L) and L+2 (P2_L)). A re-ranker over layer L's 32 look-ahead
candidates (from P1_{L-1}) scores each candidate with: its tap rank, its rank in the d=2 look-ahead (P2_{L-2}), whether
it is in the previous token's true top-2 / top-10 at L, and a co-occurrence score from this token's true top-10 at L-1
(log-odds of being in L's top-2 given each L-1 expert, counted on the training prompts). Logistic regression,
leave-one-prompt-out. Metric: top-2 precision (|pred top-2 & true top-2| / 2) on layers 8-39, vs the tap's top-2.
Upper bound shown: oracle re-ranking of the 32 candidates (true top-2 inside the candidates)."""
import os
import numpy as np

D = os.path.expanduser("~/MoEPrefetch/data/traces125")
PROMPTS = ["code", "chat", "bulgarian", "math"]
LAYERS = range(8, 40)
E = 512


def load(p):
    z = np.load(os.path.join(D, f"{p}_400_pf.npz"))
    return {k: z[k].astype(np.int64) for k in z.files}


def cooc(tr, L):
    """log-odds table [E(L-1), E(L)]: expert e' in top-10 at L-1 -> e in top-2 at L (smoothed), from training traces."""
    c = np.ones((E, E)); n = np.ones(E) * 2
    for z in tr:
        a, b = z[f"L{L - 1}"][:, :10], z[f"L{L}"][:, :2]
        for t in range(a.shape[0]):
            for e1 in a[t]:
                n[e1] += 1
                c[e1, b[t]] += 1
    p = c / (n[:, None] + E * 2 / 512)
    base = c.sum(0) / c.sum()
    return np.log(p) - np.log(base[None, :])


def features(z, L, co):
    cand = z[f"P1_{L - 1}"]                                       # [T, 32] look-ahead candidates for layer L
    T = cand.shape[0]
    rank1 = np.tile(np.arange(32), (T, 1)).astype(np.float32)
    p2 = z.get(f"P2_{L - 2}")
    rank2 = np.full((T, 32), 32.0, np.float32)
    if p2 is not None:
        for t in range(T):
            pos = {e: i for i, e in enumerate(p2[t])}
            rank2[t] = [pos.get(e, 32) for e in cand[t]]
    prev2 = np.zeros((T, 32), np.float32); prev10 = np.zeros((T, 32), np.float32)
    tl = z[f"L{L}"]
    for t in range(1, T):
        s2, s10 = set(tl[t - 1, :2]), set(tl[t - 1, :10])
        prev2[t] = [e in s2 for e in cand[t]]; prev10[t] = [e in s10 for e in cand[t]]
    cs = np.zeros((T, 32), np.float32)
    up = z[f"L{L - 1}"][:, :10]
    for t in range(T):
        cs[t] = co[up[t]][:, cand[t]].sum(0)
    F = np.stack([rank1, np.log1p(rank1), rank2, np.log1p(rank2), prev2, prev10, cs], -1)   # [T, 32, 7]
    y = np.zeros((T, 32), np.float32)
    for t in range(T):
        s = set(tl[t, :2]); y[t] = [e in s for e in cand[t]]
    return F, y, cand, tl[:, :2]


def prec2(scores, cand, true2):
    top = np.argsort(-scores, axis=1)[:, :2]
    pick = np.take_along_axis(cand, top, 1)
    return np.mean([len(set(a) & set(b)) for a, b in zip(pick, true2)]) / 2


def logreg(X, y, steps=300, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(steps):
        z = Xs @ w + b; p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (Xs.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return lambda Xn: ((Xn - mu) / sd) @ w + b


def main():
    Z = {p: load(p) for p in PROMPTS}
    res = {"tap": [], "rerank": [], "oracle32": []}
    for hold in PROMPTS:
        tr = [Z[p] for p in PROMPTS if p != hold]
        for L in LAYERS:
            co = cooc(tr, L)
            Xs, ys = [], []
            for z in tr:
                F, y, _, _ = features(z, L, co)
                Xs.append(F.reshape(-1, F.shape[-1])); ys.append(y.reshape(-1))
            f = logreg(np.concatenate(Xs), np.concatenate(ys))
            F, y, cand, true2 = features(Z[hold], L, co)
            T = F.shape[0]
            s = f(F.reshape(-1, F.shape[-1])).reshape(T, 32)
            res["tap"].append(prec2(-np.tile(np.arange(32), (T, 1)), cand, true2))
            res["rerank"].append(prec2(s, cand, true2))
            res["oracle32"].append(np.mean(y.sum(1).clip(0, 2)) / 2)
        n = len(LAYERS)
        print(f"held-out {hold:9s}: tap prec2 {np.mean(res['tap'][-n:]):.3f} | re-ranked {np.mean(res['rerank'][-n:]):.3f} "
              f"| true top-2 inside the 32 candidates {np.mean(res['oracle32'][-n:]):.3f}", flush=True)
    print(f"ALL layers 8-39: tap {np.mean(res['tap']):.3f}  re-ranked {np.mean(res['rerank']):.3f}  oracle-32 {np.mean(res['oracle32']):.3f}")


if __name__ == "__main__":
    main()
