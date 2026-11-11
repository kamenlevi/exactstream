"""Branch-predictor-style next-token expert prediction, trained online (no offline training).
For token t+1 at layer L, predict the top-8 experts from token t's routing at layers L-1, L, L+1
(available a whole token ahead). Models learn as they go, like a CPU's perceptron prefetcher.
Baseline (MoEPrefetch step 6): token t's own top-16 at layer L, recall ~0.51 at a 16-expert budget.
Uses ~/MoEPrefetch/traces/*.npz (Qwen3.6-35B-A3B, 40 layers x 256 experts, top-8)."""
import sys
import numpy as np

E, NL, B = 256, 40, 16
sets = sys.argv[1:] or ["chat", "code", "bulgarian", "realcode"]


def run(name):
    z = np.load(f"/Users/kamen/MoEPrefetch/traces/{name}.npz")
    doc = z["doc"]
    act = np.stack([z[f"L{l}_act"][:, :8] for l in range(NL)], 1).astype(np.int64)   # (T, 40, 8)
    act16 = np.stack([z[f"L{l}_act"] for l in range(NL)], 1).astype(np.int64)        # router top-16
    T = len(doc)
    # features: experts of token t at layers L-1, L, L+1 -> index (k*E + e), k in 0..2
    R = int(__import__("os").environ.get("R", "1")); C = np.zeros((NL, (2 * R + 1) * E, E), np.float32)      # co-occurrence counts f(t) -> e(t+1)
    W = np.zeros((NL, (2 * R + 1) * E, E), np.float32)      # online logistic weights
    bias = np.zeros((NL, E), np.float32)
    hits = {"temporal16": 0, "markov": 0, "logistic": 0, "markov+temporal": 0}
    n = 0
    lr = float(__import__("os").environ.get("LR", "0.05"))
    for t in range(T - 1):
        if doc[t] != doc[t + 1]:
            continue
        warm = t > 200
        for L in range(NL):
            feats = []
            for k, LL in enumerate(range(L - R, L + R + 1)):
                if 0 <= LL < NL:
                    feats.extend(k * E + act[t, LL])
            feats = np.array(feats)
            truth = act[t + 1, L]
            sm = C[L, feats].sum(0)
            sl = W[L, feats].sum(0) + bias[L]
            if warm:
                n += 8
                hits["temporal16"] += np.isin(truth, act16[t, L]).sum()
                hits["markov"] += np.isin(truth, np.argpartition(-sm, B)[:B]).sum()
                hits["logistic"] += np.isin(truth, np.argpartition(-sl, B)[:B]).sum()
                mix = np.concatenate([act[t, L], [e for e in np.argsort(-sm) if e not in act[t, L]][:B - 8]])
                hits["markov+temporal"] += np.isin(truth, mix).sum()
            # online updates
            C[L, feats[:, None], truth[None, :]] += 1
            y = np.zeros(E, np.float32); y[truth] = 1
            p = 1 / (1 + np.exp(-sl))
            g = (y - p) * lr
            W[L, feats] += g
            bias[L] += g
    print(f"{name:10s} tokens {T}: recall@{B} " + "  ".join(f"{k} {v/n:.3f}" for k, v in hits.items()), flush=True)


for s in sets:
    run(s)
