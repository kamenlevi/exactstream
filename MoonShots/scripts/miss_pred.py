"""Does the online next-token predictor (online_pred.py's logistic model) cut cache misses when used
as a whole-token-ahead prefetch source? Per-layer LRU cache holding FRAC of 256 experts (BigMoE's
exact design: every miss stalls). After token t, prefetch the predictor's top-P experts for token t+1
that aren't resident (evicting LRU), then serve token t+1. Counts demand misses and prefetch loads per token.
Usage: miss_pred.py set [FRAC] [P] [MAXTOK]"""
import sys
import numpy as np

E, NL = 256, 40
name = sys.argv[1]
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
P = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
CAP = int(round(FRAC * E))
z = np.load(f"/Users/kamen/MoEPrefetch/traces/{name}.npz")
doc = z["doc"][:MAXT]
act = np.stack([z[f"L{l}_act"][:MAXT, :8] for l in range(NL)], 1).astype(np.int64)
T = len(doc)


class LRU:
    def __init__(s):
        s.last = np.full((NL, E), -1e9)
        s.res = np.zeros((NL, E), bool)
        s.res[:, :CAP] = True                      # arbitrary warm start
        s.last[:, :CAP] = -1e8

    def touch(s, L, ids, now):
        """returns how many of ids were not resident; makes them resident."""
        miss = 0
        for e in ids:
            if not s.res[L, e]:
                miss += 1
                v = np.where(s.res[L], s.last[L], np.inf).argmin()
                s.res[L, v] = False
                s.res[L, e] = True
            s.last[L, e] = now
        return miss


base, pf = LRU(), LRU()
W = np.zeros((NL, 3 * E, E), np.float32); bias = np.zeros((NL, E), np.float32); lr = 0.05
m_base = m_pf = loads = useful = n = 0
for t in range(T - 1):
    same = doc[t] == doc[t + 1]
    warm = t > 300
    for L in range(NL):                          # serve token t
        mb = base.touch(L, act[t, L], t)
        mp = pf.touch(L, act[t, L], t)
        if warm:
            m_base += mb; m_pf += mp
    if warm:
        n += 1
    if not same:
        continue
    for L in range(NL):                          # predict token t+1, prefetch, learn
        feats = np.concatenate([k * E + act[t, LL] for k, LL in enumerate((L - 1, L, L + 1)) if 0 <= LL < NL])
        sl = W[L, feats].sum(0) + bias[L]
        pred = np.argpartition(-sl, P)[:P]
        new = [e for e in pred if not pf.res[L, e]]
        pf.touch(L, new, t + 0.5)                  # prefetched: more recent than token t's own uses
        if warm:
            loads += len(new)
            useful += np.isin(act[t + 1, L], new).sum()
        y = np.zeros(E, np.float32); y[act[t + 1, L]] = 1
        g = (y - 1 / (1 + np.exp(-sl))) * lr
        W[L, feats] += g; bias[L] += g
print(f"{name:9s} {FRAC:.0%} resident, P={P}: misses/token LRU {m_base/n:5.1f} -> with prefetch {m_pf/n:5.1f} "
      f"({1 - m_pf/m_base:+.0%} fewer); prefetch loads/token {loads/n:5.1f}, of which used next token {useful/max(loads,1):.0%}",
      flush=True)
