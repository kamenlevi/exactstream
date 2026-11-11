"""Can the cache recover frugal's missed gate mass for free? (Breakthrough, 2026-10-20 22:10.) Frugal wait-top-2 replay
on the 125B traces with router probabilities (data/expA/traces125/*_800_g.npz), per layer capacity C, only waited
experts are loaded; missing ranks 2-9 are skipped (no substitution here: the mass that is NOT computed is the metric).
Eviction policies (victim = a resident expert not used by the current token):
  lru      : least recently used (uses = computed experts)
  lfu-m    : lowest decayed routed mass: score = d*score + p_r for every expert in the token's top-20 (resident or not)
  belady   : oracle, evict the expert whose next appearance in a top-10 is farthest (upper bound for prediction)
  belady-m : oracle, evict the lowest future gate mass over the next H tokens (mass-aware bound)
Prints missed top-10 mass per token-layer (mean, P99) and loads/token.  usage: evict_sim.py [C ...]"""
import bisect, glob, sys
import numpy as np

NL = 48
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}


def run(C, pol, decay=0.97, H=64):
    loads = ntok = 0
    missed = []
    for k in ids:
        tr, wt, g20 = ids[k], w[k], gp[k]
        T = tr.shape[0]; ntok += T
        for l in range(NL):
            occ = {}
            for t in range(T):
                for r in range(10):
                    occ.setdefault(int(tr[t, l, r]), []).append(t)
            res, score, last = set(), {}, {}
            for t in range(T):
                row = [int(e) for e in tr[t, l, :10]]
                if pol == "lfu-m":
                    for e, p in zip(tr[t, l], g20[t, l]):
                        e = int(e); score[e] = score.get(e, 0.0) * 1.0 + float(p)   # decay applied lazily below
                used, miss = set(), 0.0
                for r, e in enumerate(row):
                    if r < 2:
                        if e not in res:
                            loads += 1
                            if len(res) >= C:
                                cand = [x for x in res if x not in used and x not in row[:2]]
                                if pol == "lru":
                                    v = min(cand, key=lambda x: last.get(x, -1))
                                elif pol == "lfu-m":
                                    v = min(cand, key=lambda x: score.get(x, 0.0))
                                elif pol == "belady":
                                    def nxt(x):
                                        o = occ.get(x, []); i = bisect.bisect_right(o, t)
                                        return o[i] if i < len(o) else 10 ** 9
                                    v = max(cand, key=nxt)
                                else:   # belady-m: future routed mass in the next H tokens
                                    def fm(x):
                                        o = occ.get(x, []); i = bisect.bisect_right(o, t); j = bisect.bisect_right(o, t + H)
                                        return sum(float(wt[s, l][list(tr[s, l, :10]).index(x)]) for s in o[i:j])
                                    v = min(cand, key=fm)
                                res.discard(v)
                            res.add(e)
                        used.add(e)
                    elif e in res:
                        used.add(e)
                    else:
                        miss += float(wt[t, l, r])
                for e in used:
                    last[e] = t
                if pol == "lfu-m":
                    for e in list(score):
                        score[e] *= decay
                missed.append(miss)
    m = np.array(missed)
    return loads / ntok, m.mean(), np.percentile(m, 99)


for C in [int(a) for a in sys.argv[1:]] or [51, 26]:
    print(f"C={C} ({100 * C / 512:.0f} %)", flush=True)
    for pol in ("lru", "lfu-m", "belady", "belady-m"):
        ld, mm, p99 = run(C, pol)
        print(f"  {pol:9s} loads {ld:5.1f}/token ({ld * 2.77:4.0f} MB)  missed mass {mm:.3f} / P99 {p99:.3f}", flush=True)
