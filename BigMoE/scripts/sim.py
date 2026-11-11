"""Lossless expert-cache simulation on Qwen3.6-35B-A3B routing traces (proxy for the
target's 512-expert/top-10 routing), at the target's resident fraction of experts.
Policies: LRU; LRU + look-ahead prefetch (layer L-1's residual through layer L's router,
top-B); verify batches of S consecutive tokens (MTP speculative decoding: one pass reads
the union of S tokens' experts). Every needed expert must be resident before its layer
runs (no substitution => bit-exact). Output: per-step stats -> data/sim.json + summary."""
import json, os, sys
from collections import OrderedDict
import numpy as np
TR = os.path.expanduser("~/MoEPrefetch/traces"); NL, K, E = 40, 8, 256
SETS = ["chat", "code", "bulgarian", "longctx", "realcode"]
def sim(z, C, B, S, start=200):
    T = len(z["doc"]); steps = range(start - start % S, T - S + 1, S)
    n = len(steps); stall = np.zeros(n); slay = np.zeros(n); pre = np.zeros(n); uni = np.zeros(n)
    pre_lay_max = np.zeros(n)
    for L in range(NL):
        act = z[f"L{L}_act"][:, :K]
        P = z[f"L{L-1}_pmid1"][:, :B] if (B and L >= 1) else None
        cache = OrderedDict()
        def touch(e):
            if e in cache: cache.move_to_end(e); return True
            cache[e] = None
            if len(cache) > C: cache.popitem(last=False)
            return False
        # warm the cache on the first `start` tokens
        for t in range(0, steps[0]):
            for e in act[t].tolist(): touch(e)
        for i, t0 in enumerate(steps):
            need = list(dict.fromkeys(act[t0:t0+S].reshape(-1).tolist()))
            uni[i] += len(need)
            p = 0
            if P is not None:
                for e in dict.fromkeys(P[t0:t0+S].reshape(-1).tolist()):
                    if not touch(e): p += 1
            m = sum(0 if touch(e) else 1 for e in need)
            stall[i] += m; slay[i] += m > 0; pre[i] += p; pre_lay_max[i] = max(pre_lay_max[i], p)
    return dict(stall=stall.mean(), stall_layers=slay.mean(), prefetch=pre.mean(),
                union=uni.mean() / NL, p95_stall=float(np.percentile(stall, 95)))
if __name__ == "__main__":
    fracs = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.5, 0.6, 0.7]
    zs = {s: dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in SETS}
    out = []
    for f in fracs:
        C = round(f * E)
        for B in (0, 8, 12, 16):
            for S in (1, 2, 3, 4):
                if B and S > 1 and B != 12: continue
                r = [sim(z, C, B, S) for z in zs.values()]
                m = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
                m.update(frac=f, C=C, B=B, S=S, per_set={s: round(x["stall"], 2) for s, x in zip(zs, r)})
                out.append(m)
                print(f"f={f:.2f} C={C} B={B:2d} S={S}: union/layer {m['union']:5.2f}  stall exp/step {m['stall']:6.2f} "
                      f"(of {K*NL*S}) stalled layers {m['stall_layers']:5.2f}  prefetch exp/step {m['prefetch']:6.2f}", flush=True)
    json.dump(out, open(os.path.expanduser("~/BigMoE/data/sim.json"), "w"), indent=1)
