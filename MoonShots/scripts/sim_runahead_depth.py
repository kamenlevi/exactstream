"""Can runahead make SSD streaming nearly free? Runahead only runs while a layer stalls, so as it removes stalls it also removes
its own trigger. This models the equilibrium on the MoEPrefetch proxy traces (35B: 40 layers x 256 experts, top-8; S=3 verify
unions; per-layer LRU at 55 %; prefetched experts inserted at the LRU end), with the measured SSD curve and per-layer timing.
Per layer L (in order): attention, router, stall if a needed expert is neither resident nor landed, MoE.
  look-ahead (always on, BigMoE v1): at L's router, issue L+1's top-B prediction (pmid1 from the traces, union over rows).
  runahead depth d (only when L stalls): RA ms into the stall, issue the *exact* needs of L+1..L+d (measured recall 0.96-1.0).
SSD: n reads issued together land at t + 0.2 + 0.25 n ms (measured: 1/2/3/4 experts -> 0.45/0.71/0.99/1.17 ms), FIFO across batches.
Stall time = until the last needed expert lands, + 0.07 ms gate. Prints stalled layers and stall ms per pass (x48/40 to the 125B).
Usage: sim_runahead_depth.py   (CPU only)"""
import os
from collections import OrderedDict
import numpy as np

TR = os.path.expanduser("~/MoEPrefetch/traces")
NL, K, E, S = 40, 8, 256, 3
zs = [dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in ("chat", "code", "realcode")]
C = int(0.55 * E)
LAYER = 38.0 / 48            # ms per layer of compute in an S=3 verify pass (BigMoE: 38 ms)
ATTN = 0.5 * LAYER           # share before the router
RA = 0.6                     # ms into a stall until runahead knows the next layer's routing (+0.8 per further layer)
GATE, LAT, PER = 0.07, 0.2, 0.25
PCAP = 6                     # at most this many prefetch reads per layer (BigMoE: bounded by slots)


def run(depth, B=12, lookahead=True, passes=300):
    stl, ms, n = 0, 0.0, 0
    for z in zs:
        T = len(z["doc"])
        acts = [z[f"L{L}_act"][:, :K] for L in range(NL)]
        p1 = [z[f"L{L}_pmid1"][:, :B] for L in range(NL - 1)]
        lru = [OrderedDict() for _ in range(NL)]
        for t0 in range(200, min(T - S, 200 + S * passes), S):
            n += 1
            now, ssd = 0.0, 0.0
            land = [dict() for _ in range(NL)]        # expert -> landing time, this pass

            def issue(L, es, t, demand=False, cap=None):
                nonlocal ssd
                es = [e for e in es if e not in lru[L] and e not in land[L]]
                if cap is not None:
                    es = es[:cap]
                if demand:                            # demand reads go first (software priority): no queueing
                    for i, e in enumerate(es):
                        land[L][e] = t + LAT + PER * (i + 1)
                    return
                start = max(t, ssd)                   # prefetches: FIFO among themselves
                for i, e in enumerate(es):
                    land[L][e] = start + LAT + PER * (i + 1)
                if es:
                    ssd = start + PER * len(es)
                for e in es:                          # low-priority insert: evict the current LRU victim, take its place
                    while len(lru[L]) >= C:
                        lru[L].popitem(last=False)
                    lru[L][e] = 1
                    lru[L].move_to_end(e, last=False)
            need = [list(dict.fromkeys(acts[L][t0:t0 + S].reshape(-1).tolist())) for L in range(NL)]
            for L in range(NL):
                now += ATTN
                if lookahead and L + 1 < NL:
                    issue(L + 1, list(dict.fromkeys(p1[L][t0:t0 + S].reshape(-1).tolist())), now, cap=PCAP)
                miss = [e for e in need[L] if e not in lru[L] and e not in land[L]]
                issue(L, miss, now, demand=True)
                ready = max([land[L].get(e, 0.0) for e in need[L]] + [now])
                stall = ready - now
                if stall > 0:
                    stall += GATE
                    stl += 1
                    ms += stall
                    for d in range(1, depth + 1):     # runahead during the stall, if it lasts long enough
                        t_known = now + RA + 0.8 * (d - 1)
                        if L + d < NL and t_known < now + stall:
                            issue(L + d, need[L + d], t_known, cap=PCAP)
                now += max(stall, 0.0)
                for e in need[L]:                     # true use: promote to MRU
                    lru[L][e] = 1
                    lru[L].move_to_end(e, last=True)
                while len(lru[L]) > C:
                    lru[L].popitem(last=False)
                now += LAYER - ATTN
    sc = 48 / NL
    return stl / n * sc, ms / n * sc


print("S=3 verify pass on proxy traces, 55 % resident; per pass, scaled to 48 layers (BigMoE measured: none 36.5 stalled / "
      "40.5 ms, look-ahead B=12 27 stalled / 26 ms)")
for name, kw in (("no prefetch", dict(depth=0, lookahead=False)), ("look-ahead B=12", dict(depth=0)),
                 ("+ runahead d=1", dict(depth=1)), ("+ runahead d=2", dict(depth=2)), ("+ runahead d=3", dict(depth=3)),
                 ("runahead d=2 only", dict(depth=2, lookahead=False))):
    sl, ms = run(**kw)
    print(f"  {name:18s}: stalled layers {sl:5.1f}, stall {ms:5.1f} ms/pass", flush=True)
