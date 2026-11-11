"""Exact-cache replacement policies vs LRU on the proxy traces (no prefetch: the target design
forbids mid-token prefetch). Metric: stall-causing misses and stalled layers per pass.
Policies: lru; lfu_decay (score = decayed use count); 2q (new experts enter a probation FIFO,
promoted on 2nd use); gdsf-ish (LRU tie-broken by recent frequency). -> data/sim_policy.txt"""
import os, sys
from collections import OrderedDict, defaultdict
import numpy as np
TR = os.path.expanduser("~/MoEPrefetch/traces"); NL, K, E = 40, 8, 256
SETS = ["chat", "code", "bulgarian", "longctx", "realcode"]
zs = [dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in SETS]


class LRU:
    def __init__(s, C): s.C = C; s.d = OrderedDict()
    def need(s, es):
        miss = [e for e in es if e not in s.d]
        for e in es:
            if e in s.d: s.d.move_to_end(e)
        for e in miss:
            while len(s.d) >= s.C:
                for v in s.d:
                    if v not in es: del s.d[v]; break
            s.d[e] = None
        return len(miss)


class LFUDecay:
    def __init__(s, C, decay=0.97): s.C = C; s.f = defaultdict(float); s.res = set(); s.decay = decay
    def need(s, es):
        for e in list(s.f): s.f[e] *= s.decay
        for e in es: s.f[e] += 1.0
        miss = [e for e in es if e not in s.res]
        for e in miss:
            if len(s.res) >= s.C:
                v = min((x for x in s.res if x not in es), key=lambda x: s.f[x]); s.res.discard(v)
            s.res.add(e)
        return len(miss)


class TwoQ:
    def __init__(s, C, frac_in=0.25): s.C = C; s.cin = max(1, int(C * frac_in)); s.a1 = OrderedDict(); s.am = OrderedDict()
    def need(s, es):
        miss = 0
        for e in es:
            if e in s.am: s.am.move_to_end(e)
            elif e in s.a1: del s.a1[e]; s.am[e] = None          # second use: promote
            else: miss += 1; s.a1[e] = None
        while len(s.a1) + len(s.am) > s.C:
            src = s.a1 if len(s.a1) > s.cin or not s.am else s.am
            for v in src:
                if v not in es: del src[v]; break
            else:
                src = s.am if src is s.a1 else s.a1
                for v in src:
                    if v not in es: del src[v]; break
        return miss


def run(pol, frac, S, start=200):
    C = round(frac * E); st = sl = n = 0
    for z in zs:
        caches = [pol(C) for _ in range(NL)]
        T = len(z["doc"])
        acts = [z[f"L{L}_act"][:, :K] for L in range(NL)]
        for t0 in range(0, T - S + 1, S):
            for L in range(NL):
                es = list(dict.fromkeys(acts[L][t0:t0 + S].reshape(-1).tolist()))
                m = caches[L].need(es)
                if t0 >= start: st += m; sl += m > 0
            if t0 >= start: n += 1
    return st / n, sl / n


lines = []
for S in (1, 3):
    for name, pol in [("lru", LRU), ("lfu_decay", LFUDecay), ("2q", TwoQ)]:
        st, sl = run(pol, 0.6, S)
        lines.append(f"60% S={S} {name:9s}: stall loads/pass {st:6.2f}, stalled layers/pass {sl:5.2f} (of 40)")
        print(lines[-1], flush=True)
open(os.path.expanduser("~/BigMoE/data/sim_policy.txt"), "w").write("\n".join(lines) + "\n")
