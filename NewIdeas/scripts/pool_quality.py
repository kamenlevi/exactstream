"""QUALITY POOL kill test (NewIdeas, Wed 4 Nov 2026 08:5x; pre-registered QUEUE 08:5x; CPU only, no GPU).
A stats-only twin of dm_oea_masks.masks()'s QOM pool under qom_0_live_bq's rule (cf 0, top-2 on sure rows, live cut,
the gated last row): per domain, the kept share of sure-row rank 2-9 pairs (count and MASS-weighted, ten_x_ledger's mean
router mass by rank) and the load misses, for several eviction / admission policies at the same slots per layer.
The loop mirrors the builder line for line (prefill warm-up, one pool per layer carried across a domain's windows, loads
= the check's union, resident = after the loads, resident uses refresh the pool on every row of the check); only the
pool's policy changes. LRU reproduces the builder's logs (self-check printed beside the table).
Usage (BigMoE venv): python pool_quality.py   -> ~/NewIdeas/data/ledger/pool_quality.txt
"""
import heapq, json, os, sys, time
import multiprocessing as mp
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_oea_masks as M  # noqa: E402
MASS = np.array([0.185, 0.134, 0.112, 0.098, 0.090, 0.083, 0.079, 0.076, 0.073, 0.071])
THR, KS, KU = 0.0, 2, 10


class LRU:
    def __init__(s, C): s.C, s.c = C, OrderedDict()
    def tick(s): pass
    def see(s, e, w): pass
    def has(s, e): return e in s.c
    def use(s, e, w):
        if e in s.c: s.c.move_to_end(e)
    def load(s, e, w):
        if e in s.c: s.c.move_to_end(e); return False
        s.c[e] = 1
        if len(s.c) > s.C: s.c.popitem(last=False)
        return True


class SLRU:                                   # probation (20 %) -> protected (80 %) on a second touch
    def __init__(s, C): s.C, s.pb, s.pt, s.np_ = C, OrderedDict(), OrderedDict(), max(1, int(round(0.8 * C)))
    def tick(s): pass
    def see(s, e, w): pass
    def has(s, e): return e in s.pb or e in s.pt
    def _promote(s, e):
        del s.pb[e]; s.pt[e] = 1
        if len(s.pt) > s.np_:
            d, _ = s.pt.popitem(last=False); s.pb[d] = 1; s.pb.move_to_end(d)
    def use(s, e, w):
        if e in s.pt: s.pt.move_to_end(e)
        elif e in s.pb: s._promote(e)
    def load(s, e, w):
        if s.has(e): s.use(e, w); return False
        s.pb[e] = 1
        while len(s.pb) + len(s.pt) > s.C: s.pb.popitem(last=False)
        return True


class LFU:                                    # decayed frequency; ghost=True scores every routed pair (mass-LFU)
    def __init__(s, C, d, mass, ghost):
        s.C, s.inf, s.g, s.mass, s.ghost = C, 1.0 / d, 1.0, mass, ghost
        s.sc, s.res, s.h = {}, set(), []
    def tick(s): s.g *= s.inf                 # older increments shrink relative to new ones
    def _add(s, e, w):
        v = s.sc.get(e, 0.0) + (w if s.mass else 1.0) * s.g; s.sc[e] = v
        if e in s.res: heapq.heappush(s.h, (v, e))
        if len(s.h) > 50 * s.C: s.h = [(s.sc[x], x) for x in s.res]; heapq.heapify(s.h)   # bound the lazy heap
    def see(s, e, w):
        if s.ghost: s._add(e, w)
    def has(s, e): return e in s.res
    def use(s, e, w):
        if not s.ghost and e in s.res: s._add(e, w)
    def _evict(s):
        while s.h:
            v, e = heapq.heappop(s.h)
            if e in s.res and s.sc[e] == v: s.res.discard(e); return
    def load(s, e, w):
        if e in s.res:
            if not s.ghost: s._add(e, w)
            return False
        if not s.ghost: s._add(e, w)
        s.res.add(e); heapq.heappush(s.h, (s.sc.get(e, 0.0), e))
        if len(s.res) > s.C: s._evict()       # may bypass the newcomer (admission by score)
        return True


POLICIES = {"lru": lambda C: LRU(C), "slru": lambda C: SLRU(C),
            "lfu_d99": lambda C: LFU(C, 0.99, False, False), "lfu_d999": lambda C: LFU(C, 0.999, False, False),
            "mlfu_d99": lambda C: LFU(C, 0.99, True, True), "mlfu_d999": lambda C: LFU(C, 0.999, True, True)}


def run(task):
    dom, pol, pool = task
    CL = M.pool_sizes() if pool == "m" else [int(pool)] * 48
    base = json.load(open(os.path.join(M.OUT, "baseline.json"))); G = M.dm_gate.load()
    Z = np.load(os.path.join(M.dm_gate.TD, M.dm_gate.WALK[dom]))
    P = [POLICIES[pol](CL[l]) for l in range(48)]
    n = mem = res = 0; wn = wmem = wres = 0.0; ld = ms = 0
    for i, nm in enumerate(str(x) for x in Z["names"]):
        if nm not in base: continue
        ids = np.load(os.path.join(M.OUT, "routing", nm))["ids"]; N = len(ids)
        g = np.nan_to_num(G[nm], nan=-1.0)
        for p in range(base[nm]["a"]):                             # prefill warms the pool
            for l in range(48):
                for r, e in enumerate(ids[p, l].tolist()): P[l].see(e, MASS[r]); P[l].load(e, MASS[r])
        for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            for (k, m, A, dc, nxt) in M.live_segments_nxt(k0_, m0_, A0_, dc0_):
                last, fm = min(A + m, N - 1), min(A + k, N - 1)
                rows, cfs = [], []
                for p in range(A, last + 1):
                    if p <= fm: rows.append(p); c_ = -1.0 if k == m and p == A + m else g[p]
                    else: rows.append((p + 97) % N); c_ = dc[p - A] if p - A < m else -1.0
                    if p - A == m and nxt >= 0: c_ = nxt                    # _bq: the last row's own draft
                    cfs.append(c_)
                if not rows: continue
                k0 = np.where(np.array(cfs) >= THR, KS, KU)
                for l in range(48):
                    pl = P[l]; pl.tick()
                    for j, r in enumerate(rows):                            # the router's output is known for every row
                        for rk, e in enumerate(ids[r, l].tolist()): pl.see(e, MASS[rk])
                    U = set()
                    for j, r in enumerate(rows): U.update(ids[r, l, :k0[j]].tolist())
                    for e in sorted(U):                                    # np.unique's order, as the builder
                        ld += 1; ms += pl.load(e, 1.0)
                    for j, p in enumerate(range(A, fm + 1)):
                        if k0[j] < 10:
                            for rk in range(k0[j], 10):
                                e = int(ids[p, l, rk]); w = MASS[rk]; n += 1; wn += w
                                if e in U: mem += 1; wmem += w
                                elif pl.has(e): res += 1; wres += w
                    for j, r in enumerate(rows):                            # kept resident experts are used
                        if k0[j] < 10:
                            for rk in range(k0[j], 10):
                                e = int(ids[r, l, rk])
                                if pl.has(e): pl.use(e, MASS[rk])
    return task, (n, mem, res, wn, wmem, wres, ld, ms)


def main():
    t0 = time.time()
    tasks = [(d, "lru", "m") for d in M.dm_gate.WALK] + [(d, p, "147") for d in M.dm_gate.WALK for p in POLICIES]
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "3"))) as Pp:
        R = dict(Pp.map(run, tasks, chunksize=1))
    L = ["QUALITY POOL: sure-row rank 2-9 pairs kept under QOM (qom_0_live_bq's rule), by pool policy; kept = union member "
         "+ resident; MASS-weighted = by mean router mass of the rank. ALIVE line (QUEUE 08:5x): on IFEval AND multi close "
         ">= 1/3 of the MASS gap LRU 147 -> LRU 45 GB pool, with load misses <= 1.10x LRU 147.",
         "| domain | pool | policy | kept (count) | kept (MASS) | gap closed | load misses (x LRU 147) |", "|---|---|---|---|---|---|---|"]
    for d in M.dm_gate.WALK:
        def kw(t): n, mem, res, wn, wmem, wres, ld, ms = R[t]; return 100 * (mem + res) / n, 100 * (wmem + wres) / wn, ms
        _, hi, _ = kw((d, "lru", "m")); _, lo, ms0 = kw((d, "lru", "147"))
        for t in [(d, "lru", "m")] + [(d, p, "147") for p in POLICIES]:
            c, w, ms = kw(t)
            L.append(f"| {d} | {'45 GB' if t[2] == 'm' else t[2]} | {t[1]} | {c:.1f} % | {w:.1f} % | "
                     f"{'-' if t[2] == 'm' else f'{100 * (w - lo) / max(hi - lo, 1e-9):+.0f} %'} | {ms} (x{ms / max(ms0, 1):.2f}) |")
    L.append(f"(self-check: LRU count shares must equal the builder's logs: 147 slots 75.8 / 79.0 / 80.4 %, 45 GB pool "
             f"92.9 / 90.4 / 89.1 %; {time.time() - t0:.0f} s)")
    os.makedirs(os.path.expanduser("~/NewIdeas/data/ledger"), exist_ok=True)
    open(os.path.expanduser("~/NewIdeas/data/ledger/pool_quality.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
