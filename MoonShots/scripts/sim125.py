"""Expert-streaming policies on REAL 125B verify passes (BigMoE's oracle recording: per (pass, layer) the union of routed experts;
48 layers x 512 experts, spec125 k=3), with the 35B proxy traces (S=3 unions) as a cross-check. 50 % resident per layer.
Timing per layer: ATTN, router, stall until the needed experts land (+ gate), MoE. SSD: demand reads first (no queueing), prefetch
reads FIFO; n reads issued together land at t + 0.2 + 0.25 n ms (measured). Look-ahead (BigMoE v1) is emulated from the true next
layer: each true expert kept with probability R_LA (125B: 0.75 @12), plus false positives up to B per pass. Runahead: when a stall
lasts > RA ms, L+1's true needs are prefetched RA ms into the stall. Policies on top:
  insert : where prefetched experts enter the cache: 'lru_end' (BigMoE victim prefetch) or 'srrip' (RRIP: re-reference interval 2 of 3)
  repl   : cache replacement: 'lru' or 'srrip' (scan-resistant, demand inserted at RRPV 2, promoted to 0 on hit)
  fdp    : feedback-directed throttling: per-layer prefetch cap adapted from measured accuracy (used/issued) over a window
  crit   : criticality order: prefetch reads for the nearest layer first (vs issue order)
Usage: sim125.py   (CPU only)"""
import os, struct
from collections import OrderedDict
import numpy as np

LAT, PER, GATE = 0.2, 0.25, 0.07
LAYER = 38.0 / 48
ATTN = 0.5 * LAYER
RA = 0.6


def load125(path=os.path.expanduser("~/BigMoE/data/oracle125.bin")):
    rows = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            k, n = struct.unpack("<qq", h)
            rows[k] = list(struct.unpack(f"<{n}i", f.read(4 * n)))
    seqs = sorted({k // 1024 for k in rows})
    NL = max(k % 1024 for k in rows) + 1
    passes = [[rows.get(s * 1024 + L, []) for L in range(NL)] for s in seqs]
    return passes, NL, 512


def load35(S=3, maxpass=600):
    z = dict(np.load(os.path.expanduser("~/MoEPrefetch/traces/code.npz")))
    NL = 40
    T = len(z["doc"])
    acts = [z[f"L{L}_act"][:, :8] for L in range(NL)]
    passes = [[list(dict.fromkeys(acts[L][t:t + S].reshape(-1).tolist())) for L in range(NL)]
              for t in range(0, min(T - S, S * maxpass), S)]
    return passes, NL, 256


class Cache:
    """per-layer cache: LRU or SRRIP (2-bit RRPV)."""

    def __init__(self, C, repl):
        self.C, self.repl = C, repl
        self.d = OrderedDict()          # expert -> rrpv (srrip) ; order = recency (lru)

    def __contains__(self, e):
        return e in self.d

    def hit(self, e):
        if self.repl == "lru":
            self.d.move_to_end(e)
        else:
            self.d[e] = 0

    def victim(self):
        if self.repl == "lru":
            return next(iter(self.d))
        while True:
            for e, v in self.d.items():
                if v >= 3:
                    return e
            for e in self.d:
                self.d[e] += 1

    def insert(self, e, how):
        while len(self.d) >= self.C:
            del self.d[self.victim()]
        if self.repl == "lru":
            self.d[e] = 0
            if how == "low":
                self.d.move_to_end(e, last=False)
        else:
            self.d[e] = {"high": 0, "mid": 2, "low": 3}[how]


def run(passes, NL, E, *, frac=0.5, lookahead=True, B=12, R_LA=0.75, runahead=False, insert="lru_end", repl="lru",
        fdp=False, crit=False, warm=None, seed=0):
    rng = np.random.default_rng(seed)
    C = int(frac * E)
    caches = [Cache(C, repl) for _ in range(NL)]
    cap = [B] * NL
    used_w, iss_w = [1.0] * NL, [1.0] * NL
    warm = len(passes) // 4 if warm is None else warm
    stl = ms = 0.0
    n = 0
    for p, need in enumerate(passes):
        now, ssd = 0.0, 0.0
        land = [dict() for _ in range(NL)]
        issued = [set() for _ in range(NL)]
        pend = []                                     # (layer, expert) prefetches not yet started (crit ordering)

        def prefetch(L, es, t):
            nonlocal ssd
            es = [e for e in es if e not in caches[L] and e not in land[L]][:cap[L]]
            if crit:                                  # nearer layers first: start this batch before any queued farther-layer reads
                ssd = max(t, min(ssd, t + 0.0))
            start = max(t, ssd)
            for i, e in enumerate(es):
                land[L][e] = start + LAT + PER * (i + 1)
                caches[L].insert(e, "low" if insert == "lru_end" else "mid")
                issued[L].add(e)
            if es:
                ssd = start + PER * len(es)

        for L in range(NL):
            now += ATTN
            if lookahead and L + 1 < NL:
                tru = need[L + 1]
                pred = [e for e in tru if rng.random() < R_LA]
                while len(pred) < min(B * 2, len(tru) + B):    # false positives (union of per-row top-B over ~3 rows)
                    pred.append(int(rng.integers(E)))
                prefetch(L + 1, list(dict.fromkeys(pred)), now)
            miss = [e for e in need[L] if e not in caches[L] and e not in land[L]]
            for i, e in enumerate(miss):              # demand reads: first in line
                land[L][e] = now + LAT + PER * (i + 1)
            ready = max([land[L].get(e, 0.0) for e in need[L]] + [now])
            stall = ready - now
            if stall > 0:
                stall += GATE
                if p >= warm:
                    stl += 1
                    ms += stall
                if runahead and L + 1 < NL and stall > RA:
                    prefetch(L + 1, need[L + 1], now + RA)
            now += max(stall, 0.0)
            for e in need[L]:
                if e in caches[L]:
                    caches[L].hit(e)
                else:
                    caches[L].insert(e, "high")
            if fdp and issued[L]:                     # feedback: accuracy of this layer's prefetches
                acc = len(issued[L] & set(need[L])) / len(issued[L])
                used_w[L] = 0.8 * used_w[L] + 0.2 * acc
                cap[L] = int(np.clip(round(B * 1.5 * used_w[L]), 2, 2 * B))
            now += LAYER - ATTN
        if p >= warm:
            n += 1
    sc = 48 / NL
    return stl / n * sc, ms / n * sc


if __name__ == "__main__":
    for name, (passes, NL, E) in (("125B real verify passes", load125()), ("35B proxy (code, S=3)", load35())):
        print(f"{name}: {len(passes)} passes, {NL} layers x {E} experts, 50 % resident (stall per pass, x48/{NL})")
        base = dict(passes=passes, NL=NL, E=E)
        cases = [("no prefetch", dict(lookahead=False)),
                 ("look-ahead (v1, LRU-end)", dict()),
                 ("+ runahead", dict(runahead=True)),
                 ("+ runahead + FDP throttle", dict(runahead=True, fdp=True)),
                 ("+ runahead + criticality order", dict(runahead=True, crit=True)),
                 ("+ runahead, SRRIP insert (RRPV 2)", dict(runahead=True, insert="srrip")),
                 ("+ runahead, SRRIP cache", dict(runahead=True, insert="srrip", repl="srrip")),
                 ("runahead only (no look-ahead)", dict(lookahead=False, runahead=True))]
        for tag, kw in cases:
            r = [run(**base, seed=s, **kw) for s in range(3)]
            sl = np.mean([x[0] for x in r]); msv = np.mean([x[1] for x in r])
            print(f"  {tag:36s}: stalled layers {sl:5.1f}, stall {msv:5.1f} ms/pass", flush=True)
