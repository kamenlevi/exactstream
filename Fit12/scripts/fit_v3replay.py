"""Fit12 (2026-10-25; NewIdeas' item (a)): an independent replay of NewIdeas' v3 residency numbers (their b10_resid.py run(),
data/b10r/lfu_r16.txt, 'today v3 147/46': 144.5 MB/token under LRU, 116.0 under the engine's decayed LFU 0.99 with
16-row checks accepting 10). Written from the engine (~/BigMoE/ext/bigmoe_ext.cpp: ensure(), make_room(), lfu_victim(),
lines 309-352), not from their code.
Per layer two caches: MSB parts (C_m 147) for every top-10 use, LSB parts (C_l 46) for ranks 0-1 (plan 4433333333; the
traces' ids are sorted by gate weight). A check = R rows from token t (the trace's next R tokens), accepting A (the next
check starts at t + A). Per check, layer and tier: need = the unique ids, row-major. LFU (BIGMOE_EVICT_LFU = d): w /= d, then
score[e] += w for every needed e (scores persist after eviction). Each miss evicts while the tier holds >= C: LRU = the least
recently used expert not needed now; LFU = the lowest score not needed now (the first of equals from the MRU end, as
lfu_victim). Hits and misses move to the MRU end in need order. Bytes per miss = the engine's reads (records3 index): MSB
2,162,688 B, LSB 770,048 B. Counted: checks that start at tokens 400-799 (the caches warm from token 0); MB = 1e6 B;
MB / token averaged over the 4 expA sets, as NewIdeas'. Also printed: cold start at token 400, and MB = 2^20.
Usage: ~/BigMoE/.venv/bin/python -B fit_v3replay.py   (CPU, 1 core, ~1-2 min)"""
import os, time
from collections import OrderedDict
import numpy as np

TR = os.path.expanduser("~/Breakthrough/data/expA/traces125")
SETS = ("bulgarian", "chat", "code", "math")
NL, CM, CL, MSB, LSB, T0 = 48, 147, 46, 2162688, 770048, 400


class Tier:
    def __init__(self, C, d):
        self.C, self.d, self.od, self.w = C, d, OrderedDict(), 1.0
        self.score = np.zeros(512)

    def call(self, need):
        if self.d:
            self.w /= self.d
            for e in need:
                self.score[e] += self.w
        nset, miss = set(need), 0
        for e in need:
            if e in self.od:
                self.od.move_to_end(e)
                continue
            miss += 1
            while len(self.od) >= self.C:
                v = None
                if self.d:
                    bs = 0.0
                    for x in reversed(self.od):                 # MRU -> LRU, first strictly lowest (lfu_victim)
                        if x not in nset and (v is None or self.score[x] < bs):
                            v, bs = x, self.score[x]
                else:
                    v = next((x for x in self.od if x not in nset), None)   # from the LRU end
                if v is None:
                    break
                del self.od[v]
            self.od[e] = None
        return miss


def run(ids, R, A, d, start):
    """one set: (MSB misses, LSB misses, tokens) over the checks that start at >= T0; caches warm from `start`"""
    T = ids[0].shape[0]
    bm = bl = 0
    for L in range(NL):
        m, l = Tier(CM, d), Tier(CL, d)
        x = ids[L]
        for t in range(start, T, A):
            rows = x[t:t + R]
            a = m.call(list(dict.fromkeys(rows[:, :10].reshape(-1).tolist())))
            b = l.call(list(dict.fromkeys(rows[:, :2].reshape(-1).tolist())))
            if t >= T0:
                bm, bl = bm + a, bl + b
    tok = sum(min(A, T - t) for t in range(start, T, A) if t >= T0)
    return bm, bl, tok


def main():
    t0 = time.time()
    tr = {s: [np.load(os.path.join(TR, f"{s}_800_g.npz"))[f"L{L}"][:, :10].astype(np.int64) for L in range(NL)] for s in SETS}
    print(f"[v3r] sets {', '.join(f'{s} {tr[s][0].shape[0]}' for s in SETS)} tokens; counted from token {T0}", flush=True)
    for R, A in ((1, 1), (16, 10)):
        for d in (0.0, 0.99):
            for start, lab in ((0, "warm from 0"), (T0, "cold at 400")):
                per = []
                for s in SETS:
                    bm, bl, tok = run(tr[s], R, A, d, start)
                    per.append((bm / tok, bl / tok, (bm * MSB + bl * LSB) / tok))
                mb = np.mean([p[2] for p in per])
                print(f"R {R} A {A} {'LRU' if not d else f'LFU {d}'} ({lab}): {mb / 1e6:.1f} MB/token "
                      f"({mb / 2 ** 20:.1f} MiB) | MSB misses / token {np.mean([p[0] for p in per]):.1f}, LSB "
                      f"{np.mean([p[1] for p in per]):.1f} | per set " + " ".join(f"{s} {p[2] / 1e6:.1f}" for s, p in zip(SETS, per))
                      + f" ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
