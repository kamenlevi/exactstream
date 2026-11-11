"""Cache ADMISSION (not eviction) for SSD-streamed experts, on the real 125B traces (MoEPrefetch traces125/*_800.npz, verify
windows = 4-token unions, stride 3). A missed expert must be read and computed either way. With admission control it may
go to a scratch slot instead of the cache, so the hot set is not displaced by one-off experts.
  lru       : always admit, evict LRU
  tinylfu   : W-TinyLFU-style (Einziger et al. 2015): admit only if est_freq(new) > est_freq(LRU victim); frequencies
              decay (halved every W accesses per layer); a small window cache (1 %) always admits to catch new hot experts
  lfu-evict : always admit, evict the least frequently used (decayed counts), for reference
Reports misses per pass (the SSD reads) and stalled layers per pass, after a 200-token warm-up.   CPU only.
usage: admission_sim.py [FRAC]"""
import glob, os, sys
from collections import OrderedDict
import numpy as np
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
E, K = 512, 10
C = int(FRAC * E)
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))


def run(policy, W=None):
    tot_m = tot_s = P = 0
    for f in files:
        z = np.load(f); NL = len(z.files)
        tr = [z[f"L{L}"][:, :K].astype(np.int64) for L in range(NL)]
        T = tr[0].shape[0]
        caches = [OrderedDict() for _ in range(NL)]
        freq = [np.zeros(E) for _ in range(NL)]
        win = [OrderedDict() for _ in range(NL)]
        Wn = max(1, int(0.01 * C)) if policy == "tinylfu" else 0
        acc = [0] * NL
        for start in range(0, T - 4, 3):
            st = 0; ms = 0
            for L in range(NL):
                u = [int(e) for e in np.unique(tr[L][start:start + 4])]
                cache, fr, w = caches[L], freq[L], win[L]
                miss = [e for e in u if e not in cache and e not in w]
                for e in u:
                    fr[e] += 1; acc[L] += 1
                    if e in cache: cache.move_to_end(e)
                    if e in w: w.move_to_end(e)
                if W and acc[L] >= W: fr *= 0.5; acc[L] = 0
                for e in miss:
                    if policy == "lru":
                        cache[e] = 1
                        if len(cache) > C: cache.popitem(last=False)
                    elif policy == "lfu-evict":
                        cache[e] = 1
                        if len(cache) > C:
                            v = min(cache, key=lambda x: fr[x]); del cache[v]
                    else:                                       # tinylfu: window first, then admission to the main cache
                        w[e] = 1
                        if len(w) > Wn:
                            cand, _ = w.popitem(last=False)
                            if len(cache) < C - Wn:
                                cache[cand] = 1
                            else:
                                victim = next(iter(cache))
                                if fr[cand] > fr[victim]:
                                    del cache[victim]; cache[cand] = 1
                if start >= 200:
                    ms += len(miss); st += bool(miss)
            if start >= 200:
                tot_m += ms; tot_s += st; P += 1
    return tot_m / P, tot_s / P


print(f"125B traces, {len(files)} prompts, C={C}/512 ({FRAC:.0%}), verify windows of 4 tokens")
for pol, W in (("lru", None), ("lfu-evict", 20 * C), ("tinylfu", 10 * C), ("tinylfu", 20 * C), ("tinylfu", 50 * C)):
    m, s = run(pol, W)
    print(f"  {pol:9s} {'' if W is None else f'decay/{W // C}C':9s}: misses/pass {m:6.1f}  stalled layers/pass {s:5.1f}", flush=True)
