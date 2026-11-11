"""Upper bound for stall-funded runahead slices on the real 125B routing traces (MoEPrefetch data/traces125/*_800.npz).
Verify passes = unions of 4 consecutive tokens' top-10 (stride 3, ~ spec k=3 at 3.4 tok/pass); per-layer LRU at FRAC of 512.
A stall at layer X funds a slice that perfectly protects X+1..X+D (D = 2: predictions 2 and 3 layers past the slice seed);
a protected layer does not stall and funds nothing (no idle time). Reports stalls per pass without and with slices, the share of
stalled layers that sit within D layers after a previous stall, and the same for a continuous slice (every layer protected
except the first D after a pass starts) for scale. CPU only."""
import glob, os, sys
from collections import OrderedDict
import numpy as np
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
D = int(sys.argv[2]) if len(sys.argv) > 2 else 2
E, K, C = 512, 10, int(FRAC * 512)
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
tot = dict(passes=0, stalls=0, clustered=0, left_slice=0)
for f in files:
    z = np.load(f)
    NL = len(z.files)
    tr = [z[f"L{L}"][:, :K].astype(np.int64) for L in range(NL)]
    T = tr[0].shape[0]
    lru = [OrderedDict() for _ in range(NL)]
    warm = 200
    for start in range(0, T - 4, 3):
        stalled = []
        for L in range(NL):
            u = np.unique(tr[L][start:start + 4])
            miss = sum(1 for e in u if e not in lru[L])
            for e in u:
                lru[L].pop(int(e), None); lru[L][int(e)] = True
            while len(lru[L]) > C: lru[L].popitem(last=False)
            stalled.append(miss > 0)
        if start < warm: continue
        tot["passes"] += 1
        tot["stalls"] += sum(stalled)
        tot["clustered"] += sum(1 for L in range(NL) if stalled[L] and any(stalled[L - j] for j in range(1, D + 1) if L - j >= 0))
        protect = -1                                  # stall-funded slices: a stall protects the next D layers
        for L in range(NL):
            if stalled[L] and L > protect:
                tot["left_slice"] += 1
                protect = L + D
    print(f"{os.path.basename(f)}: done", flush=True)
P = tot["passes"]
print(f"FRAC {FRAC}, D {D}, {P} verify passes (4-token unions, LRU, no prefetch):")
print(f"  stalled layers/pass {tot['stalls']/P:.1f}; of them within {D} layers after a previous stall: {tot['clustered']/tot['stalls']:.0%}")
print(f"  with stall-funded slices (perfect, protect next {D}): stalls/pass {tot['left_slice']/P:.1f} "
      f"({1 - tot['left_slice']/tot['stalls']:.0%} removed)")
