"""Working-set and cache numbers for the bytes-per-token budget (Breakthrough, 2026-10-19).
125B decode routing traces from MoEPrefetch (data/traces125/{code,chat,bulgarian,math}_800.npz: router top-20 ids per
decode token per layer; ranks 0-9 are the active experts). Eager decode, one token per step, page cache ignored.
Prints: (1) per-layer LRU hit rate vs cache size C (slots per layer), (2) distinct experts per layer touched in windows
of W tokens, (3) LRU with a reduced k (top-k only), (4) a crude 'cache-aware routing' proxy: a missing expert at rank
>= r0 is replaced by the best-ranked *cached* expert among ranks 10-19 if one exists (quality NOT modelled).
"""
import sys, numpy as np
from collections import OrderedDict
T = "/Users/kamen/MoEPrefetch/data/traces125/"
SETS = ["code", "chat", "bulgarian", "math"]
NL = 48

def load(s):
    d = np.load(T + f"{s}_800.npz")
    return np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32)  # [tok, layer, 20]

def lru(tr, C, k=10, swap_from=None):
    ntok = tr.shape[0]; miss = 0; uses = 0; swaps = 0
    for l in range(NL):
        c = OrderedDict()
        for t in range(ntok):
            ids = list(tr[t, l, :k])
            if swap_from is not None:
                alt = [e for e in tr[t, l, 10:20] if e in c]
                out = []
                for r, e in enumerate(ids):
                    if e not in c and r >= swap_from and alt:
                        out.append(alt.pop(0)); swaps += 1
                    else:
                        out.append(e)
                ids = out
            for e in ids:
                uses += 1
                if e in c: c.move_to_end(e)
                else:
                    miss += 1; c[e] = 1
                    if len(c) > C: c.popitem(last=False)
    return 1 - miss / uses, miss / ntok, swaps / ntok

trs = {s: load(s) for s in SETS}
print("tokens per set:", {s: trs[s].shape[0] for s in SETS})
print("\n(1) LRU per layer, top-10, hit rate / misses per token (of 480 uses), by C slots per layer (%% of 512)")
Cs = [4, 8, 13, 16, 26, 32, 51, 64, 96, 128, 192, 256]
print("C      " + " ".join(f"{C:>11d}" for C in Cs))
print("% res  " + " ".join(f"{100*C/512:>10.1f}%" for C in Cs))
for s in SETS:
    row = [lru(trs[s], C) for C in Cs]
    print(f"{s:7s}" + " ".join(f"{h:5.3f}/{m:5.1f}" for h, m, _ in row))

print("\n(2) distinct experts per layer touched within W consecutive tokens (mean over layers and windows), top-10")
for s in SETS:
    tr = trs[s]; out = []
    for W in [1, 4, 16, 64, 256, 800]:
        vals = []
        for l in range(NL):
            for a in range(0, tr.shape[0] - W + 1, max(1, W)):
                vals.append(len(np.unique(tr[a:a + W, l, :10])))
        out.append(f"W={W}:{np.mean(vals):.0f}")
    print(f"{s:10s}", "  ".join(out))

print("\n(3) LRU with reduced k (only top-k routed): misses per token, C=26 / 51 / 128")
for k in [10, 8, 6, 4]:
    r = [np.mean([lru(trs[s], C, k=k)[1] for s in SETS]) for C in (26, 51, 128)]
    print(f"k={k:2d}  " + "  ".join(f"{x:6.1f}" for x in r))

print("\n(4) cache-aware swap proxy (missing rank>=r0 -> best cached among ranks 10-19): misses/token, swaps/token")
for r0 in [None, 8, 6, 4]:
    r = [[lru(trs[s], C, swap_from=r0) for s in SETS] for C in (26, 51, 128)]
    print(f"r0={str(r0):4s} " + "  ".join(f"C={C}: {np.mean([x[1] for x in rr]):6.1f} miss, {np.mean([x[2] for x in rr]):5.1f} swap" for C, rr in zip((26, 51, 128), r)))
