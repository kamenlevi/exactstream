"""MSB-3 cache + LSB-1 sidecar vs whole-expert LRU at a fixed expert-cache budget (CPU; real 125B traces, MoEPrefetch
traces125: ranked router top-20 per layer, 4 prompts x 800 tokens). Eager decode: one token per pass, so the result can be
calibrated against BigMoE's measured eager bench (17 GB: 77.6 misses/token, 33.3 stalled layers/token, 16.1 tok/s).
Formats (bytes per expert): whole 4-bit 2.77 MB; MSB-3 = codes q>>1 (3 bits) + g64 scale/bias (3.5 bits/weight, 2.15 MB);
LSB-1 = the last bit + a bf16 bias delta per group (1.25 bits/weight, 0.77 MB); a record [MSB-3 | LSB-1] is contiguous,
so a top-rank use that misses both parts is one read.
Plan 4433333333 (equal accuracy on the 125B GSM8K-150): ranks 0-1 need MSB-3 + LSB-1, ranks 2-9 need MSB-3.
Policies (per layer, byte budget = GB * 1024 / 48):
  whole        : LRU of whole experts (today)
  msb3-sep     : LRU of MSB-3 parts + a separate LRU of LSB-1 parts (fraction f of the budget)
  msb3-unified : one LRU over both unit kinds, LSB-1 inserted at the cold end (SliceMoE's LSB-first eviction)
Per pass and layer: needed units, misses, a stall event if any needed unit misses, and an estimated stall from this Mac's
measured cold-read curve (scripts/ssd_qd.py: one read 0.40 ms for 2.77 MB at QD 1; reads in parallel share ~14 GB/s).
usage: msb3_cache_sim.py"""
import glob, os
from collections import OrderedDict
import numpy as np
E, K, WARM = 512, 10, 50
W, M3, L1 = 2.77, 2.15, 0.77
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
TR = []
for f in files:
    z = np.load(f)
    TR.append([z[f"L{L}"][:, :K].astype(np.int64) for L in range(len(z.files))])
NL = len(TR[0])
BW = {1: 6.95, 2: 12.7, 3: 13.4, 4: 14.0}                       # GB/s by in-flight reads (measured, whole reads)


def stall_ms(sizes):
    """sizes: MB of the reads a layer waits for, all issued together."""
    if not sizes:
        return 0.0
    n = len(sizes)
    bw = BW.get(n, 14.5)
    return 0.05 + sum(sizes) / bw                               # ~0.05 ms command latency + transfer at QD n


def run(policy, budget_mb, frac_l=0.15):
    miss = events = st = byt = 0.0
    P = 0
    for tr in TR:
        T = tr[0].shape[0]
        for L in range(NL):
            if policy == "msb3-sep":
                caches = {"M": (OrderedDict(), budget_mb * (1 - frac_l)), "L": (OrderedDict(), budget_mb * frac_l)}
            else:
                caches = {"*": (OrderedDict(), budget_mb)}
            used = {k: 0.0 for k in caches}
            for t in range(T):
                ids = [int(e) for e in tr[L][t]]
                if policy == "whole":
                    need = [(e, "W") for e in ids]
                else:
                    need = [(e, "M") for e in ids] + [(e, "L") for e in ids[:2]]
                late = []
                for u in need:
                    key = "*" if "*" in caches else ("M" if u[1] in ("M", "W") else "L")
                    c = caches[key][0]
                    if u in c:
                        c.move_to_end(u)
                    else:
                        late.append(u)
                # a top-rank use missing both parts is one contiguous read of the record
                reads, merged = [], set()
                for u in late:
                    if u[1] == "M" and (u[0], "L") in late:
                        reads.append(M3 + L1); merged.add(u[0])
                    elif u[1] == "L" and u[0] in merged:
                        continue
                    else:
                        reads.append({"W": W, "M": M3, "L": L1}[u[1]])
                for u in late:
                    key = "*" if "*" in caches else ("M" if u[1] in ("M", "W") else "L")
                    c, cap = caches[key]
                    size = {"W": W, "M": M3, "L": L1}[u[1]]
                    c[u] = size; used[key] += size
                    if policy == "msb3-unified" and u[1] == "L":
                        c.move_to_end(u, last=False)                  # LSB-first eviction
                    while used[key] > cap and c:
                        _, sz = c.popitem(last=False); used[key] -= sz
                if t >= WARM:
                    miss += len(late); events += bool(late); st += stall_ms(reads); byt += sum(reads)
            if L == 0:
                P += max(0, T - WARM)
    return miss / P, events / P, st / P, byt / P


print(f"125B traces: {len(TR)} prompts, {NL} layers, eager (1 token per pass), after {WARM} warm-up tokens")
for gb in (17, 20, 24, 34):
    b = gb * 1024 / NL
    m, e, s, by = run("whole", b)
    print(f"{gb} GB  whole-expert LRU : misses {m:5.1f}/tok, stalled layers {e:4.1f}/tok, est. stall {s:5.1f} ms/tok, "
          f"read {by:5.0f} MB/tok", flush=True)
    for pol, fl in (("msb3-unified", None), ("msb3-sep", 0.10), ("msb3-sep", 0.20)):
        m, e, s, by = run(pol, b, fl or 0.15)
        print(f"   {pol:12s}{'' if fl is None else f' f={fl:.2f}':7s}: units missed {m:5.1f}/tok, stalled layers {e:4.1f}/tok, "
              f"est. stall {s:5.1f} ms/tok, read {by:5.0f} MB/tok", flush=True)
