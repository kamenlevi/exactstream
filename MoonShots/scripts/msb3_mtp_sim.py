"""Round 25: the MSB-3 expert cache (round 20) revisited under MTP verify passes. Under eager decode it scored only ~+7 %
because stall events, not bytes, dominated (33 stalled layers/token); with committed MTP each pass commits ~2.8 tokens,
stalled layers drop to ~14/token while misses per token stay ~80, so bytes per miss now carry most of the stall time.
Trace-driven (data/verify_trace125.npz/json: 125B, committed history, depth 3, 401 verify passes of 4 rows; routing
never depends on the cache). Plan 4433333333 (GSM8K 427/450 vs 431/450 for 4-bit, n.s.): every routed expert needs its
MSB-3 part (2.15 MB: codes q>>1 + g64 scale/bias), the top-2 ranks also need LSB-1 (0.77 MB). The trace has no gate
scores (argpartition order), so 2 of each row's 10 ids, chosen by a fixed hash, stand in for the top-2 ranks.
Policies at a byte budget (per layer, GB * 1024 / 48 MB):
  whole      LRU of whole 4-bit experts (2.77 MB, BigMoE today)
  msb3-sep   LRU of MSB-3 parts + a separate LRU of LSB-1 parts (fraction f of the budget)
  msb3-uni   one LRU over both kinds, LSB-1 parts inserted at the cold end
A top-rank use that misses both parts reads one contiguous record [MSB-3 | LSB-1] (2.92 MB).
Stall model (round 20, BigMoE eager benches): 0.40 ms per stalled layer + 0.083 ms per MB read.
usage: msb3_mtp_sim.py"""
import json, os
from collections import OrderedDict, defaultdict
import numpy as np
D = os.path.expanduser("~/MoonShots/data")
z = np.load(f"{D}/verify_trace125.npz")
meta = json.load(open(f"{D}/verify_trace125.json"))
ids, rows, seqs, prompt = z["ids"], z["rows"], z["seq"], z["prompt"]
P, NL, RM, K = ids.shape
committed = defaultdict(int)
for (p, s, n) in meta["callbacks"]:
    committed[(p, s)] += n
ntok_pass = np.array([max(1, min(committed.get((int(prompt[i]), int(seqs[i])), 1), int(rows[i]))) for i in range(P)])
W, M3, L1 = 2.77, 2.15, 0.77
WARM = 60


def top2(row):
    """stand-in for the two highest-weight experts of a row (no gate scores in the trace): a fixed hash pick."""
    return set(sorted(row, key=lambda e: (e * 2654435761) % 1000003)[:2])


def run(policy, budget_mb, frac_l=0.10):
    miss_units = stalled = mb = 0.0
    ntok = 0
    caches = []
    for L in range(NL):
        if policy == "msb3-sep":
            caches.append({"M": [OrderedDict(), budget_mb * (1 - frac_l), 0.0], "L": [OrderedDict(), budget_mb * frac_l, 0.0]})
        else:
            caches.append({"*": [OrderedDict(), budget_mb, 0.0]})
    for i in range(P):
        R = int(rows[i])
        for L in range(NL):
            a = ids[i, L]
            if a[0, 0] < 0:
                continue
            rowsets = [[int(e) for e in a[r] if e >= 0] for r in range(R)]
            if policy == "whole":
                need = {(e, "W") for rs in rowsets for e in rs}
            else:
                need = {(e, "M") for rs in rowsets for e in rs} | {(e, "L") for rs in rowsets for e in top2(rs)}
            cs = caches[L]
            key = lambda u: "*" if "*" in cs else ("M" if u[1] in ("M", "W") else "L")
            late = [u for u in need if u not in cs[key(u)][0]]
            reads = []
            lset = {u[0] for u in late if u[1] == "L"}
            for u in late:
                if u[1] == "M" and u[0] in lset:
                    reads.append(M3 + L1)
                elif u[1] == "L" and (u[0], "M") in late:
                    continue
                else:
                    reads.append({"W": W, "M": M3, "L": L1}[u[1]])
            for u in need:                                  # recency: every use moves to the MRU end
                c = cs[key(u)][0]
                if u in c:
                    c.move_to_end(u)
            for u in late:
                ent = cs[key(u)]
                size = {"W": W, "M": M3, "L": L1}[u[1]]
                ent[0][u] = size; ent[2] += size
                if policy == "msb3-uni" and u[1] == "L":
                    ent[0].move_to_end(u, last=False)
                while ent[2] > ent[1] and ent[0]:
                    _, sz = ent[0].popitem(last=False); ent[2] -= sz
            if i >= WARM:
                miss_units += len(late); stalled += bool(late); mb += sum(reads)
        if i >= WARM:
            ntok += int(ntok_pass[i])
    ms = stalled * 0.40 + mb * 0.083
    return miss_units / ntok, stalled / ntok, mb / ntok, ms / ntok


print(f"verify trace: {P} passes, {int(ntok_pass[WARM:].sum())} tokens after {WARM} warm-up passes; per token:")
print("budget          policy          misses   stalled layers   MB read   est. stall ms   vs whole")
for gb in (13, 17, 21, 26):
    b = gb * 1024 / NL
    base = None
    for pol, f in (("whole", None), ("msb3-sep", 0.10), ("msb3-sep", 0.20), ("msb3-uni", None)):
        m, s, mbr, ms = run(pol, b, f or 0.10)
        base = base or ms
        lab = pol + (f" f={f:.2f}" if f else "")
        print(f"{gb:2d} GB         {lab:15s} {m:6.1f}    {s:6.2f}          {mbr:6.1f}    {ms:6.2f}          {ms / base - 1:+.0%}",
              flush=True)
