"""Per-layer timeline of a traced dh / d1 decode (run.py gen with DH_TRACE=1; ext trace events (L, pass, kind, t0, t1): kind 1
= a served row (t_seen, t_rel; for split rows t_rel = the phase-B release), kind 5 = the phase-A release of a split row,
kind 4 = a row that had misses). Base pool rows only (ext layers 0..47). Per verify pass and layer L:
  waited = t_relA - t_seen (the waited batch + loader work), late = t_relB - t_relA (the late batch; ~0 when none),
  gap = t_seen(L+1) - t_relB(L): with dh, when the GPU waited at spin B, this is the TAIL (late write, MLP read, router,
  post, commit, loader poll); with d1 the loader goes straight from L's phase B to L+1's row (gap ~ 0 if posted).
Prints medians and percentiles per prompt file, plus the SSD-busy share of the layer span.
usage: trace_tail.py data/defer/gen_code_dh_trace.json [...]"""
import json, sys
import numpy as np
for f in sys.argv[1:]:
    r = json.load(open(f))
    ev = np.array(r["ext_trace"])
    L, seq, kind, t0, t1 = ev[:, 0].astype(int), ev[:, 1].astype(int), ev[:, 2].astype(int), ev[:, 3] * 1e3, ev[:, 4] * 1e3
    W, LA, G, GL, SPAN, BUSY = [], [], [], [], [], []
    for s in np.unique(seq):
        m1 = (seq == s) & (kind == 1) & (L < 48)
        if len(np.unique(L[m1])) < 40:
            continue
        seen = dict(zip(L[m1], t0[m1])); relB = dict(zip(L[m1], t1[m1]))
        m5 = (seq == s) & (kind == 5) & (L < 48)
        relA = dict(zip(L[m5], t1[m5]))
        ls = sorted(seen)
        busy = 0.0
        for a, b in zip(ls[:-1], ls[1:]):
            ra = relA.get(a, relB[a])
            W.append(ra - seen[a]); LA.append(relB[a] - ra)
            g = seen[b] - relB[a]
            G.append(g)
            if relB[a] - ra > 0.05:                             # a layer whose late batch actually read something
                GL.append(g)
            busy += relB[a] - seen[a]
        SPAN.append(relB[ls[-1]] - seen[ls[0]]); BUSY.append(busy)
    W, LA, G, GL = map(np.array, (W, LA, G, GL))
    pc = lambda x: f"p10 {np.percentile(x, 10):.3f} / med {np.median(x):.3f} / p90 {np.percentile(x, 90):.3f}" if len(x) else "-"
    print(f"== {f}: {len(SPAN)} passes, layer span {np.mean(SPAN):.1f} ms, loader busy (seen -> phase B) {np.mean(BUSY):.1f} ms "
          f"({100 * np.mean(BUSY) / np.mean(SPAN):.0f} %)")
    print(f"   waited batch + loader (ms): {pc(W)}")
    print(f"   late batch (ms), all layers: {pc(LA)}; layers with late reads: {int((LA > 0.05).sum())} of {len(LA)}")
    print(f"   gap phase B(L) -> seen(L+1) (ms): all {pc(G)}; after a late batch {pc(GL)}")
