#!/usr/bin/env python3 -I
"""lu_batch_ssd.py (Lunatic, round 1). CPU only, recorded routing captures. Does the SSD let B parallel streams through?
Per-layer LRU with C slots (uniform, no sidecar, no M-17), warmed by each stream's prompt rows; then B streams decode side
by side in aligned 16-row blocks; a block's union per layer is read once (misses = union members not resident).
Prints SSD MB per generated row (2.15 MB per miss) and DRAM GB per generated row (union x 1.9 MB + trunk 3.44 GB per
block step), B = 1 / 2 / 4 / 8, C = 160 / 256 / 320. Single-stream absolute numbers are rough (no rejected rows, no
sidecar); the B-ratio is the point."""
import glob, os, numpy as np
from collections import OrderedDict
rng = np.random.default_rng(11)
D = os.path.expanduser("~/FableGod/decision_map/out/routing")
S = {}
for f in sorted(glob.glob(os.path.join(D, "*.npz"))):
    z = np.load(f, allow_pickle=True); ids = z["ids"].astype(np.int32); gen = np.asarray(z["gen"]); n = ids.shape[0]
    if gen.ndim == 0: st = max(int(gen) - 1, 0); P, G = ids[:st], ids[st:]
    else:
        m = gen.astype(bool); m = m[1:] if m.shape[0] == n + 1 else m[:n]; P, G = ids[~m], ids[m]
    S.setdefault(os.path.basename(f).split("_")[1], []).append((P, G))
def run(streams, C, LS=range(0, 48, 2)):
    cache = {L: OrderedDict() for L in LS}; miss = 0; dram = 0.0; rows = 0
    def touch(L, ex):
        nonlocal miss
        c = cache[L]
        for e in ex:
            if e in c: c.move_to_end(e)
            else:
                miss += 1; c[e] = 1
                if len(c) > C: c.popitem(last=False)
    for P, _ in streams:                              # warm with each prompt (prefill)
        for s in range(0, P.shape[0], 64):
            for L in LS: touch(L, np.unique(P[s:s+64, L, :]))
    miss = 0
    L0 = min(G.shape[0] for _, G in streams)
    for s in range(0, L0 - 15, 16):
        u = 0
        for L in LS:
            ex = np.unique(np.concatenate([G[s:s+16, L, :].ravel() for _, G in streams])); u += ex.size; touch(L, ex)
        dram += u * (48 / len(LS)) * 1.9e-3 + 3.44; rows += 16 * len(streams)
    return miss * (48 / len(LS)) * 2.15 / rows, dram / rows
for dom in ["code", "ifeval"]:
    for C in [160, 256, 320]:
        out = []
        for B in [1, 2, 4, 8]:
            r = [run([S[dom][i] for i in rng.choice(len(S[dom]), B, replace=False)], C) for _ in range(4 if B > 1 else 12)]
            out.append(f"B{B}: SSD {np.mean([x[0] for x in r]):5.1f} MB, DRAM {np.mean([x[1] for x in r]):4.2f} GB per row")
        print(f"{dom:6s} C {C}: " + " | ".join(out))
