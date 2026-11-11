"""S10b (b32): does the shipped JIT prefetch land in time? CPU only (numpy), reads a b32 trace dir (res.bin needed).
Per verify pass, base layer L and pool T = L + lead (every trunk layer, not just the predictor layers), with the routed
set RT at T (union over the pass's rows), the residency at L's prefetch decision (res.bin) and the misses at T's
service (miss.bin):
- not resident at the decision: routed experts the engine still had to load after L's release;
- in time: of those, resident at T's service (loaded between the decision and the service, by the lead-1 JIT
  prefetch or anything else);
- evicted: resident at the decision but a miss at T's service (pushed out in between).
Join rules as in s10b_check.py (first call of each layer per seq; passes with all trunk layers).
usage: s10b_intime.py TRACE_DIR"""
import glob, json, os, sys
import numpy as np

D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json")))
NL, E = meta["n_layers"], meta["E"]
raw = np.fromfile(os.path.join(D, "miss.bin"), np.int32)
miss, i = {}, 0
while i < len(raw):
    seq, L, n = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
    v = raw[i + 3:i + 3 + n]
    miss.setdefault((seq, L), set((v & 0xFFFF)[((v >> 16) & 1) == 1].tolist()))
    i += 3 + n
rr, i, Wd, RES = np.fromfile(os.path.join(D, "res.bin"), "<i4"), 0, meta["res"]["W"], {}
while i < len(rr):
    seq, L, nT = int(rr[i]), int(rr[i + 1]), int(rr[i + 2])
    i += 3
    for _ in range(nT):
        RES.setdefault((seq, L, int(rr[i])), i + 1)
        i += 1 + Wd
acc = {}                                       # lead -> [pass-layers, routed, not resident at decision, in time, misses, evicted]
for f in sorted(glob.glob(os.path.join(D, "p*.npz"))):
    z = np.load(f)
    S, Ls, rows, off, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    by = {}
    for j in range(len(S)):
        by.setdefault(int(S[j]), {}).setdefault(int(Ls[j]), j)
    for seq, calls in by.items():
        if not all(L in calls for L in range(NL)) or len({int(rows[calls[L]]) for L in range(NL)}) != 1:
            continue
        R = int(rows[calls[0]])
        RT = {L: set(ids[off[calls[L]]:off[calls[L]] + R].ravel().tolist()) for L in range(NL)}
        for L in range(NL):
            for lead in (1, 2, 3):
                T = L + lead
                o = RES.get((seq, L, T))
                if o is None or (seq, T) not in miss:
                    continue
                rb = np.unpackbits(rr[o:o + Wd].view(np.uint8), bitorder="little")[:E].astype(bool)
                M = RT[T] & miss[(seq, T)]
                nr = {e for e in RT[T] if not rb[e]}
                a = acc.setdefault(lead, [0] * 6)
                a[0] += 1; a[1] += len(RT[T]); a[2] += len(nr); a[3] += len(nr - M); a[4] += len(M); a[5] += len(M - nr)
for lead, a in sorted(acc.items()):
    n = max(a[0], 1)
    print(f"lead {lead}: {a[0]} pass-layers; per pass-layer: routed {a[1] / n:.2f}, not resident at L's decision "
          f"{a[2] / n:.2f}, of them in time at T {a[3] / n:.2f} ({100 * a[3] / max(a[2], 1):.1f} %), misses {a[4] / n:.2f}"
          f" (of them resident at the decision, evicted in between: {a[5] / n:.2f})")
