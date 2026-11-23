"""Reconcile Leap's frugal replay (5.5 misses/token at C 192, resid27 / TABLE_350) with the Eval suite's b12c192 measurement
(27.8 SSD reads per generated token, 22.9 waited, 40 MB/token, MTP checks). b12c192 = Breakthrough's D1 frugal wait-top-2 on
the v3 tiered store (plan 4433333333): ranks 0-1 are 4-bit = MSB-3 base part (pool C_m 192) + LSB-1 sidecar (pool C_l 77);
ranks 2-9 are 3-bit (MSB only), never read (resident substitutes from ranks 10-19, else dropped). The earlier replay had ONE
pool of whole experts and counted only MSB misses of ranks 0-1. Here: two LRU pools per layer (MSB C_m, LSB C_l), MTP-like
checks (4 rows, 3.4 tokens) and 16-row checks; reads = MSB misses (waited ranks 0-1) + LSB misses (ranks 0-1).
Usage: python3 b12rec29.py -> ~/Leap/data/b12rec29.txt"""
import glob, os
import numpy as np
from collections import OrderedDict
NL, E = 48, 512
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
sets = {os.path.basename(f).split("_")[0]: np.load(f) for f in sorted(glob.glob(T + "*_800_g.npz"))}
IDS = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1)[:, :, :20].astype(np.int32) for k, d in sets.items()}
GP = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1)[:, :, :20].astype(np.float32) for k, d in sets.items()}
u = np.zeros((NL, E))
for k in IDS:
    n = IDS[k].shape[0]
    for l in range(NL):
        np.add.at(u[l], IDS[k][:n // 2, l, :10].ravel(), GP[k][:n // 2, l, :10].ravel())
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def pools(C):
    return [OrderedDict((int(e), True) for e in np.argsort(-u[l])[:C]) for l in range(NL)]

def touch(od, e, C):
    if e in od:
        od.move_to_end(e)
        return 0
    od[e] = True
    if len(od) > C:
        od.popitem(last=False)
    return 1

def run(ids, Cm, Cl, R, Tacc, nwait=2):
    msb, lsb = pools(Cm), pools(Cl)
    n = ids.shape[0]; pos, mm, ml, gen = 0.0, 0, 0, 0
    while int(pos) + R <= n:
        s = int(pos)
        for l in range(NL):
            need = set()
            for t in range(s, s + R):
                need.update(ids[t, l, :nwait].tolist())
            for e in need:
                mm += touch(msb[l], e, Cm)
                ml += touch(lsb[l], e, Cl)
            for t in range(s, s + R):                       # resident ranks 2..19 get touched (MSB pool LRU)
                for e in ids[t, l, nwait:20].tolist():
                    if e in msb[l]:
                        msb[l].move_to_end(e)
        gen += int(pos + Tacc) - s
        pos += Tacc
    return mm / gen, ml / gen

MB_MSB, MB_LSB = 2.15, 0.77           # MB per part (MSB-3 + bf16 scale / bias ~3.5 b; LSB-1 + its bias pool ~1.25 b)
say("b12c192 reconciliation (Breakthrough traces, second halves; LRU pools warm-started by usage on first halves)")
for test in ("code", "math", "chat"):
    n = IDS[test].shape[0]; ids = IDS[test][n // 2:]
    for R, Tacc, lab in ((4, 3.4, "MTP 4-row"), (16, 9.5, "16-row")):
        for Cm, Cl in ((192, 77), (192, 192), (192, 512)):
            m, l = run(ids, Cm, Cl, R, Tacc)
            say(f"  {test:5s} {lab:9s} C_m {Cm} C_l {Cl}: MSB misses {m:5.1f} + LSB misses {l:5.1f} = {m + l:5.1f} reads/token, "
                f"~{m * MB_MSB + l * MB_LSB:4.0f} MB/token")
say("Measured (Eval suite, b12c192 run 1, MTP): 27.8 reads/token (22.9 waited), 40 MB/token")
open(os.path.expanduser("~/Leap/data/b12rec29.txt"), "w").write("\n".join(lines) + "\n")
