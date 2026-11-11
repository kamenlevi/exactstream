"""price_copy_rule.py (Fable FULL, LOSSY350): what a wide copy pass computes under each row rule, i.e. how much part A
leans on the split form. 12 code map windows, 45 GB pools (plain LRU per layer, warmed by the prompt's rows, Fable C's
conventions). Blocks = the answer's verbatim prompt-copy run (after an 8-token trigger) cut into W rows, every row drafted
with cf 1.0 (a copied row). Rules: EXACT (10 ranks), QOM0BQ (ranks 0-1 + members + residents: today's preset),
STRICT (ranks 0-1 + members: the split form's rule for sure rows). Tiles = distinct kept experts per block over 48 layers.
The npz is the project's own walk (allow_pickle for its names array only). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I price_copy_rule.py > price_copy_rule.out"""
import os, sys
from collections import OrderedDict
import numpy as np
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/FableGod/fablec")
from bytes10_replay import windows, POOLS, MR  # noqa: E402
WK = np.load(f"{H}/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)
NAMES = [str(x) for x in WK["names"]]
TRIG = 8
def copy_run(nm, N):
    wi = NAMES.index(nm); ids = WK[f"{wi}_ids"].astype(int).tolist(); ck = WK[f"{wi}_checks"]
    g0 = int(ck[0, 2]) + 1; P = ids[:g0]; A = ids[g0:][:max(0, N - g0)]
    pos = {}
    for i, t in enumerate(P): pos.setdefault(t, []).append(i)
    q = 0
    while q < len(A):
        best = 0
        for i in pos.get(A[q], ()):
            L = 0
            while q + L < len(A) and i + L < len(P) and A[q + L] == P[i + L]: L += 1
            best = max(best, L)
        if best >= 8: return g0 + q + TRIG, g0 + q + best
        q += 1
    return None
W = windows()["code"]; Cm = POOLS["45"][0]
print("wide copy passes on the 12 code map windows, 45 GB pools: tiles per block (per copied row) and loads that miss")
for Wd in (16, 32, 64, 128):
    for rule in ("EXACT", "QOM0BQ", "STRICT"):
        tiles = miss = rows = blocks = 0
        for w in W:
            ids = w["ids"]; a = w["a"]; run = copy_run(w["nm"], len(ids))
            if run is None: continue
            pools = [OrderedDict() for _ in range(48)]
            for l in range(48):
                c = pools[l]
                for e in ids[:a, l].ravel().tolist():
                    if e in c: c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > Cm[l]: c.popitem(last=False)
            s, e_ = run
            for b in range(s, e_, Wd):
                E = ids[b:min(b + Wd, e_)]; R = len(E)
                if R < min(Wd, 16) // 2: continue
                blocks += 1; rows += R; k0 = 10 if rule == "EXACT" else 2
                for l in range(48):
                    c = pools[l]; loads = set(E[:, l, :k0].ravel().tolist())
                    for x in loads:
                        if x in c: c.move_to_end(x)
                        else:
                            miss += 1; c[x] = 1
                            if len(c) > Cm[l]: c.popitem(last=False)
                    keep = set(loads)
                    if rule == "QOM0BQ":
                        for x in E[:, l, 2:].ravel().tolist():
                            if x in c: keep.add(x)
                    for x in keep:
                        if x in c: c.move_to_end(x)
                    tiles += len(keep)
        print(f"W {Wd:3d} {rule:7s}: {blocks:3d} blocks, {rows / blocks:5.1f} rows; tiles per block {tiles / blocks:6.0f} ({tiles / rows:5.1f} per row); loads that miss per block {miss / blocks:5.1f} ({miss * MR / 1e6 / rows:.2f} MB per copied token)")
