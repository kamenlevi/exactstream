"""M-17 cross-domain check (MoonShots, Sun 2026-11-01): do the per-layer pool sizes calibrated on cap45 (code + math only;
data/laysz45_<C>.json, FLOOR arm, P30) also cut SSD bytes on other domains? R-27's engine A/B at 24 GB (b36) lost most of its
replay gain on prompts that included prose and edit, so this is the first thing to rule out before an engine A/B.
Data (read only): Breakthrough's expA traces (bulgarian / chat / code / math, 800 tokens of the 125B's own routing, top-10 in
rank order; ~/Breakthrough/data/expA/traces125), cut into 16-row checks advancing 6 tokens (cap45's 6.11 kept per check) by
wall24.needs (base needs = the rows' top-10 union, sidecar needs = experts at rank 0-1 in some row); tokens 0-399 warm the
pools, checks starting at 400+ are measured. Each set replayed alone from empty pools, LRU + 30 x popularity (cap45's
popularity over all six prompts, as the live prior file), strict pools.
Per set and C (224 / 256): SSD MB per kept token at the uniform split, with cap45's per-layer sizes, and with the set's own
per-layer optimum (in-sample, Lagrangian at the same bytes: what is left on the table).
usage: [BigMoE venv] taskpolicy -b python xdom45.py (CPU, 4 workers, ~3 min) -> stdout + data/xdom45.txt"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
import laysz45 as LZ
import wall24 as W

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
R, A, B = 16, 6, 30.0
UNI = LZ.UNI
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay(task):
    name, l, key, K = task
    while os.path.exists(PAUSE):
        time.sleep(20)
    pop = POP[key][l]
    miss, st = 0, {}
    for i, (meas, lay) in enumerate(CH[name]):
        need = lay[l][0 if key == "m" else 1]
        if meas:
            miss += sum(1 for e in need if e not in st)
        for e in need:
            st[e] = float(i + 1)
        if len(st) > K:
            kf = lambda e: st[e] + B * pop[e]
            for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                del st[e]
            if len(st) > K:
                for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                    del st[e]
    return miss


t0 = time.time()
P0, _ = LZ.load()
Q = LZ.needs(P0, lambda c: 16)
POP = {}
for key in ("m", "l"):                                  # cap45's popularity over all six prompts (share of checks)
    pop, n = np.zeros((NL, LZ.E)), 0
    for _, cks, _ in Q:
        for c in cks:
            for l in range(NL):
                pop[l, list(c[key][l])] += 1
        n += len(cks)
    POP[key] = [list(map(float, pop[l] / n)) for l in range(NL)]
CH = {k: W.needs(ids, R, A) for k, ids in W.IDS.items()}
GM, GL = LZ.GM, LZ.GL
SZ = {C: json.load(open(os.path.expanduser(f"~/MoonShots/data/laysz45_{C}.json"))) for C in UNI}
tasks = [(k, l, key, K) for k in CH for key, g in (("m", GM), ("l", GL)) for l in range(NL) for K in g]
with mp.get_context("fork").Pool(4) as pool:
    res = pool.map(replay, tasks, chunksize=16)
say(f"M-17 cross-domain: expA sets {', '.join(f'{k} {sum(m for m, _ in CH[k])} measured checks' for k in CH)} "
    f"(16 rows, advance {A}); {len(tasks)} replays ({time.time() - t0:.0f} s)")
LZ.GM, LZ.GL = GM, GL
for k in CH:
    kept = A * sum(m for m, _ in CH[k])
    cm = np.zeros((NL, len(GM), 1)); cl = np.zeros((NL, len(GL), 1))
    for (n, l, key, K), m in zip(tasks, res):
        if n == k:
            (cm if key == "m" else cl)[l, (GM if key == "m" else GL).index(K), 0] = m
    LZ.CUR = {(k, "P30", "m"): cm, (k, "P30", "l"): cl}
    LZ.KEPT = np.array([kept], float)
    out = []
    for C, (um, ul) in UNI.items():
        mb = lambda i, j: (cm[np.arange(NL), i, 0].sum() * MR + cl[np.arange(NL), j, 0].sum() * LR) / kept / 1e6
        u = mb(np.full(NL, GM.index(um)), np.full(NL, GL.index(ul)))
        x = mb(np.array([GM.index(c) for c in SZ[C]["C_m"]]), np.array([GL.index(c) for c in SZ[C]["C_l"]]))
        i, j, _ = LZ.alloc((k, "P30"), (0,), NL * (um * MR + ul * LR), 160, 32)
        o = mb(i, j)
        out.append(f"C {C}: uniform {u:.1f} -> cap45 sizes {x:.1f} ({100 * (x / u - 1):+.1f} %), own optimum {o:.1f} "
                   f"({100 * (o / u - 1):+.1f} %)")
    say(f"  {k}: " + "; ".join(out))
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/xdom45.txt"), "w").write("\n".join(lines) + "\n")
