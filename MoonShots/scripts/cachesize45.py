> (a quote from the project owner, kept private)
45 GB default under ~15 SSD misses per kept token? CPU replay, the M-17 harness (scripts/laysz45.py) past C 256.
Pools per layer as the preset splits them (stream._tier_layer, BIGMOE_TIER_CL 112): C_m = (C x 2,764,800 - 112 x 770,048) //
2,162,688 MSB-3 base slots + 112 LSB sidecar slots; RAM = REF2-T's lifetime peak 38.0 GB at C 224 + 0.13 GB per C unit.
cap45's six prompts CHAINED (res0 forced into the base pool per prompt, prompts 2-6 counted), strict pools.
Policies: LRU, P30 = LRU + 30 x popularity (SEED45's leave-one-prompt-out key), MIN = Belady (farthest next use leaves
first, as seed45_replay.py), and LAYSZ = P30 with M-17's per-layer (C_m, C_l) at the same pool bytes (held-out: sizes from
the other counted prompts; floor C_m >= 160, C_l >= 32).
Rows: ALL = the check's 16 rows; KEPT = only the kept prefix + 1 (R-62P's oracle).
Per C and policy: base (MSB) misses per kept token (SEED45's metric), sidecar (LSB) misses, MB per kept token (2,162,688 /
770,048 B per miss) and the SSD-side ceiling 13 GB/s / MB per kept token.
Caveat: cap45 was captured at C 160; the routing trajectory is the same at any C (exact), but each prompt starts from its
captured C 160 res0, not from what a larger cache would hold.
usage: [BigMoE venv] taskpolicy -b python cachesize45.py (CPU, 4 workers, ~2-5 min) -> stdout + data/cachesize45.txt"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import bisect, heapq, multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
sys.argv = sys.argv[:1]
import laysz45 as LZ                                    # its run is guarded: only the functions load

NL, MR, LR, H = LZ.NL, LZ.MR, LZ.LR, LZ.H
CS = (224, 256, 278, 300, 320, 340, 360, 380, 400, 432)
KM = {C: min(512, (C * 2764800 - 112 * LR) // MR) for C in CS}
GB = {C: 38.0 + 0.13 * (C - 224) for C in CS}
GM = sorted(set(LZ.GM) | set(KM.values()))
GL = LZ.GL
TARGET_MISS, TARGET_MB, SSD = 15.0, 13000 / 350, 13000.0
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay(task):
    """misses per prompt for one layer's pool at size K: LRU + b x popularity (pol 'p') or Belady (pol 'min'), strict"""
    var, l, key, K, pol, b = task
    while os.path.exists(PAUSE):
        time.sleep(20)
    if pol == "p":
        return LZ.replay((var, l, key, K, b))
    P = DATA[var][0]
    items = []
    for p, (_, cks, res0) in enumerate(P):
        if key == "m":
            items.append((p, -1, res0[l]))
        items += [(p, t, c[key][l]) for t, c in enumerate(cks)]
    uses = {}
    for i, (_, _, need) in enumerate(items):
        for e in need:
            uses.setdefault(e, []).append(i)
    miss, st = np.zeros(len(P)), set()
    for i, (p, t, need) in enumerate(items):
        if t >= 0:
            miss[p] += sum(1 for e in need if e not in st)
        st |= need
        if len(st) > K:
            def far(e):
                u = uses[e]; k = bisect.bisect_right(u, i)
                return -(u[k] if k < len(u) else 1e9)
            for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=far):
                st.discard(e)
            if len(st) > K:
                for e in heapq.nsmallest(len(st) - K, list(st), key=far):
                    st.discard(e)
    return miss


t0 = time.time()
P0, agr = LZ.load()
DATA = {}
for var, rows_of in (("ALL", lambda c: 16), ("KEPT", lambda c: int(min(16, c["kept"] + 1)))):
    Q = LZ.needs(P0, rows_of)
    DATA[var] = (Q, {"m": LZ.popularity(Q, "m"), "l": LZ.popularity(Q, "l")})
LZ.DATA = DATA                                          # LZ.replay reads its module's DATA (the workers fork after this)
KEPT =np.array([sum(c["kept"] for c in cks) for _, cks, _ in DATA["ALL"][0]], float)
kt = KEPT[list(H)].sum()
say(f"CACHE-SIZE45: cap45 chained, {sum(len(P0[p][1]) for p in H)} checks / {kt:.0f} kept tokens counted (prompts 2-6); "
    f"rows per check ALL 16, KEPT {np.mean([c['rows'] for p in H for c in DATA['KEPT'][0][p][1]]):.2f}; load {time.time() - t0:.0f} s")
POL = {"LRU": ("p", 0.0), "P30": ("p", 30.0), "MIN": ("min", 0.0)}
tasks = [(var, l, "m", K, *POL[pn]) for var in DATA for pn in POL for l in range(NL) for K in sorted(set(KM.values()))] + \
        [(var, l, "l", 112, *POL[pn]) for var in DATA for pn in POL for l in range(NL)] + \
        [(var, l, key, K, "p", 30.0) for var in DATA for key, g in (("m", GM), ("l", GL)) for l in range(NL) for K in g]
tasks = list(dict.fromkeys(tasks))
with mp.get_context("fork").Pool(4) as pool:
    res = pool.map(replay, tasks, chunksize=8)
R = dict(zip(tasks, res))
say(f"  replays: {len(tasks)} ({time.time() - t0:.0f} s)")


def tot(var, key, K, pn, sets=H):
    return sum(R[(var, l, key, K, *POL[pn])][list(sets)].sum() for l in range(NL))


# LAYSZ: M-17's Lagrangian on the P30 curves (LZ.alloc reads LZ.CUR / LZ.KEPT / LZ.GM / LZ.GL)
LZ.GM, LZ.GL, LZ.KEPT = GM, GL, KEPT
LZ.CUR = {}
for var in DATA:
    for key, g in (("m", GM), ("l", GL)):
        LZ.CUR[(var, "P30", key)] = np.stack([np.stack([R[(var, l, key, K, "p", 30.0)] for K in g]) for l in range(NL)])


def laysz(var, C):
    """held-out (leave one counted prompt out) base / sidecar misses per kept token with per-layer sizes at C's pool bytes"""
    budget = NL * (KM[C] * MR + 112 * LR)
    mm = ml = 0.0
    for h in H:
        i, j, _ = LZ.alloc((var, "P30"), tuple(x for x in H if x != h), budget, 160, 32)
        mm += LZ.CUR[(var, "P30", "m")][np.arange(NL), i][:, h].sum()
        ml += LZ.CUR[(var, "P30", "l")][np.arange(NL), j][:, h].sum()
    return mm / kt, ml / kt


tab = {}
for var in DATA:
    say(f"{var} rows: per C (RAM = 38.0 GB at C 224 + 0.13 GB per C unit; base slots C_m + 112 sidecar slots per layer): base "
        f"misses / sidecar misses / MB per kept token / SSD ceiling at 13 GB/s")
    for C in CS:
        row = []
        for pn in ("LRU", "P30", "MIN", "LAYSZ"):
            if pn == "LAYSZ":
                m, s = laysz(var, C)
            else:
                m, s = tot(var, "m", KM[C], pn) / kt, tot(var, "l", 112, pn) / kt
            mb = (m * MR + s * LR) / 1e6
            tab[(var, C, pn)] = (m, s, mb)
            row.append(f"{pn} {m:5.1f} / {s:4.1f} / {mb:5.1f} MB / {SSD / mb:4.0f} tok/s")
        say(f"  C {C} ({GB[C]:.1f} GB, C_m {KM[C]}): " + "; ".join(row))


def first(var, pn, f, target):
    """smallest C meeting the target, linearly interpolated between grid points (None if not within the C list)"""
    prev = None
    for C in CS:
        v = f(tab[(var, C, pn)])
        if v <= target:
            if prev is None:
                return C, v
            c0, v0 = prev
            return c0 + (C - c0) * (v0 - target) / (v0 - v), v
        prev = (C, v)
    return None


for var in DATA:
    for pn in ("P30", "MIN", "LAYSZ", "LRU"):
        out = []
        for what, f, tg in (("<= 15 base misses per kept token", lambda x: x[0], TARGET_MISS),
                            (f"<= {TARGET_MB:.1f} MB per kept token (350 tok/s at 13 GB/s)", lambda x: x[2], TARGET_MB)):
            r = first(var, pn, f, tg)
            out.append(f"{what}: " + (f"C ~{r[0]:.0f} ({38.0 + 0.13 * (r[0] - 224):.1f} GB)" if r else f"not by C {CS[-1]}"))
        say(f"ANSWER {var} {pn}: " + "; ".join(out))
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/cachesize45.txt"), "w").write("\n".join(lines) + "\n")
