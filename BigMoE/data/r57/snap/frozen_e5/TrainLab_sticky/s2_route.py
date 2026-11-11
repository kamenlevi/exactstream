"""S2 (Fable B's pre-registered screen, ~/FableGod/push/ROW_WALL.md sec 5; run by MTP Head on the orchestrator's routing, Tue
11-03 evening, CPU): D2 as a ROUTE PREDICTOR. On BigMoE's 45 GB capture (cap45) through the adopted C 256 per-layer pools
(~/MoonShots/data/laysz45_256.json; LRU + 30 x popularity, res0 forced per prompt, prompts 2-6 counted: exactly Fable B's
row_split.py replay, rows capped at the live chain's 10), for each check: the exact pass's misses on rows 0-9, vs the
requests of a cheap chain over rows 1-9 one cycle early: a TOP-2 pass (ranks 0-1 routes: MSB ranks 0-1 + the LSB sidecar)
and an ALL-RESIDENT pass (all ranks 0-9 + the sidecar). Row 0 (the base union) is never seen by the cheap chain. The cheap
chain's routes = the exact routes (route drift not modelled: cap45 has no cheap-pass states), so requests per hit are 1.00
by construction; drift d is shown as a sensitivity (recall x (1 - d), d / (1 - d) wasted reads per hit).
LINE (Fable B): PASS top-2 recall >= 0.45 with <= 1.3 requests per hit AND the model >= +10 % on code; KILL recall < 0.30 OR
the model < +4 %; BETWEEN otherwise. Model = ROW_WALL sec 3 (tree_wide.py's today()): draft 12 + dense 34 + 3.0 x rows +
0.157 x misses + cheap rows x ms + dense extra + rest 5; D2 = 16 cheap rows (top-2 0.6 ms / all-resident 2.64 ms) + dense
extra 9.8, misses x (1 - recall); reported with D2's own kept +15 % assumption (the pre-verified chain) and route-only
(kept unchanged). Output: data/s2_route.txt (appended)."""
import os, sys, json, heapq, time
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import laysz45 as L                                         # read-only use (MoonShots' loader, as row_split.py)

t0 = time.time()
P, agree = L.load()
S = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))
CM, CL = S["C_m"], S["C_l"]
NL, R4 = L.NL, L.R4
Q = L.needs(P, lambda c: 16)
POP = {"m": L.popularity(Q, "m"), "l": L.popularity(Q, "l")}
B = 30.0
ROWS = 10                                                   # the live chain's verified rows (code ~9.84)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
say(f"S2 route predictor: cap45 loaded in {time.time() - t0:.0f} s; router self-check {agree:.3f}")


def replay(l, key):
    """one layer, one pool: per check -> (misses, row-0 only, named by top-2 rows 1-9, named by all-resident rows 1-9)"""
    K = CM[l] if key == "m" else CL[l]
    st, i, out = {}, 0, []
    for p, (name, cks, res0) in enumerate(P):
        pop = POP[key][p][l]
        if key == "m":
            i += 1
            for e in res0[l]:
                st[e] = float(i)
        for t, c in enumerate(cks):
            rid = c["rid"][l]
            rows = min(rid.shape[0], ROWS)
            i += 1
            need, top2, allr, row0 = set(), set(), set(), set()
            for j in range(rows):
                rr = rid[j][:R4] if key == "l" else rid[j]
                for r, e in enumerate(rr):
                    e = int(e)
                    need.add(e)
                    if j == 0:
                        row0.add(e)
                    else:
                        allr.add(e)
                        if r < R4:
                            top2.add(e)
            miss = {e for e in need if e not in st}
            out.append((p, len(miss), len((miss & row0) - allr), len(miss & top2), len(miss & allr), len(miss & top2 & row0)))
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + B * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                    del st[e]
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                        del st[e]
    return out


t1 = time.time()
RES = {k: [replay(l, k) for l in range(NL)] for k in ("m", "l")}
say(f"replay {time.time() - t1:.0f} s")
names = [n for n, _, _ in P]
N = len(RES["m"][0])
dom = ["code" if "humaneval" in names[RES["m"][0][k][0]] else "math" for k in range(N)]
counted = [k for k in range(N) if RES["m"][0][k][0] >= 1]
agg = np.zeros((N, 5))                                      # misses, row-0 only, top-2 named, all-resident named, top-2 & row 0
for key in ("m", "l"):
    for l in range(NL):
        for k in range(N):
            agg[k] += RES[key][l][k][1:]

ROW_GPU, MS_MISS, DENSE, DRAFT, REST = 3.0, 0.157, 34.0, 12.0, 5.0
def today(rows, misses, cheap=0, cheap_ms=0.6, dense_extra=0.0):
    return DRAFT + DENSE + dense_extra + ROW_GPU * rows + MS_MISS * misses + cheap * cheap_ms + REST
res = {}
for d in ("code", "math", "all"):
    ks = [k for k in counted if d == "all" or dom[k] == d]
    a = agg[ks].sum(0); n = len(ks)
    m, r0, t2, ar = a[0] / n, a[1] / n, a[2] / a[0], a[3] / a[0]
    say(f"  {d:4s} ({n} checks, rows 0-9): {m:.0f} misses per check (MSB + LSB), of them {r0:.0f} ({r0 / m:.0%}) only on row 0 (the "
        f"base union, unseen); recall of the exact pass's misses: TOP-2 chain {t2:.3f}, ALL-RESIDENT chain {ar:.3f}; requests per hit "
        f"1.00 (no drift modelled)")
    res[d] = (m, t2, ar)
base_rows, base_kept = 9.84, 10.54                          # D-23d's live code chain (ROW_WALL / tree_wide)
m, t2, ar = res["code"]
b = today(base_rows, m)
say(f"  model, code (rows {base_rows}, kept {base_kept}, the replay's {m:.0f} misses): base {b:.0f} ms -> {1e3 * base_kept / b:.1f} tok/s")
for lab, rec, ms in (("TOP-2 (0.6 ms rows)", t2, 0.6), ("ALL-RESIDENT (2.64 ms rows)", ar, 2.64)):
    for dr in (0.0, 0.1, 0.2):
        r_ = rec * (1 - dr)
        ct = today(base_rows, m * (1 - r_), 16, ms, 9.8)
        say(f"    {lab:27s} drift {dr:.1f}: recall {r_:.3f}, {dr / (1 - dr) + 1:.2f} requests per hit; cycle {ct:.0f} ms; "
            f"route-only {100 * (b / ct - 1):+.1f} %; with D2's kept +15 % {100 * (1.15 * b / ct - 1):+.1f} %")
tp = 1.15 * b / today(base_rows, m * (1 - t2), 16, 0.6, 9.8) - 1
v = "PASS" if (t2 >= 0.45 and tp >= 0.10) else ("KILL" if (t2 < 0.30 or tp < 0.04) else "BETWEEN")
say(f"  LINE (Fable B: top-2 recall >= 0.45, <= 1.3 requests per hit, model >= +10 % on code PASS; recall < 0.30 or model < +4 % "
    f"KILL): top-2 recall {t2:.3f} (no drift), model {100 * tp:+.1f} % with D2's kept +15 % ({100 * (b / today(base_rows, m * (1 - t2), 16, 0.6, 9.8) - 1):+.1f} % route-only) -> {v}")
open(os.path.expanduser("~/TrainLab/data/s2_route.txt"), "a").write("\n".join(lines) + "\n\n")
