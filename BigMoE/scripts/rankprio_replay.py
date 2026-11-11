"""Rank-priority eviction at 45 GB, replayed (the orchestrator 11-02 21:0x on cloud task 12, ~/FableGod/CLOUD_ANSWER_SSD45.md:
"rank-aware residency as a priority (protect ranks 0-1, evict 2-9 first) -1.8 ms"). CPU only, exact: misses and bytes.
The engine's pools as R57_PRESET=45 sizes them (M-17 data/seed45/laysz45_<C>.json): per layer a base pool of C_m MSB-3
records (2.16 MB, every rank) and an LSB sidecar of C_l records (0.77 MB, ranks 0-1 = the 4-bit tier), both LRU today.
Data and method as Breakthrough's dm_rankcache.py / Fable's cache_replay.py (DAY9's routing capture, the HPD-0 walks' real
check boundaries, 12 windows per domain replayed in order as one session, prompt rows one by one). The gate = the
realizable one (decision_map/dm_gate.py): row A + j of a check is top-2 when the drafter's cf[j] >= 0.99 (j < m; prompt
rows and the last row are exact top-10). Rows after the first miss: "proxy" = row p + 97's routing (a wrong draft's
stand-in; the pre-registered variant), "true" = their own.
Policies on the base pool (the sidecar holds ranks 0-1 only, LRU in every arm):
  lru    today's: every needed expert refreshed, the least recent evicted
  rank   two classes by the latest use (rank 0-1 by any row of the step = protected); evict the least recent unprotected
         first, a protected one only when none is left
  rank75 the same with the protected class capped at 3/4 of C_m (its least recent demoted to the unprotected end)
PRE-REGISTERED (11-02 21:2x, before any run): code, C 320 sizes, gate 0.99, proxy rows, rank vs lru, MB per check
(base + sidecar): PASS <= -6 % with C 256 + gate not up; KILL > -2 % (flat, as Breakthrough's C 160 total); else BETWEEN.
Also per check: layers with >= 2 base misses (the cloud's chain unit: two unprefetched misses stall a layer).
usage: .venv/bin/python scripts/rankprio_replay.py DOMAIN   (one process per domain), then
       .venv/bin/python scripts/rankprio_replay.py report   -> data/rankprio/RANKPRIO_read.txt"""
import json, os, sys, time
from collections import OrderedDict
import numpy as np

DM = os.path.expanduser("~/FableGod/decision_map/out"); TD = os.path.expanduser("~/TrainLab/data")
OUT = os.path.expanduser("~/BigMoE/data/rankprio"); os.makedirs(OUT, exist_ok=True)
SZ = {C: json.load(open(os.path.expanduser(f"~/BigMoE/data/seed45/laysz45_{C}.json"))) for C in (256, 320)}
MSB, LSB = (b / 1e6 for b in SZ[256]["slot_bytes"])      # MB per base / sidecar record
THETA, NL = 0.99, 48


def windows(dom):
    base = json.load(open(os.path.join(DM, "baseline.json")))
    Z = np.load(os.path.join(TD, f"d02b_walk_hpd0_{dom}.npz")); names = [str(x) for x in Z["names"]]
    W = []
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom:
            continue
        ids = np.load(os.path.join(DM, "routing", nm))["ids"].tolist()
        W.append(dict(nm=nm, ids=ids, a=base[nm]["a"], n1=len(base[nm]["tk"]) + base[nm]["a"],
                      checks=Z[f"{i}_checks"].tolist(), cf=Z[f"{i}_dcf"].tolist()))
    return W


class Pool:
    """one layer's pool; access(need, hi) -> misses (need: ids of the step, hi: those used at ranks 0-1)"""
    def __init__(self, C, policy):
        self.C, self.pol, self.a, self.b = C, policy, OrderedDict(), OrderedDict()   # a = protected (rank), b = the rest
        self.cap = 3 * C // 4 if policy == "rank75" else C

    def access(self, need, hi):
        a, b, m = self.a, self.b, 0
        if self.pol == "lru":
            for e in need:
                if e in b:
                    b.move_to_end(e)
                else:
                    b[e] = 1; m += 1
            while len(b) > self.C:
                b.popitem(last=False)
            return m
        for e in need:
            if e in a:
                del a[e]
            elif e in b:
                del b[e]
            else:
                m += 1
            (a if e in hi else b)[e] = 1
        while len(a) > self.cap:                         # rank75: the protected class's least recent -> unprotected
            e = next(iter(a))
            if e in need:
                break
            del a[e]; b[e] = 1
        while len(a) + len(b) > self.C:
            for seg in (b, a):
                if seg:
                    e = next(iter(seg))
                    if e not in need:
                        del seg[e]; break
            else:
                raise RuntimeError("no victim")
        return m


def replay(W, C, gate, after, policy):
    sz = SZ[C]
    P = [Pool(sz["C_m"][l], policy) for l in range(NL)]; S = [Pool(sz["C_l"][l], "lru") for l in range(NL)]
    T = dict(chk=0, kept=0, mb=0, ml=0, l2=0, l1=0, rows=0, gated=0)
    for w in W:
        ids, a, n1 = w["ids"], w["a"], w["n1"]; N = len(ids)
        for p in range(a):                               # prefill, row by row (exact top-10)
            r = ids[p]
            for l in range(NL):
                t = r[l]; h = set(t[:2])
                P[l].access(sorted(t), h); S[l].access(sorted(h), h)
        for (k, m, A), cf in zip(w["checks"], w["cf"]):
            last = min(A + m, n1 - 1); T["chk"] += 1; T["kept"] += k + 1
            rows = []
            for p in range(A, last + 1):
                j = p - A
                q = p if (after == "true" or p <= A + k) else (p + 97) % N
                g = gate and j < m and cf[j] >= THETA
                rows.append((q, 2 if g else 10)); T["rows"] += 1; T["gated"] += g
            for l in range(NL):
                need, h = set(), set()
                for q, kk in rows:
                    t = ids[q][l]; need.update(t[:kk]); h.update(t[:2])
                mb_ = P[l].access(sorted(need), h); ml_ = S[l].access(sorted(h), h)
                T["mb"] += mb_; T["ml"] += ml_; T["l2"] += mb_ >= 2; T["l1"] += mb_ >= 1
    c = T["chk"]
    return dict(checks=c, kept=T["kept"] / c, base=T["mb"] / c, side=T["ml"] / c, MB=(T["mb"] * MSB + T["ml"] * LSB) / c,
                mpk=T["mb"] / T["kept"], l2=T["l2"] / c, l1=T["l1"] / c, gated=T["gated"] / max(T["rows"], 1))


CFG = [(C, g, af, pol) for C in (320, 256) for g in (True, False) for af in ("proxy", "true") for pol in ("lru", "rank", "rank75")]

if sys.argv[1] != "report":
    dom = sys.argv[1]; W = windows(dom); t0 = time.time(); R = {}
    print(f"{dom}: {len(W)} windows, {sum(len(w['checks']) for w in W)} checks", flush=True)
    for C, g, af, pol in CFG:
        R[f"{C}|{int(g)}|{af}|{pol}"] = r = replay(W, C, g, af, pol)
        print(f"  C {C} gate {int(g)} {af} {pol}: {r['base']:.1f} base + {r['side']:.1f} side misses, {r['MB']:.0f} MB per check, "
              f"{r['l2']:.1f} layers >= 2 ({time.time() - t0:.0f} s)", flush=True)
    json.dump(R, open(os.path.join(OUT, f"rankprio_{dom}.json"), "w"), indent=1)
    sys.exit(0)

L = []
def say(s=""):
    print(s); L.append(s)
R = {d: json.load(open(os.path.join(OUT, f"rankprio_{d}.json"))) for d in ("code", "ifeval", "multi")
     if os.path.exists(os.path.join(OUT, f"rankprio_{d}.json"))}
say("Rank-priority eviction vs today's LRU at 45 GB (M-17 per-layer pools; per check: base / sidecar misses, MB at "
    f"{MSB:.2f} / {LSB:.2f} MB, layers with >= 2 base misses; ms at 0.088 ms per MB)")
say("| domain | C | gate 0.99 | after-miss rows | policy | base misses | sidecar misses | MB per check | vs lru | ~ms | layers >= 2 | layers >= 1 | base misses per kept | gated rows |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d, Rd in R.items():
    for C, g, af, pol in CFG:
        r, o = Rd[f"{C}|{int(g)}|{af}|{pol}"], Rd[f"{C}|{int(g)}|{af}|lru"]
        say(f"| {d} | {C} | {'on' if g else 'off'} | {af} | {pol} | {r['base']:.1f} | {r['side']:.1f} | {r['MB']:.1f} | "
            f"{100 * (r['MB'] / o['MB'] - 1):+.1f} % | {0.088 * r['MB']:.1f} | {r['l2']:.1f} | {r['l1']:.1f} | {r['mpk']:.2f} | "
            f"{100 * r['gated']:.0f} % |")
if "code" in R:
    c = R["code"]; o, n = c["320|1|proxy|lru"], c["320|1|proxy|rank"]; o2, n2 = c["256|1|proxy|lru"], c["256|1|proxy|rank"]
    d1, d2 = n["MB"] / o["MB"] - 1, n2["MB"] / o2["MB"] - 1
    v = "KILL" if d1 > -0.02 else "PASS" if d1 <= -0.06 and d2 <= 0 else "BETWEEN"
    say(f"\n## Verdict (pre-registered): code, C 320, gate 0.99, proxy rows: MB per check {o['MB']:.1f} -> {n['MB']:.1f} "
        f"({100 * d1:+.1f} %, ~{0.088 * (n['MB'] - o['MB']):+.1f} ms), layers >= 2 base misses {o['l2']:.1f} -> {n['l2']:.1f}; "
        f"C 256 + gate {100 * d2:+.1f} % -> {v} (PASS <= -6 % with C 256 not up, KILL > -2 %)")
open(os.path.join(OUT, "RANKPRIO_read.txt"), "w").write("\n".join(L) + "\n")
