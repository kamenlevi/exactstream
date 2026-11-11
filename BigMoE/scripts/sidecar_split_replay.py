"""Sidecar share at equal RAM, replayed (follow-up of the rank-priority replay, 11-02 21:2x; CPU, exact: misses and bytes).
The replay found the 4-bit LSB sidecar (C_l, 0.77 MB records) missing 101-166 times per check on code vs 40-108 base (C_m,
2.16 MB) misses: most of the reads. M-17 sized C_m / C_l per layer for SSD MB per kept token; if tonight's stall trace says
the chain is latency-bound (a cost per read or per stalled layer), the read count matters too. Here: per layer C_l x f and
C_m - (f - 1) x C_l x 0.77 / 2.16 (the same pool bytes), f = 1 (M-17) / 1.5 / 2 / 3, LRU both, at C 256 and C 320 sizes,
gate 0.99 on / off, "proxy" and "true" rows after the first miss; data and method as scripts/rankprio_replay.py.
Per check: base + sidecar misses (reads), MB, layers with any read (base or sidecar), layers with >= 2 reads.
usage: .venv/bin/python scripts/sidecar_split_replay.py DOMAIN | report -> data/rankprio/SIDECAR_read.txt"""
import json, os, sys, time
from collections import OrderedDict
import numpy as np

DM = os.path.expanduser("~/FableGod/decision_map/out"); TD = os.path.expanduser("~/TrainLab/data")
OUT = os.path.expanduser("~/BigMoE/data/rankprio")
SZ = {C: json.load(open(os.path.expanduser(f"~/BigMoE/data/seed45/laysz45_{C}.json"))) for C in (256, 320)}
MSB, LSB = (b / 1e6 for b in SZ[256]["slot_bytes"])
THETA, NL, FS = 0.99, 48, (1.0, 1.5, 2.0, 3.0)


def windows(dom):
    base = json.load(open(os.path.join(DM, "baseline.json")))
    Z = np.load(os.path.join(TD, f"d02b_walk_hpd0_{dom}.npz")); names = [str(x) for x in Z["names"]]
    W = []
    for i, nm in enumerate(names):
        if nm in base and base[nm]["dom"] == dom:
            W.append(dict(ids=np.load(os.path.join(DM, "routing", nm))["ids"].tolist(), a=base[nm]["a"],
                          n1=len(base[nm]["tk"]) + base[nm]["a"], checks=Z[f"{i}_checks"].tolist(), cf=Z[f"{i}_dcf"].tolist()))
    return W


def lru(c, need, C):
    m = 0
    for e in need:
        if e in c:
            c.move_to_end(e)
        else:
            c[e] = 1; m += 1
    while len(c) > C:
        c.popitem(last=False)
    return m


def sizes(C, f):
    cm, cl = SZ[C]["C_m"], SZ[C]["C_l"]
    return ([max(16, int(round(m - (f - 1) * l * LSB / MSB))) for m, l in zip(cm, cl)], [int(round(l * f)) for l in cl])


def replay(W, C, f, gate, after):
    cm, cl = sizes(C, f)
    P, S = [OrderedDict() for _ in range(NL)], [OrderedDict() for _ in range(NL)]
    T = dict(chk=0, kept=0, mb=0, ml=0, any=0, two=0)
    for w in W:
        ids, a, n1 = w["ids"], w["a"], w["n1"]; N = len(ids)
        for p in range(a):
            for l in range(NL):
                t = ids[p][l]; lru(P[l], sorted(t), cm[l]); lru(S[l], sorted(t[:2]), cl[l])
        for (k, m, A), cf in zip(w["checks"], w["cf"]):
            last = min(A + m, n1 - 1); T["chk"] += 1; T["kept"] += k + 1
            rows = [((p if (after == "true" or p <= A + k) else (p + 97) % N),
                     2 if gate and (p - A) < m and cf[p - A] >= THETA else 10) for p in range(A, last + 1)]
            for l in range(NL):
                need, h = set(), set()
                for q, kk in rows:
                    t = ids[q][l]; need.update(t[:kk]); h.update(t[:2])
                mb_, ml_ = lru(P[l], sorted(need), cm[l]), lru(S[l], sorted(h), cl[l])
                T["mb"] += mb_; T["ml"] += ml_; T["any"] += (mb_ + ml_) > 0; T["two"] += (mb_ + ml_) >= 2
    c = T["chk"]
    return dict(base=T["mb"] / c, side=T["ml"] / c, reads=(T["mb"] + T["ml"]) / c, MB=(T["mb"] * MSB + T["ml"] * LSB) / c,
                any=T["any"] / c, two=T["two"] / c, kept=T["kept"] / c,
                gb=(sum(cm) * MSB + sum(cl) * LSB) / 1e3)


CFG = [(C, f, g, af) for C in (320, 256) for g in (True, False) for af in ("proxy", "true") for f in FS]
if sys.argv[1] != "report":
    dom = sys.argv[1]; W = windows(dom); t0 = time.time(); R = {}
    for C, f, g, af in CFG:
        R[f"{C}|{f}|{int(g)}|{af}"] = r = replay(W, C, f, g, af)
        print(f"{dom} C {C} f {f} gate {int(g)} {af}: {r['base']:.1f} + {r['side']:.1f} reads, {r['MB']:.0f} MB, "
              f"{r['any']:.1f} layers with a read ({time.time() - t0:.0f} s)", flush=True)
    json.dump(R, open(os.path.join(OUT, f"sidecar_{dom}.json"), "w"), indent=1)
    sys.exit(0)
L = []
def say(s=""):
    print(s); L.append(s)
R = {d: json.load(open(os.path.join(OUT, f"sidecar_{d}.json"))) for d in ("code", "ifeval", "multi")
     if os.path.exists(os.path.join(OUT, f"sidecar_{d}.json"))}
say("Sidecar share at equal pool bytes (C_l x f, C_m reduced by the same bytes; LRU; per check). f 1 = M-17's sizes")
say("| domain | C | gate | rows | f | pool GB | base reads | sidecar reads | reads | MB | layers with a read | layers >= 2 reads |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d, Rd in R.items():
    for C, f, g, af in CFG:
        r, o = Rd[f"{C}|{f}|{int(g)}|{af}"], Rd[f"{C}|1.0|{int(g)}|{af}"]
        say(f"| {d} | {C} | {'on' if g else 'off'} | {af} | {f} | {r['gb']:.2f} | {r['base']:.1f} | {r['side']:.1f} | "
            f"{r['reads']:.1f} ({100 * (r['reads'] / o['reads'] - 1):+.0f} %) | {r['MB']:.0f} ({100 * (r['MB'] / o['MB'] - 1):+.0f} %) | "
            f"{r['any']:.1f} | {r['two']:.1f} |")
open(os.path.join(OUT, "SIDECAR_read.txt"), "w").write("\n".join(L) + "\n")
