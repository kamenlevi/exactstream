#!/usr/bin/env python3 -I
"""wall_union_width.py (FableGod, mission 14). CPU, no numpy. From DAY9's routing capture (decision_map/out/routing)
and floor_rows.json: the expert union per block at 16 / 32 / 64 rows on code under exact, the realizable gate
(cf >= 0.999 top-2; NEW-1 bands), the 125B-sure oracle and top-2 everywhere, in records and MB at the 45 GB tiers,
and the MB per kept token at the kept a chain drafter gives at each width: kept(B, a) = (a - a^(B+1)) / (1 - a),
a = 0.93 (DFlash2 today: 10.5 at 16) and a = 0.96 (DFlash3 + HPD, estimated)."""
import zipfile, struct, array, json, os
from collections import defaultdict
OUT = os.path.expanduser("~/FableGod/decision_map/out")
rows = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))); R = defaultdict(dict)
for r in rows: R[r["nm"]][r["j"]] = r
base = json.load(open(f"{OUT}/baseline.json"))
def load_npy(z, name):
    d = z.read(name); hl = struct.unpack("<H", d[8:10])[0]; body = d[10 + hl:]
    a = array.array("h"); a.frombytes(body); return a
W = {}
for f in sorted(os.listdir(f"{OUT}/routing")):
    if f not in base or base[f]["dom"] != "code": continue
    W[f] = dict(a=base[f]["a"], na=len(base[f]["tk"]), ids=load_npy(zipfile.ZipFile(f"{OUT}/routing/{f}"), "ids.npy"))
def row_k(sc, r):
    cf = r["cf"]
    if sc == "exact": return 10
    if sc == "q1_999": return 2 if cf is not None and cf >= 0.999 else 10
    if sc == "new1": return 10 if cf is None else (2 if cf >= 0.999 else 4 if cf >= 0.99 else 8 if cf >= 0.9 else 10)
    if sc == "q1_oracle": return 2 if r["cls"] == "sure" else 10
    if sc == "top2_all": return 2
def kept(B, a): return (a - a ** (B + 1)) / (1 - a)
# today's drafter keeps 10.5 at 16 rows (DS) and 10.5 at 32 (DS2: depth decay), taken as 10.5 at 64 too; a 0.96 chain
# without decay keeps 11.5 / 17.5 / 22.2 (the formula; a 0.945 chain without decay would give today's 10.5 at 16 rows).
KEPT_TODAY = {16: 10.5, 32: 10.5, 64: 10.5}
print("| scheme | B=16: records/block, MB | B=32 | B=64 | union ratio 32/16, 64/16 | MB per kept, today's drafter (16/32/64) and an a 0.96 chain |")
print("|---|---|---|---|---|---|")
for sc in ("exact", "q1_999", "new1", "q1_oracle", "top2_all"):
    res = {}
    for B in (16, 32, 64):
        nb = 0; mb = 0.0; recs = 0
        for nm, w in W.items():
            for j0 in range(0, w["na"] - B + 1, B):
                nb += 1
                for l in range(48):
                    U = {}
                    for j in range(j0, j0 + B):
                        k = row_k(sc, R[nm][j]); o = ((w["a"] + j) * 48 + l) * 10
                        for rk in range(k):
                            e = w["ids"][o + rk]
                            if e not in U or rk < U[e]: U[e] = rk
                    recs += len(U); mb += sum(2.8 if rk < 2 else 2.1 for rk in U.values())
        res[B] = (recs / nb, mb / nb, nb)
    print(f"| {sc} | {res[16][0]:.0f}, {res[16][1]:.0f} MB ({res[16][2]} blocks) | {res[32][0]:.0f}, {res[32][1]:.0f} MB | {res[64][0]:.0f}, {res[64][1]:.0f} MB | "
          f"{res[32][1] / res[16][1]:.2f}, {res[64][1] / res[16][1]:.2f} | "
          + " / ".join(f"{res[B][1] / KEPT_TODAY[B]:.0f}" for B in (16, 32, 64)) + " and " + " / ".join(f"{res[B][1] / kept(B, 0.96):.0f}" for B in (16, 32, 64)) + " |")
