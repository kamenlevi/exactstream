"""TREE16 M4 engine A/B analysis (BigMoE; pre-registered in docs/TREE16_M4_PREREG.md, Sun 15 Nov 2026, before any run).
Arms on snapshot t16b: C = t30b chain-16 (the engine today, cut 0.3), T = tree-16 (R57_TREE16=1). Runs C1, T1, T2, C2 (tags
t16bC1 / t16bT1 / t16bT2 / t16bC2 in data/r57). Whatever ran is read; missing runs are reported.
- EXACT (the pass rule): every key's tokens identical across all runs that ran, no run cut. Else M4 FAILS (no speed claim).
- Speed (headline = whole decode, runs.json "tps" = n / decode wall): per set, R = geomean over keys of mean(T tps) /
  mean(C tps); K = pooled kept per check, T / C. Prediction: S9c's kept ratio / 1.025 (the tree check's +2.5 %).
  PAYS if R_mbpp >= 1.03 and R_he >= 0.99; COSTS if R_mbpp < 0.98 or R_he < 0.97; else NEUTRAL.
  With only C1 + T1 (or one arm repeated once) the speed reading is flagged ORDER-BIASED and is not a verdict.
usage: python scripts/tree16_m4ab.py [--dir data/r57] -> data/deepdraft/tree16/m4ab.json"""
import json
import math
import os
import sys

D = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else os.path.expanduser("~/BigMoE/data/r57")
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.expanduser("~/BigMoE/data/deepdraft/tree16/m4ab.json")
TAGS = {"C1": "t16bC1", "T1": "t16bT1", "T2": "t16bT2", "C2": "t16bC2"}
S9C = {"humaneval": 1.060, "mbpp": 1.136}                # MTP Head S9c (TrainLab fa3ce76): kept per check vs chain-16
COST = 1.025                                             # Leap TREE16-TIME v2: the tree check's cost, engine-scaled


def load(tag):
    try:
        runs = json.load(open(os.path.join(D, f"{tag}_runs.json")))
        toks = json.load(open(os.path.join(D, f"{tag}_tokens.json")))
    except (OSError, ValueError):
        return None
    return {r["key"]: dict(r, out=[o for v in toks.get(r["key"], {}).values() for o in v]) for r in runs}


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


R = {k: load(t) for k, t in TAGS.items()}
ran = [k for k, v in R.items() if v]
res = dict(ran=ran, missing=[k for k in TAGS if k not in ran])
keys = sorted(set().union(*(set(R[k]) for k in ran))) if ran else []
diff, cut = [], []
for key in keys:
    outs = [R[k][key]["out"] for k in ran if key in R[k]]
    if len(outs) < len(ran) or any(o != outs[0] for o in outs) or any(len(o) != 1 for o in outs):
        diff.append(key)
    cut += [f"{k}:{key}" for k in ran if key in R[k] and (R[k][key].get("cut") or R[k][key].get("bad"))]
exact = bool(ran) and any(k.startswith("C") for k in ran) and any(k.startswith("T") for k in ran) and not diff and not cut
res.update(exact=exact, keys=len(keys), differ=diff, cut_or_bad=cut)
Cs, Ts = [k for k in ran if k.startswith("C")], [k for k in ran if k.startswith("T")]
sets = {}
if exact:
    for st in ("humaneval", "mbpp"):
        ks = [k for k in keys if k.startswith(st + "/")]
        rat, per = [], {}
        for key in ks:
            c = sum(R[a][key]["tps"] for a in Cs) / len(Cs)
            t = sum(R[a][key]["tps"] for a in Ts) / len(Ts)
            rat.append(t / c); per[key] = round(t / c, 4)
        kc = sum(R[a][k]["kept"] * R[a][k]["checks"] for a in Cs for k in ks) / sum(R[a][k]["checks"] for a in Cs for k in ks)
        kt = sum(R[a][k]["kept"] * R[a][k]["checks"] for a in Ts for k in ks) / sum(R[a][k]["checks"] for a in Ts for k in ks)
        noise = {f"{a}/{b}": round(gmean([R[a][k]["tps"] / R[b][k]["tps"] for k in ks]), 4)
                 for a, b in (("C2", "C1"), ("T2", "T1")) if a in ran and b in ran}
        t16 = [R[a][k].get("t16") or {} for a in Ts for k in ks]
        sets[st] = dict(R=round(gmean(rat), 4), K=round(kt / kc, 4), kept_C=round(kc, 3), kept_T=round(kt, 3),
                        predicted_R=round(S9C[st] / COST, 4), s9c_K=S9C[st], per_key=per, rep_ratio=noise,
                        side=sum(x.get("side", 0) for x in t16), main_end=sum(x.get("main_end", 0) for x in t16),
                        tree_checks=sum(x.get("checks", 0) for x in t16), fallback=sum(x.get("fallback", 0) for x in t16),
                        draft_ms_T=[x.get("draft_ms") for x in t16], drf_ms_C=[R[a][k]["drf"] for a in Cs for k in ks])
    he, mb = sets["humaneval"]["R"], sets["mbpp"]["R"]
    reading = "PAYS" if mb >= 1.03 and he >= 0.99 else "COSTS" if (mb < 0.98 or he < 0.97) else "NEUTRAL"
    biased = not (len(Cs) >= 2 and len(Ts) >= 2)
    res.update(sets=sets, reading=reading + (" (ORDER-BIASED: not a verdict)" if biased else ""))
else:
    res.update(reading="M4 FAILS: tokens differ or a run was cut / bad" if ran and (diff or cut) else "not run / incomplete")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"TREE16-M4AB: ran {ran}; EXACT {'PASS' if exact else 'FAIL' if (diff or cut) else 'n/a (needs a C and a T run)'} "
      f"({len(keys)} keys; differ {diff}; cut/bad {cut}); "
      + (" ".join(f"{s}: R {v['R']} (pred {v['predicted_R']}), K {v['K']} (S9c {v['s9c_K']})" for s, v in sets.items())
         if sets else "") + f"; {res['reading']}")
