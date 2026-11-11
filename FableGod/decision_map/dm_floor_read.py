"""FLOOR night read-out (Breakthrough, Tue 2026-11-03; Fable's FLOOR_125B.md section 5; pre-registered SCOREBOARD a277380).
Per run and domain: (1) band flip rates = flips on the measured rows in each drafter band (cf >= 0.999 / 0.99-0.999 /
0.9-0.99; the realizable gate dm_gate) that the run cuts; (2) the whole sequence by the 125B's own class (sure p1 >= 0.99 /
tie p1 - p2 <= 0.1 / mid), non-tie = sure + mid. Verdict on gate_bands (the headline): NEW WORLD-with-margin if code bands
<= 0.02 % (>= 0.999) and <= 0.1 % (0.99-0.999), chat bands <= 0.3 %, sure rows <= 0.1 %, non-tie <= 1 % code / <= 5 % chat
(and the bytes condition from floor_bytes on this night's routing, read beside); EDGE if the >= 0.999 band > 0.1 % or sure
rows > 0.3 %; KILL of the mechanism if the 0.99-0.999 band at top-4 / top-5 > 1 % or code non-tie > 2 %; else BETWEEN.
Usage (BigMoE venv): DM_OUT=<night dir> python dm_floor_read.py -> <night dir>/FLOOR_read.txt"""
import json, os, math
import numpy as np
import dm_gate
OUT = dm_gate.OUT
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
DOMS = sorted({b["dom"] for b in base.values()})
BANDS = ((0.999, 9.0, ">=0.999"), (0.99, 0.999, "0.99-0.999"), (0.9, 0.99, "0.9-0.99"))
CUTS = {"gate_bands": (0, 1, 2), "gate_bands+hc8w": (0, 1, 2), "gate_top2_cf0999": (0,), "gate_top5_cf099": (0, 1),
        "gate_top4_cf099": (0, 1), "gate_bands+moe_reuse_chain1_L08_35_dcf0999": (0, 1, 2), "gate_top0_cf0999_L08_35": (0,)}
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
def pct(k, n): return 100 * k / n if n else float("nan")
def wil(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c + h)
def table(name):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); T = {}
    for nm, b in base.items():
        if nm not in R: continue
        m = min(len(b["tk"]), len(R[nm]["tk"])); lp = np.array(b["lp"][:m]); p1, p2 = np.exp(lp[:, 0]), np.exp(lp[:, 1])
        c = np.where(p1 >= 0.99, 0, np.where(p1 - p2 <= 0.1, 1, 2)); f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        g = np.nan_to_num(G[nm], nan=-1.0)[b["a"]:b["a"] + m]; d = T.setdefault(b["dom"], {"b": [[0, 0] for _ in BANDS], "c": [[0, 0] for _ in range(3)], "all": [0, 0]})
        for i, (lo, hi, _) in enumerate(BANDS):
            s = (g >= lo) & (g < hi); d["b"][i][0] += int(f[s].sum()); d["b"][i][1] += int(s.sum())
        for k in range(3): d["c"][k][0] += int(f[c == k].sum()); d["c"][k][1] += int((c == k).sum())
        d["all"][0] += int(f.sum()); d["all"][1] += m
    return T
say(f"FLOOR night read-out ({OUT}): {len(base)} windows, measured rows " + ", ".join(f"{d} {sum(len(b['tk']) for b in base.values() if b['dom'] == d)}" for d in DOMS))
say("| run | domain | band >= 0.999 | band 0.99-0.999 | band 0.9-0.99 | sure | tie | mid | non-tie | all rows |")
say("|---|---|---|---|---|---|---|---|---|---|")
V = {}
for name in CUTS:
    T = table(name)
    if T is None: continue
    for dom in DOMS:
        d = T[dom]; cut = CUTS[name]
        bands = [(f"{pct(*d['b'][i]):.3f} % ({d['b'][i][0]}/{d['b'][i][1]})" if i in cut else "-") for i in range(3)]
        nt = [d["c"][0][0] + d["c"][2][0], d["c"][0][1] + d["c"][2][1]]
        say(f"| {name} | {dom} | " + " | ".join(bands) + " | " + " | ".join(f"{pct(*d['c'][k]):.2f} %" for k in range(3))
            + f" | {pct(*nt):.2f} % | {pct(*d['all']):.2f} % |")
    V[name] = T
if "gate_bands" in V:
    T = V["gate_bands"]; c = T["code"]
    b0, b1 = pct(*c["b"][0]), pct(*c["b"][1]); sure = pct(*c["c"][0]); nt = pct(c["c"][0][0] + c["c"][2][0], c["c"][0][1] + c["c"][2][1])
    chat = [d for d in DOMS if d != "code"]
    chat_ok = all(pct(*T[d]["b"][i]) <= 0.3 for d in chat for i in range(3) if T[d]["b"][i][1]) and \
        all(pct(T[d]["c"][0][0] + T[d]["c"][2][0], T[d]["c"][0][1] + T[d]["c"][2][1]) <= 5 for d in chat)
    sure_all = max(pct(*T[d]["c"][0]) for d in DOMS)
    t45 = [V[n]["code"]["b"][1] for n in ("gate_top5_cf099", "gate_top4_cf099") if n in V]
    if (t45 and max(pct(*x) for x in t45) > 1) or nt > 2: v = "KILL (of the mechanism)"
    elif b0 > 0.1 or sure_all > 0.3: v = "EDGE"
    elif b0 <= 0.02 and b1 <= 0.1 and chat_ok and sure_all <= 0.1 and nt <= 1: v = "NEW WORLD-with-margin on flips (the bytes condition: floor_bytes.out)"
    else: v = "BETWEEN"
    say(f"\nVerdict on gate_bands: code band >= 0.999 {b0:.3f} % (upper 95 % {wil(*c['b'][0]):.3f}), 0.99-0.999 {b1:.3f} %, sure rows "
        f"{sure:.2f} % (max over domains {sure_all:.2f}), code non-tie {nt:.2f} % -> {v}")
open(os.path.join(OUT, "FLOOR_read.txt"), "w").write("\n".join(lines) + "\n")
