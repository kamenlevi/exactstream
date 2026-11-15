"""THINK_1115 #4, second tiny CPU check (same map data as nearties_1115.py). Near-tie = 125B top-1 vs top-2 margin < 2 nats.
(1) lag profile of near-tie clustering; (2) the next check's band-0 outcome given the miss row's margin (coin flip vs clear miss);
(3) misses by margin threshold and whether the wrong draft was the 125B's #2 (the lossy forgiveness yield);
(4) drafter cf vs the 125B's row class (floor_rows.json); (5) a same-window bigram memo of near-ties (precision / recall)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map")); sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
import dm_gate
from bytes10_replay import windows
base = json.load(open(os.path.join(dm_gate.OUT, "baseline.json"))); W = windows()
FR = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
rows = FR if isinstance(FR, list) else (list(FR.values()) if isinstance(FR, dict) else [])
if rows and isinstance(rows[0], dict) and "cls" in rows[0]:
    print("floor_rows: P(125B class | drafter cf bucket), code / ifeval / multi")
    for dom in ("code", "ifeval", "multi"):
        R = [r for r in rows if r.get("dom") == dom and r.get("cf") is not None]
        out = []
        for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
            S = [r for r in R if lo <= float(r["cf"]) < hi]
            if S: out.append(f"cf [{lo},{hi}): n {len(S)} tie {100 * np.mean([r['cls'] == 'tie' for r in S]):.1f} % mid {100 * np.mean([r['cls'] == 'mid' for r in S]):.1f} %")
        print(f"  {dom}: " + " | ".join(out))
else:
    print("floor_rows: unexpected layout", type(FR), (rows[:1] if rows else ""))
TH = (0.25, 0.5, 1.0, 2.0)
for dom, ws in W.items():
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    lag = {L: [0, 0] for L in range(1, 7)}; b0 = {"tie": [0, 0, 0], "clear": [0, 0, 0]}   # [band-0 misses, band-0 near-ties, n]
    mis = {t: [0, 0] for t in TH}; nmiss = 0; memo = [0, 0, 0, 0]                      # memo: TP FP FN TN
    for w in ws:
        nm = w["nm"]; i = names.index(nm); b = base[nm]; a = int(b["a"]); tk = np.array(b["tk"]); lp = np.array(b["lp"])
        marg = lp[:, 0] - lp[:, 1]; nt = marg < 2.0; n = len(nt)
        for L in range(1, 7):
            for t in range(n - L):
                if nt[t]: lag[L][0] += int(nt[t + L]); lag[L][1] += 1
        seen = {}
        for t in range(2, n):
            key = (int(tk[t - 2, 0]), int(tk[t - 1, 0])); pred = seen.get(key, False)
            memo[0 if pred and nt[t] else 1 if pred else 2 if nt[t] else 3] += 1
            seen[key] = bool(nt[t])
        checks = Z[f"{i}_checks"].tolist(); dtok = Z[f"{i}_dtok"]
        for ci, (k, m, A) in enumerate(checks):
            if k >= m: continue
            r0 = A - a; rm = r0 + k
            if not (0 <= rm < n): continue
            nmiss += 1; was2 = int(dtok[ci, k] == tk[rm, 1])
            for t in TH:
                if marg[rm] < t: mis[t][0] += 1; mis[t][1] += was2
            cls = "tie" if nt[rm] else "clear"
            if ci + 1 < len(checks):
                k2, m2, A2 = checks[ci + 1]
                if A2 == A + k + 1 and rm + 1 < n:
                    b0[cls][0] += int(k2 == 0 and m2 > 0); b0[cls][1] += int(nt[rm + 1]); b0[cls][2] += 1
    print(f"{dom}: P(near-tie at t+L | near-tie at t), L=1..6: " + " ".join(f"{100 * lag[L][0] / max(lag[L][1], 1):.0f}" for L in range(1, 7)) + " %")
    for cls in ("tie", "clear"):
        c = b0[cls]
        print(f"   miss row {cls:5s} (n {c[2]}): next check band-0 miss {100 * c[0] / max(c[2], 1):.1f} %, band-0 row near-tie {100 * c[1] / max(c[2], 1):.1f} %")
    print("   misses by the miss row's margin: " + "; ".join(f"< {t} nats {100 * mis[t][0] / max(nmiss, 1):.1f} % (draft = 125B #2 in {100 * mis[t][1] / max(mis[t][0], 1):.0f} %)" for t in TH) + f"; misses {nmiss}")
    TP, FP, FN, TN = memo
    print(f"   bigram memo (same window): precision {100 * TP / max(TP + FP, 1):.1f} % recall {100 * TP / max(TP + FN, 1):.1f} % (base rate {100 * (TP + FN) / max(sum(memo), 1):.1f} %, flagged {100 * (TP + FP) / max(sum(memo), 1):.1f} %)")
