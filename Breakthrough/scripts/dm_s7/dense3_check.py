"""DENSE-3's INVALID rule (NewIdeas' QUEUE 16:5x pre-registration; Breakthrough, Wed 2026-11-04 17:0x): the arm is INVALID
if its stored top-8 ids and log-probs equal its twin's on every row (the rewrite never reached the forward). CPU, reads
the map's out/ JSONs only. Prints per domain: rows, rows identical to the twin (top-8 ids + log-probs), top-1 flips vs
the twin, mean |dlp1| vs the twin; then VERDICT VALID / INVALID.
Usage (BigMoE venv): python dense3_check.py [arm] [twin] [out_dir]
  (default: qom_0_live_ball_c147_t3+dense3 vs qom_0_live_ball_c147_t3 in ~/FableGod/decision_map/out)"""
import json, os, sys
import numpy as np
arm = sys.argv[1] if len(sys.argv) > 1 else "qom_0_live_ball_c147_t3+dense3"
twin = sys.argv[2] if len(sys.argv) > 2 else "qom_0_live_ball_c147_t3"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/FableGod/decision_map/out")
ld = lambda n: json.load(open(os.path.join(OUT, n + ".json")))
A, T = ld(arm), ld(twin)
win = [nm for nm, v in T.items() if isinstance(v, dict) and "tk" in v and nm in A]
dom = {}
for nm in win:
    a, t = A[nm], T[nm]; m = min(len(a["tk"]), len(t["tk"]))
    s = dom.setdefault(t["dom"], [0, 0, 0, 0.0, 0])
    s[0] += m; s[4] += 1
    s[1] += sum(1 for i in range(m) if a["tk"][i] == t["tk"][i] and a["lp"][i] == t["lp"][i])
    s[2] += int((np.array(a["tk"])[:m, 0] != np.array(t["tk"])[:m, 0]).sum())
    s[3] += float(np.abs(np.array(a["lp"])[:m, 0] - np.array(t["lp"])[:m, 0]).sum())
print(f"[dense3 check] {arm} vs {twin}: {len(win)} windows in both")
for d, (n, same, fl, dl, w) in dom.items():
    print(f"  {d}: {w} windows, {n} rows; identical to the twin {same}; top-1 flips vs the twin {fl} ({100 * fl / max(n, 1):.2f} %); "
          f"mean |dlp1| {dl / max(n, 1):.4f}")
rows = sum(v[0] for v in dom.values()); same = sum(v[1] for v in dom.values())
print(f"VERDICT {'INVALID' if rows and same == rows else 'VALID' if rows else 'NO DATA'}: {same}/{rows} rows identical to the twin")
