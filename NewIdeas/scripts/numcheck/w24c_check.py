"""NUMBER CHECK for W24c's gate-switch rule (NewIdeas, Mon 9 Nov 23:1x; criteria fixed BEFORE the result, board 'GATE-SWITCH RULE').
PASS iff (1) the 14-row config's HumanEval whole decode, pooled sum n / sum n/tps over all prompts AND both repeats, >= 120.0;
(2) no HE row has a non-empty 'bad' field (a PAUSED row = unclean: no switch tonight); (3) lifetime peak max(mem[-1]) < 24.000
(10^9 bytes) in every repeat; (4) the controls (first / last) agree within 2 % (else: thermal drift FLAG, not a fail).
usage: python3 -I w24c_check.py [arms: control_first rep1 rep2 control_last] [set h]   (files ~/BigMoE/data/r57/<arm><set>_100_runs.json)"""
import json, os, sys
R57 = os.path.expanduser("~/BigMoE/data/r57")
arms = sys.argv[1:5] if len(sys.argv) >= 5 else ["w13c", "w14a", "w14b", "w13d"]
st = sys.argv[5] if len(sys.argv) >= 6 else "h"
def load(a):
    p = os.path.join(R57, f"{a}{st}_100_runs.json")
    if not os.path.exists(p): return None, p
    return json.load(open(p)), p
def summ(rows):
    n = sum(r["n"] for r in rows); t = sum(r["n"] / r["tps"] for r in rows)
    return n, t, max(r["mem"][-1] for r in rows), [r["key"].split("/")[-1] + ": " + r["bad"].strip() for r in rows if r["bad"].strip()]
res = {}
for a in arms:
    d, p = load(a)
    if d is None: print(f"{a}: MISSING {p}"); continue
    n, t, pk, bad = summ(d); res[a] = (d, n, t, pk, bad)
    print(f"{a}{st}: prompts {len(d)}, tokens {n}, whole decode pooled {n / t:.2f} tok/s, peak {pk:.3f} GB, bad rows {len(bad)} {bad}")
c1, r1, r2, c2 = arms
ok = all(a in res for a in arms)
if ok:
    n = res[r1][1] + res[r2][1]; t = res[r1][2] + res[r2][2]; pooled = n / t
    peak = max(res[r1][3], res[r2][3]); bad = res[r1][4] + res[r2][4]
    keys1 = [r["key"] for r in res[r1][0]]; keys2 = [r["key"] for r in res[r2][0]]
    same_tok = sum(1 for x, y in zip(res[r1][0], res[r2][0]) if x.get("lh") == y.get("lh"))
    cc = (res[c1][1] / res[c1][2], res[c2][1] / res[c2][2]); drift = abs(cc[0] - cc[1]) / min(cc)
    print(f"14-row config pooled over {r1} + {r2}: {pooled:.2f} tok/s (each {res[r1][1] / res[r1][2]:.2f} / {res[r2][1] / res[r2][2]:.2f}); "
          f"peak {peak:.3f} GB; bad rows {len(bad)}; same prompt set {keys1 == keys2}; equal lh hashes {same_tok} / {len(keys1)}")
    print(f"controls {c1} / {c2}: {cc[0]:.2f} / {cc[1]:.2f} tok/s, drift {100 * drift:.1f} % {'FLAG (> 2 %)' if drift > 0.02 else 'ok'}")
    v = pooled >= 120.0 and not bad and peak < 24.0 and keys1 == keys2
    print(f"VERDICT: {'PASS' if v else 'FAIL'} (>= 120.0: {pooled >= 120.0}; clean rows: {not bad}; peak < 24.000: {peak < 24.0}; same prompts: {keys1 == keys2})")
else:
    print("VERDICT: NOT COMPLETE (a file is missing)")
