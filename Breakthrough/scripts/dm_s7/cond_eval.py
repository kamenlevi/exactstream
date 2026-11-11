"""Conditions for the s67 runner's extra arms (WHEN lines; Fable C's audit AUDIT_1105 sections 2-3 via the orchestrator, Thu
2026-11-05). Prints yes / no / pending. Flips = all-row top-1 flips vs the baseline (NewIdeas' map_kl.arm, read-only import),
per domain, on the arm's stored out/<arm>.json; an arm is done when it holds its expected windows (36; 12 for "+code").
  codepass <arm> ...       yes if any listed arm, done, has code flips <= 1.34 %; no once every listed arm is done and none
                           does; else pending (reach07: "only if a code arm PASSes")
  multile <arm> <thr>      yes if <arm>, done, has multi flips <= thr; no if above; pending until done (R15b: R15 with >= 0.5 of
                           margin on multi = multi <= 3.19)
Usage (BigMoE venv): python cond_eval.py codepass split_0.9 split_0.99 c3_b2d c3_b2c | multile <arm> 3.19"""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/NewIdeas/scripts")); sys.dont_write_bytecode = True
OUT = os.path.expanduser("~/FableGod/decision_map/out")

def done(a):
    p = os.path.join(OUT, a + ".json")
    if not os.path.exists(p): return False
    n = sum(1 for v in json.load(open(p)).values() if isinstance(v, dict) and "tk" in v)
    return n == (12 if a.endswith("+code") else 36)

def flips(a):
    import map_kl
    return {d: v[0] for d, v in map_kl.arm(a).items()}

kind, args = sys.argv[1], sys.argv[2:]
if kind == "codepass":
    ds = [a for a in args if done(a)]
    if any(flips(a).get("code", 99.0) <= 1.34 for a in ds): print("yes")
    else: print("no" if len(ds) == len(args) else "pending")
elif kind == "multile":
    a, thr = args[0], float(args[1])
    print(("yes" if flips(a).get("multi", 99.0) <= thr else "no") if done(a) else "pending")
else:
    raise SystemExit(f"unknown condition {kind}")
