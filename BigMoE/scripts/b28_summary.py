"""b28 summary: per arm and prompt, tok/s, verify ms per check and verify time for the same 256 tokens (ms/check x checks),
each round and the mean, as % vs base; shas must match the shipped v3 ones.  usage: b28_summary.py [data/b28]"""
import glob, json, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/data/b28")
SHA = {"code": "0b6ad61220", "prose": "7b3cf8f87d", "edit": "8b72b3d757"}
ARMS = ["base", "joint", "sb1_", "jsb"]
R = {}
for a in ARMS:
    for f in sorted(glob.glob(os.path.join(D, f"ab_{a}[0-9].json"))):
        R.setdefault(a, []).append(json.load(open(f))["prompts"])
bad = [(a, i + 1, p, r[p]["sha"]) for a in R for i, r in enumerate(R[a]) for p in r if r[p]["sha"] != SHA[p]]
print("shas:", "ALL MATCH" if not bad else f"MISMATCH {bad}")
m = lambda xs: sum(xs) / len(xs)
base = {p: {k: m([r[p][k] if k != "vt" else r[p]["verify_ms"] * r[p]["n_passes"] for r in R["base"]])
            for k in ("tok_s", "verify_ms", "vt")} for p in SHA}
print(f"{'arm':6} {'prompt':6} {'tok/s per round':>22} {'mean':>6} {'vs base':>8} {'ms/check':>9} {'vs base':>8} "
      f"{'verify ms':>9} {'vs base':>8}")
for a in ARMS:
    for p in SHA:
        rs = R.get(a, [])
        if not rs:
            continue
        ts = [r[p]["tok_s"] for r in rs]; vm = [r[p]["verify_ms"] for r in rs]
        vt = [r[p]["verify_ms"] * r[p]["n_passes"] for r in rs]
        b = base[p]
        print(f"{a.rstrip('_'):6} {p:6} {' '.join(f'{t:6.1f}' for t in ts):>22} {m(ts):6.1f} "
              f"{100 * (m(ts) / b['tok_s'] - 1):+7.1f}% {m(vm):9.1f} {100 * (m(vm) / b['verify_ms'] - 1):+7.1f}% "
              f"{m(vt):9.0f} {100 * (m(vt) / b['vt'] - 1):+7.1f}%")
    print()
