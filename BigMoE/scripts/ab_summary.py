"""A/B summary of s10_anatomy.py runs: per arm and prompt, tok/s per round, mean and % vs the first arm, verify ms per
check, and verify time for the same tokens (ms/check x checks). Prompts a quiet-mode pause or freeze touched
(clean == False, from gated.PAUSED) are shown as x and left out of the means. Shas must match the shipped v3 ones.
usage: ab_summary.py DIR ARM [ARM ...]   (files DIR/ab_<arm><round>.json; the first arm is the reference)"""
import glob, json, os, sys
D, ARMS = sys.argv[1], sys.argv[2:]
SHA = {"code": "0b6ad61220", "prose": "7b3cf8f87d", "edit": "8b72b3d757"}
R = {a: [json.load(open(f))["prompts"] for f in sorted(glob.glob(os.path.join(D, f"ab_{a}[0-9].json")))] for a in ARMS}
bad = [(a, i + 1, p, r[p]["sha"]) for a in R for i, r in enumerate(R[a]) for p in r if r[p]["sha"] != SHA.get(p)]
print("shas:", "ALL MATCH" if not bad else f"MISMATCH {bad}")
m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
ok = lambda r, p: r[p].get("clean", True)


def means(a, p):
    rs = [r for r in R[a] if p in r and ok(r, p)]
    return (m([r[p]["tok_s"] for r in rs]), m([r[p]["verify_ms"] for r in rs]),
            m([r[p]["verify_ms"] * r[p]["n_passes"] for r in rs]), len(rs))


print(f"{'arm':7} {'prompt':6} {'tok/s per round':>24} {'mean':>6} {'vs ref':>7} {'ms/check':>9} {'vs ref':>7} "
      f"{'verify ms':>9} {'vs ref':>7}  n")
for a in ARMS:
    for p in SHA:
        if not any(p in r for r in R[a]):
            continue
        ts, vm, vt, n = means(a, p)
        b = means(ARMS[0], p)
        per = " ".join(f"{r[p]['tok_s']:6.1f}" if ok(r, p) else f"{'x':>6}" for r in R[a] if p in r)
        print(f"{a:7} {p:6} {per:>24} {ts:6.1f} {100 * (ts / b[0] - 1):+6.1f}% {vm:9.1f} {100 * (vm / b[1] - 1):+6.1f}% "
              f"{vt:9.0f} {100 * (vt / b[2] - 1):+6.1f}%  {n}")
    print()
