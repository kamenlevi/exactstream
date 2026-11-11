"""Re-derive CERT's arithmetic (EvalSuite RESULT_cert 2cfed64) from its discordant counts and decode.jsonl rows.
Independent of tools/cert_report.py: exact two-sided McNemar from scratch, deltas, pools, token-identity counts."""
import json, os, sys
from math import comb
R = os.path.expanduser("~/EvalSuite/results")

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

lines = {  # set/order: (n, arm only, ref only) as printed in RESULT_cert
    "humaneval fwd": (164, 0, 4), "humaneval rev": (164, 2, 2), "math fwd": (100, 2, 2), "math rev": (100, 1, 3),
    "mbpp fwd": (100, 1, 4), "mbpp rev": (100, 5, 3), "ifeval fwd": (100, 3, 4), "ifeval rev": (100, 4, 6)}
for k, (n, b, c) in lines.items():
    print(f"{k:14s} n {n} delta {100*(b-c)/n:+.2f} pt  p {mcnemar(b, c):.4f}")
core = [v for k, v in lines.items() if not k.startswith("ifeval")]
for name, ls in (("pooled 3 sets", core), ("pooled 4 sets", list(lines.values()))):
    n = sum(x[0] for x in ls); b = sum(x[1] for x in ls); c = sum(x[2] for x in ls)
    print(f"{name}: n {n} arm only {b} ref only {c} delta {100*(b-c)/n:+.2f} pt p {mcnemar(b, c):.4f}")

def load(run):
    p = os.path.join(R, run, "decode.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))}
ref = {}
for run in ("ref2tnax", "ref2tnaxn.r3", "ref2tnaxn.r2"):
    for k, r in load(run).items():
        ref.setdefault(k, r)  # first wins, as cert_report's load(*REFS) order is unknown here; checked below
for run in ("cert_fwd", "cert_rev"):
    A = load(run)
    by = {}
    for k, r in A.items():
        s = r.get("set"); t = by.setdefault(s, [0, 0])
        if k in ref:
            t[1] += 1; t[0] += r["completion_ids"] == ref[k]["completion_ids"]
    print(run, {s: f"{a}/{n}" for s, (a, n) in sorted(by.items())})
A, Rn = load("cert_chat"), load("ref2tnax")
ks = sorted(k for k in A if A[k].get("set") == "ifeval" and k in Rn)
same = sum(A[k]["completion_ids"] == Rn[k]["completion_ids"] for k in ks)
print(f"cert_chat: {same} of {len(ks)} IFEval rows token-identical to ref2tnax")
f, r_ = load("cert_fwd"), load("cert_rev")
for s in ("humaneval", "math", "mbpp", "ifeval"):
    ks = [k for k in f if f[k].get("set") == s and k in r_]
    print(f"fwd==rev {s}: {sum(f[k]['completion_ids'] == r_[k]['completion_ids'] for k in ks)}/{len(ks)}")
