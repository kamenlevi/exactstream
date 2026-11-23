"""NATIVE-FRI RUN 2 pooled (Leap, Sun 2026-11-08 20:0x, for KERNELS R23 / NewIdeas' number check): whole decode = total
tokens / total decode wall (n / tps per run) over the 13 prompts kept in all four arms (agent/text23 flagged in two arms),
per arm and per kernel (both passes). The screen of 11-06 recorded no prefill wall (pfs), so whole request and first
token cannot be derived. Input data/natfri/r57/nf*_192_runs.json -> stdout (data/natfri/natfri_pooled.txt)."""
import json
arms = {"nfdc": "3c-N pass 1", "nfnc": "NAX6-N pass 1", "nfncb": "NAX6-N pass 2", "nfdcb": "3c-N pass 2"}
R = {a: json.load(open(f"data/natfri/r57/{a}_192_runs.json")) for a in arms}
keep = set.intersection(*[{r["key"] for r in R[a] if not r.get("bad")} for a in arms])
pool = lambda rs: (sum(r["n"] for r in rs), sum(r["n"] for r in rs) / sum(r["n"] / r["tps"] for r in rs), max(r["mem"][6] for r in rs))
print(f"NATIVE-FRI RUN 2 pooled whole decode, {len(keep)} prompts kept in all four arms (C 192, quiet, 11-06 02:58-03:22)")
for a, lab in arms.items():
    n, wd, pk = pool([r for r in R[a] if r["key"] in keep]); print(f"  {a:6s} {lab:14s} {n} tokens, whole decode {wd:.2f} tok/s, lifetime peak {pk:.2f} GB")
g = {k: pool([r for a in v for r in R[a] if r["key"] in keep]) for k, v in (("3c-N", ("nfdc", "nfdcb")), ("NAX6-N", ("nfnc", "nfncb")))}
for k, (n, wd, pk) in g.items():
    print(f"  {k} (both passes): whole decode {wd:.2f} tok/s, peak {pk:.2f} GB")
print(f"  NAX6-N / 3c-N whole decode: x{g['NAX6-N'][1] / g['3c-N'][1]:.4f} (the report's steady-state tpsw ratio: x1.0665)")
print("  whole request / first token: not derivable (no pfs field in this screen's runs.json)")
