"""R-60 verdict from data/r60/<arm>.jsonl (r60_screen.py): tokens identical across arms on every prompt (and the math rows
= b12llff2's), tokens per verify pass (all prompts pooled: total tokens / total passes) per arm vs stock, decode tok/s
(shared GPU, rough), lifetime peaks. PASS: tokens identical AND d22 tokens per pass >= +3 % over stock (then a quiet
confirmation, then the default on NewIdeas' word); KILL: under +1 %. Usage: [BigMoE venv] r60_sum.py [arms...]"""
import json, os, sys

D = os.path.expanduser("~/Breakthrough/data/r60")
arms = sys.argv[1:] or [a for a in ("stock", "d22", "d22q8") if os.path.exists(os.path.join(D, f"{a}.jsonl"))]
R = {a: {r["prompt"]: r for r in map(json.loads, open(os.path.join(D, f"{a}.jsonl")))} for a in arms}
base = R["stock"]
print("R-60 MTPDROP (shared GPU, rough timings): tokens per pass by prompt")
print("prompt".ljust(26) + "".join(a.rjust(22) for a in arms))
for p in base:
    print(p.ljust(26) + "".join((f"{R[a][p]['tok_per_pass']:.3f} ({R[a][p]['ntok']} tok)" if p in R[a] else "-").rjust(22) for a in arms))
for a in arms:
    rows = list(R[a].values())
    tok, pas = sum(r["ntok"] for r in rows), sum(r["passes"] for r in rows)
    dec = sum(r["decode_s"] for r in rows)
    same = all(p in R[a] and R[a][p]["sha"] == base[p]["sha"] for p in base)
    refok = all(r["ref_equal"] in (None, True) for r in rows)
    print(f"{a:6s}: tokens per pass {tok / pas:.3f} ({100 * (tok / pas / (sum(r['ntok'] for r in base.values()) / sum(r['passes'] for r in base.values())) - 1):+.1f} % "
          f"vs stock) | decode {tok / dec:.2f} tok/s | tokens = stock's on every prompt {same} | math rows = b12llff2 {refok} | "
          f"lifetime max {max(r['lifetime'] for r in rows):.3f} GB")
for a in arms:
    if a == "stock":
        continue
    rows = list(R[a].values())
    g = (sum(r["ntok"] for r in rows) / sum(r["passes"] for r in rows)) / (sum(r["ntok"] for r in base.values()) / sum(r["passes"] for r in base.values())) - 1
    same = all(p in R[a] and R[a][p]["sha"] == base[p]["sha"] for p in base)
    v = "FAIL (tokens differ)" if not same else "PASS" if g >= 0.03 else "KILL" if g < 0.01 else "BETWEEN"
    print(f"VERDICT {a}: {v} (tokens per pass {100 * g:+.1f} %; PASS >= +3 %, KILL < +1 %)")
