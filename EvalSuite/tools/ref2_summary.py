"""R-40 REF2 rows (docs/PREREG_ref2_2026-10-28.md): speed per set next to the baseline's (information: different days),
REF2's per-path kernel calls per row and per token (a decode with no experts calls did not run REF2), footprint, and
provenance per chunk. Usage: .venv/bin/python tools/ref2_summary.py [BASE]   (default ref; results/<v>/decode.jsonl)"""
import json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = sys.argv[1] if len(sys.argv) > 1 else "ref"
SETS = ("math", "ifeval", "humaneval")


def rows(v):
    return {r["key"]: r for r in map(json.loads, open(os.path.join(ROOT, "results", v, "decode.jsonl")))}


a, b = rows(BASE), rows("ref2")
print(f"ref2: {len(b)} rows; baseline {BASE}: {len(a)} rows\n")

print("1. speed (tokens / row seconds, prefill included; different days: information only)")
for s in SETS + ("all",):
    ks = [k for k in b if (s == "all" or b[k]["set"] == s) and k in a]
    if not ks:
        continue
    f = lambda R: (sum(len(R[k]["completion_ids"]) for k in ks), sum(R[k]["seconds"] for k in ks))
    (na, sa), (nb, sb) = f(a), f(b)
    print(f"  {s:9s} rows {len(ks):3d}  {BASE} {na / sa:6.2f} tok/s ({na} tokens, {sa:7.0f} s)   ref2 {nb / sb:6.2f} tok/s "
          f"({nb} tokens, {sb:7.0f} s)   ratio {(nb / sb) / (na / sa):.3f}")

print("\n2. REF2 kernel calls (ref2.counts(), counted at graph build)")
rs = [r for r in b.values() if "ref2" in r]
none = [r["key"] for r in rs if not r["ref2"]["calls"].get("experts")]
print(f"  rows with counters {len(rs)}/{len(b)}; rows with no experts calls: {len(none)}" + (f" -> {none[:10]}" if none else ""))
paths = sorted({p for r in rs for p in r["ref2"]["calls"]})
for p in paths:
    per = [r["ref2"]["calls"].get(p, 0) / max(1, r["ref2"]["n"]) for r in rs]
    print(f"  {p:12s} per generated token: median {statistics.median(per):8.1f}  min {min(per):8.1f}  max {max(per):8.1f}")

print("\n3. footprint (GB, row peak = max of the sampled max and the lifetime peak)")
pk = sorted(((max(r["mem"]["max"], r["mem"]["life"]), k) for k, r in b.items() if "mem" in r), reverse=True)
if pk:
    print(f"  ref2 max row peak {pk[0][0]:.2f} ({pk[0][1]}); median {statistics.median(p for p, _ in pk):.2f}")

print("\n4. provenance per chunk (Leap / BigMoE / EvalSuite heads and REF2 file hashes)")
cp = os.path.join(ROOT, "results", "ref2", "chunks.jsonl")
ch = [json.loads(l) for l in open(cp)] if os.path.exists(cp) else []
pv = [{k: x for k, x in c["provenance"].items() if k.startswith(("sha/leap/", "leap", "bigmoe", "mlx", "mtplx"))} for c in ch]
uniq = sorted({json.dumps(p, sort_keys=True) for p in pv})
print(f"  {len(ch)} chunks, {len(uniq)} distinct" + "".join(f"\n    {u}" for u in uniq))
