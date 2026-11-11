"""Token-identity of two decode runs, row by row (same key): per set, how many rows are identical, and where the others
first differ. Usage: .venv/bin/python tools/same_tokens.py b12llu2 b12ll   (results/<v>/decode.jsonl)"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(v):
    return {r["key"]: r for r in map(json.loads, open(os.path.join(ROOT, "results", v, "decode.jsonl")))}


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


a, b = rows(sys.argv[1]), rows(sys.argv[2])
by = {}
for k in a.keys() & b.keys():
    by.setdefault(a[k]["set"], []).append(k)
total = same = 0
for s, keys in sorted(by.items()):
    diff = sorted((first_diff(a[k]["completion_ids"], b[k]["completion_ids"]), k) for k in keys
                  if a[k]["completion_ids"] != b[k]["completion_ids"])
    total, same = total + len(keys), same + len(keys) - len(diff)
    print(f"{s}: {len(keys) - len(diff)}/{len(keys)} rows token-identical"
          + (f"; the others first differ at token {', '.join(str(i) for i, _ in diff[:12])}" if diff else ""))
print(f"all: {same}/{total} rows token-identical ({sys.argv[1]} vs {sys.argv[2]})")
