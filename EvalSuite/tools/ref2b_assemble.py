"""REF2b reference rows (docs/PREREG_ref2b_2026-10-30.md): check the decoded rows, then complete results/ref2b.
REF2b changes REF2's AR only in steps whose padded attention has more than 512 keys: the step making completion token
i has N = n_prompt + i + 15, so tokens with n_prompt + i < 498 must equal results/ref2's. Checks:
  1. every key of results/ref2b_keys.txt is decoded;
  2. the 3 spot-check rows (results/ref2b_spot.txt: rows that end just short of the boundary) equal results/ref2's;
  3. no decoded row differs from results/ref2 before its first bucketed step (n_prompt + i >= 498).
If all pass, the other rows are copied from results/ref2 (they never reach a bucketed step), marked copied_from, so
results/ref2b/decode.jsonl holds all 210 core rows. Any failure: nothing is copied, exit 1.
Usage: .venv/bin/python tools/ref2b_assemble.py > results/ref2b_regen.txt"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
old = {r["key"]: r for r in map(json.loads, open(os.path.join(R, "ref2", "decode.jsonl")))}
path = os.path.join(R, "ref2b", "decode.jsonl")
new = {r["key"]: r for r in map(json.loads, open(path))}
keys = open(os.path.join(R, "ref2b_keys.txt")).read().split()
spot = open(os.path.join(R, "ref2b_spot.txt")).read().split()
ok = True
miss = [k for k in keys if k not in new]
print(f"1. decoded: {len(keys) - len(miss)}/{len(keys)}" + (f"; missing {miss}" if miss else ""))
ok &= not miss
for k in spot:
    same = k in new and new[k]["completion_ids"] == old[k]["completion_ids"]
    print(f"2. spot check {k} (n_prompt {old[k]['n_prompt']}, {len(old[k]['completion_ids'])} tokens): identical {same}")
    ok &= same
early, diff, same_n = [], [], 0
for k in keys:
    if k in spot or k not in new:
        continue
    a, b, P = new[k]["completion_ids"], old[k]["completion_ids"], old[k]["n_prompt"]
    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
    if i is None:
        same_n += 1
    elif P + i < 498:
        early.append((k, P, i))
    else:
        diff.append((k, P, i, P + i, len(a), len(b)))
print(f"3. rows that reach a bucketed step: {same_n} identical to results/ref2, {len(diff)} differ after it, "
      f"{len(early)} differ BEFORE it" + (f": {early}" if early else ""))
for k, P, i, ab, la, lb in diff:
    print(f"   {k}: first difference at token {i} (n_prompt {P}, n_prompt + i = {ab}); length {la} vs {lb}")
ok &= not early
if not ok:
    print("STOP: a check failed; nothing copied")
    sys.exit(1)
rest = [k for k in old if k not in new]
with open(path, "a") as f:
    for k in rest:
        f.write(json.dumps(dict(old[k], copied_from="ref2")) + "\n")
print(f"copied {len(rest)} rows from results/ref2 (no bucketed step); results/ref2b/decode.jsonl now has "
      f"{len(new) + len(rest)} rows")
