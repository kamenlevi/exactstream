#!/usr/bin/env python3
"""check_tokens.py <run_tokens.json> <reference_tokens.json> [--arm gate+t] [--history]
Compares a run's tokens with a reference, key by key, as prefixes (the shorter length; an exact config must match its
reference token for token). Both files: {key: {arm: [[tokens], ...]}} (the decode loop's tokens.json, keys in run order);
the last output of each key counts. Exit 0 = every checked key prefix-equal, 1 = a difference (first differing position
printed), 2 = nothing to check.

--history (the lossy and chat configs): their tokens depend on what the process decoded before (the relaxed acceptance
takes the drafter's proposals, and the drafter's state carries across prompts), so a key is checked only when the run saw
the same earlier prompts as the reference: the run's keys must be the reference's first keys in the same order, and
every earlier key must have matched in full (same tokens, same length). The first key whose history differs and every
key after it are reported as information only. Exact configs do not depend on history: no flag, every shared key counts."""
import json, sys
args = [x for x in sys.argv[1:] if not x.startswith("--")]
arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "gate+t"
if "--arm" in sys.argv:
    args.remove(arm)
hist = "--history" in sys.argv
run, ref = (json.load(open(p)) for p in args[:2])
def last(d, key):
    v = d[key]
    v = v.get(arm) or next(iter(v.values())) if isinstance(v, dict) else v
    return v[-1] if v and isinstance(v[0], list) else v
bad = checked = 0
if not hist:
    shared = sorted(set(run) & set(ref))
    if not shared:
        print("no shared keys"); sys.exit(2)
    for k in shared:
        a, b = last(run, k), last(ref, k)
        n = min(len(a), len(b))
        d = next((i for i in range(n) if a[i] != b[i]), None)
        checked += 1
        if d is None:
            print(f"  {k}: == ({n} tokens compared; run {len(a)}, reference {len(b)})")
        else:
            bad += 1
            print(f"  {k}: DIFFERS at token {d} of {n} (run {a[d]}, reference {b[d]})")
    print(f"{checked - bad}/{checked} keys prefix-equal")
    sys.exit(1 if bad else 0)
rk, fk = list(run), list(ref)
why = None                                          # set once the history differs: later keys are information only
for i, k in enumerate(rk):
    if why is None and (i >= len(fk) or fk[i] != k):
        why = f"the run's key {i + 1} is {k}, the reference's is {fk[i] if i < len(fk) else 'none'}"
    a = last(run, k)
    b = last(ref, k) if k in ref else []
    n = min(len(a), len(b))
    d = next((j for j in range(n) if a[j] != b[j]), None)
    if why is not None:
        st = "==" if d is None else f"differs at token {d}"
        print(f"  {k}: information only ({why}); {st} over {n} tokens")
        continue
    checked += 1
    if d is not None:
        bad += 1
        print(f"  {k}: DIFFERS at token {d} of {n} (run {a[d]}, reference {b[d]}; same earlier prompts as the reference)")
        why = f"key {k} differed"
    else:
        print(f"  {k}: == ({n} tokens compared; run {len(a)}, reference {len(b)})")
        if len(a) != len(b):
            why = f"{k} ran {len(a)} of the reference's {len(b)} tokens, so later prompts saw a different history"
if not checked:
    print("nothing checked (the run's first key is not the reference's first key)"); sys.exit(2)
print(f"{checked - bad}/{checked} keys prefix-equal with the reference's history"
      + (f"; {len(rk) - checked} information only" if len(rk) > checked else ""))
sys.exit(1 if bad else 0)
