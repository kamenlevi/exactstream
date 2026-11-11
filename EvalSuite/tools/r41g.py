"""b12llff3 verdict (docs/PREREG_b12llff3_2026-10-31.md, decision 5): runner 24's 29 agent rows in ONE process vs
b12llff (R-41C). PASS = 29/29 token-identical to b12llff AND the process's lifetime peak <= 11.95 GB. Information: the
per-row footprint next to b12llff2 (R-41D) and b12llff, SSD MB per token. Usage: .venv/bin/python tools/r41g.py"""
from r41c import LINE, first_diff, memory, order, rows, ssd

V, BASE, PREV = "b12llff3", "b12llff", "b12llff2"

keys = order(("agent",))[:29]
A, B, C = rows(V), rows(BASE), rows(PREV)
have = [k for k in keys if k in A]
d = [k for k in have if A[k]["completion_ids"] != B[k]["completion_ids"]]
print(f"{V} vs {BASE} on runner 24's 29 agent rows: {len(have)}/29 decoded\n")
if have:
    print(f"1. tokens: {len(have) - len(d)}/{len(have)} identical to {BASE}" + "".join(
        f"\n  {k}: from token {first_diff(A[k]['completion_ids'], B[k]['completion_ids'])}" for k in d))
    print("\n2. footprint (GB; row peak = max of the sampled max and the lifetime peak)")
    top = memory(V, set(keys))
    print(f"\n   per row, start / max / life: {V} | {PREV} (R-41D) | {BASE} (R-41C)")
    for k in have:
        a, c, b = A[k].get("mem", {}), C.get(k, {}).get("mem", {}), B[k].get("mem", {})
        print(f"   {k[-30:]:30s} {a.get('start', 0):5.2f} / {a.get('max', 0):5.2f} / {a.get('life', 0):5.2f} | "
              f"{c.get('start', 0):5.2f} / {c.get('max', 0):5.2f} / {c.get('life', 0):5.2f} | "
              f"{b.get('start', 0):5.2f} / {b.get('max', 0):5.2f} / {b.get('life', 0):5.2f}")
    ok = len(have) == 29 and not d and top <= LINE
    print(f"\n{V}: {'PASS' if ok else 'not PASS'} (29/29 identical to {BASE}: {len(have) == 29 and not d}; "
          f"the process <= {LINE} GB: {top <= LINE})")
    print("\n3. SSD MB per decoded token (information):",
          {x: {s: round(y, 1) for s, y in ssd(x, keys).items()} for x in (V, PREV, BASE)})
