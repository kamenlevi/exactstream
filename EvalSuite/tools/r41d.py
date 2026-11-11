"""R-41D fold arms + R-45's fix (docs/PREREG_r41d_2026-10-29.md): each arm vs its R-41C baseline on runner 24's 45
rows (tokens, footprint per process, per-row footprint next to the baseline's, SSD MB per token).
Usage: .venv/bin/python tools/r41d.py [b12llfx2] [b12llff2]"""
import sys
from r41c import LINE, first_diff, memory, order, rows, ssd

BASE = {"b12llfx2": "b12llfx", "b12llff2": "b12llff"}


def arm(v):
    keys = order(("agent",))[:29] + order(("math",))[:16]
    A, B = rows(v), rows(BASE[v])
    have = [k for k in keys if k in A]
    d = [k for k in have if A[k]["completion_ids"] != B[k]["completion_ids"]]
    print(f"R-41D {v} vs {BASE[v]} on runner 24's 45 rows: {len(have)}/45 decoded\n")
    if not have:
        return
    print(f"1. tokens: {len(have) - len(d)}/{len(have)} identical to {BASE[v]}" + "".join(
        f"\n  {k}: from token {first_diff(A[k]['completion_ids'], B[k]['completion_ids'])}" for k in d))
    print("\n2. footprint (GB; row peak = max of the sampled max and the lifetime peak)")
    top = memory(v, set(keys))
    print(f"\n   per row, start / max / life: {v} | {BASE[v]} (R-41C)")
    for k in have:
        a, b = A[k].get("mem", {}), B[k].get("mem", {})
        print(f"   {k[-34:]:34s} {a.get('start', 0):5.2f} / {a.get('max', 0):5.2f} / {a.get('life', 0):5.2f} | "
              f"{b.get('start', 0):5.2f} / {b.get('max', 0):5.2f} / {b.get('life', 0):5.2f}")
    ok = len(have) == 45 and not d and top <= LINE
    print(f"\n{v}: {'PASS' if ok else 'not PASS'} (45/45 identical to {BASE[v]}: {len(have) == 45 and not d}; "
          f"every process <= {LINE} GB: {top <= LINE})")
    print("\n3. SSD MB per decoded token (information):",
          {x: {s: round(y, 1) for s, y in ssd(x, keys).items()} for x in (v, BASE[v])})


if __name__ == "__main__":
    for v in sys.argv[1:] or list(BASE):
        if v not in BASE:
            raise SystemExit(__doc__)
        arm(v)
        print()
