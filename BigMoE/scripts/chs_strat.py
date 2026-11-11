"""Per-check read-out for the CHAT150 A/Bs (Leap / NewIdeas Sun 00:4x: L's chat is run-to-run nondeterministic, so score the
check, not only whole decode). usage: python -I scripts/chs_strat.py tag [tag ...]  (data/r57/<tag>_256_checks.json)
Per arm: checks, median / mean ck (the check wall) and cycle (ck + drf), misses, svc, MB per check, kept per check; then ck by the
check's row count (1-2, 3, 4, 5+) and by miss band (0-29, 30-59, 60-99, 100+), so arms compare at equal rows."""
import json, statistics as S, sys

RB = (("1-2", 1, 2), ("3", 3, 3), ("4", 4, 4), ("5+", 5, 99))
MB = (("0-29", 0, 29), ("30-59", 30, 59), ("60-99", 60, 99), ("100+", 100, 10 ** 6))


def load(t):
    C = [c for c in json.load(open(f"data/r57/{t}_256_checks.json")) if c["key"].startswith("ifeval")]
    rows = []
    for c in C:
        for i in range(len(c["ck"])):
            rows.append(dict(ck=c["ck"][i], drf=c["drf"][i] if i < len(c["drf"]) else 0.0, miss=c["miss"][i], svc=c["svc"][i],
                             mb=c["mb"][i], kept=c["kept"][i], rows=c["rows"][i]))
    return rows


def fmt(xs):
    return f"{S.mean(xs):5.1f} ({len(xs):3d})" if xs else "    -      "


out = []
for t in sys.argv[1:]:
    try:
        R = load(t)
    except (OSError, ValueError, KeyError) as e:
        out.append(f"{t}: MISSING ({type(e).__name__})")
        continue
    ck = [r["ck"] for r in R]
    out.append(f"{t:7s} {len(R):4d} checks: ck median {S.median(ck):5.1f} mean {S.mean(ck):5.1f}; cycle median "
               f"{S.median([r['ck'] + r['drf'] for r in R]):5.1f}; misses {S.mean([r['miss'] for r in R]):5.1f}; svc "
               f"{S.mean([r['svc'] for r in R]):4.1f}; MB/check {S.mean([r['mb'] for r in R]):5.1f}; kept {S.mean([r['kept'] for r in R]):.2f}; "
               f"rows {S.mean([r['rows'] for r in R]):.2f}")
    out.append("          ck by rows   " + "  ".join(f"{n}: {fmt([r['ck'] for r in R if lo <= r['rows'] <= hi])}" for n, lo, hi in RB))
    out.append("          ck by misses " + "  ".join(f"{n}: {fmt([r['ck'] for r in R if lo <= r['miss'] <= hi])}" for n, lo, hi in MB))
print("\n".join(out))
