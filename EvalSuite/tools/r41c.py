"""R-41C fold confirmation (docs/PREREG_r41c_2026-10-28.md). Subcommands:
  diffkeys A B SETS   keys (set order) whose completion_ids differ between results/A and results/B (rows in both)
  base                build results/base_r41c/decode.jsonl: b12ll's core rows; where b12llff differs from b12ll and
                      b12llfx (the exact default) differs from b12ll too, b12llfx's row (drift in the exact default)
  a                   part A: b12llfx vs b12llfoff (and b12ll) on runner 24's 45 rows, footprint per process, and
                      b12llox's follow-up rows if any
  b                   part B: b12llff's tokens vs b12ll and the baseline, footprint per process, SSD MB per token
Usage: .venv/bin/python tools/r41c.py <subcommand> [args]"""
import json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = 11.95
CORE = ("math", "ifeval", "humaneval")


def rows(v):
    p = os.path.join(ROOT, "results", v, "decode.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def order(sets):
    out = []
    for s in sets:
        out += [json.loads(l)["key"] for l in open(os.path.join(ROOT, "data", "items", f"dec_{s}.jsonl"))]
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def diffkeys(A, B, sets):
    a, b = rows(A), rows(B)
    return [k for k in order(sets) if k in a and k in b and a[k]["completion_ids"] != b[k]["completion_ids"]]


def peak(r):
    m = r.get("mem", {})
    return max(m.get("max", 0), m.get("life", 0))


def memory(v, keys=None):
    """per process (chunks.jsonl, in order; rows map to chunks by count) and over all rows"""
    R = [json.loads(l) for l in open(os.path.join(ROOT, "results", v, "decode.jsonl"))]
    cp = os.path.join(ROOT, "results", v, "chunks.jsonl")
    ch = [json.loads(l) for l in open(cp)] if os.path.exists(cp) else []
    print(f"  {v}: {len(ch)} processes with a chunk record, {len(R)} rows")
    i = 0
    for c in ch:
        rs = R[i:i + c["rows"]]
        i += c["rows"]
        if not rs:
            continue
        sets = sorted({r["set"] for r in rs})
        rp = max(peak(r) for r in rs)
        print(f"    process of {c['rows']:3d} rows ({','.join(sets)}): lifetime peak {c['footprint_peak_gb']:.2f} GB, "
              f"max row peak {rp:.2f}, footprint at the first row's start {rs[0]['mem'].get('start', 0):.2f}, "
              f"at the last row's start {rs[-1]['mem'].get('start', 0):.2f}"
              + ("  OVER 11.95" if max(rp, c["footprint_peak_gb"]) > LINE else ""))
    if i != len(R):
        print(f"    {len(R) - i} rows without a chunk record (a process stopped before its record)")
    sel = [r for r in R if keys is None or r["key"] in keys]
    over = [(r["key"], r["n_prompt"], r.get("ssd", {}).get("cm"), round(peak(r), 2)) for r in sel if peak(r) > LINE]
    top = max((max(peak(r) for r in sel), *(c["footprint_peak_gb"] for c in ch)), default=0)
    print(f"    max over rows and processes: {top:.2f} GB; rows over {LINE}: {len(over)}" + (f" -> {over}" if over else ""))
    return top


def ssd(v, keys):
    R = rows(v)
    out = {}
    for s in ("agent",) + CORE:
        rs = [R[k] for k in keys if k in R and R[k]["set"] == s and "ssd" in R[k]]
        n = sum(r["ssd"].get("n", 0) for r in rs)
        if n:
            out[s] = sum(r["ssd"].get("disk_mb", 0) for r in rs) / n
    return out


def cmd_base():
    ll, ff, fx = rows("b12ll"), rows("b12llff"), rows("b12llfx")
    keys = order(CORE)
    miss = [k for k in keys if k not in ll or k not in ff]
    assert not miss, f"core rows missing in b12ll or b12llff: {miss[:5]}"
    dk = [k for k in keys if ff[k]["completion_ids"] != ll[k]["completion_ids"]]
    nofx = [k for k in dk if k not in fx]
    assert not nofx, f"b12llfx has not decoded {len(nofx)} of the rows where b12llff differs from b12ll: {nofx[:5]}"
    drift = [k for k in dk if fx[k]["completion_ids"] != ll[k]["completion_ids"]]
    os.makedirs(os.path.join(ROOT, "results", "base_r41c"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "base_r41c", "decode.jsonl"), "w") as f:
        for k in keys:
            f.write(json.dumps(dict(fx[k] if k in drift else ll[k], base_from="b12llfx" if k in drift else "b12ll"),
                               ensure_ascii=False) + "\n")
    print(f"base_r41c: {len(keys)} core rows; b12llff differs from b12ll on {len(dk)}; b12llfx differs from b12ll on "
          f"{len(drift)} of those (its rows replace b12ll's: drift of the exact default)")
    for k in drift:
        print(f"  drift {k}: b12llfx vs b12ll from token {first_diff(fx[k]['completion_ids'], ll[k]['completion_ids'])}")


def cmd_a():
    keys = order(("agent",))[:29] + order(("math",))[:16]
    fx, off, ll, ox = rows("b12llfx"), rows("b12llfoff"), rows("b12ll"), rows("b12llox")
    have = [k for k in keys if k in fx]
    print(f"R-41C part A: b12llfx (fold-exact) on runner 24's 45 rows: {len(have)}/45 decoded\n")
    print("1. tokens")
    for name, B in (("b12llfoff", off), ("b12ll", ll)):
        d = [k for k in have if fx[k]["completion_ids"] != B[k]["completion_ids"]]
        print(f"  vs {name}: {len(have) - len(d)}/{len(have)} identical" + "".join(
            f"\n    {k}: from token {first_diff(fx[k]['completion_ids'], B[k]['completion_ids'])}" for k in d))
    d = [k for k in have if fx[k]["completion_ids"] != off[k]["completion_ids"]]
    if d:
        print("  follow-up, b12llox (old envs on the new code) on the differing rows:")
        for k in d:
            if k not in ox:
                print(f"    {k}: not decoded yet")
                continue
            print(f"    {k}: b12llox {'=' if ox[k]['completion_ids'] == off[k]['completion_ids'] else '!='} b12llfoff, "
                  f"{'=' if ox[k]['completion_ids'] == fx[k]['completion_ids'] else '!='} b12llfx")
    print("\n2. footprint (GB; row peak = max of the sampled max and the lifetime peak)")
    top = memory("b12llfx", set(keys))
    ok = len(have) == 45 and not d and top <= LINE
    print(f"\nPart A: {'PASS' if ok else 'not PASS'} (45/45 identical to b12llfoff: {len(have) == 45 and not d}; "
          f"every process <= {LINE} GB: {top <= LINE})")
    print("\n3. SSD MB per decoded token (information):",
          {v: {s: round(x, 1) for s, x in ssd(v, keys).items()} for v in ("b12llfx", "b12llff", "b12llfoff")})


def cmd_b():
    ff, ll, fx = rows("b12llff"), rows("b12ll"), rows("b12llfx")
    base = rows("base_r41c")
    print(f"R-41C part B: b12llff (fold-fast) rows {len(ff)}/239\n")
    print("1. tokens (information)")
    for s in ("agent",) + CORE:
        ks = [k for k in order((s,)) if k in ff and k in ll]
        d = [k for k in ks if ff[k]["completion_ids"] != ll[k]["completion_ids"]]
        dx = [k for k in ks if k in fx and ff[k]["completion_ids"] != fx[k]["completion_ids"]]
        print(f"  {s:9s} identical to b12ll {len(ks) - len(d)}/{len(ks)}; to b12llfx {len([k for k in ks if k in fx]) - len(dx)}"
              f"/{len([k for k in ks if k in fx])}" + (f"; baseline rows from b12llfx (drift): "
              f"{sum(1 for k in ks if base.get(k, {}).get('base_from') == 'b12llfx')}" if s in CORE and base else ""))
    print("\n2. footprint (GB)")
    memory("b12llff")
    print("\n3. SSD MB per decoded token (information):",
          {v: {s: round(x, 1) for s, x in ssd(v, order(("agent",) + CORE)).items()} for v in ("b12llff", "b12llfx", "b12ll")})


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "diffkeys":
        print("\n".join(diffkeys(sys.argv[2], sys.argv[3], sys.argv[4].split(","))))
    elif c == "base":
        cmd_base()
    elif c == "a":
        cmd_a()
    elif c == "b":
        cmd_b()
    else:
        raise SystemExit(__doc__)
