"""Fit12 B1 (2026-10-24): where a pass's loader stall goes, from the per-call log (v3j ext, BIGMOE_CALLLOG csv).
Rows: base L < SIDE0 - 1, MTP head SIDE0 - 1, LSB sidecar SIDE0 + L. Per kind: calls and stalled calls per pass, mean stalled
call time and misses; per trunk layer and pass: the sidecar row's start after its base row's release (sequential service)
and how often the sidecar stalls; joint mode: paired share and partner wait. Stdlib only.
Usage: python3 fit_clog.py SIDE0 file.csv [file.csv ...]"""
import csv, statistics as st, sys

SIDE0 = int(sys.argv[1])


def kind(L):
    return "base" if L < SIDE0 - 1 else ("mtp" if L == SIDE0 - 1 else "side")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


for path in sys.argv[2:]:
    R = [dict(L=int(r["L"]), seq=int(r["seq"]), m=int(r["misses"]), pr=int(r["paired"]), ts=float(r["t_seen"]),
              tr=float(r["t_rel"]), tw=float(r["t_wait"])) for r in csv.DictReader(open(path))]
    if not R:
        print(f"{path}: empty")
        continue
    seqs = {r["seq"] for r in R if kind(r["L"]) == "base"}
    npass = max(1, len(seqs))
    print(f"== {path}: {len(R)} calls, {npass} passes with base rows")
    for k in ("base", "side", "mtp"):
        rs = [r for r in R if kind(r["L"]) == k]
        sl = [r for r in rs if r["m"] > 0]
        d = [1e3 * (r["tr"] - r["ts"]) for r in sl]
        print(f"  {k:4s}: {len(rs) / npass:5.1f} calls/pass, {len(sl) / npass:5.1f} stalled/pass, stalled call "
              f"{st.mean(d) if d else 0:.3f} ms (p50 {q(d, .5):.3f}, p90 {q(d, .9):.3f}), {st.mean([r['m'] for r in sl]) if sl else 0:.1f} "
              f"misses each, {sum(d) / npass:.1f} ms/pass")
    base = {(r["seq"], r["L"]): r for r in R if kind(r["L"]) == "base"}
    side = {(r["seq"], r["L"] - SIDE0): r for r in R if kind(r["L"]) == "side"}
    gap, both, sstall, pr, tw = [], 0, 0, 0, []
    for key, b in base.items():
        s = side.get(key)
        pr += b["pr"]
        tw.append(1e6 * b["tw"])
        if s is None:
            continue
        both += 1
        gap.append(1e6 * (s["ts"] - b["tr"]))
        sstall += s["m"] > 0
    print(f"  layers with both rows: {both} ({both / npass:.1f}/pass); sidecar stalls in {sstall / max(1, both):.0%}; "
          f"sidecar seen after base release: p10 {q(gap, .1):.1f} us, p50 {q(gap, .5):.1f}, p90 {q(gap, .9):.1f}")
    print(f"  joint: paired {pr / max(1, len(base)):.0%} of base rows, partner wait mean {st.mean(tw) if tw else 0:.1f} us")
    span = max(r["tr"] for r in R) - min(r["ts"] for r in R)
    print(f"  span {span:.1f} s; loader busy (all calls) {sum(r['tr'] - r['ts'] for r in R) / span:.0%}")
