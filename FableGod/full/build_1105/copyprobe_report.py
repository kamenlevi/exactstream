"""copyprobe_report.py (Fable FULL, LOSSY350 T-A1): read a copy probe's JSONL (copyprobe.py) and print the timing table and
the verdict. CPU, no model. usage: python -I copyprobe_report.py <{TAG}_copyprobe.jsonl> [out.txt]
Per copied token = pass ms / rows (a W-row copy pass yields W tokens when every row is confirmed: W - 1 copies + the bonus).
Lines (pre-registered in INSTALL.md, Thu 5 Nov, before any run):
  confirmed  = the share of copied rows the exact wide passes confirm (argmax == the copied token); needs >= 0.90.
  DIRECT     = the best wide STRICT pass that really ran strict at HEAD (s or t, wc > 0) and itself confirms >= 0.90 of
               its copied rows, cold-first median.
  SLOPE      = the fast path extended by its own measured row cost: intercept + W x slope from the warm g4 / g8 / g16
               passes, plus W x the cold extra per row of a first-touch g16 pass.
  PASS    confirmed >= 0.90 AND (DIRECT or SLOPE) <= 3.0 ms per copied token at W 32 or <= 1.8 at W 64.
  KILL    confirmed < 0.80, OR every one of DIRECT and SLOPE at both widths >= 4.5 ms per copied token.
  BETWEEN anything else. A PASS by SLOPE alone means: the rows are cheap, the wide fast path is still to be built."""
import json, statistics as S, sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return S.median(xs) if xs else float("nan")


def fit(xy):
    """least squares y = a + b x over [(x, y)]; (b, a) or None"""
    if len({x for x, _ in xy}) < 2:
        return None
    n = len(xy); sx = sum(x for x, _ in xy); sy = sum(y for _, y in xy)
    sxx = sum(x * x for x, _ in xy); sxy = sum(x * y for x, y in xy)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return b, (sy - b * sx) / n


def report(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    pts = [r for r in recs if r.get("kind") == "point"]
    ps = [r for r in recs if r.get("kind") == "pass"]
    bad = [r for r in ps if r.get("err")] + [r for r in recs if r.get("kind") == "error"]
    ok = [r for r in ps if not r.get("err") and "acc" in r]
    L = [f"COPY PROBE {path}: {len(pts)} probe points in {len({r['run'] for r in pts})} runs, {len(ok)} passes, {len(bad)} failed"]
    if pts:
        L.append(f"  arm: FAST {pts[0].get('fast')}, split {pts[0].get('split')}, D17 {pts[0].get('d17')}; copied tokens left at the "
                 f"points: median {med([r['left'] for r in pts]):.0f} (min {min(r['left'] for r in pts)})")
    for r in bad[:6]:
        L.append(f"  FAILED {r.get('v', '')}{r.get('rows', '')} in {r.get('run')}: {r.get('err')}")
    keys = sorted({(r["v"], r["rows"]) for r in ok}, key=lambda k: ("gxst".index(k[0]), k[1]))
    L.append("  pass   | n | cold-first ms (misses, uses) | warm ms | per copied token cold / warm | confirmed | strict built")
    T = {}
    for v, rows in keys:
        rs = [r for r in ok if (r["v"], r["rows"]) == (v, rows)]
        first = [r for r in rs if r.get("first")]
        warm = [r for r in rs if r["rep"] == 1]
        conf = sum(r["agree"] for r in rs) / max(1, sum(r["n"] for r in rs))
        built = (v in "gx") or all(r.get("wc", 0) > 0 for r in rs)
        T[(v, rows)] = dict(cold=med([r["ms"] for r in first]), warm=med([r["ms"] for r in warm]), conf=conf, built=built,
                            nfirst=len(first), strict=all(r.get("strict") for r in rs))
        t = T[(v, rows)]
        cf = (f"{t['cold']:7.1f} ({med([r['miss'] for r in first]):.0f}, {med([r['uses'] for r in first]):.0f}; {len(first)} first)"
              if first else "      - (never first)")
        L.append(f"  {v}{rows:<5d} | {len(rs):3d} | {cf} | {t['warm']:7.1f} | "
                 f"{(format(t['cold'] / rows, '5.2f') if first else '    -')} / {t['warm'] / rows:5.2f} | {conf:.3f} | "
                 f"{'yes' if t['strict'] and built else 'NO (ran exact)' if v in 'st' else 'exact' if not t['strict'] else 'yes'}")
    # the fast path's row cost: per probe point, warm g passes against rows
    sl, ic = [], []
    for p in pts:
        xy = [(r["rows"], r["ms"]) for r in ok if r["run"] == p["run"] and r["p"] == p["p"] and r["v"] == "g" and r["rep"] == 1]
        f = fit(xy)
        if f:
            sl.append(f[0]); ic.append(f[1])
    slope, icpt = med(sl), med(ic)
    g16 = T.get(("g", 16))
    cold_row = (g16["cold"] - g16["warm"]) / 16 if g16 and g16["nfirst"] else float("nan")
    L.append(f"  fast path (gate, <= 16 rows, {len(sl)} points): warm {slope:.2f} ms per strict copied row + {icpt:.1f} ms; a first-touch "
             f"g16 pass costs {cold_row:.2f} ms per row more than a warm one ({g16['nfirst'] if g16 else 0} first passes)")
    pred = {}
    for W in (32, 64):
        warm = icpt + W * slope
        cold = warm + W * (cold_row if cold_row == cold_row else 0.0)
        pred[W] = cold / W
        L.append(f"  SLOPE, W {W}: {warm:.0f} ms warm, {cold:.0f} ms first-touch = {cold / W:.2f} ms per copied token "
                 f"(today's 16-row gate pass on the same rows: {g16['cold'] / 16 if g16 else float('nan'):.2f}, no drafter in either)")
    direct = {}
    for W in (32, 64):
        c = []
        for v in "st":
            t = T.get((v, W))
            if not t or not t["built"] or t["conf"] < 0.90:           # a strict pass that does not confirm the copy does
                continue                                             # not count, however fast (a wrong kernel is fast too)
            if t["nfirst"]:
                c.append((t["cold"] / W, f"{v}{W}"))
            else:                                                    # never first (t): its warm pass + a first touch's extra,
                s_ = T.get(("s", W))                                 # read from s at the same width, else from g16 per row
                extra = (s_["cold"] - s_["warm"]) if s_ and s_["nfirst"] else W * (cold_row if cold_row == cold_row else 0.0)
                c.append(((t["warm"] + extra) / W, f"{v}{W}, first touch estimated"))
        direct[W] = min(c)[0] if c else float("nan")
        x = T.get(("x", W))
        L.append(f"  DIRECT, W {W}: {direct[W]:.2f} ms per copied token ({min(c)[1] if c else 'no strict wide pass ran'}; the exact "
                 f"wide pass x{W}: {(x['cold'] if x['nfirst'] else x['warm']) / W if x else float('nan'):.2f})")
    wide = ([T[k]["conf"] for k in T if k[1] >= 32 and k[0] == "x"]        # the model's own word on the copy: the exact wide
            or [T[k]["conf"] for k in T if k[1] >= 32 and k[0] == "s"])     # passes (else s); t never decides this
    conf = min(wide) if wide else (g16["conf"] if g16 else float("nan"))
    best = {W: min([x for x in (direct[W], pred[W]) if x == x] or [float("nan")]) for W in (32, 64)}
    fin = [x for W in (32, 64) for x in (direct[W], pred[W]) if x == x]
    how = "DIRECT" if any(direct[W] == direct[W] and direct[W] <= lim for W, lim in ((32, 3.0), (64, 1.8))) else "SLOPE only (the wide fast path is still to be built)"
    if not fin or conf != conf:
        v = "NO VERDICT (no timing rows)"
    elif conf < 0.80 or min(fin) >= 4.5:
        v = "KILL"
    elif conf >= 0.90 and (best[32] <= 3.0 or best[64] <= 1.8):
        v = f"PASS by {how}"
    else:
        v = "BETWEEN"
    L.append(f"Verdict T-A1 (pre-registered Thu 5 Nov): {v}; confirmed {conf:.3f} (>= 0.90), per copied token W 32 {best[32]:.2f} "
             f"(<= 3.0), W 64 {best[64]:.2f} (<= 1.8); KILL at >= 4.5 everywhere or confirmed < 0.80")
    return "\n".join(L)


if __name__ == "__main__":
    import os
    if not os.path.exists(sys.argv[1]):
        print(f"COPY PROBE {sys.argv[1]}: MISSING (the probe arm did not run or wrote nothing)"); sys.exit(0)
    out = report(sys.argv[1])
    print(out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out + "\n")
