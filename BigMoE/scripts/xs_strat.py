"""Per-check stratified read-out of an xsafe A/B (Leap Sat 15:4x; information beside the pre-registered per-prompt cycle line).

X and B keep different numbers of tokens per check under FAST (QOM moves acceptance pass to pass), so their checks mix accept
types and miss loads differently, and a per-prompt median cycle blends those. This splits every check >= 2 of both passes by
accept type (full: kept == rows; partial: kept < rows) x miss band (misses <= MB or > MB) and reports X - B medians of the cycle
(c0 to the next c0), check + commit (ck) and draft (drf), with n per arm, plus the pooled and the class-weighted cycle delta
(weights: each class's share of all checks of both arms).
usage: python scripts/xs_strat.py <xsafe name> [C=256] [MB=100]   -> prints, and writes data/xs_<name>/xs_strat.txt
"""
import json
import statistics as S
import sys

N = sys.argv[1]
C = sys.argv[2] if len(sys.argv) > 2 else "256"
MB = int(sys.argv[3]) if len(sys.argv) > 3 else 100


def checks(tags):
    out = []
    for t in tags:
        for c in json.load(open(f"data/r57/{t}_{C}_checks.json")):
            if c.get("paused"):
                continue
            c0 = c.get("c0") or []
            for i in range(2, len(c["ck"])):
                out.append(dict(full=c["kept"][i] >= c["rows"][i], hi=c["miss"][i] > MB, ck=c["ck"][i], drf=c["drf"][i],
                                cyc=(c0[i + 1] - c0[i]) if i + 1 < len(c0) else None))
    return out


med = lambda xs: S.median(xs) if xs else float("nan")
B, X = checks([f"{N}b1", f"{N}b2"]), checks([f"{N}x1", f"{N}x2"])
L = [f"xsafe {N}: per-check read-out, checks >= 2 of both passes (B {len(B)}, X {len(X)} checks), miss band {MB}; X - B medians, ms"]
tot = len(B) + len(X)
wsum = 0.0
for full in (True, False):
    for hi in (False, True):
        b = [r for r in B if r["full"] == full and r["hi"] == hi]
        x = [r for r in X if r["full"] == full and r["hi"] == hi]
        if not b and not x:
            continue
        d = {k: med([r[k] for r in x if r[k] is not None]) - med([r[k] for r in b if r[k] is not None]) for k in ("cyc", "ck", "drf")}
        w = (len(b) + len(x)) / tot
        if d["cyc"] == d["cyc"]:
            wsum += w * d["cyc"]
        L.append(f"  {'full' if full else 'partial':7s} misses {'>' if hi else '<='} {MB:3d} (share {100 * w:4.1f} %, n {len(b)} / {len(x)}): "
                 f"cycle {d['cyc']:+.2f}, check + commit {d['ck']:+.2f}, draft {d['drf']:+.2f}")
cb, cx = [r["cyc"] for r in B if r["cyc"] is not None], [r["cyc"] for r in X if r["cyc"] is not None]
L.append(f"  pooled: cycle median {med(cx) - med(cb):+.2f} (mean {S.mean(cx) - S.mean(cb):+.2f}), check + commit "
         f"{med([r['ck'] for r in X]) - med([r['ck'] for r in B]):+.2f}; class-weighted cycle {wsum:+.2f} ms "
         f"({100 * wsum / med(cb):+.2f} % of B's median cycle {med(cb):.2f})")
L.append(f"  full-accept share: B {100 * sum(r['full'] for r in B) / max(len(B), 1):.1f} %, X {100 * sum(r['full'] for r in X) / max(len(X), 1):.1f} %")
open(f"data/xs_{N}/xs_strat.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
