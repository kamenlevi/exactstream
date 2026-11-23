"""Paired comparison of two eval tags (evalx.py output) over the same windows.
For a band (default 23-30) and for kept/32: the pooled rate of each arm, the difference, a per-window sign test
(two-sided binomial over windows that differ) and a bootstrap 95 % CI of the pooled difference (resampling windows).
usage: compare.py A.json "run|set|mode" B.json "run|set|mode" [--band 23-30] [--line "PASS if >= +2.0 pt"]
       (B is the candidate, A the control; positive = the candidate is better)"""
import argparse, json, math, sys
import numpy as np


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(pa, pb, key, rng):
    ws = sorted(set(pa) & set(pb))
    ra = np.array([pa[w][key][0] for w in ws], float); va = np.array([pa[w][key][1] for w in ws], float)
    rb = np.array([pb[w][key][0] for w in ws], float); vb = np.array([pb[w][key][1] for w in ws], float)
    ok = (va > 0) & (vb > 0)
    ra, va, rb, vb = ra[ok], va[ok], rb[ok], vb[ok]
    A, B = ra.sum() / max(va.sum(), 1), rb.sum() / max(vb.sum(), 1)
    d = rb / vb - ra / va
    wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(ra), len(ra))
        bs.append(rb[i].sum() / max(vb[i].sum(), 1) - ra[i].sum() / max(va[i].sum(), 1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(ok.sum()), "A": A, "B": B, "diff": B - A, "ci95": [float(lo), float(hi)], "wins": wins,
            "losses": losses, "p_sign": binom_two_sided(min(wins, losses), wins + losses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fa"); ap.add_argument("ta"); ap.add_argument("fb"); ap.add_argument("tb")
    ap.add_argument("--band", default="23-30"); ap.add_argument("--line", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    A, B = json.load(open(a.fa)), json.load(open(a.fb))
    pa, pb = A["per"][a.ta], B["per"][a.tb]
    rng = np.random.default_rng(1114)
    out = {"A": a.ta, "B": a.tb, "line": a.line}
    for key in [a.band, "0-0", "kept"]:
        r = paired(pa, pb, key, rng)
        out[key] = r
        unit = "" if key == "kept" else " %"
        if key == "kept":
            print(f"kept/32: A {r['A']:.3f} -> B {r['B']:.3f} ({r['diff']:+.3f}, CI {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}); "
                  f"windows B better {r['wins']} / worse {r['losses']} (sign p {r['p_sign']:.3f}, n {r['n']})")
        else:
            print(f"band {key}: A {100 * r['A']:.2f} -> B {100 * r['B']:.2f} ({100 * r['diff']:+.2f} pt, CI "
                  f"{100 * r['ci95'][0]:+.2f}..{100 * r['ci95'][1]:+.2f}); windows B better {r['wins']} / worse {r['losses']} "
                  f"(sign p {r['p_sign']:.3f}, n {r['n']})")
    if a.line:
        print(f"pre-registered line: {a.line}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
