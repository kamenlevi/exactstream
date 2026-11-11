"""Price the first checks of each prompt in v1' (S1b's B arm): what they cost in whole decode, and what they are made of
(misses, demand SSD busy, gate service, graph build, eval wait, draft) vs the prompt's own later checks."""
import json, math, statistics as st
D = "/Users/kamen/BigMoE/data/r57/"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
F = ("ck", "drf", "miss", "io", "svc", "bld", "wt", "mb")
for tags in (("s1bb1_256", "s1bb2_256"),):
    agg = {i: {f: [] for f in F} for i in (0, 1, "rest")}
    sav0, sav01, base, new0, new01 = [], [], [], [], []
    for tag in tags:
        runs = {r["key"]: r for r in json.load(open(D + tag + "_runs.json"))}
        for c in json.load(open(D + tag + "_checks.json")):
            r = runs[c["key"]]
            n = len(c["ck"])
            for f in F:
                v = c.get(f) or []
                if len(v) < n:
                    continue
                agg[0][f].append(v[0]); agg[1][f].append(v[1])
                agg["rest"][f].append(st.median(v[2:]))
            dec = r["n"] / r["tps"] * 1e3
            mck, mdr = st.median(c["ck"][2:]), st.median(c["drf"][2:])
            s0 = (c["ck"][0] - mck) + (c["drf"][0] - mdr)
            s1 = (c["ck"][1] - mck) + (c["drf"][1] - mdr)
            base.append(r["tps"]); new0.append(r["n"] / (dec - s0) * 1e3); new01.append(r["n"] / (dec - s0 - s1) * 1e3)
            sav0.append(100 * s0 / dec); sav01.append(100 * (s0 + s1) / dec)
    print(f"{'piece':6s} {'check 0':>9s} {'check 1':>9s} {'later (median per prompt, then median)':>40s}")
    for f in F:
        a, b, z = (st.median(agg[k][f]) if agg[k][f] else float('nan') for k in (0, 1, "rest"))
        print(f"{f:6s} {a:9.2f} {b:9.2f} {z:12.2f}")
    print(f"\nwhole decode now (geo) {geo(base):6.1f}; if check 0 cost a later check: {geo(new0):6.1f} "
          f"({geo(new0) / geo(base) - 1:+.1%}); if checks 0 and 1 did: {geo(new01):6.1f} ({geo(new01) / geo(base) - 1:+.1%})")
    print(f"share of decode wall: check 0 excess median {st.median(sav0):.1f} %, checks 0+1 {st.median(sav01):.1f} %")
