import json, math, statistics as st
D = "/Users/kamen/BigMoE/data/r57/"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
P = {"B": ["s6b1_256", "s6b2_256"], "W": ["s6w1_256", "s6w2_256"]}
R = {t: {r["key"]: r for r in json.load(open(D + t + "_runs.json"))} for ts in P.values() for t in ts}
C = {t: {c["key"]: c for c in json.load(open(D + t + "_checks.json"))} for ts in P.values() for t in ts}
keys = sorted(set.intersection(*[set(v) for v in R.values()]))
for t in R:
    print(f"{t}: whole geo {geo([R[t][k]['tps'] for k in keys]):6.1f}  check0 ck median {st.median(C[t][k]['ck'][0] for k in keys):6.1f} ms  "
          f"later ck median {st.median(st.median(C[t][k]['ck'][2:]) for k in keys):5.1f}  check1 {st.median(C[t][k]['ck'][1] for k in keys):5.1f}")
def arm(a, f):
    return geo([st.mean(R[t][k][f] for t in P[a]) for k in keys])
for f in ("tps", "tpsw"):
    print(f"{f}: B {arm('B', f):6.1f} W {arm('W', f):6.1f} W/B {arm('W', f) / arm('B', f) - 1:+.1%}")
