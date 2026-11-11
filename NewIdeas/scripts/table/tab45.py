import json, math, statistics as st, os
D = "/Users/kamen/BigMoE/data/r57/"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
def arm(tags):
    R = [{r["key"]: r for r in json.load(open(D + t + "_256_runs.json"))} for t in tags if os.path.exists(D + t + "_256_runs.json")]
    keys = sorted(set.intersection(*[set(x) for x in R]))
    kept = st.mean(r["kept"] for x in R for r in x.values())
    return geo([st.mean(x[k]["tps"] for x in R) for k in keys]), len(keys), len(R), kept
for name, tags in (("E5 E HE", ["e5h1", "e5h2"]), ("E5 EB HE", ["e5hb1", "e5hb2"]), ("E5 E MBPP", ["e5m1", "e5m2"]),
                   ("E5 EB MBPP", ["e5mb1", "e5mb2"]), ("S5' B HE", ["spb1", "spb2"]), ("S5' X HE", ["spx1", "spx2"]),
                   ("S5' B MBPP", ["spmb1", "spmb2"]), ("S5' X MBPP", ["spmx1", "spmx2"])):
    try:
        w, n, p, k = arm(tags)
        print(f"{name:11s} whole decode {w:6.1f}  ({n} prompts, {p} passes, mean kept/check {k:.2f})")
    except Exception as e:
        print(f"{name:11s} n/a ({e.__class__.__name__}: {e})")
