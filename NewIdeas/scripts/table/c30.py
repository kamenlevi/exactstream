import json, math, statistics as st
D = "/Users/kamen/BigMoE/data/r57/"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
for name, tags in (("C30 HE", ["c30h1_160", "c30h2_160"]), ("C30 MBPP", ["c30m1_160", "c30m2_160"])):
    R = [{r["key"]: r for r in json.load(open(D + t + "_runs.json"))} for t in tags]
    keys = sorted(set.intersection(*[set(x) for x in R]))
    per = [geo([x[k]["tps"] for k in keys]) for x in R]
    print(f"{name}: whole decode {geo([st.mean(x[k]['tps'] for x in R) for k in keys]):.1f} (passes {per[0]:.1f}, {per[1]:.1f}); "
          f"peak {max(max(r['mem'][-1] for r in x.values()) for x in R):.1f} GB (mem[-1] lifetime)")
