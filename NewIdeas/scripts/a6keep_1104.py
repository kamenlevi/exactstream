import json, math
D = "/Users/kamen/BigMoE/data/r57/"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
arms = ["cmb0", "cmb1", "cmb1c", "cmb1t", "cmb2", "cmb3"]
data = {}
for a in arms:
    try:
        runs = json.load(open(f"{D}{a}_160_runs.json"))
    except FileNotFoundError:
        continue
    data[a] = {r["key"]: r for r in runs if not r.get("bad")}
common = set.intersection(*[set(v) for v in data.values()])
code = sorted(k for k in common if k.startswith("humaneval"))
print(f"prompts in all arms: {len(common)} (code {len(code)})")
for a, d in data.items():
    allk = sorted(common)
    print(f"{a:6s} all: whole {geo([d[k]['tps'] for k in allk]):6.1f} steady {geo([d[k]['tpsw'] for k in allk]):6.1f} | "
          f"code: whole {geo([d[k]['tps'] for k in code]):6.1f} steady {geo([d[k]['tpsw'] for k in code]):6.1f}")
for base in ("cmb1c", "cmb1t", "cmb1"):
    if base in data and "cmb2" in data:
        b, k = data[base], data["cmb2"]
        for name, keys in (("all", sorted(common)), ("code", code)):
            w = geo([k[x]["tps"] for x in keys]) / geo([b[x]["tps"] for x in keys]) - 1
            s = geo([k[x]["tpsw"] for x in keys]) / geo([b[x]["tpsw"] for x in keys]) - 1
            print(f"cmb2 (keep) vs {base} [{name}]: whole decode {w:+.1%}, steady {s:+.1%}")
