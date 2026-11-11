"""Any r57 A/B (ARM_B, ARM_N = comma lists of pass tags in ~/BigMoE/data/r57). the registered steady metric (tpsw = kept / (c2 - c0), checks >= 2) leaves out the gap between c2 and
the next c0. Price that gap per arm from the whole decode wall (dec = n / tps) minus the timed check + draft time, and
form a full-period steady rate: tokens after the first two checks / (dec - the first two checks' ck + drf - 2 gaps)."""
import json, math, statistics as st

D = "/Users/kamen/BigMoE/data/r57/"
import os
ARMS = {"B": os.environ["ARM_B"].split(","), "NW": os.environ["ARM_N"].split(",")}
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
full, regd = {}, {}
for arm, tags in ARMS.items():
    for tag in tags:
        runs = {r["key"]: r for r in json.load(open(D + tag + "_runs.json"))}
        checks = {c["key"]: c for c in json.load(open(D + tag + "_checks.json"))}
        gaps, nchk = 0.0, 0
        fp, rg = {}, {}
        for k, c in checks.items():
            r = runs[k]
            dec = r["n"] / r["tps"] * 1e3
            s = [a + b for a, b in zip(c["ck"], c["drf"])]
            gaps += dec - sum(s); nchk += len(s)
        g = gaps / nchk                                   # mean untimed ms per check (host work + any sync outside c0..c2)
        for k, c in checks.items():
            r = runs[k]
            dec = r["n"] / r["tps"] * 1e3
            s = [a + b for a, b in zip(c["ck"], c["drf"])]
            tok = r["n"] - c["kept"][0] - c["kept"][1]
            fp[k] = 1e3 * tok / (dec - s[0] - s[1] - 2 * g)
            rg[k] = r["tpsw"]
        full[tag], regd[tag] = fp, rg
        print(f"{arm:2s} {tag}: untimed gap {g:5.2f} ms/check | geo tpsw (registered) {geo(rg.values()):6.1f} | "
              f"geo full-period steady {geo(fp.values()):6.1f}")
keys = sorted(full[ARMS["B"][0]])
def arm_geo(src, arm):
    return geo([st.mean(src[t][k] for t in ARMS[arm]) for k in keys])
for name, src in (("registered tpsw", regd), ("full-period steady", full)):
    b, n = arm_geo(src, "B"), arm_geo(src, "NW")
    print(f"{name:20s}: B {b:6.1f}  NW {n:6.1f}  NW/B {n / b - 1:+.1%}")
wins = sum(1 for k in keys if st.mean(full[t][k] for t in ARMS["NW"]) > st.mean(full[t][k] for t in ARMS["B"]))
print(f"full-period: NW faster on {wins}/12 prompts")
