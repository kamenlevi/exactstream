import re, math, sys
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
for f in ("tab12e", "tab12f"):
    path = f"/Users/kamen/Fit12/data/scr/tab12/{f}.txt"
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        print(f, "missing"); continue
    per = {}
    for l in lines:
        m = re.search(r"\] (\S+) (\S+) theta .*decode ([\d.]+) s, ([\d.]+) tok/s", l)
        if m:
            key, arm, dec, tps = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
            per.setdefault(arm, {})[key] = tps
    for arm, d in per.items():
        he = [v for k, v in d.items() if k.startswith("humaneval")]
        mb = [v for k, v in d.items() if k.startswith("mbpp")]
        print(f"{f} arm {arm}: HE n={len(he)} geo {geo(he):.2f} | MBPP n={len(mb)} geo {geo(mb):.2f}")
