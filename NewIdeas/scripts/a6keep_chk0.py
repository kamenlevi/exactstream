import json, statistics as st
D = "/Users/kamen/BigMoE/data/r57/"
def chk0(tag):
    out = {}
    try:
        cs = json.load(open(D + tag + "_checks.json"))
    except FileNotFoundError:
        return out
    for c in cs:
        n = len(c["ck"])
        if n < 4:
            continue
        b = c.get("bld") or [float("nan")] * n
        out[c["key"]] = (c["ck"][0], st.median(c["ck"][2:]), b[0] if len(b) == n else float("nan"),
                         st.median(b[2:]) if len(b) == n else float("nan"), c["miss"][0], st.median(c["miss"][2:]))
    return out
for tag in ("cmb1c_160", "cmb2_160", "s1bb1_256", "s1bb2_256"):
    o = chk0(tag)
    code = {k: v for k, v in o.items() if k.startswith("humaneval")}
    oth = {k: v for k, v in o.items() if not k.startswith("humaneval")}
    for name, d in (("code", code), ("other", oth)):
        if not d:
            continue
        ex = [v[0] - v[1] for v in d.values()]
        bx = [v[2] - v[3] for v in d.values()]
        print(f"{tag:10s} {name:5s} n={len(d):2d}: check0 ck {st.median(v[0] for v in d.values()):6.1f} vs later {st.median(v[1] for v in d.values()):6.1f} "
              f"(excess med {st.median(ex):6.1f}); build excess med {st.median(bx):6.1f}; misses {st.median(v[4] for v in d.values()):5.0f} vs {st.median(v[5] for v in d.values()):5.0f}")
