import json, sys
def desc(o, depth, maxd=3):
    if depth > maxd: return type(o).__name__
    if isinstance(o, dict):
        ks = list(o.keys())
        s = f"dict[{len(ks)}] keys={ks[:12]}"
        if ks:
            s += " first=" + desc(o[ks[0]], depth+1, maxd)
        return s
    if isinstance(o, list):
        s = f"list[{len(o)}]"
        if o: s += " of " + desc(o[0], depth+1, maxd)
        return s
    if isinstance(o, str): return f"str({o[:40]!r})"
    return f"{type(o).__name__}({o})" if not isinstance(o, float) else f"float({o:.4g})"
for p in sys.argv[1:]:
    try:
        o = json.load(open(p))
        print("==", p, "\n  ", desc(o, 0))
    except Exception as e:
        print("==", p, "ERR", e)
