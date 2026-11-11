import json, numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load(); fr = json.load(open("/root/350-tps/FableGod/floor_rows.json")); F = {(r["nm"], r["j"]): r for r in fr}
import collections
c = collections.Counter(); ex = []
for w in W:
    gen = w["gen"]
    for ch in checks_of(w):
        for i in range(ch["R"]):
            row = int(ch["rows"][i]); cf = ch["cf"][i]; rec = F.get((w["nm"], row - gen))
            if rec is None: continue
            kind = "kept" if i < ch["kept"] else "wasted"; kind += "/bonus" if i == ch["L"] else ""
            fcf = rec["cf"]
            if fcf is None: c[(kind, "floor None", cf == cf)] += 1
            elif not (cf == cf): c[(kind, "mine nan", "floor %.2f" % fcf)] += 1
            elif abs(fcf - cf) < 1e-4: c[(kind, "match")] += 1
            else:
                c[(kind, "mismatch")] += 1
                if len(ex) < 6: ex.append((w["nm"], i, ch["kept"], ch["L"], round(cf, 4), round(fcf, 4), rec["band"]))
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:14]: print(k, v)
print("examples (nm, i, kept, L, my cf, floor cf, floor band):"); [print("  ", e) for e in ex]
