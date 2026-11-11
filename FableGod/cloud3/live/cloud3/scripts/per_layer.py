import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of, distinct, rule_k
W = load()
for dom in ["code", "ifeval", "multi"]:
    ex = np.zeros(48); q = np.zeros(48); n = 0; sing = np.zeros(48); tot = np.zeros(48)
    for w in W:
        if w["dom"] != dom: continue
        for ch in checks_of(w):
            I = w["ids"][ch["rows"]]
            ex += distinct(I, np.full(ch["R"], 10)); q += distinct(I, rule_k(ch, "q1", thr=0.5, ks=2)); n += 1
            for l in range(48):
                c = {}
                for r in range(ch["R"]):
                    for e in I[r, l]: c[int(e)] = c.get(int(e), 0) + 1
                sing[l] += sum(1 for v in c.values() if v == 1); tot[l] += len(c)
    print(f"{dom}: live-cut union per layer, exact:", " ".join(f"{x/n:.0f}" for x in ex))
    print(f"{dom}: Q1 0.5 strict per layer:         ", " ".join(f"{x/n:.0f}" for x in q))
    print(f"{dom}: singleton share per layer:       ", " ".join(f"{s/t:.2f}"[1:] for s, t in zip(sing, tot)))
    bands = {"0-7": range(0, 8), "8-35": range(8, 36), "36-47": range(36, 48)}
    print("   band means exact / Q1 0.5:", {b: (round(ex[list(r)].mean()/n, 1), round(q[list(r)].mean()/n, 1)) for b, r in bands.items()})
