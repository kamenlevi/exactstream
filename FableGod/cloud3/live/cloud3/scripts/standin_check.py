"""The rejected rows' share of the union with the map's wrong-draft stand-in convention (row p + 97) vs the true-path rows."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of, distinct
W = load()
for dom in ["code", "ifeval", "multi"]:
    for standin in [False, True]:
        U = 0; UK = 0; K = 0
        for w in W:
            if w["dom"] != dom: continue
            N = w["ids"].shape[0]
            for ch in checks_of(w):
                rows = ch["rows"].copy()
                if standin:
                    for i in range(ch["kept"], ch["R"]): rows[i] = (ch["rows"][i] + 97) % N
                I = w["ids"][rows]
                U += distinct(I, np.full(ch["R"], 10)).sum(); UK += distinct(I[:ch["kept"]], np.full(ch["kept"], 10)).sum(); K += ch["kept"]
        print(f"{dom:6s} stand-in {standin!s:5s}: distinct per kept token {U/K:6.1f}, kept-only {UK/K:6.1f}, rejected-row share {1-UK/U:.3f}")
