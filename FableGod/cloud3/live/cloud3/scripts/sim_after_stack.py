"""Where the remaining tiles sit after the stack (c3_b2d + reach 0.7): attribution of the distinct experts of each (check,
layer) to A = a kept row's rank 0-1, B = a kept row's kept tail (ranks 2+ of its band), C = rejected rows only; and the
share of singletons."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_b2d_reach import kk_for
W = load()
for d in ["code", "ifeval", "multi"]:
    cat = np.zeros(3); single = 0; tot = 0; K = 0
    for w in W:
        if w["dom"] != d: continue
        for ch in checks_of(w):
            I = w["ids"][ch["rows"]]; kk = kk_for(ch, 0.7); kept = ch["kept"]; K += kept
            for l in range(48):
                cnt = {}; a01 = set(); bt = set()
                for r in range(ch["R"]):
                    k = int(kk[r, l]); es = I[r, l, :k].tolist()
                    for e in es: cnt[e] = cnt.get(e, 0) + 1
                    if r < kept: a01.update(es[:2]); bt.update(es[2:])
                for e, c in cnt.items():
                    tot += 1; single += c == 1
                    cat[0 if e in a01 else 1 if e in bt else 2] += 1
    print(f"{d}: tiles per kept token {tot/K:.1f}; A kept rank 0-1 {cat[0]/tot:.3f}, B kept tails {cat[1]/tot:.3f}, C rejected-only {cat[2]/tot:.3f}; singletons {single/tot:.3f}")
