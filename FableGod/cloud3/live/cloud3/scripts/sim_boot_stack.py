import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import union_counts
from sim_b2d_reach import kk_for
W = load(); rng = np.random.default_rng(0)
for tau, name in [(None, "c3_b2d"), (0.7, "c3_b2d_reach07")]:
    line = f"  {name:16s}"
    for d in ["code", "ifeval", "multi"]:
        per = []
        for w in W:
            if w["dom"] != d: continue
            a = b = 0
            for ch in checks_of(w):
                I = w["ids"][ch["rows"]]; a += union_counts(I, kk_for(ch, tau)).sum(); b += union_counts(I, np.full((ch["R"], 48), 10)).sum()
            per.append((a, b))
        per = np.array(per, float); pt = per[:, 0].sum() / per[:, 1].sum()
        bs = [per[i, 0].sum() / per[i, 1].sum() for i in (rng.integers(0, len(per), len(per)) for _ in range(2000))]
        line += f" | {d}: x{pt:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] windows {np.min(per[:,0]/per[:,1]):.2f}-{np.max(per[:,0]/per[:,1]):.2f}"
    print(line, flush=True)
