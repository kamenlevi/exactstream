# WALL_L test 3: SLOT-CASCADE plausibility. Lookahead rows computed inside a check's own expert union (the empty slots
# of its tiles, zero extra groups): how much of each lookahead row's routed mass does the union of the check's kept
# (exact) rows hold, per layer? Rows after the kept ones are the 125B's own continuation (teacher-forced map windows).
# Rank weights = cloud3 P2 section 4's renormalised profile (identical across domains).
import numpy as np, glob, os
D = os.path.expanduser('~/FableGod/cloud3/macpub/data/decision_map_moonshots')
WR = np.array([0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010])
for dom in ('code', 'ifeval', 'multi'):
    cov = {s: [] for s in (1, 2, 4, 8)}; r0 = {s: [] for s in (1, 2, 4, 8)}; newg = {s: [] for s in (1, 2, 4, 8)}
    for f in sorted(glob.glob(os.path.join(D, f'tf_{dom}_*.npz'))):
        z = np.load(f); ids = z['ids']
        for (A, R, kept) in z['checks']:
            if A + kept + 8 > ids.shape[0]: continue
            ex = ids[A:A + kept]                                        # the check's exact (kept) rows
            U = [set(ex[:, l, :].ravel().tolist()) for l in range(48)]
            for s in (1, 2, 4, 8):                                      # the s-th lookahead row after the kept rows
                row = ids[A + kept + s - 1]
                m = np.array([[e in U[l] for e in row[l]] for l in range(48)])   # [48, 10]
                cov[s].append((m * WR).sum(1).mean()); r0[s].append(m[:, 0].mean())
                newg[s].append((~m[:, 0]).sum())                        # rank-0 experts missing: groups to add
    print(f"{dom:7s} lookahead row s: routed mass inside the kept rows' union / rank-0 inside / rank-0 groups to add per row: " +
          ' | '.join(f"s{s} {np.mean(cov[s]) * 100:4.1f} % / {np.mean(r0[s]) * 100:4.1f} % / {np.mean(newg[s]):4.1f}" for s in (1, 2, 4, 8)))
