# WALL_L test 2c: the chat wall is check length. Distinct (expert, layer) groups per row for blocks of R consecutive
# rows of the 125B's own text (teacher-forced top-10 ids, the 36 map windows): the cost per kept token if every row of
# an R-row check were kept (perfect drafting, no waste). Compare with today's live-cut cost (cloud3 P2 / test 2).
import numpy as np, glob, os
D = os.path.expanduser('~/FableGod/cloud3/macpub/data/decision_map_moonshots')
res = {}
for f in sorted(glob.glob(os.path.join(D, '*.npz'))):
    z = np.load(f); dom = str(z['domain']); ids = z['ids']
    for R in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        n = ids.shape[0] // R
        d = 0
        for b in range(n):
            blk = ids[b * R:(b + 1) * R]
            d += sum(len(np.unique(blk[:, l, :])) for l in range(48))
        a = res.setdefault((dom, R), [0, 0]); a[0] += d; a[1] += n * R
for dom in ('code', 'ifeval', 'multi'):
    print(f"{dom:7s} groups per row, all rows kept, R-row checks: " +
          ' | '.join(f"R{R} {res[(dom, R)][0] / res[(dom, R)][1]:5.0f}" for R in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)))
