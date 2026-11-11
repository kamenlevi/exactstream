# WALL_L test 2: the wall across streams. Merge the live-cut checks of B different windows (B concurrent streams of
# one user: check j of each stream in one pass) and count distinct (expert, layer) groups per kept token, EXACT rows
# (top-10 everywhere), on the 36 decision-map windows (MoonShots export: teacher-forced top-10 ids per row and layer,
# the HPD-0 walk's live-cut checks). Same metric as cloud3 REPORT_P2 ("dpk"). Also the Q1 0.5 sure-row cut on top.
import numpy as np, glob, os, itertools
D = os.path.expanduser('~/FableGod/cloud3/macpub/data/decision_map_moonshots')
W = []
for f in sorted(glob.glob(os.path.join(D, '*.npz'))):
    z = np.load(f)
    W.append(dict(nm=os.path.basename(f), dom=str(z['domain']), ids=z['ids'].astype(np.int16), checks=z['checks'],
                  dcf=z['dcf']))

def checks(w):
    out = []
    for (A, R, kept), dc in zip(w['checks'], w['dcf']):
        L = R - 1
        if A + R > w['ids'].shape[0]:
            continue
        cf = np.full(R, np.nan, np.float32); cf[:L] = dc[:L]
        if L < 15: cf[L] = dc[L]
        out.append((np.arange(A, A + R), int(kept), cf))
    return out

CH = [checks(w) for w in W]
doms = sorted(set(w['dom'] for w in W))
print('windows per domain:', {d: sum(w['dom'] == d for w in W) for d in doms},
      '| live-cut checks per window: median', int(np.median([len(c) for c in CH])))

def merged(group, q1=None):
    """group = window indices; pass j merges check j of every window. -> (distinct groups, kept, real (row, expert) pairs)"""
    n = min(len(CH[g]) for g in group)
    dist = kept = pairs = 0
    for j in range(n):
        rows_k = []
        for g in group:
            rows, k, cf = CH[g][j]
            ks = np.full(len(rows), 10)
            if q1 is not None:
                ks[np.nan_to_num(cf, nan=-1) >= q1] = 2
            rows_k.append((W[g]['ids'][rows], ks)); kept += k
        for l in range(48):
            s = set()
            for I, ks in rows_k:
                for r in range(len(I)):
                    s.update(I[r, l, :ks[r]].tolist()); pairs += ks[r]
            dist += len(s)
    return dist, kept, pairs

rng = np.random.default_rng(5)
for label, pool in [(d, [i for i, w in enumerate(W) if w['dom'] == d]) for d in doms] + [('mixed', list(range(len(W))))]:
    for q1 in (None, 0.5):
        base = None
        line = []
        for B in (1, 2, 4, 8, 12):
            if B > len(pool):
                continue
            tot = np.zeros(3)
            for rep in range(6 if B > 1 else 1):
                if B == 1:
                    groups = [[i] for i in pool]
                else:
                    p = list(rng.permutation(pool)); groups = [p[i:i + B] for i in range(0, len(p) - B + 1, B)]
                for g in groups:
                    tot += merged(g, q1)
            dpk = tot[0] / tot[1]; rows_per_group = tot[2] / tot[0]
            base = base or dpk
            line.append(f"B{B} {dpk:6.1f} (x{dpk / base:.2f}, {rows_per_group:.1f} rows/group)")
        print(f"{label:7s} {'exact  ' if q1 is None else 'Q1 0.5 '} groups per kept token: " + ' | '.join(line), flush=True)
