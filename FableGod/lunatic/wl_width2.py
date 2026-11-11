# WALL_L test 2b: width x the perfect stop (only the kept rows cost experts: the bound of any stop rule) on chat and code.
import numpy as np, runpy, sys
sys.argv = ['x']
g = runpy.run_path('wl_width.py', run_name='lib')           # reuses W, CH, merged() definitions (prints test 2 again)
W, CH = g['W'], g['CH']
def merged_stop(group):
    n = min(len(CH[x]) for x in group); dist = kept = 0
    for j in range(n):
        rk = []
        for x in group:
            rows, k, cf = CH[x][j]; rk.append(W[x]['ids'][rows[:k]]); kept += k
        for l in range(48):
            s = set()
            for I in rk: s.update(I[:, l, :].ravel().tolist())
            dist += len(s)
    return dist, kept
rng = np.random.default_rng(5)
print('\n== width x perfect stop (kept rows only), exact top-10')
for d in ('code', 'ifeval', 'multi'):
    pool = [i for i, w in enumerate(W) if w['dom'] == d]; base = None; line = []
    for B in (1, 2, 4, 8, 12):
        tot = np.zeros(2)
        for rep in range(6 if B > 1 else 1):
            p = list(rng.permutation(pool)) if B > 1 else pool
            for i in range(0, len(p) - B + 1, B): tot += merged_stop(p[i:i + B])
        dpk = tot[0] / tot[1]; base = base or dpk
        line.append(f"B{B} {dpk:6.1f}")
    print(f"{d:7s} " + ' | '.join(line), flush=True)
