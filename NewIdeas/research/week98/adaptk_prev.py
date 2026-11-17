#!/usr/bin/env python3
"""ADAPTIVE K + the previous check's outcome (NewIdeas, Tue 17 Nov; label from date in the output).
Does "the previous check was fully accepted" add signal beyond path_p? Rule: 32 rows if path_p(pop 16) >= tau, where
tau = tau_full after a fully accepted check and tau_part otherwise. KERN2, E14 costs, L 0.66; grid tuned in-sample, read
CROSS-SET (each set at the other set's best pair). The baseline is the frozen single-tau rule (tau 0.9 and the in-set best).
"""
import os, sys, time, itertools
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tree_audit as TA
import adaptk_draft as AD
A16, OV, R32, L = 62.49, 4.0, 1.4372, 0.66
T = {16: 1.025 * A16 + OV, 32: R32 * 1.04 * A16 + OV}
TAUS = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
print(f"run {time.strftime('%a %H:%M')}")
data = {}
for s, (tag, _rt, pub32, _p16) in TA.SETS.items():
    W = TA.load_windows(tag, pub32)
    z = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)
    idx, PP, A, Hh, mono = AD.dense({k: z[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p")})
    data[s] = (AD.steps(W, idx), A, Hh, PP[:, 15].copy())
def run(s, tf, tp):
    S, A, Hh, p16 = data[s]; kept = tm = 0.0; n32 = nchk = 0
    for ids in S:
        n = len(ids); p = 1; full = False
        while p < n:
            i = ids[p]
            if i < 0:
                r, a = 16, 0
            else:
                r = 32 if p16[i] >= (tf if full else tp) else 16; a = int(A[r][i])
            kept += a + 1; tm += T[r] + (L * float(Hh[r][i]) if i >= 0 else 0.0); nchk += 1; n32 += r == 32
            full = (a == r - 1); p += a + 1
    return 1000 * kept / tm, n32 / nchk
res = {}
for s in data:
    base = run(s, 2.0, 2.0)[0]
    single = {t: run(s, t, t)[0] for t in TAUS}
    grid = {(tf, tp): run(s, tf, tp)[0] for tf, tp in itertools.product(TAUS, TAUS)}
    bs = max(single, key=single.get); bg = max(grid, key=grid.get)
    res[s] = (base, single, grid, bs, bg)
    print(f"{s}: fixed16 {base:.1f}; single tau best {bs} {single[bs]:.1f} ({100 * (single[bs] / base - 1):+.1f} %), frozen 0.9 "
          f"{single[0.9]:.1f} ({100 * (single[0.9] / base - 1):+.1f} %); two-tau best (full {bg[0]}, part {bg[1]}) {grid[bg]:.1f} "
          f"({100 * (grid[bg] / base - 1):+.1f} %)")
for s in data:
    o = [x for x in data if x != s][0]
    base, single, grid, bs, bg = res[s]; tf, tp = res[o][4]; ts = res[o][3]
    v, sh = run(s, tf, tp)
    print(f"{s}: CROSS two-tau (from {o}: full {tf}, part {tp}) {v:.1f} ({100 * (v / base - 1):+.1f} %, 32-row checks {100 * sh:.0f} %) "
          f"vs CROSS single tau {ts} {single[ts]:.1f} ({100 * (single[ts] / base - 1):+.1f} %)")
