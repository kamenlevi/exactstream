"""R-32 LFUT (MoonShots Tue 20:3x; exact, CPU replay): tune the engine's LFU at 24 GB. (1) decay d per check (today 0.99),
(2) score by the number of ROWS that use an expert in the check instead of presence. wall24 traces, 16-row checks accepting
10 and 4-row checks accepting 3, tokens 400+, exact stack C_m 159 / C_l 44, W24-ec + 11-bit miss sizes.
PASS (written first): >= -5 % bytes / token vs today's LFU (d 0.99, presence) at both check shapes; KILL < -2 %.
usage: [BigMoE venv] taskpolicy -b python lfut24.py -> stdout + data/lfut24.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

MS, LS = 1900544, 720896


def needs_rows(ids, R, A):                       # like wall24.needs but keeps per-expert row counts
    n, out, pos = ids.shape[0], [], 0
    while pos + R <= n:
        lay = []
        for l in range(W.NL):
            cnt, best = {}, {}
            for r in range(R):
                for k, e in enumerate(ids[pos + r, l]):
                    e = int(e)
                    cnt[e] = cnt.get(e, 0) + 1
                    if e not in best or k < best[e]:
                        best[e] = k
            lay.append((cnt, {e: c for e, c in cnt.items() if best[e] < 2}))
        out.append((W.MEAS[0] <= pos < W.MEAS[1], lay))
        pos += A
    return out


def run(seq, C, d, rows):
    miss, R, score, w = [], set(), {}, 1.0
    for cnt in seq:
        w /= d
        for e, c in cnt.items():
            score[e] = score.get(e, 0.0) + w * (c if rows else 1)
        N = cnt.keys()
        m = [e for e in N if e not in R]
        miss.append(len(m))
        for e in m:
            if len(R) >= C:
                vic = [v for v in R if v not in cnt]
                if vic:
                    R.discard(min(vic, key=lambda v: score[v]))
            R.add(e)
        if len(R) > C:
            R = set(sorted(R, key=lambda v: -score[v])[:C])
    return miss


lines = []
for RR, AA in ((16, 10), (4, 3)):
    ch = {k: needs_rows(ids, RR, AA) for k, ids in W.IDS.items()}
    res = {}
    for d in (0.95, 0.98, 0.99, 0.995, 0.999):
        for rows in (False, True):
            per = []
            for k, c in ch.items():
                meas = [m for m, _ in c]
                tot = 0
                for l in range(W.NL):
                    mm = run([lay[l][0] for _, lay in c], 159, d, rows)
                    ml = run([lay[l][1] for _, lay in c], 44, d, rows)
                    tot += sum(a * MS + b * LS for a, b, f in zip(mm, ml, meas) if f)
                per.append(tot / (AA * sum(meas)))
            res[(d, rows)] = float(np.mean(per))
    base = res[(0.99, False)]
    for (d, rows), v in res.items():
        s = f"{RR}/{AA} checks, d {d}, {'row-count' if rows else 'presence'} scores: {v / 1e6:.1f} MB / token ({100 * (v / base - 1):+.1f} %)"
        print(s, flush=True); lines.append(s)
open(os.path.expanduser("~/MoonShots/data/lfut24.txt"), "w").write("\n".join(lines) + "\n")
