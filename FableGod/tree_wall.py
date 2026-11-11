#!/usr/bin/env python3 -I
"""tree_wall.py (FableGod, mission 28). CPU replay of exact tree policies on MTP Head's D-24 teacher-forced ranks
(~/TrainLab/data/d24/{code,gsm,mgsm,ifeval,multi}_b5.npz: per answer rk[pos, depth] = the rank of the 125B's token in
the native MTP head's (b5) candidates at depth 1..15 along the true prefix, p1[pos, depth] = the head's top-1 probability).
A tree covers the true path at depth d if the true token is among the candidates the tree offers there: the chain's
top-1 (rank 0), or the siblings at a fork (rank <= k - 1). A rescued sibling's continuation is drafted from the true
token, so its ranks are the teacher-forced ranks at the next depths. Rows after the first uncovered depth are wasted
but paid. Policies:
  chain K15, cut t 0.3 (L1-14's running-confidence cut)                      the baseline (D-24's numbers reproduced)
  leaves: #2 (+#3) leaf rows at every drafted depth with p1 < tau (D-24's)    reproduced
  LEAF-HARD: #2 and #3 leaf rows at the FIRST depth with p1 < tau only       design 1 (fused leaves: no fork cost)
  FORK-HARD: at the first depth with p1 < tau, sibling #2 with a continuation of m rows + a #3 leaf   design 2
  WIDE-1: top-3 at depth 1 with continuations of m rows each, then FORK-HARD along the survivor      design 3 (chat)
  ORACLE-FORK: the fork placed at the chain's first miss (placement upper bound)
Costs per check at the floor engine [P/G]: dense constant 17 ms (the cloud's line), 1.0 ms per row up to 16 (experts at
the mission-27 stack), 1.6 ms per row beyond 16 (experts + dense tiles + recurrence / attention rows), 0.3 ms per fork
(the fused tree scan: the branch runs on a register copy of the parent's state inside the verify kernel, zero state bytes; the 12 attention calls remain; the boundary-capture form at 1.2 ms is a second column), 0.3 ms per leaf position (the attention
calls), 2.2 ms per drafted step (D-23's MTP step; branches draft in one batch, so steps = the deepest path). Today's
model for reference (D-23's at C 224): F per type + 7.82 ms per row + 2.2 per step."""
import numpy as np, os, time
t0 = time.time()
D = os.path.expanduser("~/TrainLab/data/d24")
SETS = ["code", "gsm", "mgsm", "ifeval", "multi"]
F_TODAY = dict(code=84.1, gsm=66.1, mgsm=66.1, ifeval=50.1, multi=50.1)
K = 15

def load(s):
    z = np.load(f"{D}/{s}_b5.npz", allow_pickle=True)
    n = len(z["keys"])
    return [(z[f"rk_{i}"].astype(np.int32), z[f"p1_{i}"].astype(np.float32)) for i in range(n)]

def cut_depth(p, t, mode):
    """drafted rows of the chain: 'prod' = stop when the running product of p1 < t; 'step' = stop at the first p1 < t"""
    prod = 1.0
    for j in range(K):
        prod = prod * p[j] if mode == "prod" else p[j]
        if prod < t:
            return max(j, 1)
    return K

def chain_run(rk, start, n):
    """leading rank-0 depths among rk[start:start+n]"""
    r = 0
    for j in range(start, min(start + n, K)):
        if rk[j] == 0: r += 1
        else: break
    return r

def replay(rk, p, pol, t=0.3, mode="prod", tau=0.8, m=4, B=16, leaf3=True, k1=3, m1=4):
    """returns tokens, rows, forks, leaf positions, drafted steps for one block start"""
    d = cut_depth(p, t, mode)
    if pol == "chain":
        d = min(d, B - 1); r = chain_run(rk, 0, d)
        return r + 1, d + 1, 0, 0, d
    if pol == "leaves":                                  # D-24's: leaves at every drafted depth with p1 < tau
        d = min(d, B - 1); rows = d + 1; tok = None; leafpos = 0
        for j in range(d):
            nleaf = m if p[j] < tau else 0                # here m = leaves per position (1 or 2)
            rows += nleaf; leafpos += nleaf > 0
            if rk[j] == 0: continue
            tok = j + 1 + (1 if rk[j] <= nleaf else 0); break
        if tok is None: tok = d + 1
        return tok, rows, 0, leafpos, d
    if pol in ("leafhard", "forkhard", "oracle"):
        if pol == "oracle":
            f = next((j for j in range(K) if rk[j] != 0), None)
        else:
            f = next((j for j in range(K) if p[j] < tau), None)
        d = min(d, B - 1)
        if f is None or f >= d:                           # no fork inside the drafted chain
            r = chain_run(rk, 0, d); return r + 1, d + 1, 0, 0, d
        br = (2 if pol == "leafhard" else 1 + m + (1 if leaf3 else 0))
        dch = min(d, B - 1 - br)                          # the chain yields rows to the branch inside the budget
        if dch <= f:                                      # the fork does not fit: plain chain
            r = chain_run(rk, 0, d); return r + 1, d + 1, 0, 0, d
        mm = 0 if pol == "leafhard" else m
        rows = dch + 1 + br; forks = 0 if pol == "leafhard" else 1; leafpos = 1 if pol == "leafhard" else 0
        steps = max(dch, f + 1 + mm)
        r = chain_run(rk, 0, dch)
        if r > f:                                         # the chain was right at the fork: branch wasted
            return r + 1, rows, forks, leafpos, steps
        if r < f:                                         # the chain died before the fork
            return r + 1, rows, forks, leafpos, steps
        # r == f: the chain is wrong at the fork depth
        if rk[f] == 1:                                    # sibling #2 right
            return f + 1 + chain_run(rk, f + 1, mm) + 1, rows, forks, leafpos, steps
        if rk[f] == 2 and (pol == "leafhard" or leaf3):   # #3 leaf right
            return f + 1 + 1, rows, forks, leafpos, steps
        return f + 1, rows, forks, leafpos, steps
    if pol == "wide1":                                    # top-k1 at depth 1 with continuations m1 each, then forkhard on the survivor
        rows = 1 + k1 * (1 + m1); forks = k1 - 1
        if rows > B: return replay(rk, p, "forkhard", t, mode, tau, m, B, leaf3)
        steps = 1 + m1
        if rk[0] >= k1:                                   # not covered at depth 1
            return 1, rows, forks, 0, steps
        # covered: the survivor continues with m1 teacher-forced rows (for the chain, the drafted chain is the survivor)
        rem = B - rows
        r = 1 + chain_run(rk, 1, m1)
        if r < 1 + m1 or rem < 3:                         # died inside the continuation, or no budget for a second fork
            return r + 1, rows, forks, 0, steps
        # second level: the survivor's chain continues up to the cut; forkhard at its first hard depth
        d2 = min(cut_depth(p, t, mode), B - 1)
        if d2 <= 1 + m1:
            return r + 1, rows, forks, 0, steps
        f = next((j for j in range(1 + m1, d2) if p[j] < tau), None)
        br = 1 + m + (1 if leaf3 else 0)
        dch = min(d2, B - br)
        if f is None or f >= dch or dch - (1 + m1) <= 0:
            dch = min(d2, B); rr = chain_run(rk, 1 + m1, dch - (1 + m1))
            return r + rr + 1, dch, forks, 0, max(steps, dch)
        rows2 = dch + br; steps2 = max(dch, f + 1 + m); forks += 1
        rr = chain_run(rk, 1 + m1, dch - (1 + m1)); rp = 1 + m1 + rr
        if rp > f: return rp + 1, rows2, forks, 0, steps2
        if rp < f: return rp + 1, rows2, forks, 0, steps2
        if rk[f] == 1: return f + 1 + chain_run(rk, f + 1, m) + 1, rows2, forks, 0, steps2
        if rk[f] == 2 and leaf3: return f + 2, rows2, forks, 0, steps2
        return f + 1, rows2, forks, 0, steps2
    raise ValueError(pol)

def score(set_, data, **kw):
    """sequential passes per answer: the next block starts where the last pass ended (D-24's tokens per pass)"""
    tk = rows = fk = lp = st = 0.0; n = 0
    for rk_a, p_a in data:
        pos = 0; N = rk_a.shape[0]
        while pos < N:
            a, b, c, d, e = replay(rk_a[pos], p_a[pos], **kw)
            tk += a; rows += b; fk += c; lp += d; st += e; n += 1
            pos += max(a, 1)
    tk, rows, fk, lp, st = tk / n, rows / n, fk / n, lp / n, st / n
    floor = 17 + 1.0 * min(rows, 16) + 1.6 * max(rows - 16, 0) + 0.3 * fk + 0.3 * lp + 2.2 * st     # fused tree scan: a fork costs its 12 attention calls only
    floor1 = 17 + 1.0 * min(rows, 16) + 1.6 * max(rows - 16, 0) + 0.3 * fk + 0.3 * lp + 1.0 * st
    floor_bc = 17 + 1.0 * min(rows, 16) + 1.6 * max(rows - 16, 0) + 1.2 * fk + 0.3 * lp + 2.2 * st  # boundary-capture fork (226 MB of state per fork)
    today = F_TODAY[set_] + 7.82 * rows + 2.2 * st
    floor0 = 17 + 2.2 * st                                   # rows and forks free: the bound
    return dict(tok=tk, rows=rows, forks=fk, leafpos=lp, steps=st, floor=floor, floor1=floor1, today=today, tps=1e3 * tk / floor, tps1=1e3 * tk / floor1, tps0=1e3 * tk / floor0, tps_bc=1e3 * tk / floor_bc, tps_today=1e3 * tk / today)

print("## 0. data")
DATA = {s: load(s) for s in SETS}
for s in SETS:
    rk = np.concatenate([a for a, _ in DATA[s]]); vals, cnt = np.unique(rk, return_counts=True)
    print(f"  {s}: {len(DATA[s])} answers, {rk.shape[0]} block starts x 15 depths; rank values {dict(zip(vals.tolist(), cnt.tolist()))}")

print("\n## 1. Reproducing D-24's table (tokens per pass / rows): chain K15, cut K15 t 0.3 by the two cut rules, cut + 2 leaves at p < 0.8")
print("  | set | chain K15 (D-24) | cut prod (D-24 cut K15) | cut step | cut + 2 leaf p<0.8 (D-24 K15: ifeval 2.56/4.59, multi 2.75/4.99, mgsm 5.28/8.52, gsm 7.30/11.43, code 9.68/12.64) |")
D24 = dict(code=(10.16, 15.67, 9.24, 10.72), gsm=(7.70, 15.76, 6.91, 8.92), mgsm=(5.74, 15.91, 4.94, 6.40), ifeval=(2.74, 15.63, 2.34, 2.97), multi=(2.91, 15.60, 2.54, 3.25))
for s in SETS:
    c = score(s, DATA[s], pol="chain", t=0.0); cp = score(s, DATA[s], pol="chain", t=0.3, mode="prod"); cs = score(s, DATA[s], pol="chain", t=0.3, mode="step")
    l2 = score(s, DATA[s], pol="leaves", t=0.3, mode="prod", tau=0.8, m=2)
    print(f"  | {s} | {c['tok']:.2f} / {c['rows']:.2f} (D-24 {D24[s][0]} / {D24[s][1]}) | {cp['tok']:.2f} / {cp['rows']:.2f} (D-24 {D24[s][2]} / {D24[s][3]}) | {cs['tok']:.2f} / {cs['rows']:.2f} | {l2['tok']:.2f} / {l2['rows']:.2f} |")

print("\n## 2. Hard-position detection by the head's own p1 (per drafted depth along the true path): P(miss | p1 < tau), share of misses at p1 < tau, share of depths flagged")
for s in SETS:
    rk = np.concatenate([a for a, _ in DATA[s]]); p = np.concatenate([b for _, b in DATA[s]])
    miss = rk != 0
    line = f"  {s}: misses {100 * miss.mean():.1f} % of depths; at misses rank 1 {100 * (rk == 1).sum() / miss.sum():.0f} %, rank 2 {100 * (rk == 2).sum() / miss.sum():.0f} %; "
    for tau in (0.5, 0.8, 0.9, 0.95):
        fl = p < tau
        line += f"tau {tau}: flagged {100 * fl.mean():.0f} %, P(miss | flag) {100 * miss[fl].mean():.0f} %, misses caught {100 * (miss & fl).sum() / miss.sum():.0f} %; "
    print(line)

print("\n## 2b. What a rescue is worth: at the chain's first miss (sequential passes), the share rescued by #2 / #3 and the teacher-forced run that follows a rescued token (the continuation a branch could keep), by set")
for s in SETS:
    nmiss = resc = run_sum = run_n = 0; runs = []
    for rk_a, p_a in DATA[s]:
        pos = 0; N = rk_a.shape[0]
        while pos < N:
            rk = rk_a[pos]; d = min(cut_depth(p_a[pos], 0.3, "prod"), K)
            r = chain_run(rk, 0, d)
            if r < d:
                nmiss += 1
                if rk[r] in (1, 2):
                    resc += 1; rr = chain_run(rk, r + 1, 6); runs.append(rr)
            pos += r + 1
    runs = np.array(runs) if runs else np.zeros(1)
    print(f"  {s}: passes ending in a miss inside the cut chain {nmiss}; rescued by #2 or #3 {100 * resc / max(nmiss, 1):.0f} %; the run after a rescued token (next 6 depths, teacher-forced): mean {runs.mean():.2f} tokens, 0 in {100 * (runs == 0).mean():.0f} %, >= 3 in {100 * (runs >= 3).mean():.0f} %")

print("\n## 3. Policies at the floor engine (tokens per pass, rows, forks, drafted steps; ms per check; tok/s; gain vs the cut chain; halved gain)")
hdr = "  | set | policy | tokens | rows | forks | steps | floor ms | tok/s | gain | halved | tok/s at 1.0 ms per draft step (gain) | gain with rows and forks free (bound) | gain with a boundary-capture fork (1.2 ms) | today's model tok/s (gain) |"
for s in SETS:
    base = score(s, DATA[s], pol="chain", t=0.3, mode="prod")
    print(f"\n  {s}: cut chain K15 t 0.3 = {base['tok']:.2f} tokens, {base['rows']:.2f} rows, {base['floor']:.1f} ms, {base['tps']:.0f} tok/s at the floor engine ({base['tps_today']:.1f} on today's model)")
    print(hdr); print("  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pols = [("chain K15 no cut", dict(pol="chain", t=0.0))]
    for tau in (0.8, 0.9, 0.95):
        pols.append((f"LEAF-HARD #2+#3 at first p1<{tau}", dict(pol="leafhard", t=0.3, tau=tau)))
    for tau in (0.8, 0.9):
        for m in (1, 2, 3, 6):
            pols.append((f"FORK-HARD #2+{m} (+#3) at first p1<{tau}, B16", dict(pol="forkhard", t=0.3, tau=tau, m=m, B=16)))
            pols.append((f"FORK-HARD #2+{m} (+#3) at first p1<{tau}, B32", dict(pol="forkhard", t=0.3, tau=tau, m=m, B=32)))
    pols.append(("ORACLE-FORK #2+6 (+#3) at the first miss, B32", dict(pol="oracle", t=0.3, m=6, B=32)))
    pols.append(("ORACLE-FORK #2+6 (+#3) at the first miss, B16", dict(pol="oracle", t=0.3, m=6, B=16)))
    for m1 in (3, 4):
        for tau in (0.8, 0.9):
            pols.append((f"WIDE-1 top-3 at depth 1 +{m1} each, then FORK-HARD #2+3 at p1<{tau}, B16", dict(pol="wide1", t=0.3, tau=tau, m=3, B=16, k1=3, m1=m1)))
            pols.append((f"WIDE-1 top-3 at depth 1 +{m1} each, then FORK-HARD #2+6 at p1<{tau}, B32", dict(pol="wide1", t=0.3, tau=tau, m=6, B=32, k1=3, m1=m1)))
    for name, kw in pols:
        r = score(s, DATA[s], **kw); g = r["tps"] / base["tps"] - 1
        print(f"  | {s} | {name} | {r['tok']:.2f} | {r['rows']:.1f} | {r['forks']:.2f} | {r['steps']:.1f} | {r['floor']:.1f} | {r['tps']:.0f} | {100 * g:+.1f} % | {100 * g / 2:+.1f} % | {r['tps1']:.0f} ({100 * (r['tps1'] / base['tps1'] - 1):+.1f} %) | rows free {100 * (r['tps0'] / base['tps0'] - 1):+.1f} % | capture fork {100 * (r['tps_bc'] / base['tps_bc'] - 1):+.1f} % | {r['tps_today']:.1f} ({100 * (r['tps_today'] / base['tps_today'] - 1):+.1f} %) |")
print(f"\n({time.time() - t0:.1f} s, CPU only)")
