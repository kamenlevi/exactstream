"""SHAPE CHANNELS (Lunatic, BREAK98c, Tue 17 Nov 11:5x). CPU only, read-only, one process, < 1 GB, ~1 min.
Prices CODE || NARRATIVE: the answer decoded as two causal channels in the same checks. The code channel never attends
to the answer's comments / docstrings; each narrative span (a comment or docstring) attends to everything before it and
starts as soon as the code before it is decided. So a wording decision no longer blocks the code, and the spans resolve
their decisions in parallel with the code's. Decision accounting on the t30b cells (chain-32 walks, the cells' own
right flags; lim800_slack's loader and lexer):
  coupled  = today's single walk (checks = sequential 125B sweeps; rows = 32 per check).
  channels = code chain over the code positions (depth mapped to the coupled cells; a chain stops where the coupled
             depth would pass 30: conservative) + one chain per narrative span; sweeps = the critical path
             max(code checks, start_s + span checks); rows per sweep = the code chain's + every active span's.
Approximation: misses are the coupled world's (the head drafted code with comments in context). PLAN TRANSFER, the
risk that a comment carried the code's decision: the code miss rate on the first code line after a comment vs other
code lines (depth-0 and depth 0-15 cells).
usage: ~/BigMoE/.venv/bin/python3 -I shape_channels.py"""
import os
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/shape_channels.out"'))
NARR = (1, 2, 4)

DMAX = 31; MATCHED = False
def chain(rt, i, positions, lim_end=None):
    """walk one check from anchor index i over `positions` (original answer positions); returns (kept, rows, next i).
    MATCHED (code chain only): the m-th drafted code position is judged by the cell at the same depth m that predicts
    it (rt[pos - m, m]) instead of the coupled depth from the anchor (pos - anchor), which over-penalises NL gaps."""
    p0 = positions[i]; acc = 0
    while acc < DMAX and i + acc < len(positions):
        pos = positions[i + acc]
        a, d = (pos - acc, acc) if MATCHED else (p0, pos - p0)
        if a < 0 or d > 30 or rt[a, d] < 0: break
        if rt[a, d] != 1: break
        acc += 1
    rows = min(DMAX, len(positions) - i) + 1
    if i + acc >= len(positions): return acc, rows, len(positions)          # span / channel finished, no bonus needed
    return acc + 1, rows, i + acc + 1

def answer(A, rule):
    rt = A["rt"].astype(np.int8).copy(); n0 = A["n0"]
    rt[(rt == 0) & rule(A)] = 1
    S0 = A["S"][:n0, 0].astype(int)
    # coupled: the standard walk from anchor 0
    p = 0; checks = 0
    while p < n0:
        acc = 0
        while acc < DMAX and p + acc < n0 and rt[p, acc] == 1: acc += 1
        checks += 1; p += acc + 1
    # channels
    code = [q for q in range(n0) if S0[q] not in NARR]
    spans = []; q = 0
    while q < n0:
        if S0[q] in NARR:
            s = q
            while q < n0 and S0[q] in NARR: q += 1
            spans.append((s, q))
        else: q += 1
    # code chain: frontier after each check
    i = 0; front = []; crow = []
    global MATCHED
    while i < len(code):
        MATCHED = MATCH_CODE; k, r, i = chain(rt, i, code); MATCHED = False; crow.append(r)
        front.append(code[i - 1] if i > 0 else -1)                              # last decided code position
    Cc = len(front)
    srows = {}; finish = []
    for (s, e) in spans:
        start = next((c + 1 for c in range(Cc) if front[c] >= s - 1), Cc) if s > 0 and S0[s - 1] not in NARR else 0
        # a span starts when the code before it is decided (check index `start`, 0 = before any check)
        if s > 0 and not any(S0[x] not in NARR for x in range(s)): start = 0
        pos = list(range(s, e)); j = 0; t = start
        while j < len(pos):
            k, r, j = chain(rt, j, pos); srows[t] = srows.get(t, 0) + r; t += 1
        finish.append(t)
    sweeps = max([Cc] + finish)
    rows = [(crow[t] if t < Cc else 0) + srows.get(t, 0) for t in range(sweeps)]
    return dict(n=n0, coupled=checks, sweeps=sweeps, Cc=Cc, rows=rows, ncode=len(code), nspans=len(spans))

def plan_transfer(W):
    """code miss rates: first code line after a comment line vs other code lines"""
    out = {"after": [0, 0, 0, 0], "other": [0, 0, 0, 0]}
    for A in W:
        n0 = A["n0"]; S0 = A["S"][:n0, 0].astype(int); y0 = A["Y"][:n0, 0]
        line = np.cumsum([0] + ["\n" in txt(t) for t in y0[:-1]])            # line index of each token
        lines_state = {}
        for q in range(n0):
            lines_state.setdefault(line[q], set()).add(S0[q])
        is_comment_line = {L: (1 in st_ and not (st_ - {0, 1})) for L, st_ in lines_state.items()}
        for q in range(n0):
            if S0[q] in NARR or not txt(y0[q]).strip(): continue
            L = line[q]; prev = L - 1
            while prev >= 0 and prev not in lines_state: prev -= 1
            key = "after" if prev >= 0 and is_comment_line.get(prev, False) else "other"
            if A["rt"][q, 0] >= 0: out[key][0] += 1; out[key][1] += int(A["rt"][q, 0] == 0)
            for d in range(16):
                if q - d >= 0 and A["rt"][q - d, d] >= 0: out[key][2] += 1; out[key][3] += int(A["rt"][q - d, d] == 0)
    return out

def c_bound(R):  return (25.2 + max(0, R - 32) * 0.25) if R >= 32 else 19.2 + (R - 16) * 0.375                         # C6 + int8 bound, 0.25 ms per extra row
def c_week(R):   return (36 + (R - 16) * 0.75) if R >= 16 else 36.0        # this week: 16 rows ~36, 32 rows ~48

import sys
CONFIGS = [(31, False), (31, True), (15, False), (15, True)]
for tag in ("heplus24", "mbppbank"):
    W = load(tag)
    say(f"== {tag}: {len(W)} answers, {sum(A['n0'] for A in W)} tokens")
    for (DMAX, MATCH_CODE), (name, rule) in [(c, r) for c in CONFIGS for r in (("exact", lambda A: np.zeros(A["rt"].shape, bool)),
                       ("cc1", lambda A: np.nan_to_num(A["DL"], nan=-99) >= -1))]:
        name = f"chain-{DMAX+1} {'matched' if MATCH_CODE else 'strict '} {name}"
        R = [answer(A, rule) for A in W]
        N = sum(r["n"] for r in R); C = sum(r["coupled"] for r in R); Sw = sum(r["sweeps"] for r in R)
        Cc = sum(r["Cc"] for r in R); rows = [x for r in R for x in r["rows"]]
        tb_c = sum(r["coupled"] * c_bound(DMAX + 1) for r in R); tw_c = sum(r["coupled"] * c_week(DMAX + 1) for r in R)
        tb_s = sum(c_bound(x) for x in rows); tw_s = sum(c_week(max(16, x)) for x in rows)
        say(f"   {name:22s} coupled: {C} checks, kept {N/C:5.2f}/check | channels: {Sw} sweeps ({Cc} on the code chain), "
            f"{N/Sw:5.2f} tokens/sweep (x{C/Sw:4.2f} fewer sweeps); rows/sweep mean {np.mean(rows):5.1f}, p90 {np.percentile(rows, 90):4.0f}, max {max(rows)}")
        say(f"         check-only tok/s, coupled vs channels: at the bound {1000*N/tb_c:4.0f} -> {1000*N/tb_s:4.0f} (x{tb_c/tb_s:4.2f}); "
            f"this week's check {1000*N/tw_c:4.0f} -> {1000*N/tw_s:4.0f} (x{tw_c/tw_s:4.2f})")
    pt = plan_transfer(W)
    for k in ("after", "other"):
        a = pt[k]
        say(f"   code tokens on {'the first line after a comment' if k == 'after' else 'other code lines':31s}: n {a[0]:6d}  "
            f"depth-0 miss {100*a[1]/max(1,a[0]):4.1f} %  depth 0-15 miss {100*a[3]/max(1,a[2]):4.1f} %")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
