"""DRIFT BITS (Lunatic, THINK_1116c, Mon 16 Nov 18:4x). CPU only, read-only, one process, < 0.5 GB.
Same target, different depth: for each answer position p, the head's guess from anchor p-1 at depth 0 uses the 125B's
TRUE state (= the LAB true-state head, WEEK98 'LAB line'); the guess from anchor p-1-d at depth d uses its OWN carried
state for d steps. Per depth band:
  - accuracy fresh vs own on the SAME targets, and the information drift destroys: mean log2 p_fresh(y) - log2 p_own(y)
    (bits per position, the head's probability of the 125B's token y; TOPK p_y);
  - DRIFT errors (fresh right, own wrong): where y sits in the own guess (rank 1 = the 2nd choice; in the top-8;
    beyond), and whether a FREE analog exists: an earlier position in the answer whose preceding >= 4 / >= 8 tokens
    match the ones before p (its true 125B state is already stored), and whether that earlier occurrence's next token
    is y (what lookup alone would give);
  - damage after doubt: own accuracy at depth d when the chain passed a step with head p1 < 0.5 vs not.
usage: ~/BigMoE/.venv/bin/python3 -I drift_bits.py"""
import os, sys
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/drift_bits.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
BANDS = ((1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30))
for tag in ("heplus24", "mbppbank"):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int); right = tz["right"].astype(bool); py = tz["p_y"].astype(np.float64)
    yr = tz["y_rank"].astype(int); p1 = tz["top_p"][:, 0].astype(np.float64); yv = tz["y"].astype(np.int64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    acc = {b: [[], []] for b in BANDS}; bits = {b: [] for b in BANDS}
    drift = {b: {"n": 0, "r1": 0, "r8": 0, "a4": 0, "a8": 0, "a4y": 0, "a8y": 0} for b in BANDS}
    doubt = {b: [[0, 0], [0, 0]] for b in BANDS}
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        R = np.full((n0, DEP), -1, np.int8); PY = np.full((n0, DEP), np.nan); YR = np.full((n0, DEP), 99)
        P1 = np.full((n0, DEP), np.nan)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            R[:c, j] = right[s:t][m]; PY[:c, j] = py[s:t][m]; YR[:c, j] = yr[s:t][m]; P1[:c, j] = p1[s:t][m]
        y0 = yv[s:t][dw == 0]                      # y0[i] = the token predicted from anchor i at depth 0 (position i+1)
        # analog tables: for each position index i (token y0[i]), does the 4 / 8 tokens before it occur earlier?
        def analog(i, k):
            if i < k: return None
            ctx = tuple(y0[i - k:i])
            for q in range(i - 1, k - 1, -1):
                if tuple(y0[q - k:q]) == ctx: return int(y0[q])
            return None
        for b in BANDS:
            lo, hi = b
            for d in range(lo, hi + 1):
                for i in range(d, n0):             # target = y0[i]; fresh = anchor i depth 0; own = anchor i-d depth d
                    fr, ow = R[i, 0], R[i - d, d]
                    if fr < 0 or ow < 0: continue
                    acc[b][0].append(fr); acc[b][1].append(ow)
                    pf, po = PY[i, 0], PY[i - d, d]
                    if pf > 0 and po > 0: bits[b].append(np.log2(pf) - np.log2(po))
                    if fr == 1 and ow == 0:
                        D = drift[b]; D["n"] += 1
                        D["r1"] += YR[i - d, d] <= 1; D["r8"] += YR[i - d, d] <= 7
                        a4, a8 = analog(i, 4), analog(i, 8)
                        D["a4"] += a4 is not None; D["a8"] += a8 is not None
                        D["a4y"] += a4 == int(y0[i]); D["a8y"] += a8 == int(y0[i])
                    if ow >= 0 and d >= 2:
                        prior = P1[i - d, :d]
                        hit = int(np.nanmin(prior) < 0.5) if np.isfinite(prior).all() else 0
                        doubt[b][hit][0] += 1; doubt[b][hit][1] += int(ow)
    say(f"== {tag} (same targets; fresh = true-state guess, own = carried state)")
    for b in BANDS:
        f, o = np.mean(acc[b][0]), np.mean(acc[b][1]); D = drift[b]; n = max(1, D["n"])
        dn = doubt[b]
        say(f"   depth {b[0]:2d}-{b[1]:2d}: acc fresh {100*f:5.2f} own {100*o:5.2f} gap {100*(f-o):4.2f} pt | bits lost "
            f"{np.mean(bits[b]):.3f}/pos | drift errors {100*D['n']/len(acc[b][0]):4.2f} % of targets: y = 2nd choice "
            f"{100*D['r1']/n:4.1f} %, in top-8 {100*D['r8']/n:4.1f} %; analog >=4 {100*D['a4']/n:4.1f} % (its next = y "
            f"{100*D['a4y']/n:4.1f} %), >=8 {100*D['a8']/n:4.1f} % ({100*D['a8y']/n:4.1f} %) | own acc after a p1<0.5 step "
            f"{100*dn[1][1]/max(1,dn[1][0]):5.2f} (n {dn[1][0]}) vs none {100*dn[0][1]/max(1,dn[0][0]):5.2f}")
open(OUT, "w").write("\n".join(lines) + "\n")
