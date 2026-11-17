"""BREAK98b limits, screen 3 (Lunatic, Tue 17 Nov 08:4x). CPU only, read-only.

Does an independent model's knowledge ADD to the head's own candidates? (1) At the chain's first misses: is y the head's
#2, the auditor's top-1, both, or neither (and in the head's top-8)? (2) AUDITOR SIBLINGS: instead of overriding, the
auditor's disagreeing pick (margin >= tau) hangs as a sibling of the spine, with the head's continuation under it to
the block end (no re-anchor: the teacher-forced cells are exact for it). The check accepts the spine to its first
miss, then the sibling if it is the 125B's token, then its continuation to the next miss (+ bonus). Rows = 32 + the
siblings and their continuations. Price at the bound: 32-row check 25.2 ms + (R - 32) x m (m = 0.25 ms, the exact-row
floor: dense 3.73 G MAC at ~24 TMAC/s int8 est. 0.15 + union 0.06-0.10 + cores 0.04) + overhead 1.5 + auditor pass c +
draft 3.0 ms per 31 levels x (31 + the longest continuation). Best-first trees at equal rows (limit_rows.out) are the
bar; the head's own best-first K-curve is optimal for its own information, so a win here = new information.
usage: ~/BigMoE/.venv/bin/python3 -I limit_auditree.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"; MS = f"{H}/MoonShots/week98"
OUT = f"{H}/FableGod/lunatic/limit_auditree.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31; M = 0.25
COST = {"2b": 3.8, "4b": 7.6}
BF = {"heplus24": {31: 16.26, 47: 18.86, 63: 20.77}, "mbppbank": {31: 9.03, 47: 9.73, 63: 10.19}}   # limit_rows.out

for tag, nm in (("heplus24", "heplus"), ("mbppbank", "mbpp")):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    files = [str(f) for f in tz["files"]]; dep = tz["depth"].astype(int); fi = tz["fi"].astype(int); row = tz["row"].astype(int)
    y = tz["y"].astype(np.int64); right = tz["right"].astype(np.int8); tid = tz["top_ids"].astype(np.int64); nc = len(dep)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    WIN = []
    for s, t_ in zip(st, en):
        dw = dep[s:t_]; n0 = int((dw == 0).sum()); C = np.full((n0 + 1, DEP), -1, np.int64)
        for j in range(DEP):
            idx = s + np.flatnonzero(dw == j); C[:len(idx), j] = idx
        WIN.append(C)
    say(f"== {tag}")
    for aud in ("2b", "4b"):
        R = np.load(f"{MS}/rows{aud}_{nm}.npz")
        key = {(os.path.basename(str(n)), int(r)): (int(t), float(mg)) for n, r, t, mg in zip(R["fname"], R["row"], R["top1"], R["margin2b"])}
        top1 = np.full(nc, -1, np.int64); mg = np.full(nc, -1.0)
        for c in range(nc):
            v = key.get((files[fi[c]], int(row[c])))
            if v is not None: top1[c], mg[c] = v
        miss = []
        for C in WIN:
            n = C.shape[0]; p = 1
            while p < n:
                a = 0
                while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
                if a < 31 and C[p, a] >= 0: miss.append(C[p, a])
                p += a + 1
        miss = np.array(miss)
        h2 = tid[miss, 1] == y[miss]; au = top1[miss] == y[miss]; h8 = (tid[miss] == y[miss, None]).any(1)
        say(f"   {aud} at the chain's {len(miss)} first misses: y = head #2 {100*h2.mean():4.1f} % | y = auditor top-1 {100*au.mean():4.1f} % | "
            f"auditor right where head #2 is not {100*(au & ~h2).mean():4.1f} % | either {100*(au | h2).mean():4.1f} % | y in head top-8 {100*h8.mean():4.1f} %")
        for tau in (1.0, 2.0, 4.0):
            sib = (top1 >= 0) & (top1 != tid[:, 0]) & (mg >= tau); sibok = top1 == y
            tok = 0; nchk = 0; ms = 0.0; rows_sum = 0
            for C in WIN:
                n = C.shape[0]; p = 1
                while p < n:
                    ks = [k for k in range(31) if C[p, k] >= 0 and sib[C[p, k]]]
                    Rw = 32 + sum(31 - k for k in ks)
                    longest = max([30 - k for k in ks], default=0)
                    a = 0
                    while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
                    if a < 31 and C[p, a] >= 0 and sib[C[p, a]] and sibok[C[p, a]]:
                        a += 1
                        while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
                    kept = a + 1
                    tok += kept; nchk += 1; rows_sum += Rw
                    ms += 25.2 + max(0, Rw - 32) * M + 1.5 + COST[aud] + 3.0 * (31 + longest) / 31
                    p += kept
            say(f"   {aud} siblings tau {tau:3.1f}: kept {tok/nchk:5.2f} at {rows_sum/nchk:5.1f} rows/check -> {1000*tok/ms:3.0f} tok/s at the bound "
                f"(best-first own trees: K31 {BF[tag][31]} / K47 {BF[tag][47]} / K63 {BF[tag][63]})")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
