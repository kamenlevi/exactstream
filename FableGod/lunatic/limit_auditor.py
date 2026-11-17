"""BREAK98b limits, screen 2 (Lunatic, Tue 17 Nov 08:4x). CPU only, read-only.

(c) "a drafter with more knowledge": how good is a bigger, independent model at the ONE thing the floor says the chain
lacks, the 125B's decision at the chain's first misses? MoonShots' per-row picks of the 2B and 4B (rows{2b,4b}_<set>.npz:
the model's greedy top-1 and margin at every answer row of the TRUE text, the 125B's argmax tk0 beside it) are exactly
what a batched audit pass over the draft sees up to the first miss (the draft equals the true text there).
  decider: override the head's draft at a row when the auditor's top-1 differs from it with margin >= tau;
  phi = the override is right at the chain's first miss; psi = an override at a right row (ends the block).
  FIX-r walk (no re-anchor; exact on the teacher-forced cells), priced at the bound like floor_bar: a round = the
  auditor's pass c ms + the head's re-draft 3.0 ms per 31 levels. c: 2B 1.06 GB -> 3.8 ms, 4B ~2.1 GB -> 7.6 ms [P].
Join: topk cell (fi, row) <-> rows file (fname, row); checked by tk0 == the cell's y.
usage: ~/BigMoE/.venv/bin/python3 -I limit_auditor.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"; MS = f"{H}/MoonShots/week98"
OUT = f"{H}/FableGod/lunatic/limit_auditor.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
CHECK = 7.6 + 16.4 + 1.2 + 1.5 + 3.0
COST = {"2b": 3.8, "4b": 7.6}

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
    # first misses and right rows of the plain chain walk
    miss, pre = [], []
    for C in WIN:
        n = C.shape[0]; p = 1
        while p < n:
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: pre.append(C[p, a]); a += 1
            if a < 31 and C[p, a] >= 0: miss.append(C[p, a])
            p += a + 1
    miss = np.array(miss); pre = np.array(pre)
    say(f"== {tag}: plain chain-32 walk, {len(miss)} first misses, {len(pre)} right rows before them")
    for aud in ("2b", "4b"):
        fpath = f"{MS}/rows{aud}_{nm}.npz"
        if not os.path.exists(fpath):
            say(f"   {aud}: no rows file"); continue
        R = np.load(fpath)
        key = {(os.path.basename(str(n)), int(r)): (int(t), float(mg), int(k0)) for n, r, t, mg, k0 in
               zip(R["fname"], R["row"], R["top1"], R["margin2b"], R["tk0"])}
        top1 = np.full(nc, -1, np.int64); mg = np.full(nc, np.nan); k0 = np.full(nc, -1, np.int64)
        for c in range(nc):
            v = key.get((files[fi[c]], int(row[c])))
            if v is not None: top1[c], mg[c], k0[c] = v
        have = top1 >= 0
        if have.sum() == 0:
            say(f"   {aud}: no rows joined"); continue
        say(f"   {aud}: joined {100*have.mean():.1f} % of cells; tk0 == cell y on {100*(k0[have] == y[have]).mean():.2f} % of them; "
            f"auditor top-1 == y on all joined cells {100*(top1[have] == y[have]).mean():.1f} %")
        dr = tid[:, 0]
        hm = have[miss]; hp = have[pre]
        say(f"      at first misses (joined {100*hm.mean():.0f} %): auditor right {100*(top1[miss][hm] == y[miss][hm]).mean():5.1f} % | "
            f"at right rows: auditor disagrees with the (right) draft {100*(top1[pre][hp] != dr[pre][hp]).mean():5.2f} %")
        for tau in (0.0, 1.0, 2.0, 4.0):
            ov = have & (top1 != dr) & (np.nan_to_num(mg, nan=-1) >= tau); ok = top1 == y
            phi = (ov[miss] & ok[miss]).mean(); psi = ov[pre].mean()
            res = []
            for r in (1, 2):
                tok = 0; nchk = 0; ms = 0.0
                for C in WIN:
                    n = C.shape[0]; p = 1
                    while p < n:
                        k = 0; acc = 0; used = 0; extra = 0.0
                        while acc < 31 and k < 31 and C[p, k] >= 0:
                            c = C[p, k]
                            if used < r and ov[c]:
                                used += 1; extra += COST[aud] + 3.0 * (31 - acc) / 31
                                if not ok[c]: break
                                acc += 1; k += 1; continue
                            if right[c] == 1: acc += 1; k += 1; continue
                            break
                        tok += acc + 1; nchk += 1; ms += CHECK + extra; p += acc + 1
                res.append(f"r{r} kept {tok/nchk:5.2f} ({1000*tok/ms:3.0f} tok/s)")
            say(f"      tau {tau:3.1f}: phi {100*phi:5.1f} %  psi {100*psi:5.2f} %  | " + " | ".join(res))
    base_tok = 0; base_n = 0
    for C in WIN:
        n = C.shape[0]; p = 1
        while p < n:
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
            base_tok += a + 1; base_n += 1; p += a + 1
    say(f"   plain chain-32: kept {base_tok/base_n:5.2f} ({1000*base_tok/(base_n*CHECK):3.0f} tok/s at the bound)")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
