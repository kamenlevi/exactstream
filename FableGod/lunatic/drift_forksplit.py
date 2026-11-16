"""DRIFT FORK-SPLIT CHECK (Lunatic, THINK_1116c, Mon 16 Nov 19:0x). CPU only, read-only. Fork damage says the head
needs a TRUE state right after its first doubtful step. The only cheap true state is a 125B row, so: split a check at the
first fork. Wave 1 = rows up to and including the fork (depth k, the first own-chain step with head p1 < 0.5); if
the fork guess is accepted, wave 2 re-drafts from the fork's true state (the own chain of anchor a + k + 1) for the
remaining rows of the 32. Chain walk, exact acceptance; priced at Leap's C6 + int8 bound like lim800_wide (dense
max(7.6, 4.9 R/32), experts by the code union law, cores 1.2 R/32, overhead 1.5 per wave, draft 3 ms per 31 levels).
Variants: split only when the fork is at depth <= kmax.
usage: ~/BigMoE/.venv/bin/python3 -I drift_forksplit.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/drift_forksplit.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
F = {1: 480, 2: 406, 3: 365, 4: 337, 6: 300, 8: 276, 12: 241, 16: 218, 24: 185, 32: 163}
def f_of(R):
    ks = sorted(F)
    if R in F: return F[R]
    lo = max(k for k in ks if k <= R); hi = min(k for k in ks if k >= R)
    return F[lo] + (F[hi] - F[lo]) * (np.log(R) - np.log(lo)) / (np.log(hi) - np.log(lo))
def wave_ms(r):
    return max(7.6, 4.9 * r / 32) + 16.4 * (r * f_of(r)) / (32 * f_of(32)) + 1.2 * r / 32 + 1.5 + 3.0 * r / 31
for tag in ("heplus24", "mbppbank"):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int); right = tz["right"].astype(int); p1 = tz["top_p"][:, 0].astype(np.float64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    say(f"== {tag}")
    for kmax in (-1, 4, 8, 12, 30):
        tok = 0; ms = 0.0; nchk = 0; nsplit = 0
        for s, t in zip(st, en):
            dw = dep[s:t]; n0 = int((dw == 0).sum())
            R = np.full((n0 + 1, DEP), -1, np.int8); P1 = np.full((n0 + 1, DEP), np.nan)
            for j in range(DEP):
                m = dw == j; c = int(m.sum()); R[:c, j] = right[s:t][m]; P1[:c, j] = p1[s:t][m]
            p = 1
            while p < R.shape[0]:
                row = R[p]; pr = P1[p]
                fk = np.flatnonzero(np.isfinite(pr) & (pr < 0.5))
                k = int(fk[0]) if len(fk) else 99
                acc = 0
                while acc < 31 and row[acc] == 1: acc += 1
                if kmax < 0 or k > kmax or k >= 30:
                    tok += acc + 1; ms += wave_ms(32); nchk += 1; p += acc + 1; continue
                if acc <= k:                               # the check ends at or before the fork: wave 1 only
                    tok += acc + 1; ms += wave_ms(k + 1); nchk += 1; p += acc + 1; continue
                q = p + k + 1                              # fork accepted: wave 2 from the fork's true state
                nsplit += 1
                if q >= R.shape[0]:
                    tok += k + 2; ms += wave_ms(k + 1); nchk += 1; p = q + 1; continue
                r2 = R[q]; cap = 31 - (k + 1); a2 = 0
                while a2 < cap and r2[a2] == 1: a2 += 1
                tok += (k + 1) + a2 + 1; ms += wave_ms(k + 1) + wave_ms(31 - k); nchk += 1; p = q + a2 + 1
        lab = "plain chain-32" if kmax < 0 else f"split at first fork, depth <= {kmax}"
        say(f"   {lab:34s} kept/check {tok/nchk:6.2f}  splits {nsplit:5d}/{nchk:5d}  tok/s at the bound {1000*tok/ms:6.0f}")
open(OUT, "w").write("\n".join(lines) + "\n")
