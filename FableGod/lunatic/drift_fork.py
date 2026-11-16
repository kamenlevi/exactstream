"""DRIFT FORK CONTROL (Lunatic, THINK_1116c, Mon 16 Nov 18:4x). CPU only, read-only. drift_bits.py found own-state
accuracy ~75 % after the chain passed a doubtful step (head p1 < 0.5) vs ~96 % without. Is that drift or difficulty?
Same split, now with the FRESH (true-state) guess on the same targets: if fresh also drops, doubt marks hard code;
if fresh holds, the carried state is damaged at forks. Also: the gap by how many doubtful steps were passed, and how
long ago (steps since the last doubtful step).
usage: ~/BigMoE/.venv/bin/python3 -I drift_fork.py"""
import os, sys
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/drift_fork.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
for tag in ("heplus24", "mbppbank"):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int); right = tz["right"].astype(bool); p1 = tz["top_p"][:, 0].astype(np.float64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    tab = {}      # key -> [n, fresh right, own right]
    def add(k, fr, ow):
        v = tab.setdefault(k, [0, 0, 0]); v[0] += 1; v[1] += int(fr); v[2] += int(ow)
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        R = np.full((n0, DEP), -1, np.int8); P1 = np.full((n0, DEP), np.nan)
        for j in range(DEP):
            m = dw == j; c = int(m.sum()); R[:c, j] = right[s:t][m]; P1[:c, j] = p1[s:t][m]
        for d in range(2, DEP):
            for i in range(d, n0):
                fr, ow = R[i, 0], R[i - d, d]
                if fr < 0 or ow < 0: continue
                pr = P1[i - d, :d]
                if not np.isfinite(pr).all(): continue
                dbt = np.flatnonzero(pr < 0.5)
                band = "2-6" if d <= 6 else "7-14" if d <= 14 else "15-30"
                cls = "none" if len(dbt) == 0 else ("1" if len(dbt) == 1 else "2+")
                add(("n", band, cls), fr, ow)
                if len(dbt):
                    ago = d - 1 - int(dbt[-1])          # steps between the last doubtful step and this guess
                    g = "0-1" if ago <= 1 else "2-4" if ago <= 4 else "5-9" if ago <= 9 else "10+"
                    add(("ago", band, g), fr, ow)
                # the target itself is doubtful for the FRESH head? (difficulty marker at the target)
                add(("tgt", band, "fresh p1<0.5" if P1[i, 0] < 0.5 else "fresh p1>=0.5"), fr, ow)
    say(f"== {tag}: accuracy fresh (true state) vs own on the same targets, by doubtful steps passed on the own chain")
    for kind, keys in (("n", ("none", "1", "2+")), ("ago", ("0-1", "2-4", "5-9", "10+")), ("tgt", ("fresh p1>=0.5", "fresh p1<0.5"))):
        for band in ("2-6", "7-14", "15-30"):
            row = []
            for k in keys:
                v = tab.get((kind, band, k))
                if v: row.append(f"{k}: n {v[0]:6d} fresh {100*v[1]/v[0]:5.1f} own {100*v[2]/v[0]:5.1f} gap {100*(v[1]-v[2])/v[0]:4.1f}")
            label = {"n": "doubtful steps passed", "ago": "steps since last doubt", "tgt": "target hard for fresh"}[kind]
            say(f"   {label:22s} depth {band:5s} | " + " | ".join(row))
open(OUT, "w").write("\n".join(lines) + "\n")
