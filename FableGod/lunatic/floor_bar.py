"""BREAK98 information floor, screen 3 (Lunatic, Tue 17 Nov 08:0x). CPU only, read-only, t30b topk cells.

THE DECIDER BAR. floor_fix.out: 1-3 located 125B decisions per block beat the true state at all 31 rows. floor_lens.out:
the 125B's own mid-layer taps (linear tuned lens) do not supply them before layer ~46. So any future decider (the
widened mid-layer states of tonight's capture, the 2B auditor, a trained probe) is judged by two numbers on the chain's
own rows, and its price per round:
  phi = P(the decider overrides the draft at the chain's first miss AND picks the 125B's token there)
  psi = P(the decider overrides a row where the draft is RIGHT) (a false fix ends the block there)
  c   = ms per round (its own pass over the re-drafted tail), + the head's re-draft 3.0 ms per 31 levels.
Synthetic decider: independent draws per row; r rounds; no re-anchor (the drifted chain continues, teacher-forced
cells: exact). Chain-32, walk-strict, priced at Leap's C6 + int8 bound (check 29.7 ms incl. draft 3 + overhead 1.5).
Reference points at the same bound (IDEAS_1116c): own chain-32 462 / 232 tok/s, tree-32 566 / 321 (HE+ / MBPP).
usage: ~/BigMoE/.venv/bin/python3 -I floor_bar.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/floor_bar.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
DEP = 31
F = {1: 480, 2: 406, 3: 365, 4: 337, 6: 300, 8: 276, 12: 241, 16: 218, 24: 185, 32: 163}
def f_of(R):
    ks = sorted(F)
    if R in F: return F[R]
    lo = max(k for k in ks if k <= R); hi = min(k for k in ks if k >= R)
    return F[lo] + (F[hi] - F[lo]) * (np.log(R) - np.log(lo)) / (np.log(hi) - np.log(lo))
U32 = 32 * f_of(32)
CHECK = max(7.6, 4.9) + 16.4 + 1.2 + 1.5 + 3.0

def windows(tag):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep = tz["depth"].astype(int); right = tz["right"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    out = []
    for s, t in zip(st, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        R = np.full((n0 + 1, DEP), -1, np.int8)
        for j in range(DEP):
            m = dw == j; c = int(m.sum()); R[:c, j] = right[s:t][m]
        out.append(R)
    return out

def run(W, phi, psi, c, r, seeds=4):
    toks = []; mss = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        tok = 0; nchk = 0; ms = 0.0
        for R in W:
            n = R.shape[0]; p = 1
            while p < n:
                k = 0; acc = 0; used = 0; extra = 0.0
                while acc < 31 and k < 31 and R[p, k] >= 0:
                    if R[p, k] == 1:
                        if used < r and rng.random() < psi:      # false fix on a right row: the block ends here
                            used += 1; extra += c + 3.0 * (31 - acc) / 31
                            break
                        acc += 1; k += 1; continue
                    if used < r:
                        used += 1; extra += c + 3.0 * (31 - acc) / 31
                        if rng.random() < phi:
                            acc += 1; k += 1; continue          # fixed: the chain continues teacher-forced
                    break
                tok += acc + 1; nchk += 1; ms += CHECK + extra; p += acc + 1
        toks.append(tok / nchk); mss.append(1000 * tok / ms)
    return np.mean(toks), np.mean(mss)

for tag, ref in (("heplus24", (462, 566)), ("mbppbank", (232, 321))):
    W = windows(tag)
    say(f"== {tag}: chain-32 + a decider, kept (tok/s at the bound); own chain {ref[0]}, tree-32 {ref[1]} tok/s")
    for r in (1, 2):
        for c in (2.0, 4.0, 8.0):
            for psi in (0.001, 0.005, 0.02):
                row = []
                for phi in (0.3, 0.5, 0.7, 0.9):
                    k, t = run(W, phi, psi, c, r)
                    row.append(f"phi {phi:.1f}: {k:5.2f} ({t:3.0f})")
                say(f"   r={r} c={c:3.0f} ms psi {100*psi:3.1f} %: " + " | ".join(row))
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
