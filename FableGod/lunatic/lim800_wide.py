"""LIM800 INERT-WIDE (Lunatic, THINK_1116b, Mon 16 Nov 18:1x). CPU only, read-only, one process, < 0.5 GB.
Under the frame's lossy rules (CC d1 + INERT: comment / docstring / prose rows accepted, code rows strict at d1), rows in
an INERT span are accepted whatever is drafted, so they only fill KV: a check that STARTS inside an INERT span can be
as wide as the span (COPYRUN, TRIED 1471, generalised from prompt copies to every INERT span). Walk:
  - fixed: chain-32 checks (cap 31), the frame;
  - wide-W: if the check's start row is INERT, the check is sized min(W, run + 32) rows: the whole INERT run ahead, then
    into the code rows that follow with CC d1 until the first miss or the width W; else a normal 32-row check.
Prices (ms) at Leap's C6 + int8 bound (check25_bound_real.txt) and the frame's per-check terms:
  dense = max(7.6 bytes, 4.9 * R / 32 int8 compute); experts = 16.4 * U(R) / U(32), U(R) = R * f(R), f from the code
  union law (lunatic wl_chatlen: 163 at 32, x0.75 per doubling beyond); cores = 1.2 * R / 32; overhead 1.5 per check;
  draft = 3.0 ms per 31 levels, one level per drafted row (chain; copies are NOT made free here).
  Efficiency 100 % (bound) and 85 % (check only).
usage: ~/BigMoE/.venv/bin/python3 -I lim800_wide.py"""
import os, sys
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/lim800_wide.out"'))

F = {1: 480, 2: 406, 3: 365, 4: 337, 6: 300, 8: 276, 12: 241, 16: 218, 24: 185, 32: 163}
def f_of(R):
    if R in F: return F[R]
    if R < 32:
        ks = sorted(F); lo = max(k for k in ks if k <= R); hi = min(k for k in ks if k >= R)
        return F[lo] + (F[hi] - F[lo]) * (np.log(R) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return 163 * 0.75 ** np.log2(R / 32)
def check_ms(R, eff=1.0):
    dense = max(7.6, 4.9 * R / 32); experts = 16.4 * (R * f_of(R)) / (32 * f_of(32)); cores = 1.2 * R / 32
    return (dense + experts + cores) / eff

def walk_wide(A, W):
    """returns a list of (kept, rows, levels) per check"""
    rt = A["rt"].copy(); rt[(rt == 0) & ((np.nan_to_num(A["DL"], nan=-99) >= -1) | np.isin(A["S"], (1, 2, 4)))] = 1
    n0 = A["n0"]; S0 = A["S"][:n0, 0]; out = []; p = C98.R0
    while p < rt.shape[0]:
        # the check starting at anchor p predicts tokens p, p+1, ... (depth j -> token index p + j)
        if W > 32 and p < n0 and S0[p] in (1, 2, 4):
            e = p
            while e < n0 and S0[e] in (1, 2, 4) and e - p < W - 1: e += 1
            run = e - p                       # INERT rows, all accepted
            k = run; q = e                    # go on into code rows from anchor q with the chain cells of q
            Wc = min(W, run + 32)
            while k < Wc - 1 and q < rt.shape[0] and (q - e) < 31 and rt[e, q - e] == 1:
                k += 1; q += 1
            k = min(k, W - 1)
            rows = W if k >= W - 1 else min(W, k + 1 + 0)   # a check is sized before it runs: W rows
            R = int(min(W, run + 32)); out.append((min(k, R - 1) + 1, R, R - 1)); p += min(k, R - 1) + 1
        else:
            k = 0
            while k < 31 and rt[p, k] == 1: k += 1
            out.append((k + 1, 32, 31)); p += k + 1
    return out

for tag in ("heplus24", "mbppbank"):
    W_ = load(tag)
    say(f"== {tag}")
    base = None
    for W in (32, 48, 64, 96, 128):
        recs = []
        for A in W_: recs += walk_wide(A, W)
        recs = np.array(recs, float)
        n_tok = recs[:, 0].sum(); n_chk = len(recs)
        for eff in (1.0, 0.85):
            t = sum(check_ms(int(r), eff) + 1.5 + 3.0 * lv / 31 for _, r, lv in recs)
            tps = 1000 * n_tok / t
            if W == 32 and eff == 1.0: base = tps
            if W == 32 and eff == 0.85: base85 = tps
            rel = tps / (base if eff == 1.0 else base85)
            say(f"   W {W:3d} eff {int(eff*100):3d} %: checks {n_chk:5d}  kept/check {n_tok/n_chk:6.2f}  wide checks "
                f"{int((recs[:,1] > 32).sum()):4d}  tok/s {tps:6.0f}  x{rel:5.3f}")
    say()
say("check_ms at the bound: " + ", ".join(f"R{R} {check_ms(R):.1f}" for R in (16, 32, 48, 64, 96, 128)))
open(OUT, "w").write("\n".join(lines) + "\n")
