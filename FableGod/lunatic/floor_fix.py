"""BREAK98 information floor, screen 1 (Lunatic, Tue 17 Nov 07:5x). CPU only, read-only, t30b topk cells.

Question: the head lacks ~10 bits per block on HE+ (drift_bits.out). Does it need the 125B state at EVERY drafted
position, or only at a few? Schemes, all chain-32 (31 guesses, depths 0..30), walk-strict, exact acceptance:
  OWN      the head's carried state (teacher-forced cells), today's chain.
  LAB      the true state at every position (truth_rt[a, d] = rt[a + d, 0]).
  PER-P    the true state injected every P depths (re-anchor at depths P, 2P, ...), own drift in between.
  FIX-r    a batched partial pass over the whole draft locates the FIRST miss; the miss is fixed (perfect lens = the
           125B's own argmax there), the head re-anchors on that row's state and re-drafts the tail; up to r rounds.
           The final full check then verifies everything. Variants: LAB-quality fix (the head's guess from the true
           state, ~97.9 %), no re-anchor (fix the token, keep the drifted state), fix success phi < 1.
Re-anchor index: anchor q's depth-a guess is x[q+2+a], verified by row q+1+a; after fixing it the next guess is the
depth-0 guess of anchor q + a + 1 (as drift_forksplit.py).
Price at Leap's C6 + int8 bound (check25_bound_real.txt): full check = dense 7.6 + experts 16.4 * U(R)/U(32) + cores
1.2 R/32 + overhead 1.5 + draft 3.0 per 31 levels. A round = layers 1..L re-read for the n re-drafted rows:
(L/48) * (7.6 + 16.4 U(n)/U(32) + 1.2 n/32) + 0.5 launch + 0.3 head re-guess + 3.0 n/31 redraft. Layers 1..L of the
first partial pass are reused by the final check, so round 1's full-draft pass is not extra.
usage: ~/BigMoE/.venv/bin/python3 -I floor_fix.py"""
import os
import numpy as np
H = os.path.expanduser("~"); CE = f"{H}/BigMoE/data/deepdraft/ceiling"
OUT = f"{H}/FableGod/lunatic/floor_fix.out"; lines = []
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
def check_ms(r=32):
    return max(7.6, 4.9 * r / 32) + 16.4 * (r * f_of(r)) / U32 + 1.2 * r / 32 + 1.5 + 3.0 * r / 31
def round_ms(L, n):
    n = max(1, n)
    return (L / 48) * (7.6 + 16.4 * (n * f_of(n)) / U32 + 1.2 * n / 32) + 0.5 + 0.3 + 3.0 * n / 31

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

def run(W, scheme, r=0, P=0, L=16, phi=1.0, reanchor=True, labfix=False, seed=0):
    rng = np.random.default_rng(seed)
    tok = 0; nchk = 0; ms = 0.0; nround = 0; nfixok = 0
    for R in W:
        n = R.shape[0]
        p = 1
        while p < n:
            acc = 0; extra = 0.0
            if scheme == "own":
                row = R[p]
                while acc < 31 and row[acc] == 1: acc += 1
            elif scheme == "lab":
                while acc < 31 and p + acc < n and R[p + acc, 0] == 1: acc += 1
            elif scheme == "per":                       # true state at depths 0, P, 2P, ...
                while acc < 31:
                    q = p + (acc // P) * P; k = acc % P
                    if q >= n or R[q, k] != 1: break
                    acc += 1
            elif scheme == "fix":
                q = p; base = 0; used = 0; k = 0       # q = state source, its depth 0 = block depth base
                while acc < 31:
                    if q >= n: break
                    if R[q, k] == 1:
                        acc += 1; k += 1
                        continue
                    if used >= r: break               # out of rounds: the final check ends here
                    used += 1; nround += 1
                    extra += round_ms(L, 31 - acc)
                    ok = (R[q + k, 0] == 1) if (labfix and q + k < n) else True
                    if ok and phi < 1.0: ok = rng.random() < phi
                    if not ok: break
                    nfixok += 1
                    acc += 1                           # the fixed guess at block depth acc-1
                    if reanchor:
                        q = q + k + 1; k = 0
                    else:
                        k += 1
                    if k >= 31: break
            kept = acc + 1
            tok += kept; nchk += 1; ms += check_ms(32) + extra; p += kept
    return tok / nchk, 1000 * tok / ms, nround / nchk, nfixok / max(1, nround)

for tag in ("heplus24", "mbppbank"):
    W = windows(tag)
    say(f"== {tag}: chain-32 (31 guesses), walk-strict, priced at the C6 + int8 bound")
    k0, t0, _, _ = run(W, "own"); say(f"   OWN (today's carried state)                    kept {k0:6.2f}  tok/s@bound {t0:5.0f}")
    k1, t1, _, _ = run(W, "lab"); say(f"   LAB (true state at all 31 positions)           kept {k1:6.2f}  tok/s@bound {t1:5.0f}   (no channel cost)")
    for P in (2, 4, 8, 16):
        k, t, _, _ = run(W, "per", P=P)
        say(f"   PER-{P:<2d} true state every {P:2d} depths ({(30 // P):2d} injections) kept {k:6.2f}  share of LAB gain {100*(k-k0)/(k1-k0):5.1f} %")
    say("   FIX-r: perfect locator + perfect fix + re-anchor (rounds r; price for L = 8 / 16 / 24 layers)")
    for r in (1, 2, 3, 4, 31):
        res = [run(W, "fix", r=r, L=L) for L in (8, 16, 24)]
        k = res[0][0]
        say(f"     r={r:2d}: kept {k:6.2f} (LAB gain share {100*(k-k0)/(k1-k0):5.1f} %), rounds/check {res[0][2]:4.2f}, "
            f"tok/s@bound L8 {res[0][1]:4.0f} L16 {res[1][1]:4.0f} L24 {res[2][1]:4.0f}")
    say("   FIX-r variants at L = 16:")
    for r in (1, 2, 3):
        a = run(W, "fix", r=r, L=16, labfix=True)
        b = run(W, "fix", r=r, L=16, reanchor=False)
        say(f"     r={r}: LAB-quality fix + re-anchor kept {a[0]:6.2f} ({a[1]:4.0f} tok/s, fix ok {100*a[3]:4.1f} %) | "
            f"perfect fix, NO re-anchor kept {b[0]:6.2f} ({b[1]:4.0f})")
    for phi in (0.9, 0.7, 0.5):
        row = []
        for r in (1, 2, 3):
            a = run(W, "fix", r=r, L=16, phi=phi, labfix=True, seed=1)
            row.append(f"r={r} kept {a[0]:6.2f} ({a[1]:4.0f})")
        say(f"     fix success phi {phi:.1f} (x LAB quality), L16: " + " | ".join(row))
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
