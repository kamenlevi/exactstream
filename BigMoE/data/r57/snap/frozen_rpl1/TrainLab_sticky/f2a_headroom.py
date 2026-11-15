"""F2A headroom (MTP Head, Sun 11-08 00:xx, CPU): what a better chat head would keep per check. NewIdeas PROGRAM CHAT 150 F2a.
A better head is simulated on D-24's teacher-forced chain (head b5, 30 IFEval + 24 multi held-out answers): each miss
flips to a hit with a probability that depends on how sure the 125B is of that token (its top-1 p at that position, from
data/d02a's top-32 logprobs). Flipped tokens get the head's p raised to >= 0.9 (a better head is sure where it is right),
so the running-p cut lets the chain run on. Walk and costs as sticky/chat_ngram.py (chat's fitted wall).
CAVEAT: the stored answers are not the 125B's own greedy text (its top-1 matches 82 % of IFEval tokens, 93 % multi), so
the hit rates are against the answer text; the engine's 12-IFEval kept (2.38 per check, N45) is close to the replay's 2.34.
-> data/f2a_headroom.txt"""
import json, os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
BM = os.path.expanduser("~/BigMoE/data/r57")
L_ = []


def say(s=""):
    print(s, flush=True); L_.append(s)


rows_, drf_, wall_ = [], [], []
for t in ("nie1", "nie2"):
    for c in json.load(open(os.path.join(BM, f"{t}_256_checks.json"))):
        for i in range(2, len(c["c0"]) - 1):
            rows_.append(c["rows"][i]); drf_.append(c["drf"][i]); wall_.append(c["c0"][i + 1] - c["c0"][i])
rows_, drf_, wall_ = map(np.array, (rows_, drf_, wall_))
F, R = np.linalg.lstsq(np.stack([np.ones(len(rows_)), rows_], 1), wall_ - drf_, rcond=None)[0]
d0, d1 = np.polyfit(rows_, drf_, 1)[::-1]


def cost_today(rows, steps):
    return F + R * rows + (d0 + d1 * steps if steps else 0.0)


def cost_f1(rows, steps, check=30.0, rrow=R, dstep=0.6):
    """NewIdeas' F1: a ~30 ms check at today's median 3 rows, + verify per extra row, + a fused 0.6 ms per drafted row"""
    return check + rrow * (rows - 3) + dstep * steps


rng = np.random.default_rng(0)
sets = {}
for st in ("ifeval", "multi"):
    z = np.load(os.path.join(TD, "d24", f"{st}_b5.npz"), allow_pickle=True)
    A = []
    for i, key in enumerate(z["keys"]):
        y = np.load(os.path.join(TD, "d02a", f"{key}.npz"))
        ids, gen = z[f"ids_{i}"], int(z["gen"][i]); a0 = max(0, gen - 2)
        lp = y["lp"].astype(np.float32)
        rk, p1 = z[f"rk_{i}"].copy(), z[f"p1_{i}"].copy()
        tp = np.full(rk.shape, np.nan, np.float32)                 # the 125B's top-1 p for the token at (anchor, depth)
        for s in range(rk.shape[0]):
            for d in range(rk.shape[1]):
                x = a0 + s + 1 + d
                if rk[s, d] >= 0 and 0 <= x - 1 < len(lp):
                    tp[s, d] = np.exp(lp[x - 1, 0])
        A.append((rk, p1, tp, len(ids), gen))
    sets[st] = A


def walk_all(A, flip, t=0.3, K=15, cost=cost_today):
    """flip = (f_hi p1 >= 0.8, f_mid 0.5-0.8, f_lo < 0.5); -> kept / check, tok/s, depth-1 hit, rows / check"""
    T = C = M = RW = 0.0; h1 = n1 = 0
    for rk0, p0, tp, n, gen in A:
        rk, p = rk0.copy(), p0.copy()
        u = rng.random(rk.shape)
        fr = np.where(tp >= 0.8, flip[0], np.where(tp >= 0.5, flip[1], flip[2]))
        fl = (rk > 0) & (u < fr)
        rk[fl] = 0; p[fl] = np.maximum(p[fl], 0.9)
        v = rk[:, 0] >= 0; h1 += (rk[v, 0] == 0).sum(); n1 += v.sum()
        a0 = max(0, gen - 2); Lw = n - gen; pos = 1
        while pos < Lw:
            s = gen + pos - 2 - a0
            if 0 <= s < len(rk) and (rk[s] >= 0).any():
                r_, q = rk[s], p[s]; L = int((r_ >= 0).sum())
                miss = np.where(r_[:L] != 0)[0]; kstar = int(miss[0]) + 1 if len(miss) else L + 1
                m = int((np.cumprod(np.nan_to_num(q[:min(K, L)], nan=0.0)) >= t).sum())
                k, rows, steps = min(kstar, m + 1), m + 1, min(m + 1, min(K, L))
            else:
                k, rows, steps = 1, 1, 0
            k = max(1, min(k, Lw - pos))
            T += k; C += 1; M += cost(rows, steps); RW += rows; pos += k
    return T / C, T / M * 1e3, h1 / n1, RW / C


say(f"F2A headroom (CPU; data/f2a_headroom.txt). Chat wall today: {F:.1f} + {R:.2f} x rows + draft ({d0:.2f} + {d1:.2f} x steps) ms;")
say("F1 = NewIdeas' ~30 ms check at 3 rows + the same verify per extra row + a fused 0.6 ms per drafted row.")
say("scenario (share of misses fixed at the 125B's p1 >= 0.8 / 0.5-0.8 / < 0.5) | set | depth-1 hit | kept/check (rows) "
    "t 0.3 | t 0.15 | tok/s today t 0.3 | tok/s with F1, best t")
SC = [("today (head b5)", (0, 0, 0)), ("modest", (0.5, 0.25, 0.1)), ("strong", (0.8, 0.5, 0.2)),
      ("all sure tokens", (1.0, 0.5, 0.2)), ("near-perfect", (1.0, 0.8, 0.5))]
for nm, fl in SC:
    for st in ("ifeval", "multi"):
        k3, tp3, h1, r3 = walk_all(sets[st], fl, 0.3)
        k15, _, _, r15 = walk_all(sets[st], fl, 0.15)
        best = max((walk_all(sets[st], fl, t, cost=cost_f1)[1], t) for t in (0.1, 0.15, 0.2, 0.3, 0.4))
        say(f"{nm:16s} {str(fl):16s} | {st:6s} | {h1:.3f} | {k3:4.2f} ({r3:3.1f}) | {k15:4.2f} ({r15:3.1f}) | {tp3:5.1f} | "
            f"{best[0]:5.1f} (t {best[1]})")
say("")
say("Reading: today's replay keeps 2.34 (IFEval) at t 0.3; NewIdeas' F2a target is ~4.5 kept at ~6 rows (150 tok/s with F1).")
open(os.path.join(TD, "f2a_headroom.txt"), "w").write("\n".join(L_) + "\n")
