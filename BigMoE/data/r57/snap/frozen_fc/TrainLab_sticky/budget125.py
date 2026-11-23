"""TrainLab (10-22): the implementable per-pass union budget on the 125B traces, next to Chat1000's hard caps. Per
16-row block and layer the U experts with the largest sum over rows of w^2 (w = top-10-normalised gate weights over the
row's top-10) form the pass set A; each row takes its top-10 within A (its top-20 members by weight first, then fill-ins
from A below its top-20, whose weight is unknown: charged 0 ("lo") or the row's 20th weight ("hi")). Chat1000's error
energy (selected-but-not-true count w^2, true-left-out count w^2), unions, KLD law. CPU only. -> data/budget125.txt"""
import glob, os
import numpy as np
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
import sys
NL, K = 48, 10
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
US = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16, 22, 32, 40, 49]
hard = {"bulgarian", "chat"}
cap = {}
for l in open('/Users/kamen/Chat1000/data/switchcap_hard125.txt').read().splitlines()[1:]:
    p = l.split('|'); n, m = p[0].split(); cap.setdefault(n, []).append((float(p[2].split()[0]), float(p[3])))


def law(name, e):
    return 0.0045 + 29.9 * e ** 1.5 if name in hard else -0.0003 + 6.49 * e ** 1.5


def cap_at(name, u):
    if B != 16:
        return None
    pts = sorted(cap[name])
    for (u0, e0), (u1, e1) in zip(pts, pts[1:]):
        if u0 <= u <= u1:
            return np.exp(np.log(e0) + (u - u0) / (u1 - u0) * (np.log(e1) - np.log(e0)))
    return None


out = []
for f in fs:
    name = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)            # [T, NL, 20]
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
    w = g / g[:, :, :K].sum(-1, keepdims=True)
    T = ids.shape[0] // B * B
    nat = np.mean([len(np.unique(ids[b:b + B, l, :K])) for b in range(0, T, B) for l in range(NL)])
    print(f"{name:9s} natural U({B}) {nat:.1f}", flush=True); out.append(f"{name:9s} natural U({B}) {nat:.1f}")
    for U in US:
        Elo = Ehi = 0.0; un = []
        for b in range(0, T, B):
            for l in range(NL):
                c = {}
                for t in range(b, b + B):
                    for e, x in zip(ids[t, l, :K], w[t, l, :K]):
                        c[int(e)] = c.get(int(e), 0.0) + x * x
                A = set(sorted(c, key=c.get, reverse=True)[:U])
                used = set()
                for t in range(b, b + B):
                    inA = np.array([int(e) in A for e in ids[t, l]])
                    pick = np.flatnonzero(inA)[:K]                                    # top-20 members in A, by weight
                    nf = K - len(pick)                                                # fill-ins below the top-20
                    true_out = (w[t, l, :K][~inA[:K]] ** 2).sum()
                    extra = (w[t, l, pick[pick >= K]] ** 2).sum()
                    Elo += true_out + extra
                    Ehi += true_out + extra + nf * w[t, l, -1] ** 2
                    used |= {int(e) for e in ids[t, l, pick]}
                un.append(min(U, len(A)) if True else len(used))
        n = T * NL
        u16 = float(np.mean(un)); elo, ehi = Elo / n, Ehi / n
        z = cap_at(name, u16)
        s = (f"{name:9s} U {U:2d}: U({B}) {u16:5.1f} | E {elo:.4f}-{ehi:.4f} | law KLD {law(name, elo):.3f}-{law(name, ehi):.3f}"
             + (f" | hard cap at equal U: E {z:.4f} (law {law(name, z):.3f}) -> cap / budget {z / ehi:.2f}-{z / elo:.2f}x"
                if z else " | (outside the caps' U(16) range)"))
        print(s, flush=True); out.append(s)
open(os.path.expanduser(f"~/TrainLab/data/budget125{'' if B == 16 else f'_B{B}'}.txt"), "w").write("\n".join(out) + "\n")
