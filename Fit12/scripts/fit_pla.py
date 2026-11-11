"""Fit12 PLA (Mon 26 Oct 2026; pre-registered in ~/Fit12/PLAN.md 'Rapid screens, batch 1'): per-layer pool sizes vs the
uniform C_m 51 / C_l 20 of the 12 GB line, at the same pool bytes. Engine-faithful LRU (fit_v3replay.Tier with d = 0, the
12 GB engine's default), expA traces, checks of R rows accepting A (MTP-like 4 / 3), caches warm from token 0, misses
counted from token 400. Per layer and set: misses at every grid size; allocations are chosen by a Lagrangian (convex hull of
the miss curves) on 2 sets and judged on the other 2, both ways. Control: the best uniform (C_m, C_l) split of the same bytes.
Usage: ~/BigMoE/.venv/bin/python -B /Users/kamen/Fit12/scripts/fit_pla.py   (CPU, 1 core, a few minutes)"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_v3replay as V

R, A = int(os.environ.get("F12_R", "4")), int(os.environ.get("F12_A", "3"))
GM, GL = list(range(24, 97, 3)), list(range(8, 41, 2))
CM0, CL0, MSB, LSB, NL = 51, 20, V.MSB, V.LSB, V.NL
BUDGET = NL * (CM0 * MSB + CL0 * LSB)


def misses(x, C, ranks):
    t_, miss = V.Tier(C, 0.0), 0
    for t in range(0, x.shape[0], A):
        m = t_.call(list(dict.fromkeys(x[t:t + R, :ranks].reshape(-1).tolist())))
        if t >= V.T0:
            miss += m
    return miss


def alloc(Mm, Ml):
    """Lagrangian (the lower convex hull of every layer's miss curves): per layer, C_m and C_l minimize misses + lam * slots
    (separable, as both are in bytes); lam bisected to the largest pool that fits the budget"""
    def pick(lam):
        im = [int(np.argmin(Mm[L] + lam * np.array(GM))) for L in range(NL)]
        il = [int(np.argmin(Ml[L] + lam * np.array(GL))) for L in range(NL)]
        return im, il, sum(GM[i] * MSB for i in im) + sum(GL[j] * LSB for j in il)
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[2] <= BUDGET:
            hi = mid
        else:
            lo = mid
    return pick(hi)[:2]


def main():
    t0 = time.time()
    tr = {s: [np.load(os.path.join(V.TR, f"{s}_800_g.npz"))[f"L{L}"][:, :10].astype(np.int64) for L in range(NL)]
          for s in V.SETS}
    T = tr[V.SETS[0]][0].shape[0]
    tok = sum(min(A, T - t) for t in range(0, T, A) if t >= V.T0)
    print(f"[pla] R {R} A {A}; sets {', '.join(V.SETS)} ({T} tokens, {tok} counted per set); grids C_m {GM[0]}-{GM[-1]} "
          f"step 3, C_l {GL[0]}-{GL[-1]} step 2; budget {BUDGET / 1e9:.3f} GB (= {CM0} / {CL0} uniform)", flush=True)
    Mm = {s: np.array([[misses(tr[s][L], C, 10) for C in GM] for L in range(NL)]) for s in V.SETS}
    print(f"[pla] MSB curves done ({time.time() - t0:.0f} s)", flush=True)
    Ml = {s: np.array([[misses(tr[s][L], C, 2) for C in GL] for L in range(NL)]) for s in V.SETS}
    print(f"[pla] LSB curves done ({time.time() - t0:.0f} s)", flush=True)
    np.savez(os.path.expanduser("~/Fit12/data/pla_curves.npz"), GM=GM, GL=GL, **{f"m_{s}": Mm[s] for s in V.SETS},
             **{f"l_{s}": Ml[s] for s in V.SETS})

    def mb(sets, im, il):          # read MB / token, mean over the sets
        return np.mean([sum(Mm[s][L][im[L]] * MSB + Ml[s][L][il[L]] * LSB for L in range(NL)) / tok for s in sets]) / 1e6

    u_m, u_l = [GM.index(CM0)] * NL, [GL.index(CL0)] * NL
    folds = ((("bulgarian", "chat"), ("code", "math")), (("code", "math"), ("bulgarian", "chat")), (V.SETS, V.SETS))
    for fit, test in folds:
        Fm, Fl = sum(Mm[s] for s in fit), sum(Ml[s] for s in fit)
        im, il = alloc(Fm, Fl)
        uni = [(cm, cl) for cm in range(len(GM)) for cl in range(len(GL)) if GM[cm] * MSB + GL[cl] * LSB <= CM0 * MSB + CL0 * LSB]
        bu = min(uni, key=lambda p: mb(fit, [p[0]] * NL, [p[1]] * NL))
        base = mb(test, u_m, u_l)
        pl, us = mb(test, im, il), mb(test, [bu[0]] * NL, [bu[1]] * NL)
        cms, cls = [GM[i] for i in im], [GL[i] for i in il]
        used = sum(c * MSB for c in cms) + sum(c * LSB for c in cls)
        print(f"[pla] fit {'+'.join(fit)} -> judged on {'+'.join(test)}: uniform 51/20 {base:.1f} MB/token | per-layer "
              f"{pl:.1f} ({(pl / base - 1) * 100:+.1f} %) | best uniform split {GM[bu[0]]}/{GL[bu[1]]} {us:.1f} "
              f"({(us / base - 1) * 100:+.1f} %) | per-layer pool {used / 1e9:.3f} GB, C_m {min(cms)}-{max(cms)}, C_l "
              f"{min(cls)}-{max(cls)}", flush=True)
        print(f"[pla]   C_m by layer {cms}", flush=True)
        print(f"[pla]   C_l by layer {cls}", flush=True)
    print(f"[pla] done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
