"""Fit12 R-28 LAYSZ-X (Tue 27 Oct 2026; NewIdeas QUEUE R-28 + 20:23 message; PLAN.md 'R-28', written before any run):
cross-check of MoonShots' R-27 (static per-layer pool sizes; at 12 GB -20.7 % held out under 16 / 10 LFU, -5.8 % under 4 / 3
LRU) in Fit12's PLA replay (fit_v3replay.Tier: LRU d = 0 = what bt_b12 ships, LFU d = 0.99 = BigMoE fast24's), expA traces,
misses from token 400, the pool bytes of uniform C_m 51 / C_l 20. Check shapes: R 4 / A 3 (bt_b12's MTP path, k_mtp 3) and
R 16 / A 10 (the shape of K1's 256.4 and of R-27). PLA's grid was C_m 24-96 / C_l 8-40; here R-27's range: C_m 1, 11, ..., 231
and C_l 2-80 step 2 (51 and 20 included). Per shape and policy: uniform 51 / 20, and the Lagrangian per-layer allocation fitted
on calibration sets and judged held out (leave one set out, 4 folds; PLA's two 2-set folds). Also R-27's published C_m
(MoonShots data/laysz_51_20.txt, calibrated on all 4 sets under 16 / 10 LFU, so in-sample) with C_l fitted to the rest of the
budget. F12_LAYSZ_SMOKE=1: 2 layers, 2 sets (a code check). Usage: [BigMoE venv] fit_laysz.py [TAG]   (CPU, 1 core)"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_v3replay as V

SMOKE = os.environ.get("F12_LAYSZ_SMOKE") == "1"
NL = 2 if SMOKE else V.NL
GM, GL = [1] + list(range(11, 232, 10)), list(range(2, 81, 2))
CM0, CL0, MSB, LSB = 51, 20, V.MSB, V.LSB
BUDGET = NL * (CM0 * MSB + CL0 * LSB)
POL, SHAPES = {"LRU": 0.0, "LFU": 0.99}, ((4, 3), (16, 10))
R27F = os.path.expanduser("~/MoonShots/data/laysz_51_20.txt")


def misses(x, C, ranks, d, R, A):
    t_, miss = V.Tier(C, d), 0
    for t in range(0, x.shape[0], A):
        m = t_.call(list(dict.fromkeys(x[t:t + R, :ranks].reshape(-1).tolist())))
        if t >= V.T0:
            miss += m
    return miss


def alloc(Mm, Ml, budget, im_fixed=None):
    """PLA's Lagrangian (lower convex hull of each layer's miss curves; misses + lam * slots, both in bytes); im_fixed keeps C_m"""
    def pick(lam):
        im = im_fixed or [int(np.argmin(Mm[L] + lam * np.array(GM))) for L in range(NL)]
        il = [int(np.argmin(Ml[L] + lam * np.array(GL))) for L in range(NL)]
        return im, il, sum(GM[i] * MSB for i in im) + sum(GL[j] * LSB for j in il)
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[2] <= budget:
            hi = mid
        else:
            lo = mid
    return pick(hi)[:2]


def main():
    t0 = time.time()
    sets = V.SETS[:2] if SMOKE else V.SETS
    tr = {s: [np.load(os.path.join(V.TR, f"{s}_800_g.npz"))[f"L{L}"][:, :10].astype(np.int64) for L in range(NL)] for s in sets}
    T = tr[sets[0]][0].shape[0]
    r27 = [int(v) for v in [l for l in open(R27F) if l.startswith("per-layer C_m")][0].split(":")[1].split()][:NL]
    im27, um, ul = [GM.index(c) for c in r27], [GM.index(CM0)] * NL, [GL.index(CL0)] * NL
    print(f"[laysz] sets {', '.join(sets)} ({T} tokens); C_m {GM[0]}-{GM[-1]} ({len(GM)} sizes), C_l {GL[0]}-{GL[-1]} "
          f"({len(GL)}); budget {BUDGET / 1e9:.3f} GB (= {CM0} / {CL0} uniform); R-27 C_m {sum(r27)} slots", flush=True)
    folds = [([x for x in sets if x != s], [s]) for s in sets] + ([] if SMOKE else [(["bulgarian", "chat"], ["code", "math"]),
                                                                                  (["code", "math"], ["bulgarian", "chat"])])
    for R, A in SHAPES:
        tok = sum(min(A, T - t) for t in range(0, T, A) if t >= V.T0)
        M = {}
        for p, d in POL.items():
            M[p] = ({s: np.array([[misses(tr[s][L], C, 10, d, R, A) for C in GM] for L in range(NL)]) for s in sets},
                    {s: np.array([[misses(tr[s][L], C, 2, d, R, A) for C in GL] for L in range(NL)]) for s in sets})
            print(f"[laysz] R {R} A {A} {p} curves done ({time.time() - t0:.0f} s; {tok} tokens counted per set)", flush=True)
        np.savez(os.path.expanduser(f"~/Fit12/data/laysz_curves_r{R}a{A}{'_smoke' if SMOKE else ''}.npz"), GM=GM, GL=GL,
                 **{f"{p}_{k}_{s}": M[p][i][s] for p in POL for i, k in enumerate("ml") for s in sets})

        def mb(p, ss, im, il):
            Mm, Ml = M[p]
            return np.mean([sum(Mm[s][L][im[L]] * MSB + Ml[s][L][il[L]] * LSB for L in range(NL)) / tok for s in ss]) / 1e6

        res = {p: [] for p in POL}
        for fit, test in folds:
            base = mb("LRU", test, um, ul)
            for p in POL:
                im, il = alloc(sum(M[p][0][s] for s in fit), sum(M[p][1][s] for s in fit), BUDGET)
                u, pl = mb(p, test, um, ul), mb(p, test, im, il)
                cms, cls = [GM[i] for i in im], [GL[i] for i in il]
                used = sum(c * MSB for c in cms) + sum(c * LSB for c in cls)
                res[p].append((len(fit) == len(sets) - 1, pl / base - 1, pl / u - 1, u / base - 1))
                print(f"[laysz] R {R} A {A} {p} fit {'+'.join(fit)} -> held out {'+'.join(test)}: uniform 51/20 {u:.1f} "
                      f"MB/token ({(u / base - 1) * 100:+.1f} % vs LRU uniform {base:.1f}) | per-layer {pl:.1f} ({(pl / u - 1) * 100:+.1f}"
                      f" % vs {p} uniform, {(pl / base - 1) * 100:+.1f} % vs LRU uniform) | pool {used / 1e9:.3f} GB, C_m "
                      f"{min(cms)}-{max(cms)} ({sum(c > 1 for c in cms)} layers > 1), C_l {min(cls)}-{max(cls)}", flush=True)
                print(f"[laysz]   C_m by layer {cms}", flush=True)
        base = mb("LRU", sets, um, ul)
        for p in POL:
            loo = [r for r in res[p] if r[0]]
            print(f"[laysz] SUMMARY R {R} A {A} {p}: uniform 51/20 all sets {mb(p, sets, um, ul):.1f} MB/token (LRU {base:.1f}); "
                  f"leave-one-set-out mean per-layer {np.mean([r[1] for r in loo]) * 100:+.1f} % vs LRU uniform "
                  f"({np.mean([r[2] for r in loo]) * 100:+.1f} % vs {p} uniform; {p} uniform itself "
                  f"{np.mean([r[3] for r in loo]) * 100:+.1f} %); 2-set folds "
                  f"{', '.join(f'{r[1] * 100:+.1f}' for r in res[p] if not r[0]) or 'n/a'} % vs LRU uniform", flush=True)
            _, il = alloc(sum(M[p][0][s] for s in sets), sum(M[p][1][s] for s in sets), BUDGET, im_fixed=im27)
            pl = mb(p, sets, im27, il)
            used = sum(GM[i] * MSB for i in im27) + sum(GL[j] * LSB for j in il)
            print(f"[laysz]   R-27's C_m + fitted C_l under R {R} A {A} {p} (in-sample, all sets): {pl:.1f} MB/token "
                  f"({(pl / base - 1) * 100:+.1f} % vs LRU uniform), pool {used / 1e9:.3f} GB", flush=True)
    print(f"[laysz] done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
