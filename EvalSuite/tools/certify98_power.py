"""CERTIFY98 sample sizes (docs/PREREG_certify98_2026-11-15.md): how many held-out windows put every band's 95 % lower
bound at >= 97.5 % when the truth is 98 % per guess at every depth. CPU only.
Clustering comes from t30b's MBPP 100 cells (~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz; depth-major per
problem): each problem's per-band error is scaled so the pooled error is 2 % (M1: the relative spread kept), or its
spread around 2 % is doubled (M2, the stress case). Window sizes come from each set's bank files (split.json).
The lower bound is the pre-registered one: the ratio estimate minus t(K-1, 0.95) x the window-clustered (CR1) SE.
Within a window, the depths of one band are treated as one draw per anchor (fully correlated, conservative).
  .venv/bin/python tools/certify98_power.py [reps]"""
import json, os, sys
import numpy as np

def t95(df):
    """t(df) 0.95 quantile, Cornish-Fisher (error < 1e-3 for df >= 10; no scipy in this venv)"""
    z = 1.6448536
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]          # dd_ttt.py BANDS
H = os.path.expanduser("~")
CELLS = {"mbpp": f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz",
         "heplus": f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus.npz"}


def load(name):
    global dep, right, st, en
    d = np.load(CELLS[name])
    dep, right = d["depth"], d["right"].astype(float)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))


load("mbpp")


def band_table():
    """per problem x band: errors and cells (problems with cells in every band)"""
    E, N = [], []
    for s, t in zip(st, en):
        e, n = [], []
        for lo, hi in BANDS:
            m = (dep[s:t] >= lo) & (dep[s:t] <= hi)
            e.append(1 - right[s:t][m].mean() if m.sum() else np.nan); n.append(m.sum())
        E.append(e); N.append(n)
    E, N = np.array(E), np.array(N)
    return E, N


def cr1(acc, w):
    """ratio estimate and CR1 SE over windows (acc per window, w cells per window)"""
    K = len(acc); p = (acc * w).sum() / w.sum()
    se = np.sqrt(K / (K - 1) * ((w * (acc - p)) ** 2).sum()) / w.sum()
    return p, se


def observed():
    E, N = band_table()
    print("observed t30b MBPP 100 (cluster = problem):")
    for j, (lo, hi) in enumerate(BANDS):
        ok = N[:, j] > 0; acc, w = 1 - E[ok, j], N[ok, j]
        p, se = cr1(acc, w)
        se_iid = np.sqrt(p * (1 - p) / w.sum())
        print(f"  band {lo}-{hi}: {100 * p:.2f} % over {int(w.sum())} cells in {ok.sum()} problems; clustered SE "
              f"{100 * se:.3f} pt vs {100 * se_iid:.3f} iid -> design effect {(se / se_iid) ** 2:.1f}")


def truth(E, mode, target=0.02):
    """per problem x band error at a 2 % truth in every band, over the problems with cells in every band (window sizes
    are drawn apart from the error profiles, so the plain mean over problems is the truth)"""
    E = E[~np.isnan(E).any(1)]
    T = np.empty_like(E)
    for j in range(E.shape[1]):
        e = E[:, j]; ok = ~np.isnan(e); w = np.ones(ok.sum())
        t = e[ok] * target / e[ok].mean()
        if mode == "M2":
            t = np.clip(target + 2 * (t - target), 0, 1); t *= target / t.mean()
        T[ok, j] = np.clip(t, 0, 1); T[~ok, j] = np.nan
    return T


def power(T, sizes, K, reps, rng, line=0.975, shift=0.0):
    tq = t95(K - 1)
    allpass, bandpass = 0, np.zeros(len(BANDS))
    for _ in range(reps):
        i = rng.integers(0, len(T), K); R = rng.choice(sizes, K)
        good = True
        for j, (lo, hi) in enumerate(BANDS):
            eff = np.maximum(R - lo, 1)                              # one draw per anchor in this band
            cells = np.array([sum(max(r - dd, 0) for dd in range(lo, hi + 1)) for r in R], float)
            e = np.clip(T[i, j] + shift, 0, 1)
            acc = 1 - rng.binomial(eff, e) / eff
            p, se = cr1(acc, np.maximum(cells, 1))
            ok_ = p - tq * se >= line
            bandpass[j] += ok_; good &= ok_
        allpass += good
    return allpass / reps, bandpass / reps


def heplus_measured(sizes, reps, rng):
    """HE+ windows' own per-band spread (t30b is at 97.95 % at depth 0 there), scaled to 98 % in every band"""
    load("heplus"); E, _ = band_table(); load("mbpp")
    T = truth(E, "M1")
    print(f"\nHE+ measured ({len(T)} window profiles; band-0 per-window error sd at the 98 % truth "
          f"{np.std(T[:, 0]) * 100:.2f} pt): P(every band's 95 % LCB >= line) when the truth is 98 % flat")
    for line in (0.975, 0.97):
        row = []
        for K in (60, 100, 140, 164, 300):
            a, b = power(T, sizes, K, reps, rng, line=line)
            row.append(f"K {K}: {a:.2f} (band 0 {b[0]:.2f})")
        print(f"  line {100 * line:.1f}: " + "; ".join(row))
    for line, sh in ((0.975, 0.005), (0.97, 0.01)):
        a, _ = power(T, sizes, 140, reps, rng, line=line, shift=sh)
        print(f"  line {100 * line:.1f}, K 140, truth {100 * (0.98 - sh):.1f} % flat: certified {a:.3f}")


def main(reps=1000):
    rng = np.random.default_rng(15)
    observed()
    E, N = band_table()
    s = json.load(open(f"{H}/BigMoE/data/deepdraft/bank/split.json"))
    B = f"{H}/datasets/deepdraft/bank"
    sizes = {}
    for k in ("heplus", "mbpp", "val"):
        a = []
        for f in s[k]:
            z = np.load(os.path.join(B, f), mmap_mode="r")
            a.append(z["wd"].shape[0] - (int(z["gen"]) - 1))
        sizes[k] = np.array(a)
    heplus_measured(sizes["heplus"], reps, rng)
    for mode in ("M1", "M2"):
        T = truth(E, mode)
        sd = np.nanstd(T[:, 0]) * 100
        print(f"\n{mode}: {len(T)} problem profiles; truth per band " + " ".join(f"{100 * (1 - T[:, j].mean()):.2f}" for j in range(len(BANDS))))
        print(f"\n{mode} ({'relative spread kept' if mode == 'M1' else 'spread doubled'}; band-0 per-window error sd at "
              f"the 98 % truth {sd:.2f} pt): P(every band's 95 % LCB >= 97.5 %) when the truth is 98 % flat")
        for k, lab in (("heplus", "HE+ sizes"), ("mbpp", "MBPP sizes"), ("val", "everyday/val sizes")):
            row = []
            for K in (24, 40, 60, 100, 150, 200, 300):
                a, b = power(T, sizes[k], K, reps, rng)
                row.append(f"K {K}: {a:.2f} (band 0 {b[0]:.2f})")
            print(f"  {lab}: " + "; ".join(row))
        a, _ = power(T, sizes["heplus"], 150, reps, rng, shift=0.01)
        b, _ = power(T, sizes["mbpp"], 150, reps, rng, shift=0.01)
        print(f"  false certification at a 97 % truth, K 150: HE+ sizes {a:.3f}, MBPP sizes {b:.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
