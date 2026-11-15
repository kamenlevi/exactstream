"""CERTIFY98 amendment 4 sizing: the STRICT conditional accuracy at the decode walk's check starts (what the engine pays
on) and kept/32, when the truth is a flat p per guess there. CPU only, numpy only, ~1 min.
Model: each window w has an error rate e_w at check starts (HumanEval+'s measured per-window depth-0 spread, times one
scale); the walk draws each check's run L ~ Geometric(e_w) capped at 31 and by the window's end; the strict per-band
counts and kept/32 follow, scored with certify98.py's CR1 bounds. The truth is the POOLED walk-strict accuracy at depth
0, as the gate measures it: check starts gather in the windows that miss, so the scale is set by bisection until the
pooled depth-0 accuracy over 3,000 simulated windows equals the truth (a mean per-window error of 1 - truth would read
far lower, as t30b's own walk does: HumanEval+ 91.6 % at its check starts vs 97.9 % at every anchor).
  .venv/bin/python tools/certify98_power_strict.py [reps]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C
import certify98_power as P

BANDS = C.GATE                    # amendment 4's gate bands (0-2 merged)


def walk(e, n, rng):
    """one window of n anchors at error e: strict (num, den) per depth and the kept list"""
    num, den, kept, p = np.zeros(31), np.zeros(31), [], 0
    while p < n:
        avail = min(31, n - 1 - p)                                  # cells exist up to the window's end
        L = 0
        while L < avail and rng.random() >= e:
            L += 1
        for k in range(min(L + 1, 31)):                             # depths 0..L: L rights, then the miss or the end
            den[k] += 1
            if k < L:
                num[k] += 1
        kept.append(L + 1); p += L + 1
    return num, den, kept


def calibrate(E0, sizes, truth, rng, n=3000):
    """the scale at which the pooled walk-strict depth-0 accuracy equals the truth"""
    lo, hi = 0.0, (1 - truth) / E0.mean()
    for _ in range(18):
        mid = (lo + hi) / 2
        i = rng.integers(0, len(E0), n); R = rng.choice(sizes, n)
        Cn = [walk(min(1.0, E0[j] * mid), int(m), rng) for j, m in zip(i, R)]
        acc = sum(x[0][0] for x in Cn) / sum(x[1][0] for x in Cn)
        lo, hi = (mid, hi) if acc > truth else (lo, mid)
    return (lo + hi) / 2


def sim(E0, sizes, K, truth, line, kl, reps, rng, scale):
    ok_b, ok_k, ok_all = np.zeros(len(BANDS)), 0, 0
    for _ in range(reps):
        i = rng.integers(0, len(E0), K); R = rng.choice(sizes, K)
        Cn = [walk(min(1.0, E0[j] * scale), int(n), rng) for j, n in zip(i, R)]
        good = True
        for b, (lo, hi) in enumerate(BANDS):
            st = C.strict_stats([(x[0], x[1]) for x in Cn], lo, hi)
            ok = st["lcb"] >= line; ok_b[b] += ok; good &= ok
        y = np.array([sum(x[2]) for x in Cn], float); x_ = np.array([len(x[2]) for x in Cn], float)
        p, se = C.cr1(y, x_)
        kok = p - C.t_one(K - 1) * se >= kl; ok_k += kok
        ok_all += good and kok
    return ok_b / reps, ok_k / reps, ok_all / reps


def main(reps=200):
    rng = np.random.default_rng(15)
    P.load("heplus"); E, _ = P.band_table(); P.load("mbpp")
    E0 = E[~np.isnan(E).any(1)][:, 0]                                # HE+ per-window depth-0 error (t30b, 97.95 %)
    s = json.load(open(os.path.expanduser("~/BigMoE/data/deepdraft/bank/split.json")))
    B = os.path.expanduser("~/datasets/deepdraft/bank")
    sizes = np.array([np.load(os.path.join(B, f), mmap_mode="r")["wd"].shape[0] - (int(np.load(os.path.join(B, f), mmap_mode="r")["gen"]) - 1)
                      for f in s["heplus"]])
    print("truth flat at the check starts; HumanEval+ window sizes and per-window spread; lines: every strict band's 95 % "
          "lower bound >= the line, and kept/32's >= the line's flat kept")
    for truth in (0.98, 0.97):
        scale = calibrate(E0, sizes, truth, rng)
        print(f" truth {100 * truth:.1f} at the check starts: scale {scale:.3f} (mean per-window error {100 * scale * E0.mean():.2f} %)")
        for line in (97.0, 97.5):
            kl = C.kept_line(line)
            for K in (80, 140, 300):
                b, k, a = sim(E0, sizes, K, truth, line, kl, reps, rng, scale)
                print(f"  truth {100 * truth:.1f}, line {line} (kept {kl:.2f}), K {K}: P(all strict bands) {b.min():.2f} "
                      f"[0-2 {b[0]:.2f}, 3-6 {b[1]:.2f}, 23-30 {b[-1]:.2f}]; P(kept) {k:.2f}; P(both) {a:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
