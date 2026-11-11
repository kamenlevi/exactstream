"""M-21b STOPCLS (MoonShots, Mon 2026-11-02 22:4x; the orchestrator 22:5x after M-21's oracle; pre-registered in
~/NewIdeas/QUEUE.md M-21b before any run): a learned stop classifier for the check (the published form: DISCO's
continue-drafting classifier, SpecDec++'s acceptance head). TRIED: CALSTOP (calibrated per-depth keep model) was KILL in
substance; new here: the history features and tuning on the SSD-aware objective.
Model: logistic P(draft j accepted | drafts 1..j-1 accepted) on [1, logit cf_j, log cumprod cf, depth, depth^2, the last two
checks' kept, logit cf_j x depth] (the walks carry no top-2 margin), trained on the real uncut walks (checks_walk) of half
the windows; 2 folds by window, never by row. Rule: verify while the predicted survival S_j = prod p >= mu (adding row j+1
buys S_j expected tokens for one row's cost), mu tuned on the train fold. Widths 1-16 (today's block-16 drafter).
Beside it: t* (the cumprod-cf cut with its threshold re-tuned on the same folds: retuning vs learning) and the hindsight
oracle (verify exactly through the first wrong draft: the ceiling). Baseline: fixed 16 + t 0.3 (today).
Harness: bwidth45 / wide45 (coupled walks: right / wrong per position from the real uncut walk, per-domain acceptance rates
from the capture's own uncut walks where unknown, one uniform draw per position and seed; C 256 + M-17 sizes byte replay,
EXACT, windows chained per domain). Cost, SSD-aware: dense + GDN / attention per row + draft 6.3 + SSD MB at 12 GB/s + the
expert union at 174 GB/s (today) or x2.15 (NAX6). Tuning replays only the train fold's windows (their own chain).
Line: pooled code SSD-aware tok/s (today's kernels) on the test folds vs fixed 16 + t 0.3: >= +2 % PASS, < +1 % KILL, else
BETWEEN; chat (IFEval, multi) reported separately.
usage: [BigMoE venv] taskpolicy -b python stop45.py -> data/stop45.txt (under run_guarded.sh)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B
import laysz45 as LZ
import wide45 as W4
import bwidth45 as BW

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SEEDS = int(os.environ.get("STOP45_SEEDS", "2"))
DOMS = ("code", "ifeval", "multi")
MUS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
D16 = 6.3
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def rates_of(Zs):
    """per-depth acceptance of the capture's own uncut walks: P(draft d right | drafts 1..d-1 right)"""
    reach, acc = np.zeros(15), np.zeros(15)
    for z in Zs:
        for a, r, k in z["checks_walk"].tolist():
            for d in range(1, min(r - 1, 15) + 1):
                if k >= d:
                    reach[d - 1] += 1
                    acc[d - 1] += k > d
    return acc / np.maximum(reach, 1)


def feats(conf, a, J, rates, k1, k2):
    cf = conf[a:a + J].copy()
    nan = np.isnan(cf)
    cf[nan] = rates[:J][nan]
    cf = np.clip(cf, 1e-4, 1 - 1e-6)
    lg = np.log(cf / (1 - cf))
    d = np.arange(1, J + 1) / 16.0
    return np.stack([np.ones(J), lg, np.cumsum(np.log(cf)), d, d * d, np.full(J, k1 / 16), np.full(J, k2 / 16), lg * d], 1)


def train_set(Zs, rates):
    X, y = [], []
    for z in Zs:
        conf, ks = z["conf"].astype(float), [8, 8]
        for a, r, k in z["checks_walk"].tolist():
            J = min(r - 1, 15)
            if J > 0:
                F = feats(conf, a, J, rates, ks[-1], ks[-2])
                for d in range(1, J + 1):
                    if k >= d:
                        X.append(F[d - 1]); y.append(float(k > d))
            ks.append(k)
    return np.array(X), np.array(y)


def fit(X, y, l2=1e-2, it=30):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y) + l2 * w)
    return w


def walk(z, rule, rates, u, w=None, mu=0.3):
    conf, right, a, end, n = BW.prep(z)
    out, ks = [], [8, 8]
    while a < end:
        R_ = min(16, n - a)
        J = R_ - 1
        acc = lambda j: (right[a + j] == 1.0) if not np.isnan(right[a + j]) else u[a + j] < rates[j - 1]
        if J <= 0:
            rows = R_
        elif rule == "oracle":
            L = 0
            while L < J and acc(L + 1):
                L += 1
            rows = min(L + 1, R_)
        else:
            if rule == "t":
                c = conf[a:a + J]
                S = np.cumprod(np.where(np.isnan(c), rates[:J], c))
            else:
                S = np.cumprod(1 / (1 + np.exp(-np.clip(feats(conf, a, J, rates, ks[-1], ks[-2]) @ w, -30, 30))))
            below = np.nonzero(S < mu)[0]
            rows = 1 + int(below[0]) if len(below) else R_
        kept = rows
        for j in range(1, rows):
            if not acc(j):
                kept = j
                break
        out.append((a, rows, kept, rows == R_))
        ks.append(kept)
        a += kept
    return out


def evaluate(pool, ix, Z, PREW, POP, policy, rates, seeds):
    """one chain over windows ix (each with its own (rule, w, mu)): (checks, kept, rows, ms today, ms NAX6) summed
    over seeds, and per window (kept, ms today)"""
    tot, per = np.zeros(5), {i: np.zeros(2) for i in ix}
    for seed in range(seeds):
        wks = {i: walk(Z[i], *policy[i][:1], rates, np.random.default_rng(10_000 * seed + i).random(Z[i]["conf"].shape[0] + 1),
                       *policy[i][1:]) for i in ix}
        Qw = {i: W4.needs_of(Z[i], wks[i], W4.GATES["EXACT"]) for i in ix}
        jobs = [(key, B.SIZES["C_m" if key == "m" else "C_l"][l],
                 [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, c[key][l]) for t, c in enumerate(Qw[i])] for i in ix],
                 [POP[key][i][l] for i in ix]) for key in ("m", "l") for l in range(NL)]
        mbw = {i: 0.0 for i in ix}
        for (key, *_), miss in zip(jobs, pool.map(W4.replay_arr, jobs, chunksize=4)):
            for k, i in enumerate(ix):
                mbw[i] += miss[k] * (MR if key == "m" else LR) / 1e6
        for i in ix:
            ug = sum(sum(len(c["m"][l]) * MR + len(c["l"][l]) * LR for l in range(NL)) for c in Qw[i]) / 1e9
            fx = sum(W4.dense(x[1]) + 0.35 * x[1] + D16 for x in wks[i]) + mbw[i] / 12.0
            ms, ms6 = fx + ug * 1e3 / W4.GPU_GBS, fx + ug * 1e3 / (W4.GPU_GBS * W4.NAX6)
            kept = sum(x[2] for x in wks[i])
            tot += (len(wks[i]), kept, sum(x[1] for x in wks[i]), ms, ms6)
            per[i] += (kept, ms)
        del Qw, jobs
    return tot, per


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)                        # forked while the parent is still small
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in names],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in names]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    say(f"M-21b STOPCLS on Breakthrough's capture: {len(names)} windows, {SEEDS} seeds, mu grid {MUS}; SSD-aware tok/s (today / NAX6)")
    for dom in DOMS:
        ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
        rates = rates_of([Z[i] for i in ix])
        folds = (ix[0::2], ix[1::2])
        pol = {"t 0.3": {i: ("t", None, 0.3) for i in ix}, "t*": {}, "learned": {}, "oracle": {i: ("oracle", None, 0.0) for i in ix}}
        notes = []
        for f in (0, 1):
            tr, te = folds[f], folds[1 - f]
            w = fit(*train_set([Z[i] for i in tr], rates))
            for name, rule, wt in (("t*", "t", None), ("learned", "learned", w)):
                best = max(MUS, key=lambda mu: (lambda t: t[1] / t[3])(
                    evaluate(pool, tr, Z, PREW, POP, {i: (rule, wt, mu) for i in tr}, rates, 1)[0]))
                pol[name].update({i: (rule, wt, best) for i in te})
                notes.append(f"{name} fold {f} mu {best}")
        R = {k: evaluate(pool, ix, Z, PREW, POP, p, rates, SEEDS) for k, p in pol.items()}
        b = R["t 0.3"][0]
        say(f"  {dom} ({len(ix)} windows; acceptance rates from its own walks, mean {rates.mean():.3f}; {', '.join(notes)}):")
        for k, (t, per) in R.items():
            g, g6 = (t[1] / t[3]) / (b[1] / b[3]) - 1, (t[1] / t[4]) / (b[1] / b[4]) - 1
            say(f"    {k:8s} kept / check {t[1] / t[0]:5.2f}, rows / check {t[2] / t[0]:5.2f}, tok/s {1e3 * t[1] / t[3]:6.1f} / "
                f"{1e3 * t[1] / t[4]:6.1f}: {100 * g:+5.1f} % / {100 * g6:+5.1f} % vs t 0.3")
        g = (R["learned"][0][1] / R["learned"][0][3]) / (b[1] / b[3]) - 1
        if dom == "code":
            say(f"  code verdict (learned vs t 0.3, SSD-aware today): {100 * g:+.2f} % -> "
                + ("PASS" if g >= 0.02 else "KILL" if g < 0.01 else "BETWEEN"))
        per_l, per_b = R["learned"][1], R["t 0.3"][1]
        say("    per window (learned vs t 0.3): " + ", ".join(
            f"{names[i].split('_', 2)[2]} {100 * ((per_l[i][0] / per_l[i][1]) / (per_b[i][0] / per_b[i][1]) - 1):+.1f} %" for i in ix))
        say(f"    ({time.time() - t0:.0f} s)")
    pool.close(); pool.join()
    open(os.path.join(LZ.OUT, "stop45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
