"""M-21 BWIDTH (MoonShots, Mon 2026-11-02 22:3x; the orchestrator 22:5x from Fable A's ASSUMPTION_HUNT, mission 31; pre-registered
in ~/NewIdeas/QUEUE.md M-21 before any run): a branch predictor for the check width. Consecutive checks' accepted lengths
correlate 0.60 on code, so the outcome history (not the drafter's confidence) picks the next check's rows: 8 / 16 / 32.
Policies: fixed 16 (today), fixed 8 / 32 (context), 'last2' (8 after two short checks, 32 after two long ones, else 16),
'sat2' (2-bit saturating counter: start 1, long +1, short -1; 0 -> 8, 1-2 -> 16, 3 -> 32), 'oracle' (the best width per
check in hindsight under the flat cost, myopic: the ceiling). long = the check verified its full width with no cut and
kept every row; short = kept <= width / 4. The engine's t 0.3 cut stays on inside the chosen width.
Walks: wide45.py's machinery on Breakthrough's 12 code windows: right / wrong per position from the real uncut walk,
MTP Head's per-depth rates where unknown (base16 for 8 / 16-row checks, base32 for 32; variant match16 / match32), one
uniform draw per position and seed, shared by every width so the policies see the same luck.
Cost, flat (the orchestrator's): ms = M0 + 2.0 x verified rows + draft (block 8 / 16 / 32: 5.4 / 6.3 / 8.2 ms), M0 so that
fixed 16 = C50's 96.8 ms cycle. SSD-aware (beside it): M-18c's model on this replay's own bytes (C 256 + M-17 sizes, EXACT).
Line: pooled code tok/s of the better predictor vs fixed 16, flat model: >= +2 % PASS, < +0.5 % KILL, else BETWEEN.
usage: [BigMoE venv] taskpolicy -b python bwidth45.py -> data/bwidth45.txt (run under run_guarded.sh)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B
import laysz45 as LZ
import wide45 as W4

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SEEDS = int(os.environ.get("BWIDTH_SEEDS", "4"))
DRAFT = {8: 6.3 * 0.85, 16: 6.3, 32: 6.3 * 1.3}
CYC16, ROW = 96.8, 2.0
POLICIES = ("fixed16", "fixed8", "fixed32", "last2", "sat2", "oracle")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def prep(z):
    conf, ck = z["conf"].astype(float), z["checks_walk"].tolist()
    n = conf.shape[0]
    right = np.full(n + 1, np.nan)
    for a, r, k in ck:
        right[a + 1:a + k] = 1.0
        if k < r:
            right[a + k] = 0.0
    return conf, right, ck[0][0], min(ck[-1][0] + ck[-1][2], n), n


def check(a, R, conf, right, u, n, rates):
    """(rows, kept) of one check of width R at anchor a, the t 0.3 cut on, acceptance coupled to the real walk"""
    rate = rates[32 if R == 32 else 16]
    R_ = min(R, n - a)
    rows = R_
    if R_ > 1:
        c = np.where(np.isnan(conf[a:a + R_ - 1]), rate[:R_ - 1], conf[a:a + R_ - 1])
        below = np.nonzero(np.cumprod(c) < 0.3)[0]
        if len(below):
            rows = 1 + int(below[0])
    for j in range(1, rows):
        r = right[a + j]
        if not (r == 1.0 if not np.isnan(r) else u[a + j] < rate[j - 1]):
            return rows, j
    return rows, rows


def walk(z, pol, rates, u, m0):
    conf, right, a, end, n = prep(z)
    out, hist, s = [], [], 1
    while a < end:
        if pol == "oracle":
            best = max((8, 16, 32), key=lambda R: (lambda rk: rk[1] / (m0 + ROW * rk[0] + DRAFT[R]))(check(a, R, conf, right, u, n, rates)))
            R = best
        elif pol.startswith("fixed"):
            R = int(pol[5:])
        elif pol == "last2":
            R = 8 if hist[-2:] == [-1, -1] else 32 if hist[-2:] == [1, 1] else 16
        else:
            R = (8, 16, 16, 32)[s]
        rows, kept = check(a, R, conf, right, u, n, rates)
        full = rows == min(R, n - a)
        o = 1 if (full and kept == rows) else -1 if kept <= R // 4 else 0
        hist.append(o)
        s = min(3, s + 1) if o == 1 else max(0, s - 1) if o == -1 else s
        out.append((a, rows, kept, full, R))
        a += kept
    return out


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)                        # forked while the parent is still small
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    code = [w for w in names if w.startswith("tf_code")]
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in code],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in code]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in code]
    PREW = [W4.prefill_of(z) for z in Z]
    RATES = {"base": {16: W4.rates("base16"), 32: np.nan_to_num(W4.rates("base32"), nan=0.0)},
             "match": {16: W4.rates("match16"), 32: np.nan_to_num(W4.rates("match32"), nan=0.0)}}
    say(f"M-21 BWIDTH on Breakthrough's capture: {len(code)} code windows, {SEEDS} seeds; flat cost: M0 + {ROW} ms x rows + "
        f"draft {DRAFT}; fixed 16 = {CYC16} ms")
    for var, rates in RATES.items():
        T = {p: np.zeros(6) for p in POLICIES}                   # checks, kept, rows, flat ms, SSD-aware ms (today), (NAX6)
        PW = {p: np.zeros((len(code), 2)) for p in POLICIES}      # per window: kept, flat ms
        wid = {p: {8: 0, 16: 0, 32: 0} for p in POLICIES}
        for seed in range(SEEDS):
            U = [np.random.default_rng(10_000 * seed + k).random(z["conf"].shape[0] + 1) for k, z in enumerate(Z)]
            base = [walk(z, "fixed16", rates, U[k], 0.0) for k, z in enumerate(Z)]
            r16 = sum(x[1] for wk in base for x in wk) / sum(len(wk) for wk in base)
            m0 = CYC16 - ROW * r16 - DRAFT[16]                   # fixed 16's pooled modeled cycle = C50's 96.8 ms
            for p in POLICIES:
                wks = base if p == "fixed16" else [walk(z, p, rates, U[k], m0) for k, z in enumerate(Z)]
                Qw = [W4.needs_of(z, [x[:4] for x in wks[k]], W4.GATES["EXACT"]) for k, z in enumerate(Z)]
                jobs = [(key, B.SIZES["C_m" if key == "m" else "C_l"][l],
                         [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[k]] + [(t, c[key][l]) for t, c in enumerate(Qw[k])]
                          for k in range(len(code))], [POP[key][k][l] for k in range(len(code))]) for key in ("m", "l") for l in range(NL)]
                mb = 0.0
                for (key, *_), miss in zip(jobs, pool.map(W4.replay_arr, jobs, chunksize=4)):
                    mb += miss.sum() * (MR if key == "m" else LR) / 1e6
                U_gb = sum(sum(len(c["m"][l]) * MR + len(c["l"][l]) * LR for l in range(NL)) for q in Qw for c in q) / 1e9
                ck = sum(len(wk) for wk in wks)
                kept = sum(x[2] for wk in wks for x in wk)
                rows = sum(x[1] for wk in wks for x in wk)
                flat = sum(m0 + ROW * x[1] + DRAFT[x[4]] for wk in wks for x in wk)
                fixed = sum(W4.dense(x[1]) + 0.35 * x[1] + DRAFT[x[4]] for wk in wks for x in wk) + mb / 12.0
                T[p] += (ck, kept, rows, flat, fixed + U_gb * 1e3 / W4.GPU_GBS, fixed + U_gb * 1e3 / (W4.GPU_GBS * W4.NAX6))
                for k, wk in enumerate(wks):
                    PW[p][k] += (sum(x[2] for x in wk), sum(m0 + ROW * x[1] + DRAFT[x[4]] for x in wk))
                for x in (y for wk in wks for y in wk):
                    wid[p][x[4]] += 1
                del Qw, jobs
            say(f"  [{var}] seed {seed} done ({time.time() - t0:.0f} s)")
        say("")
        say(f"  rates {var}: policy | kept / check | rows / check | widths 8 / 16 / 32 (% of checks) | tok/s flat | SSD-aware today / NAX6 "
            f"| vs fixed 16 (flat, SSD today, SSD NAX6)")
        b = T["fixed16"]
        for p in POLICIES:
            t = T[p]
            n = sum(wid[p].values())
            tf, ts, tn6 = (1e3 * t[1] / t[i] for i in (3, 4, 5))
            bf, bs, bn6 = (1e3 * b[1] / b[i] for i in (3, 4, 5))
            say(f"  {p:8s} | {t[1] / t[0]:6.2f} | {t[2] / t[0]:6.2f} | {100 * wid[p][8] / n:4.0f} / {100 * wid[p][16] / n:4.0f} / "
                f"{100 * wid[p][32] / n:4.0f} | {tf:6.1f} | {ts:6.1f} / {tn6:6.1f} | {100 * (tf / bf - 1):+5.1f} %, {100 * (ts / bs - 1):+5.1f} %, "
                f"{100 * (tn6 / bn6 - 1):+5.1f} %")
        best = max(("last2", "sat2"), key=lambda p: T[p][1] / T[p][3])
        g = (T[best][1] / T[best][3]) / (b[1] / b[3]) - 1
        say(f"  [{var}] better predictor {best}: {100 * g:+.2f} % flat -> " + ("PASS" if g >= 0.02 else "KILL" if g < 0.005 else "BETWEEN")
            + f" (oracle ceiling {100 * ((T['oracle'][1] / T['oracle'][3]) / (b[1] / b[3]) - 1):+.1f} %)")
        say("  per window, flat tok/s vs fixed 16: " + ", ".join(
            f"{w[13:]} {100 * ((PW[best][k][0] / PW[best][k][1]) / (PW['fixed16'][k][0] / PW['fixed16'][k][1]) - 1):+.1f} %"
            for k, w in enumerate(code)))
        say("")
    pool.close(); pool.join()
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "bwidth45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
