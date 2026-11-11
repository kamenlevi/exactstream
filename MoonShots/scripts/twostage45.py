"""M-22 TWOSTAGE (MoonShots, Mon 2026-11-02 23:0x; the orchestrator 23:0x: Fable C's C1-3, ~/FableGod/fablec/INVENT.md;
pre-registered in ~/NewIdeas/QUEUE.md M-22 before any run): the two-stage check. Rows 0..s-1 run as a normal check + commit;
the tail s..n-1 runs from the committed state only if all s are accepted. s = 1 + the first depth where cumprod(cf) < P*,
inside today's t 0.3 chain of n rows. Kept is unchanged by construction (a rejected prefix means a rejected tail): only the
cost moves. Exact by REF2-T's batch invariance (Fable C).
Harness: M-21b's (stop45 walks with the t 0.3 rule, coupled acceptance, per-domain rates; Breakthrough's 36 windows, chains
per domain), byte replay per pass (band45's LRU + 30 x popularity, strict) at C 256 and C 320 (M-17 per-layer sizes).
Cost per pass: F + rows x 0.61 ms (dense 0.264 + GDN / attention 0.35) + the pass's SSD MB / 12 GB/s + its expert union at
174 GB/s; draft 6.3 ms once per check. F = 13.73 ms (D32) for a check's first pass; the second pass costs F2. Scenarios
(Fable C's): 'today' F2 34 ms, P* 0.4; '17 ms dense' F2 17 ms, P* 0.6. Max form: F + max(SSD, union + rows); add form:
F + SSD + union + rows.
Line (Fable C's): modelled code cycle (tok/s at equal kept) vs the one-pass check, max form: >= +5 % PASS, < +2 % KILL, else
BETWEEN, per scenario; chat (IFEval, multi) separately.
usage: [BigMoE venv] taskpolicy -b python twostage45.py -> data/twostage45.txt (under run_guarded.sh)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B
import laysz45 as LZ
import wide45 as W4
import stop45 as S45

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SEEDS = int(os.environ.get("TWOSTAGE_SEEDS", "2"))
F1, ROWMS, D16, SSD_GBS = 13.73, 0.2644 + 0.35, 6.3, 12.0
SCEN = {"today": (34.0, 0.4), "17 ms dense": (17.0, 0.6)}
SIZES = {256: B.SIZES, 320: json.load(open(os.path.join(LZ.OUT, "laysz45_320.json")))}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay_calls(task):
    """band45.replay's pool (LRU + 30 x popularity, strict) returning the misses of every counted call, per window"""
    key, K, calls, pops = task
    while os.path.exists(LZ.PAUSE):
        time.sleep(20)
    out, st, i = [], {}, 0
    for p, cl in enumerate(calls):
        pop, mw = pops[p], []
        for t, need in cl:
            need = frozenset(need.tolist())
            i += 1
            if t >= 0:
                mw.append(sum(1 for e in need if e not in st))
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + 30.0 * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                    del st[e]
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                        del st[e]
        out.append(np.array(mw, dtype=np.int32))
    return out


def passes(z, wk, pstar, rates):
    """per check: its passes as (r0, r1) row ranges; pstar None = one pass"""
    conf = z["conf"].astype(float)
    out = []
    for a, rows, kept, full in wk:
        s = rows
        if pstar is not None and rows > 1:
            c = conf[a:a + rows - 1]
            below = np.nonzero(np.cumprod(np.where(np.isnan(c), rates[:rows - 1], c)) < pstar)[0]
            if len(below):
                s = 1 + int(below[0])
        out.append([(0, rows)] if s >= rows else [(0, s)] + ([(s, rows)] if kept >= s + 1 else []))
    return out


def pass_needs(z, wk, P):
    ids = z["ids"]
    return [[{"m": [np.unique(ids[a + r0:a + r1, l]).astype(np.int16) for l in range(NL)],
              "l": [np.unique(ids[a + r0:a + r1, l, :2]).astype(np.int16) for l in range(NL)], "rows": r1 - r0}
             for r0, r1 in ps] for (a, *_), ps in zip(wk, P)]


def cost(pool, ix, Z, PREW, POP, WK, Pw, C):
    """(kept, ms max form, ms add form) over windows ix; the second pass of a check takes F2 (summed separately)"""
    Q = {i: pass_needs(Z[i], WK[i], Pw[i]) for i in ix}
    flat = {i: [p for ck in Q[i] for p in ck] for i in ix}
    jobs = [(key, SIZES[C]["C_m" if key == "m" else "C_l"][l],
             [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, p[key][l]) for t, p in enumerate(flat[i])] for i in ix],
             [POP[key][i][l] for i in ix]) for key in ("m", "l") for l in range(NL)]
    mb = {i: np.zeros(len(flat[i])) for i in ix}
    for (key, *_), res in zip(jobs, pool.map(replay_calls, jobs, chunksize=4)):
        for k, i in enumerate(ix):
            mb[i] += res[k] * (MR if key == "m" else LR) / 1e6
    out = {}
    for i in ix:
        mx_, ad_, n2, j = 0.0, 0.0, 0, 0
        for ck in Q[i]:
            for q, p in enumerate(ck):
                ssd = mb[i][j] / SSD_GBS
                un = sum(len(p["m"][l]) * MR + len(p["l"][l]) * LR for l in range(NL)) / 1e9 * 1e3 / W4.GPU_GBS
                rw = ROWMS * p["rows"]
                f = F1 if q == 0 else 0.0
                mx_ += f + max(ssd, un + rw)
                ad_ += f + ssd + un + rw
                n2 += q
                j += 1
            mx_ += D16; ad_ += D16
        out[i] = (sum(x[2] for x in WK[i]), mx_, ad_, n2)
    return out


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in names],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in names]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    say(f"M-22 TWOSTAGE on Breakthrough's capture: {len(names)} windows, {SEEDS} seeds; scenarios {SCEN}; tok/s gain at equal "
        f"kept vs the one-pass check (max form / add form)")
    for dom in ("code", "ifeval", "multi"):
        ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
        rates = S45.rates_of([Z[i] for i in ix])
        for Cs in (256, 320):
            R = {}
            for seed in range(SEEDS):
                WK = {i: S45.walk(Z[i], "t", rates, np.random.default_rng(10_000 * seed + i).random(Z[i]["conf"].shape[0] + 1),
                                  None, 0.3) for i in ix}
                for name, pst in [("one pass", None)] + [(k, v[1]) for k, v in SCEN.items()]:
                    Pw = {i: passes(Z[i], WK[i], pst, rates) for i in ix}
                    for i, v in cost(pool, ix, Z, PREW, POP, WK, Pw, Cs).items():
                        R.setdefault(name, {}).setdefault(i, np.zeros(4))
                        R[name][i] += v
            base = sum(R["one pass"].values())
            say(f"  {dom} C {Cs}: one pass kept {base[0]:.0f}, {1e3 * base[0] / base[1]:.1f} tok/s (max) / {1e3 * base[0] / base[2]:.1f} (add)")
            for name, (F2, pst) in SCEN.items():
                t = sum(R[name].values())
                gm = (base[1]) / (t[1] + F2 * t[3]) - 1
                ga = (base[2]) / (t[2] + F2 * t[3]) - 1
                verdict = (" -> " + ("PASS" if gm >= 0.05 else "KILL" if gm < 0.02 else "BETWEEN")) if dom == "code" else ""
                say(f"    {name:12s} (F2 {F2:.0f} ms, P* {pst}): second passes {t[3]:.0f}; gain max form {100 * gm:+.1f} %, add form "
                    f"{100 * ga:+.1f} %{verdict}")
        say(f"    ({time.time() - t0:.0f} s)")
    pool.close(); pool.join()
    open(os.path.join(LZ.OUT, "twostage45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
