"""M-18d Q1-24 (MoonShots, Mon 2026-11-02 23:0x; pre-registered by NewIdeas 22:00, QUEUE M-18d before any run): does Q1
(top-2 experts on drafter-sure verify rows, cf >= 0.99) cut the 24 GB line's SSD bytes as much as at 45 GB?
24 GB default (BigMoE scripts/fast24.py, ext/bigmoe_ext.cpp): plan4433333333 (= v3: ranks 0-1 MSB + LSB sidecar, 2-9 MSB),
C_m 147 + C_l 46 per layer, decayed LFU eviction (BIGMOE_EVICT_LFU=0.99: per gated call the weight grows by 1 / 0.99 and
every needed expert's count gains it; the victim is the lowest count not needed now), narrow MTP verify passes.
Option A (NewIdeas to confirm; MTP per-row confidences don't exist for these windows): Breakthrough's 12 code windows,
4-row checks (anchor + 3 drafts), acceptance coupled to the real walk's right / wrong (DFlash2's: conservative for MTP,
whose depth-1 is right 96.8 % on code), the Q1 gate on DFlash2's cf as the MTP stand-in.
Arms: v3 exact verify, and b16 (bigmoe/budget.py: per pass and layer, the 16 experts with the largest summed w^2 over the
rows' top-k weights, each row keeping its experts inside that set; the capture has no gate weights, so w ~ 1 / (rank + 1)
normalised over the top-10, a flagged stand-in). Each with and without Q1 (sure rows top-2, inside the set for b16).
Line (NewIdeas'): code SSD MB per kept token, Q1 vs the same replay without Q1: <= x0.70 PASS, > x0.90 KILL, else BETWEEN.
usage: [BigMoE venv] taskpolicy -b python q1_24.py -> data/q1_24.txt (under run_guarded.sh)"""
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
import stop45 as S45

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
CM, CL, DECAY, ROWS, BUDGET = 147, 46, 0.99, 4, 16
SEEDS = int(os.environ.get("Q124_SEEDS", "4"))
WR = 1 / np.arange(1, 11)
WR = WR / WR.sum()                                               # rank -> stand-in gate weight
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


MTP = os.environ.get("Q124_GATE") == "mtp"                      # option B: MTP Head's per-anchor export (Tue 06:57)
MTPDIR = os.path.expanduser(os.environ.get("Q124_MTPDIR", "~/TrainLab/data/mtp_map_code_v2"))   # v1 was one row late (MTP Head 15:37)


def walk4_mtp(z, m):
    """option B: 4-row checks walked with the native MTP chain's own right / wrong per depth (teacher-forced, equal to the
    engine's chain up to its first miss); each check carries its rows' gate = the chain's top-1 probability per depth"""
    ck = z["checks_walk"].tolist()
    n = z["conf"].shape[0]
    a, end, out = ck[0][0], min(ck[-1][0] + ck[-1][2], n), []
    while a < end:
        rows = min(ROWS, n - a)
        kept = rows
        for d in range(1, rows):
            if m["right"][a, d - 1] != 1:
                kept = d
                break
        out.append((a, rows, kept, True, np.r_[m["q"][a, :rows - 1].astype(float), np.nan]))
        a += kept
    return out


def walk4(z, rates, u):
    conf, right, a, end, n = BW.prep(z)
    out = []
    while a < end:
        rows = min(ROWS, n - a)
        kept = rows
        for j in range(1, rows):
            r = right[a + j]
            if not (r == 1.0 if not np.isnan(r) else u[a + j] < rates[j - 1]):
                kept = j
                break
        out.append((a, rows, kept, True))
        a += kept
    return out


def needs4(z, wk, q1, budget):
    """per check: base / sidecar needs per layer (int16 arrays); q1 = sure rows top-2; budget = the b16 pass set"""
    ids, conf = z["ids"], z["conf"].astype(float)
    out = []
    for a, rows, kept, full, *g in wk:
        cf = g[0] if g else conf[a:a + rows].copy()              # option B: the MTP chain's probabilities
        cf[rows - 1] = np.nan                                    # the last row of a pass verifies nothing
        k = np.array([2 if (q1 and c >= 0.99) else 10 for c in cf])
        rid = ids[a:a + rows]
        m, s = [], []
        for l in range(NL):
            r_ = rid[:, l]
            use = np.arange(10)[None, :] < k[:, None]
            if budget:
                e, w2 = r_[use], np.broadcast_to(WR, r_.shape)[use] ** 2
                tot = {}
                for x, v in zip(e.tolist(), w2.tolist()):
                    tot[x] = tot.get(x, 0.0) + v
                keep = set(sorted(tot, key=tot.get, reverse=True)[:budget])
                use = use & np.isin(r_, list(keep))
            m.append(np.unique(r_[use]).astype(np.int16))
            s.append(np.unique(r_[:, :2][use[:, :2]]).astype(np.int16))
        out.append({"m": m, "l": s, "kept": kept, "rows": rows})
    return out


def replay_lfu(task):
    """decayed-LFU pool (BigMoE's BIGMOE_EVICT_LFU=d) for one layer and key: misses per window; -1 calls uncounted"""
    K, calls = task
    while os.path.exists(LZ.PAUSE):
        time.sleep(20)
    miss = np.zeros(len(calls))
    cnt, w, st = np.zeros(512), 1.0, set()
    for p, cl in enumerate(calls):
        for t, need in cl:
            need = set(need.tolist())
            w /= DECAY
            for e in need:
                cnt[e] += w
            if t >= 0:
                miss[p] += len(need - st)
            st |= need
            if len(st) > K:
                for e in sorted((e for e in st if e not in need), key=lambda e: cnt[e])[:len(st) - K]:
                    st.discard(e)
            if w > 1e200:
                cnt /= w; w = 1.0
    return miss


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.startswith("tf_code"))
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    rates = S45.rates_of(Z)
    say(f"M-18d Q1-24 (option {'B' if MTP else 'A'}) on Breakthrough's {len(names)} code windows: {ROWS}-row checks, C_m {CM} + C_l {CL}, decayed LFU "
        f"{DECAY}, {SEEDS} seeds; acceptance rates from the walks (depths 1-3: {rates[0]:.3f} {rates[1]:.3f} {rates[2]:.3f})")
    if MTP:
        MD = [dict(np.load(os.path.join(MTPDIR, w + ".npz"))) for w in names]
        say(f"  option B: the native MTP chain's own acceptance and top-1 probabilities (MTP Head's export, {MTPDIR}); one walk")
    T = {}
    for seed in range(1 if MTP else SEEDS):
        wks = ([walk4_mtp(z, MD[k]) for k, z in enumerate(Z)] if MTP else
               [walk4(z, rates, np.random.default_rng(10_000 * seed + k).random(z["conf"].shape[0] + 1)) for k, z in enumerate(Z)])
        for arm, budget in (("v3", 0), ("b16", BUDGET)):
            for q1 in (False, True):
                Qw = [needs4(z, wks[k], q1, budget) for k, z in enumerate(Z)]
                jobs = [(CM if key == "m" else CL, [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[k]] +
                                                   [(t, c[key][l]) for t, c in enumerate(Qw[k])] for k in range(len(Z))])
                        for key in ("m", "l") for l in range(NL)]
                mb = 0.0
                for (K, _), (key, l), miss in zip(jobs, [(k_, l) for k_ in ("m", "l") for l in range(NL)], pool.map(replay_lfu, jobs, chunksize=4)):
                    mb += miss.sum() * (MR if key == "m" else LR) / 1e6
                kept = sum(c["kept"] for q in Qw for c in q)
                ck = sum(len(q) for q in Qw)
                T.setdefault((arm, q1), np.zeros(3))
                T[(arm, q1)] += (mb, kept, ck)
                del Qw, jobs
        say(f"  seed {seed} done ({time.time() - t0:.0f} s)")
    say("  arm | rule | SSD MB per kept token | kept per check | Q1 / no-Q1")
    for arm in ("v3", "b16"):
        b = T[(arm, False)][0] / T[(arm, False)][1]
        for q1 in (False, True):
            mb, kept, ck = T[(arm, q1)]
            r = (mb / kept) / b
            say(f"  {arm:4s} | {'Q1 0.99' if q1 else 'no Q1':7s} | {mb / kept:6.1f} | {kept / ck:5.2f} | x{r:.3f}"
                + (" -> " + ("PASS" if r <= 0.70 else "KILL" if r > 0.90 else "BETWEEN") if q1 else ""))
    pool.close(); pool.join()
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, ("q1_24_mtp_" + os.path.basename(MTPDIR) + ".txt") if MTP else "q1_24.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
