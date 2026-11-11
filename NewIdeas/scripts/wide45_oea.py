"""NewIdeas copy (Tue 3 Nov 06:5x, U-32 pre-registered in ~/NewIdeas/QUEUE.md): MoonShots' wide45.py with the union-stack
rules (sure rows k0 2, the rest k0 8 plus the check's union; optionally 3-bit sure rows) and arms 16 cut / 32 cut.
Reads MoonShots' / Breakthrough's / TrainLab's data; writes only ~/NewIdeas/data/band45_oea/wide45_oea.txt.
Original docstring follows.
M-18c WIDE-Q1 (MoonShots, Mon 2026-11-02 21:2x; pre-registered by NewIdeas 20:48 in ~/NewIdeas/QUEUE.md M-18c, before any
run): does Q1 (top-2 experts on drafter-sure rows, DFlash2 cf >= 0.99) make 32-row checks pay on code at the 45 GB route
(C 256 + the M-17 per-layer sizes)? D32: dense x1.24 for 32 rows vs 16; NAX6 pays only at wide checks.
Walks: Breakthrough's DAY9 capture, its 12 HumanEval windows (teacher-forced rank-sorted routing for every position, the
cf of the draft each row verifies, the real 16-row walk after the engine's t 0.3 cut). Each arm re-walks a window from its
first check: blocks of R rows (16 or 32), the t 0.3 cut on the cf ('cut') or none (MATCH's always-R, 'always'); draft j is
accepted iff the real walk's draft at that position was right (wrong -> rejected); positions the real walk never drafted
(its anchors) draw from MTP Head's per-depth code rates (~/TrainLab/data/d02b_walk_m_base16.txt for 16 rows, _match32 for
32: MATCH's 32-row walk, HumanEval+ 12.35 kept vs base16 9.57). Right / wrong per position comes from the capture's uncut
walk (checks_walk; its anchors follow the uncut acceptance), so '16 always' reproduces it exactly (checked below). Gate of
row j = the capture's cf at that position (NaN = exact; the last row of a full block has no draft); 'deep sure' also takes
rows 16-31 at top-2 (NewIdeas: MATCH-copied rows treated as sure). Bytes: band45's chained replay (LRU + 30 x popularity,
one leave-one-window-out prior from the capture's EXACT needs, strict, prompts as uncounted prefill).
Cost per check, ms: SSD MB at 12 GB/s (fully exposed); expert compute = the check's expert union at 174 GB/s (today's kernels,
M-16's beta) or with NAX6 (B3: 1.41 vs 3.03 ms per added row, x2.15); dense 13.73 + 0.264 x rows (D32: 17.96 / 22.19 ms at
16 / 32); GDN / attention 0.35 x rows (B3); draft 6.3 ms for block 16 (D7b), block 32 x1.3 (8.2 ms, MoonShots' estimate).
Implied tok/s = kept per check / modeled ms (no other fixed cost: the 16-row model lands at ~100 ms, C50 measures 94-187).
Line (NewIdeas): code implied steady tok/s, 32 rows + Q1 vs 16 rows + Q1: >= +10 % PASS, <= +3 % KILL, else BETWEEN.
usage: [BigMoE venv] taskpolicy -b python wide45.py -> data/wide45.txt (WIDE45_SEEDS=1 for the peak smoke run)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
MYOUT = os.path.expanduser("~/NewIdeas/data/band45_oea")
import band45 as B
import laysz45 as LZ

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SEEDS = int(os.environ.get("WIDE45_SEEDS", "4"))               # 1 = the smoke run that measures the peak
TL = os.path.expanduser("~/TrainLab/data")
GPU_GBS, NAX6 = 174.0, 3.03 / 1.41
DRAFT = {16: 6.3, 32: 6.3 * 1.3}
ARMS = (("16 cut", 16, "base16", True), ("32 cut", 32, "match32", True))
GATES = {"EXACT": lambda j, c: (10, 2), "Q1 0.9": lambda j, c: (2, 2) if c >= 0.9 else (10, 2),
         "Q1 0.9 3b": lambda j, c: (2, 0) if c >= 0.9 else (10, 2), "HYB 0.9 u8": lambda j, c: (2, 2) if c >= 0.9 else (8, 2), "HYB 0.9 u8 3b": lambda j, c: (2, 0) if c >= 0.9 else (8, 2),
         "HYB 0.9 u8 3b deep sure": lambda j, c: (2, 0) if (c >= 0.9 or j >= 16) else (8, 2)}
lines = []
dense = lambda r: 13.73 + 0.2644 * r                      # D32: 17.96 / 22.19 ms at 16 / 32 rows


def say(s):
    print(s, flush=True); lines.append(s)


def rates(arm):
    """MTP Head's measured per-depth acceptance on code (P(draft d right | drafts 1..d-1 right))"""
    txt = open(os.path.join(TL, f"d02b_walk_m_{arm}.txt")).read().split("\n")
    i = next(k for k, s in enumerate(txt) if s.strip().startswith("code:"))
    return np.array([float(x) for x in txt[i + 1].split(":")[1].split()])


def walk(z, R, rate, cut, rng):
    """[(anchor row, rows verified, kept, full block)] for one window"""
    conf, ck = z["conf"].astype(float), z["checks_walk"].tolist()            # the real walk, uncut: (anchor, rows, kept)
    n = conf.shape[0]
    right = np.full(n + 1, np.nan)
    for a, r, k in ck:
        right[a + 1:a + k] = 1.0
        if k < r:
            right[a + k] = 0.0
    a, end, out = ck[0][0], min(ck[-1][0] + ck[-1][2], n), []      # min(): past the last routing row rows = 0 and
    while a < end:                                                    # the walk looped forever (Mon 21:43 / 22:12 blow-ups)
        R_ = min(R, n - a)
        rows = R_
        if cut and R_ > 1:
            c = np.where(np.isnan(conf[a:a + R_ - 1]), rate[:R_ - 1], conf[a:a + R_ - 1])
            below = np.nonzero(np.cumprod(c) < 0.3)[0]
            if len(below):
                rows = 1 + int(below[0])
        kept = rows
        for j in range(1, rows):
            r = right[a + j]
            if not (r == 1.0 if not np.isnan(r) else rng.random() < rate[j - 1]):
                kept = j
                break
        out.append((a, rows, kept, rows == R_))
        a += kept
    return out


def needs_of(z, wk, gate):
    """per check: base / sidecar needs per layer as compact int16 id arrays (sets are built per call in the workers;
    Mon 21:43 frozensets for all combos at once reached ~122 GB)"""
    ids, conf = z["ids"], z["conf"].astype(float)
    out, col = [], np.arange(10)[None, :]
    for a, rows, kept, full in wk:
        cf = conf[a:a + rows].copy()
        if full:
            cf[rows - 1] = np.nan
        kr = np.array([gate(j, cf[j]) for j in range(rows)])
        km, kl = col < kr[:, :1], col < kr[:, 1:2]
        rid = ids[a:a + rows]
        out.append({"m": [np.unique(rid[:, l][km]).astype(np.int16) for l in range(NL)],
                    "l": [np.unique(rid[:, l][kl]).astype(np.int16) for l in range(NL)], "kept": kept, "rows": rows})
    return out


def counts_of(z):
    """laysz45.popularity's inputs from one window's EXACT needs (cut rows, top-10 / top-2): checks using each expert"""
    ids = z["ids"]
    cm, cl, lay = np.zeros((NL, 512)), np.zeros((NL, 512)), np.arange(NL)[None, :, None]
    for a, n, k in z["checks"].tolist():
        rid = ids[a:a + n]                                       # a window's last check can run past its routing rows
        for c, r in ((cm, 10), (cl, 2)):
            m = np.zeros((NL, 512), bool)
            m[np.broadcast_to(lay, (len(rid), NL, r)), rid[:, :, :r]] = True
            c += m
    return cm, cl, len(z["checks"])


def prefill_of(z):
    ids, gen = z["ids"], int(z["gen"])
    return [([np.unique(ids[i:min(i + 128, gen - 1), l]).astype(np.int16) for l in range(NL)],
             [np.unique(ids[i:min(i + 128, gen - 1), l, :2]).astype(np.int16) for l in range(NL)]) for i in range(0, gen - 1, 128)]


def replay_arr(task):
    key, K, calls, pops = task
    return B.replay((key, K, [[(t, frozenset(need.tolist())) for t, need in cl] for cl in calls], pops))


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)                        # forked while the parent is still small
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {}
    for w in names:
        C[w] = counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz")))
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    W = [(w,) for w in names if w.startswith("tf_code")]
    wi = list(range(len(W)))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for (w,) in W],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for (w,) in W]}
    Z = {i: {k: v for k, v in np.load(os.path.join(B.CAPDIR, W[i][0] + ".npz")).items()} for i in wi}
    W = [(w, w) for (w,) in W]                                   # (domain slot, name) as band45's tuples index [1]
    PREW = [prefill_of(Z[i]) for i in wi]
    R8 = {a: rates(a) for a in ("base16", "match32")}
    say(f"M-18c WIDE-Q1 on Breakthrough's capture: {len(wi)} code windows; MATCH code rates base16 (15 depths, mean "
        f"{R8['base16'].mean():.3f}), match32 (31 depths, mean {R8['match32'].mean():.3f})")
    same = tot = 0
    for i in wi:
        wk = walk(Z[i], 16, R8["base16"], False, np.random.default_rng(0))
        real = [tuple(x) for x in Z[i]["checks_walk"].tolist()]
        same += sum(1 for x, y in zip(wk, real) if x[:3] == y); tot += len(real)
    say(f"  check: '16 always' reproduces the real (uncut) walk's checks (anchor, rows, kept): {same} / {tot}")
    ST, MB, nrep = {}, {}, 0
    for seed in range(SEEDS):                                    # one (seed, arm, rule) combo in memory at a time
        for name, R, ra, cut in ARMS:
            wks = {i: walk(Z[i], R, R8[ra], cut, np.random.default_rng(1000 * seed + R + cut)) for i in wi}
            for g, gate in GATES.items():
                if g.endswith("deep sure") and R == 16:
                    continue
                Qw = [needs_of(Z[i], wks[i], gate) for i in wi]
                for k, i in enumerate(wi):
                    U = sum(sum(len(c["m"][l]) * MR + len(c["l"][l]) * LR for l in range(NL)) for c in Qw[k])
                    st = ST.setdefault((name, g, W[i][1]), np.zeros(5))
                    st += (len(Qw[k]), sum(c["kept"] for c in Qw[k]), sum(c["rows"] for c in Qw[k]), U, 0)
                jobs = [(key, B.SIZES["C_m" if key == "m" else "C_l"][l],
                         [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[k]] + [(t, c[key][l]) for t, c in enumerate(Qw[k])]
                          for k in wi], [POP[key][k][l] for k in wi]) for key in ("m", "l") for l in range(NL)]
                for (key, *_), miss in zip(jobs, pool.map(replay_arr, jobs, chunksize=4)):
                    for k, i in enumerate(wi):
                        MB[(name, g, W[i][1])] = MB.get((name, g, W[i][1]), 0.0) + miss[k] * (MR if key == "m" else LR) / 1e6
                nrep += len(jobs)
                del Qw, jobs
        say(f"  seed {seed} done: {nrep} replays, {time.time() - t0:.0f} s")
    pool.close(); pool.join()

    def stats(name, g, wins):
        ck, kept, rows, U, _ = sum(ST[(name, g, w)] for w in wins)
        mb = sum(MB[(name, g, w)] for w in wins)
        R = 32 if name.startswith("32") else 16
        base = dense(rows / ck) + 0.35 * rows / ck + DRAFT[R] + mb / ck / 12.0
        ms = {f: base + U / ck / 1e9 * 1e3 / (GPU_GBS * f) for f in (1.0, NAX6)}
        return kept / ck, rows / ck, mb / kept, U / ck / 1e9, ms

    wins =[W[i][1] for i in wi]
    say("")
    say("  arm        | rule              | kept / check | rows / check | SSD MB / kept | union GB / check | modeled ms / check "
        "(today / NAX6) | tok/s (today / NAX6)")
    T = {}
    for name, R, _, _ in ARMS:
        for g in GATES:
            if (name, g, wins[0]) not in ST:
                continue
            k, r, mbk, u, ms = stats(name, g, wins)
            T[(name, g)] = {f: 1e3 * k / ms[f] for f in ms}
            say(f"  {name:<10} | {g:<17} | {k:6.2f} | {r:6.2f} | {mbk:6.1f} | {u:5.2f} | {ms[1.0]:6.1f} / {ms[NAX6]:6.1f} | "
                f"{T[(name, g)][1.0]:6.1f} / {T[(name, g)][NAX6]:6.1f}")
    say("")
    say("  32 rows vs 16 rows, same rule, pooled code (tok/s ratio today / NAX6):")
    for g in GATES:
        for a32, a16 in (("32 cut", "16 cut"), ("32 always", "16 always"), ("32 always", "16 cut")):
            g16 = g if (a16, g) in T else g.replace(" deep sure", "")
            if (a32, g) in T:
                say(f"    {g:<17} {a32:<9} vs {a16:<9} ({g16}): " + " / ".join(f"{100 * (T[(a32, g)][f] / T[(a16, g16)][f] - 1):+.1f} %"
                                                                          for f in (1.0, NAX6)))
    say("")
    say("  per window, HYB 0.9 u8 3b, '32 cut' vs '16 cut' (kept 16 -> 32, tok/s ratio today / NAX6); 'deep sure' NAX6 last:")
    rat = []
    for w in wins:
        k16, _, _, _, m16 = stats("16 cut", "HYB 0.9 u8 3b", [w])
        k32, _, _, _, m32 = stats("32 cut", "HYB 0.9 u8 3b", [w])
        _, _, _, _, m32d = stats("32 cut", "HYB 0.9 u8 3b deep sure", [w])
        k32d = stats("32 cut", "HYB 0.9 u8 3b deep sure", [w])[0]
        r = [k32 / m32[f] / (k16 / m16[f]) - 1 for f in (1.0, NAX6)] + [k32d / m32d[NAX6] / (k16 / m16[NAX6]) - 1]
        rat.append(r)
        say(f"    {w:<24} kept {k16:5.2f} -> {k32:5.2f}: {100 * r[0]:+6.1f} % / {100 * r[1]:+6.1f} % ; deep sure {100 * r[2]:+6.1f} %")
    rat = np.array(rat)
    say(f"    median over windows: {100 * np.median(rat[:, 0]):+.1f} % / {100 * np.median(rat[:, 1]):+.1f} % ; deep sure "
        f"{100 * np.median(rat[:, 2]):+.1f} %")
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(MYOUT, "wide45_oea.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
