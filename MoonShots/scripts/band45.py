> (a quote from the project owner, kept private)
expert SSD reads and MB per kept token at C 256 with the adopted M-17 per-layer sizes when the check's drafter-sure rows
verify with fewer experts. ROUTE_350 Q1: top-2 (ranks 0-1 at today's 4-bit tier, the model's own gate weights) on rows
whose drafter confidence is >= theta, exact top-10 elsewhere. Q2 / NEW-1: bands >= 0.999 top-2, 0.99-0.999 top-4,
0.9-0.99 top-8 with ranks 0-3 at 4 bits, < 0.9 exact top-10. Bytes only: the flip rates are Breakthrough's map.
Routing: cap45 (BigMoE's 45 GB capture: six prompts chained, the real check rows incl. the rejected ones; ranks from the
8-bit router as laysz45.load). Drafter confidence: cap45 has none, so every check takes a real DFlash2 cf vector of a
D-23 check (~/TrainLab/data/d23_runs.jsonl, arm dfl, C 224, same type, same kept; nearest kept if none), SEEDS draws.
Row j verifies drafted token j+1, so its gate is cf[j]; row 15 of a full check has no draft after it and stays exact.
Rows per check follow the engine's t 0.3 cut through the transplanted check (rows = kept + its rows - kept, as laysz45's
SHORT; D-23's rows = 1 + the first i with cumprod(cf[:i+1]) < 0.3). Pools: per-layer C_m / C_l from laysz45_256.json
(md5 ee76e2a7...), LRU + 30 x popularity with one fixed prior per seed (leave-one-prompt-out shares of the EXACT needs:
the engine's prior table does not change with the band rule), strict, res0 forced per prompt (laysz45.replay rules).
Tok/s: C50's measured default (BigMoE data/c50/c45l_256.txt, C 256 + L, OOO, K-73: per prompt median cycle, steady
tok/s; 63.6 = the median over the six prompts) with each prompt's cycle cut by its saved SSD bytes at SSD_GBS (fully
exposed: the check is SSD-bound at C 256) and, as the upper bound, also by the saved expert RAM bytes of the check's
union at the GPU rate that M-16's beta (2.64 ms per row) implies over cap45's union growth per row. Kept per check is
held fixed (no flips).
Capture mode (Breakthrough's DAY9 capture, Mon 10:4x: one npz per map window, teacher-forced rank-sorted ids for all rows,
DFlash2 cf per row = the draft that row verifies, NaN = no draft, the walk's checks after the t 0.3 cut): the same rules
on its checks, windows chained per domain with the prompt rows as uncounted prefill chunks (no res0), prior = leave-one-
window-out shares of the EXACT needs; plus the oracle gate (the 125B's own p1). Tok/s for code: C50's two code prompts
with the capture's code ratios, and the six-prompt median with cap45's math ratios (data/band45.json).
usage: [BigMoE venv] taskpolicy -b python band45.py [capture [dir]]  -> stdout + data/band45.txt | data/band45_capture.txt"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, multiprocessing as mp, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laysz45 as LZ

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
SIZES = json.load(open(os.path.join(LZ.OUT, "laysz45_256.json")))
CAPDIR = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
CACHE = os.path.join(LZ.OUT, "band45_cap45.pkl")         # laysz45.load() output (rank-sorted ids), ~3 MB, not committed
SEEDS = int(os.environ.get("BAND45_SEEDS", "6"))
SSD_GBS, BETA = 12.0, 2.64                              # SSD GB/s at deep queue (M-16 0.157 ms per read ~ 13.8; C50 ~10.5)
TYPES = {"humaneval": "code", "math500": "math", "gsm8k": "math"}
# C50 c45l_256 per prompt (cap45's order): kept per check, median cycle ms, steady tok/s, SSD MB per kept token
C50 = {"humaneval_HumanEval_68": (15.41, 187.2, 84.7, 94.2), "math500_test_prealgebra_1640": (5.20, 97.6, 53.1, 60.4),
       "humaneval_HumanEval_91": (6.97, 93.9, 62.1, 67.8), "math500_test_algebra_694": (7.43, 113.2, 68.0, 55.2),
       "math500_test_algebra_1282": (6.07, 88.9, 64.2, 37.2), "gsm8k_708": (6.39, 96.0, 63.0, 57.4)}


def q1(theta, r4=2):
    return lambda c: (2, r4) if c >= theta else (10, 2)


def q2(mid_r4):
    return lambda c: (2, 2) if c >= 0.999 else (4, 2) if c >= 0.99 else (8, mid_r4) if c >= 0.9 else (10, 2)


RULES = {"EXACT": lambda c: (10, 2), "Q1 0.9": q1(0.9), "Q1 0.95": q1(0.95), "Q1 0.99": q1(0.99), "Q1 0.999": q1(0.999),
         "Q2": q2(4), "Q2 mid 3-bit": q2(2),           # (experts per row, of which ranks at 4 bits = sidecar reads)
         "Q1 0.99 3b": q1(0.99, 0), "Q1 0.999 3b": q1(0.999, 0)}   # sure rows at the resident 3-bit part only (no LSB read)
BANDS = ((0.999, ">= 0.999"), (0.99, "0.99-0.999"), (0.9, "0.9-0.99"), (-0.5, "< 0.9"), (-2.0, "no draft"))
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def d23(arm="dfl"):
    """type -> [(rows, kept, cf[15])] over D-23's valid runs of one drafter arm"""
    out = {}
    for r in map(json.loads, open(LZ.D23)):
        if r["arm"] == arm and not r.get("invalid") and not r.get("paused"):
            for pc, cf in zip(r["pc"], r["cf"]):
                out.setdefault(r["type"], []).append((int(pc[0]), int(pc[1]), np.asarray(cf, float)))
    return out


def transplant(P, D, seed):
    """per prompt, per check: (rows after the t 0.3 cut, conf per row [16]; -1 = no draft after the row)"""
    rng = np.random.default_rng(seed)
    T = []
    for name, cks, _ in P:
        pool = D[TYPES[name.split("_")[0]]]
        kk = np.array([k for _, k, _ in pool])
        out = []
        for c in cks:
            n, k = len(c["rid"][0]), c["kept"]
            d = np.abs(kk - k)
            r2, k2, cf = pool[rng.choice(np.nonzero(d == d.min())[0])]
            conf = np.full(16, -1.0)
            conf[:15] = cf
            out.append((int(min(n, max(k, k + r2 - k2))), conf))
        T.append(out)
    return T


def needs(P, T, rule):
    """per check and layer: base-pool needs (union of each row's first k experts) and sidecar needs (ranks < r4)"""
    Q = []
    for (name, cks, res0), tr in zip(P, T):
        out = []
        for c, (n, conf) in zip(cks, tr):
            kr = [rule(conf[j]) for j in range(n)]
            m, s = [], []
            for l in range(NL):
                rid = c["rid"][l]
                sm, sl = set(), set()
                for j, (k, r4) in enumerate(kr):
                    sm.update(rid[j, :k].tolist()); sl.update(rid[j, :r4].tolist())
                m.append(frozenset(sm)); s.append(frozenset(sl))
            out.append({"m": m, "l": s, "kept": c["kept"], "rows": n})
        Q.append((name, out, res0))
    return Q


def replay(task):
    """laysz45.replay's rules for one layer's pool: misses per prompt"""
    key, K, calls, pops = task
    while os.path.exists(LZ.PAUSE):
        time.sleep(20)
    miss = np.zeros(len(calls))
    st, i = {}, 0
    for p, cl in enumerate(calls):
        pop = pops[p]
        for t, need in cl:
            i += 1
            if t >= 0:
                miss[p] += sum(1 for e in need if e not in st)
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + 30.0 * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                    del st[e]
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                        del st[e]
    return miss


def tasks_of(Q, POP):
    out = []
    for key in ("m", "l"):
        for l in range(NL):
            calls = [([(-1, res0[l])] if key == "m" else []) + [(t, c[key][l]) for t, c in enumerate(cks)]
                     for _, cks, res0 in Q]
            out.append((key, SIZES["C_m" if key == "m" else "C_l"][l], calls, [POP[key][p][l] for p in range(len(Q))]))
    return out


def load_capture(d):
    """[(domain, window, checks [{rid {l: [rows, 10]}, kept, conf [rows], p1 [rows]}], prefill chunks [rows, 48, 10])]"""
    W = []
    for f in sorted(x for x in os.listdir(d) if x.endswith(".npz")):
        z = np.load(os.path.join(d, f), allow_pickle=False)
        ids, conf, p1, gen = z["ids"], z["conf"].astype(float), z["p1"].astype(float), int(z["gen"])
        cks = [{"rid": {l: ids[a:a + n, l] for l in range(NL)}, "kept": int(k), "conf": conf[a:a + n], "p1": p1[a:a + n]}
               for a, n, k in z["checks"].tolist()]
        W.append((str(z["domain"]), f[:-4], cks, [ids[i:min(i + 128, gen - 1)] for i in range(0, gen - 1, 128)]))
    return sorted(W, key=lambda w: (w[0], w[1]))


def capture_needs(W, rules):
    """per rule: [(window, checks with m / l needs per layer, None)], mean union bytes per check, prefill chunk needs"""
    Q, U = {}, {}
    for r, (src, f) in rules.items():
        Q[r] = []
        for dm, nm, cks, pre in W:
            out = []
            for c in cks:
                m, s = [], []
                kr = [f(x) for x in c[src]]
                for l in range(NL):
                    rid = c["rid"][l]
                    sm, sl = set(), set()
                    for j, (k, r4) in enumerate(kr):
                        sm.update(rid[j, :k].tolist()); sl.update(rid[j, :r4].tolist())
                    m.append(frozenset(sm)); s.append(frozenset(sl))
                out.append({"m": m, "l": s, "kept": c["kept"], "rows": len(kr)})
            Q[r].append((nm, out, None))
            U[(r, nm)] = np.mean([sum(len(c["m"][l]) * MR + len(c["l"][l]) * LR for l in range(NL)) for c in out])
    PRE = [([frozenset(np.unique(ch[:, l]).tolist()) for l in range(NL)], [frozenset(np.unique(ch[:, l, :2]).tolist())
            for l in range(NL)]) for _, _, _, pre in W for ch in pre]
    pi = 0; PREW = []
    for _, _, _, pre in W:
        PREW.append(PRE[pi:pi + len(pre)]); pi += len(pre)
    return Q, U, PREW


def capture_calls(Q, PREW, wi, key, l):
    """one domain chain's calls for one layer's pool: per window, the prefill chunks (uncounted) and the checks"""
    return [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, c[key][l]) for t, c in enumerate(Q[i][1])] for i in wi]


def capture_main(d):
    t0 = time.time()
    W = load_capture(d)
    doms = sorted(set(w[0] for w in W))
    say(f"DAY9 Q1 / Q2 bytes on Breakthrough's capture ({d}): " + ", ".join(
        f"{dm} {sum(1 for w in W if w[0] == dm)} windows / {sum(len(w[2]) for w in W if w[0] == dm)} checks / "
        f"{sum(c['kept'] for w in W if w[0] == dm for c in w[2])} kept" for dm in doms))
    rules = {r: ("conf", f) for r, f in RULES.items()}
    rules.update({"Q1 0.99 oracle p1": ("p1", q1(0.99)), "Q1 0.999 oracle p1": ("p1", q1(0.999))})
    Q, U, PREW = capture_needs(W, rules)
    POP = {key: LZ.popularity(Q["EXACT"], key) for key in ("m", "l")}
    jobs, meta = [], []
    for r in rules:
        for dm in doms:
            wi = [i for i, w in enumerate(W) if w[0] == dm]
            for key in ("m", "l"):
                for l in range(NL):
                    jobs.append((key, SIZES["C_m" if key == "m" else "C_l"][l], capture_calls(Q[r], PREW, wi, key, l),
                                 [POP[key][i][l] for i in wi]))
                    meta.append((r, dm, key))
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(replay, jobs, chunksize=8)
    M = {}
    for (r, dm, key), m in zip(meta, res):
        M[(r, dm, key)] = M.get((r, dm, key), 0) + m
    say(f"  {len(jobs)} replays, {time.time() - t0:.0f} s")
    for dm in doms:
        rows = [(x, y) for w in W if w[0] == dm for c in w[2] for x, y in zip(c["conf"], c["p1"])]
        cf = np.array([x for x, _ in rows]); p1 = np.array([y for _, y in rows])
        say(f"  {dm} verified rows: {len(cf)}, drafter cf " + ", ".join(
            f"{lab} {100 * np.mean(sel):.1f} %" for lab, sel in ((">= 0.999", cf >= 0.999), ("0.99-0.999", (cf >= 0.99) & (cf < 0.999)),
                                                                ("0.9-0.99", (cf >= 0.9) & (cf < 0.99)), ("< 0.9", cf < 0.9),
                                                                ("no draft", np.isnan(cf))))
            + f"; oracle p1 >= 0.99 {100 * np.mean(p1 >= 0.99):.1f} %")
    js = json.load(open(os.path.join(LZ.OUT, "band45.json"))) if os.path.exists(os.path.join(LZ.OUT, "band45.json")) else None
    gpu_gbs = js["gpu_gbs"] if js else 174.0
    say("")
    say("  rule               | domain | SSD reads per check (base + LSB) | MB per kept token | expert union GB per check | "
        "implied steady tok/s: C50 code prompts HE68 / HE91 (SSD only), six-prompt median (SSD / +GPU)")
    base = {}
    for r in rules:
        for dm in doms:
            wi = [i for i, w in enumerate(W) if w[0] == dm]
            ck = sum(len(W[i][2]) for i in wi); kp = sum(c["kept"] for i in wi for c in W[i][2])
            mm, ll = M[(r, dm, "m")].sum(), M[(r, dm, "l")].sum()
            mb = (mm * MR + ll * LR) / 1e6 / kp
            un = sum(U[(r, W[i][1])] * len(W[i][2]) for i in wi) / ck / 1e9
            if r == "EXACT":
                base[dm] = (mm + ll, mb, un)
            a0, b0, u0 = base[dm]
            tp = ""
            if dm == "code":
                rat, urat = mb / b0, un / u0
                imp = {}
                for use_gpu in (False, True):
                    v = []
                    for n, (kept, cyc, st, mbk) in C50.items():
                        q = rat if n.startswith("humaneval") else (js["ratio"].get(r, {}).get(n, 1.0) if js else 1.0)
                        uq = urat if n.startswith("humaneval") else (js["uratio"].get(r, {}).get(n, 1.0) if js else 1.0)
                        d_gpu = (1 - uq) * 5.30e3 / gpu_gbs if use_gpu else 0.0
                        v.append((n, st * cyc / (cyc - mbk * kept * (1 - q) / SSD_GBS - d_gpu)))
                    imp[use_gpu] = v
                tp = (f"{imp[False][0][1]:.1f} / {imp[False][2][1]:.1f}, median {np.median([x for _, x in imp[False]]):.1f} / "
                      f"{np.median([x for _, x in imp[True]]):.1f}")
            say(f"  {r:<18} | {dm:<6} | {(mm + ll) / ck:6.1f} ({mm / ck:5.1f} + {ll / ck:5.1f}) {100 * ((mm + ll) / a0 - 1):+6.1f} % | "
                f"{mb:6.1f} {100 * (mb / b0 - 1):+6.1f} % | {un:5.2f} {100 * (un / u0 - 1):+6.1f} % | {tp}")
    say("  (code prompts take the capture's code ratios, math prompts cap45's ratios for that rule (1.0 if none); +GPU = the "
        f"union ratio x 5.30 GB (cap45's EXACT union per check) at {gpu_gbs:.0f} GB/s)")
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "band45_capture.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if os.path.exists(LZ.PAUSE):
        sys.exit("PAUSE set: not running")
    if sys.argv[1:2] == ["capture"]:
        capture_main(sys.argv[2] if len(sys.argv) > 2 else CAPDIR)
        sys.exit(0)
    t0 = time.time()
    if os.path.exists(CACHE):
        P, agr = pickle.load(open(CACHE, "rb"))
    else:
        P, agr = LZ.load()
        pickle.dump((P, agr), open(CACHE, "wb"))
    D = d23()
    names = [n for n, _, _ in P]
    typ = [TYPES[n.split("_")[0]] for n in names]
    KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float)
    NCK = np.array([len(cks) for _, cks, _ in P], float)
    say(f"DAY9 Q1 / Q2 bytes (band45): cap45 {', '.join(f'{n} {len(c)}' for n, c, _ in P)} checks, {KEPT.sum():.0f} kept "
        f"tokens; router agreement {100 * agr:.2f} %; D-23 dfl checks {', '.join(f'{t} {len(v)}' for t, v in D.items())}; "
        f"C 256 per-layer sizes C_m {min(SIZES['C_m'])}-{max(SIZES['C_m'])}, C_l {min(SIZES['C_l'])}-{max(SIZES['C_l'])}; "
        f"{SEEDS} seeds; load {time.time() - t0:.0f} s")
    M, U, SH = {}, {}, {}                                 # M: (rule, seed, key) -> misses per prompt (summed over layers)
    with mp.get_context("fork").Pool(4) as pool:
        for s in range(SEEDS):                            # one seed at a time: a seed's needs are ~0.4 GB of sets
            T = transplant(P, D, s)
            Qs = {r: needs(P, T, f) for r, f in RULES.items()}
            POP = {key: LZ.popularity(Qs["EXACT"], key) for key in ("m", "l")}
            jobs, meta = [], []
            for r, Q in Qs.items():
                for p, (_, cks, _) in enumerate(Q):       # union bytes per check (RAM traffic of the experts) and rows
                    U.setdefault((r, p), []).append((np.mean([sum(len(c["m"][l]) for l in range(NL)) * MR
                                                              + sum(len(c["l"][l]) for l in range(NL)) * LR for c in cks]),
                                                     np.mean([c["rows"] for c in cks])))
                ts = tasks_of(Q, POP)
                jobs += ts; meta += [(r, key) for key, *_ in ts]
            for p, tr in enumerate(T):                    # band shares of the verified rows
                for n, conf in tr:
                    for j in range(n):
                        SH.setdefault((typ[p], next(lab for th, lab in BANDS if conf[j] >= th)), []).append(1)
            for (r, key), m in zip(meta, pool.map(replay, jobs, chunksize=8)):
                M[(r, s, key)] = M.get((r, s, key), 0) + m
            del Qs, jobs
            say(f"  seed {s}: {len(meta)} replays, {time.time() - t0:.0f} s")
    for t in ("code", "math"):
        nt = sum(len(v) for (tt, _), v in SH.items() if tt == t)
        say(f"  {t} verified rows by drafter band: " + ", ".join(f"{lab} {100 * len(SH.get((t, lab), [])) / nt:.1f} %"
                                                            for _, lab in BANDS))
    # GPU rate from M-16's beta: expert union MB added per verified row (EXACT, all prompts) / 2.64 ms
    ue = np.array([np.mean([u for u, _ in U[("EXACT", p)]]) for p in range(len(P))])
    ru = np.array([np.mean([n for _, n in U[("EXACT", p)]]) for p in range(len(P))])
    grow = []
    for _, cks, _ in P:
        for c in cks[:40]:
            prev = 0
            for n in range(1, len(c["rid"][0]) + 1):
                b = sum(len(np.unique(c["rid"][l][:n])) * MR + len(np.unique(c["rid"][l][:n, :2])) * LR for l in range(NL))
                grow.append((n, b - prev)); prev = b
    g = np.array(grow, float)
    mb_row = g[(g[:, 0] >= 6) & (g[:, 0] <= 10), 1].mean() / 1e6
    gpu_gbs = mb_row / BETA
    say(f"  EXACT check: {np.average(ru, weights=NCK):.2f} rows, expert union {np.average(ue, weights=NCK) / 1e9:.2f} GB per "
        f"check; union growth at rows 6-10 {mb_row:.0f} MB per row -> GPU expert rate {gpu_gbs:.0f} GB/s at beta {BETA} ms")
    mb0 = np.array([(M[("EXACT", s, "m")] * MR + M[("EXACT", s, "l")] * LR) for s in range(SEEDS)]).mean(0) / 1e6 / KEPT
    say("  EXACT replay vs C50 engine, MB per kept token: " + "; ".join(
        f"{n.split('_', 1)[1]} {mb0[p]:.1f} / {C50[n][3]:.1f}" for p, n in enumerate(names)))
    say("")
    say(f"  rule          | type | SSD reads per check (base + LSB)      | MB per kept token          | expert union GB per check | "
        f"implied steady tok/s (SSD / +GPU)")
    base = {}
    for r in RULES:
        mm = np.array([M[(r, s, "m")] for s in range(SEEDS)])          # [seeds, prompts]
        ll = np.array([M[(r, s, "l")] for s in range(SEEDS)])
        mb = (mm * MR + ll * LR) / 1e6                                  # MB per prompt
        un = np.array([[U[(r, p)][s][0] for p in range(len(P))] for s in range(SEEDS)])
        if r == "EXACT":
            base = dict(mm=mm, ll=ll, mb=mb, un=un)
        tps = {}
        for k, use_gpu in (("ssd", False), ("gpu", True)):
            out = []
            for s in range(SEEDS):
                v = []
                for p, n in enumerate(names):
                    kept, cyc, st, mbk = C50[n]
                    d_ssd = mbk * kept * (1 - mb[s, p] / base["mb"][s, p]) / SSD_GBS        # MB / (GB/s) = ms
                    d_gpu = (base["un"][s, p] - un[s, p]) / 1e6 / gpu_gbs if use_gpu else 0.0
                    v.append(st * cyc / (cyc - d_ssd - d_gpu))
                out.append(np.median(v))
            tps[k] = np.array(out)
        for t, sel in (("code", [p for p in range(len(P)) if typ[p] == "code"]),
                       ("math", [p for p in range(len(P)) if typ[p] == "math"]), ("all", list(range(len(P))))):
            ck, kp = NCK[sel].sum(), KEPT[sel].sum()
            a = (mm[:, sel].sum(1) + ll[:, sel].sum(1)) / ck
            b_ = mb[:, sel].sum(1) / kp
            b0 = base["mb"][:, sel].sum(1) / kp
            a0 = (base["mm"][:, sel].sum(1) + base["ll"][:, sel].sum(1)) / ck
            u_ = (un[:, sel] * NCK[sel]).sum(1) / ck / 1e9
            u0 = (base["un"][:, sel] * NCK[sel]).sum(1) / ck / 1e9
            tp = (f"{tps['ssd'].mean():.1f} / {tps['gpu'].mean():.1f}" if t == "all" else "")
            say(f"  {r:<13} | {t:<4} | {a.mean():6.1f} ({mm[:, sel].sum(1).mean() / ck:5.1f} + {ll[:, sel].sum(1).mean() / ck:4.1f})"
                f" {100 * (a / a0 - 1).mean():+6.1f} % [{100 * (a / a0 - 1).min():+.1f}, {100 * (a / a0 - 1).max():+.1f}] | "
                f"{b_.mean():5.1f} {100 * (b_ / b0 - 1).mean():+6.1f} % [{100 * (b_ / b0 - 1).min():+.1f}, "
                f"{100 * (b_ / b0 - 1).max():+.1f}] | {u_.mean():5.2f} {100 * (u_ / u0 - 1).mean():+6.1f} % | {tp}")
    say("")
    say(f"  per prompt (rule, prompt: MB per kept token vs EXACT, implied steady tok/s SSD-only):")
    for r in ("Q1 0.99", "Q1 0.999", "Q2"):
        mm = np.array([M[(r, s, "m")] for s in range(SEEDS)]); ll = np.array([M[(r, s, "l")] for s in range(SEEDS)])
        mb = (mm * MR + ll * LR) / 1e6
        say("    " + r + ": " + "; ".join(
            f"{n.split('_', 1)[1]} {100 * (mb[:, p] / base['mb'][:, p] - 1).mean():+.1f} % "
            f"{C50[n][2] * C50[n][1] / (C50[n][1] - C50[n][3] * C50[n][0] * (1 - (mb[:, p] / base['mb'][:, p]).mean()) / SSD_GBS):.1f}"
            for p, n in enumerate(names)))
    js = {"gpu_gbs": gpu_gbs, "ratio": {}, "uratio": {}}
    for r in RULES:
        js["uratio"][r] = {n: float(np.mean([U[(r, p)][s][0] / U[("EXACT", p)][s][0] for s in range(SEEDS)]))
                           for p, n in enumerate(names)}
        mb = np.array([(M[(r, s, "m")] * MR + M[(r, s, "l")] * LR) for s in range(SEEDS)])
        js["ratio"][r] = {n: float((mb[:, p] / np.array([(M[("EXACT", s, "m")] * MR + M[("EXACT", s, "l")] * LR)
                                                          for s in range(SEEDS)])[:, p]).mean()) for p, n in enumerate(names)}
    json.dump(js, open(os.path.join(LZ.OUT, "band45.json"), "w"), indent=1)
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "band45.txt"), "w").write("\n".join(lines) + "\n")
