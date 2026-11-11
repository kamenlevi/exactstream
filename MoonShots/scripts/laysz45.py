"""M-17 LAYSZ45 (MoonShots, Sun 2026-11-01; NewIdeas 14:5x hand-out; QUEUE M-17, lines written first): R-27's non-uniform
per-layer pool sizes on the 45 GB default. Exact (residency only).
The default's tier pools per layer (stream._tier_layer): C_m MSB-3 base slots (2,162,688 B; every rank reads its MSB part)
+ C_l LSB sidecar slots (770,048 B; ranks 0-1 read the LSB part too); uniform C 224 = 246 + 112, C 256 = 287 + 112.
Per layer and per prompt: miss curves of both pools over a size grid, under BigMoE's live rule (LRU + 30 x popularity,
popularity = leave-one-prompt-out share of checks, as seed45_lru.py; the sidecar pool with its own rank 0-1 popularity),
cap45's six prompts CHAINED (res0 forced into the base pool per prompt, prompts 2-6 counted), STRICT pools (trimmed to the
size after every check: a check's own experts beyond it are staging, not cache). Base pool needs = the check's union of
top-10 experts; sidecar needs = the union of rank 0-1 experts (ranks from the real 8-bit router, as runahead_mass.py).
A Lagrangian (+ greedy fill of the leftover bytes) picks per-layer (C_m, C_l) within the uniform pools' bytes on the
calibration prompts; scored on the held-out one (leave one counted prompt out); SSD bytes = MSB misses x 2,162,688 + LSB
misses x 770,048 per kept token (records3 parts; M-01 is not in the 45 GB stack).
Arms: FLOOR (C_m >= 160 = the gate's MAXK, C_l >= 32 = 16 rows x 2 ranks: today's ext, no staging; the verdict arm),
FREE (C_m >= 64, C_l >= 0) and RESPLIT (one C_m / C_l for all layers). Policies: P30 (LRU + 30 x popularity, the verdict)
and LRU (no prior, reference). Checks: 16 = cap45 as captured (the verdict); SHORT = the default's draft truncation
modelled (each check keeps its first min(16, kept + d) rows, d drawn from D-23's code + math checks at C 224), P30 only.
Lines (QUEUE M-17): PASS = FLOOR (P30, 16) held-out SSD MB per kept token <= -8 % vs uniform at both C 224 and C 256;
KILL = > -3 % at both; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python laysz45.py (CPU, 4 workers, ~5-10 min) -> stdout + data/laysz45.txt,
data/laysz45_<C>.json (BIGMOE_TIER_SIZES files: FLOOR arm, P30, 16-row checks, calibrated on all counted prompts)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, multiprocessing as mp, struct, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
MD = os.path.expanduser("~/BigMoE/models/bare")
D23 = os.path.expanduser("~/TrainLab/data/d23_runs.jsonl")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
OUT = os.path.expanduser("~/MoonShots/data")
NL, E, R4 = 48, 512, 2                                  # ranks 0-1 are 4-bit (tiers 4433333333)
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")      # the capture run's order (cap45_run.txt)
MR, LR = 2162688, 770048                                # slot RAM = SSD bytes per miss (records3's MSB / LSB parts)
CS = [int(c) for c in os.environ.get("LAYSZ45_C", "224,256").split(",")]    # NewIdeas 20:4x: also C 300 / 320
UNI = {c: ((c * 2764800 - 112 * LR) // MR, 112) for c in CS}  # stream._tier_layer's split: 224 -> 246 + 112, 256 -> 287 + 112
POLS = {"P30": 30.0, "LRU": 0.0}
RUNS = tuple(tuple(r.split(":")) for r in os.environ.get("LAYSZ45_RUNS", "16:P30,16:LRU,SHORT:P30").split(","))
GM = sorted(set(range(64, E + 1, 8)) | {246, 287} | {u[0] for u in UNI.values()})
GL = list(range(0, 257, 8))
ARMS = {"FLOOR": (160, 32), "FREE": (64, 0)}
H = (1, 2, 3, 4, 5)                                     # counted prompts (prompt 1, 5 checks, is the warm-up)
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
heads = {}


def tensor(name):
    f = os.path.join(MD, idx[name])
    if f not in heads:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            heads[f] = (8 + n, json.loads(fh.read(n)))
    start, h = heads[f]
    m = h[name]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[m["dtype"]]
    return np.memmap(f, dtype=dt, mode="r", offset=start + m["data_offsets"][0], shape=tuple(m["shape"]))


def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def router(l):
    """layer l's 8-bit router, dequantized fp32 [512, 2560]"""
    w, s, b = (tensor(f"language_model.model.layers.{l}.mlp.gate.{k}") for k in ("weight", "scales", "biases"))
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
    sh = np.arange(0, 32, bits, dtype=np.uint32)
    q = ((np.asarray(w)[..., None] >> sh) & np.uint32((1 << bits) - 1)).reshape(w.shape[0], -1).astype(np.float32)
    o, n = q.shape
    return (q.reshape(o, n // 64, 64) * f32(s)[..., None] + f32(b)[..., None]).reshape(o, n)


def load():
    """per prompt: checks with per-layer rank-sorted ids [rows, 10] and kept tokens, and res0"""
    WR = [router(l) for l in range(NL)]
    P, agree = [], [0, 0]
    for name in ORDER:
        z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
        seq, L, rows, roff, xoff, ids, X = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"], z["x"]
        checks, order = {}, []
        for n, (s, l, r) in enumerate(zip(seq, L, rows)):
            s, l, r = int(s), int(l), int(r)
            if s not in checks:
                order.append(s)
            c = checks.setdefault(s, {"rid": {}, "kept": None})
            if l in c["rid"]:                            # the commit replay: all hits, its rows = kept tokens
                c["kept"] = r
                continue
            rid = ids[roff[n]:roff[n] + r].astype(np.int64)                  # [r, 10], argpartition order
            lg = f32(X[xoff[n]:xoff[n] + r]) @ WR[l].T
            agree[0] += int((np.argsort(-lg, 1)[:, :10][:, :, None] == rid[:, None, :]).any(2).sum()); agree[1] += rid.size
            c["rid"][l] = np.take_along_axis(rid, np.argsort(-np.take_along_axis(lg, rid, 1), 1, kind="stable"), 1).astype(np.int16)
        cks = [checks[s] for s in order if len(checks[s]["rid"]) == NL and checks[s]["kept"] is not None]
        P.append((name, cks, [frozenset(np.nonzero(z["res0"][l] >= 0)[0].tolist()) for l in range(NL)]))
    return P, agree[0] / agree[1]


def needs(P, rows_of):
    """per check and layer: base-pool needs (union of the first n rows' top-10) and sidecar needs (their ranks 0-1)"""
    Q = []
    for name, cks, res0 in P:
        out = []
        for c in cks:
            n = rows_of(c)
            out.append({"m": [frozenset(np.unique(c["rid"][l][:n]).tolist()) for l in range(NL)],
                        "l": [frozenset(np.unique(c["rid"][l][:n, :R4]).tolist()) for l in range(NL)],
                        "kept": c["kept"], "rows": min(n, len(c["rid"][0]))})
        Q.append((name, out, res0))
    return Q


def popularity(P, key):
    out = []
    for i in range(len(P)):
        pop, n = np.zeros((NL, E)), 0
        for j, (_, cks, _) in enumerate(P):
            if j != i:
                for c in cks:
                    for l in range(NL):
                        pop[l, list(c[key][l])] += 1
                n += len(cks)
        out.append([list(map(float, pop[l] / n)) for l in range(NL)])
    return out


def replay(task):
    """misses per prompt for one layer's pool (key m = base, l = sidecar) at size K, LRU + b x popularity, strict"""
    var, l, key, K, b = task
    while os.path.exists(PAUSE):
        time.sleep(20)
    P, POP = DATA[var]
    miss = np.zeros(len(P))
    st, i = {}, 0
    for p, (_, cks, res0) in enumerate(P):
        pop = POP[key][p][l]
        calls = ([(-1, res0[l])] if key == "m" else []) + [(t, c[key][l]) for t, c in enumerate(cks)]
        for t, need in calls:
            i += 1
            if t >= 0:
                miss[p] += sum(1 for e in need if e not in st)
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + b * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                    del st[e]
                if len(st) > K:                          # strict: the check's own experts beyond K were staging
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                        del st[e]
    return miss


def cost(run, sets):
    """calibration bytes per kept token per layer and grid size, both pools"""
    k = KEPT[list(sets)].sum()
    return (CUR[run + ("m",)][:, :, list(sets)].sum(2) * MR / k, CUR[run + ("l",)][:, :, list(sets)].sum(2) * LR / k)


def score(run, sets, i, j):
    m, l = CUR[run + ("m",)], CUR[run + ("l",)]
    return float((m[np.arange(NL), i][:, list(sets)].sum() * MR + l[np.arange(NL), j][:, list(sets)].sum() * LR)
                 / KEPT[list(sets)].sum())


def alloc(run, sets, budget, mmin, lmin):
    """per-layer grid indices (i into GM, j into GL) minimizing calibration bytes within budget: Lagrangian (binary search
    on lambda), then greedy fill of the leftover bytes by the best saving per byte"""
    cm, cl = cost(run, sets)
    ok_m, ok_l = np.array(GM) >= mmin, np.array(GL) >= lmin
    rm, rl = np.array(GM, float) * MR, np.array(GL, float) * LR
    am, al = np.where(ok_m, 0, np.inf), np.where(ok_l, 0, np.inf)

    def pick(lam):
        i = np.argmin(cm + lam * rm[None, :] + am[None, :], 1)
        j = np.argmin(cl + lam * rl[None, :] + al[None, :], 1)
        return i, j, rm[i].sum() + rl[j].sum()
    lo, hi = 0.0, 1e-6
    while pick(hi)[2] > budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[2] > budget:
            lo = mid
        else:
            hi = mid
    i, j, used = pick(hi)
    i, j = i.copy(), j.copy()
    while True:
        best = None
        for l in range(NL):
            for which, r, c, ok, cur in (("m", rm, cm, ok_m, i), ("l", rl, cl, ok_l, j)):
                for n in range(cur[l] + 1, len(r)):     # any larger size of this pool that fits
                    d = r[n] - r[cur[l]]
                    if ok[n] and used + d <= budget and c[l, n] < c[l, cur[l]]:
                        v = (c[l, cur[l]] - c[l, n]) / d
                        if best is None or v > best[0]:
                            best = (v, which, l, n, d)
        if best is None:
            break
        _, which, l, n, d = best
        (i if which == "m" else j)[l] = n
        used += d
    return i, j, used


if __name__ == "__main__":                              # cachesize45.py imports the functions
    if os.path.exists(PAUSE):
        sys.exit("PAUSE set: not running")
    t0 = time.time()
    P0, agr = load()
    dd = np.array([c[0] - c[1] for r in map(json.loads, open(D23)) if r["arm"] == "dfl" and r["type"] in ("code", "math")
                   for c in r["pc"]], int)
    rng = np.random.default_rng(0)
    DATA = {}
    for var, rows_of in (("16", lambda c: 16), ("SHORT", lambda c: int(min(16, c["kept"] + rng.choice(dd))))):
        Q = needs(P0, rows_of)
        DATA[var] = (Q, {"m": popularity(Q, "m"), "l": popularity(Q, "l")})
    KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in DATA["16"][0]], float)
    say(f"M-17 LAYSZ45: cap45 {', '.join(f'{n} {len(c)}' for n, c, _ in P0)} checks; kept tokens (prompts 2-6) "
        f"{KEPT[list(H)].sum():.0f}, {KEPT[list(H)].sum() / sum(len(P0[p][1]) for p in H):.2f} per check; router top-10 agreement "
        f"with the captured ids {100 * agr:.2f} %; load {time.time() - t0:.0f} s")
    for var, (Q, _) in DATA.items():
        cc = [c for p in H for c in Q[p][1]]
        say(f"  checks {var}: rows {np.mean([c['rows'] for c in cc]):.2f} per check; needs per layer: base pool "
            f"{np.mean([len(c['m'][l]) for c in cc for l in range(NL)]):.1f}, sidecar {np.mean([len(c['l'][l]) for c in cc for l in range(NL)]):.1f}")
    tasks = [(var, l, key, K, POLS[pol]) for var, pol in RUNS for key, g in (("m", GM), ("l", GL)) for l in range(NL) for K in g]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(replay, tasks, chunksize=16)
    CUR = {}                                                # (var, pol, key) -> misses [NL, grid, prompts]
    for (var, l, key, K, b), m in zip(tasks, res):
        pol = next(k for k, v in POLS.items() if v == b)
        g = GM if key == "m" else GL
        CUR.setdefault((var, pol, key), np.zeros((NL, len(g), len(P0))))[l, g.index(K)] = m
    say(f"  replays: {len(tasks)} ({time.time() - t0:.0f} s)")
    res_tab = {}
    for run in RUNS:
        var, pol = run
        for C, (um, ul) in UNI.items():
            budget = NL * (um * MR + ul * LR)
            ui, uj = np.full(NL, GM.index(um)), np.full(NL, GL.index(ul))
            u_all = score(run, H, ui, uj)
            mpt = CUR[run + ("m",)][:, GM.index(um), list(H)].sum() / KEPT[list(H)].sum()
            lpt = CUR[run + ("l",)][:, GL.index(ul), list(H)].sum() / KEPT[list(H)].sum()
            say(f"{var}-row checks, {pol}, C {C} ({um} + {ul}, {budget / 1e9:.2f} GB of pools): uniform {u_all / 1e6:.1f} MB per kept "
                f"token (base misses {mpt:.1f} + sidecar misses {lpt:.1f} per kept token)")
            for arm, (mmin, lmin) in tuple(ARMS.items()) + (("RESPLIT", (160, 0)),):
                ho_b = ho_u = 0.0
                per = []
                for h in H:
                    cal = tuple(x for x in H if x != h)
                    if arm == "RESPLIT":                     # one (C_m, C_l) for all layers, the best on calibration
                        cm, cl = cost(run, cal)
                        cand = []
                        for jj, cL in enumerate(GL):
                            fit = [ii for ii, cM in enumerate(GM) if cM >= mmin and NL * (cM * MR + cL * LR) <= budget]
                            if fit:
                                ii = max(fit)
                                cand.append((cm[:, ii].sum() + cl[:, jj].sum(), ii, jj))
                        _, ii, jj = min(cand)
                        i, j = np.full(NL, ii), np.full(NL, jj)
                    else:
                        i, j, _ = alloc(run, cal, budget, mmin, lmin)
                    b, u = score(run, (h,), i, j), score(run, (h,), ui, uj)
                    ho_b += b * KEPT[h]; ho_u += u * KEPT[h]
                    per.append(b / u - 1)
                g = ho_b / ho_u - 1
                res_tab[run + (C, arm)] = g
                say(f"    {arm}: held-out {ho_u / KEPT[list(H)].sum() / 1e6:.1f} -> {ho_b / KEPT[list(H)].sum() / 1e6:.1f} MB per "
                    f"kept token ({100 * g:+.1f} %; per prompt {', '.join(f'{100 * v:+.1f}' for v in per)})"
                    + (f"; one size for all: C_m {GM[ii]}, C_l {GL[jj]} (last fold)" if arm == "RESPLIT" else ""))
            for arm, (mmin, lmin) in ARMS.items():
                i, j, used = alloc(run, H, budget, mmin, lmin)
                ins = score(run, H, i, j) / u_all - 1
                cmv, clv = np.array(GM)[i], np.array(GL)[j]
                say(f"    {arm} sizes (calibrated on prompts 2-6, in-sample {100 * ins:+.1f} %; pools {used / 1e9:.2f} of "
                    f"{budget / 1e9:.2f} GB): C_m {cmv.min()}-{cmv.max()} (median {np.median(cmv):.0f}), C_l {clv.min()}-"
                    f"{clv.max()} (median {np.median(clv):.0f}); MSB share of pool bytes {100 * (cmv * MR).sum() / used:.1f} % "
                    f"(uniform {100 * um * MR / (um * MR + ul * LR):.1f} %)")
                say(f"      C_m by layer: {' '.join(map(str, cmv))}")
                say(f"      C_l by layer: {' '.join(map(str, clv))}")
                if (arm, var, pol) == ("FLOOR", "16", "P30"):
                    fn = os.path.join(OUT, f"laysz45_{C}.json")
                    json.dump({"C_m": [int(x) for x in cmv], "C_l": [int(x) for x in clv], "budget_bytes": int(budget),
                               "used_bytes": int(used), "uniform": [um, ul], "policy": "LRU + 30 x popularity (seed45)",
                               "floor": [mmin, lmin], "slot_bytes": [MR, LR], "in_sample": ins,
                               "held_out": res_tab[run + (C, arm)]}, open(fn, "w"), indent=1)
                    say(f"      wrote {fn} (BIGMOE_TIER_SIZES format)")
    g = [res_tab[("16", "P30", C, "FLOOR")] for C in UNI]
    call = "PASS" if all(v <= -0.08 for v in g) else ("KILL" if all(v > -0.03 for v in g) else "BETWEEN")
    say(f"Lines: FLOOR (P30, 16-row checks) held-out {', '.join(f'C {C} {100 * v:+.1f} %' for C, v in zip(UNI, g))} -> {call} "
        f"(PASS <= -8 % at both, KILL > -3 % at both)" + ("; SHORT checks: " + ", ".join(f"C {C} {100 * res_tab[('SHORT', 'P30', C, 'FLOOR')]:+.1f} %"
        for C in UNI) if ("SHORT", "P30") in RUNS else ""))
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, "laysz45.txt" if CS == [224, 256] else f"laysz45_c{'_'.join(map(str, CS))}.txt"), "w").write("\n".join(lines) + "\n")
