"""M-38 FORESIGHT (MoonShots, Wed 2026-11-04; the orchestrator's FAR-AHEAD FORESIGHT screen, 20:5x): how much of the
LRU -> OPT miss gap survives when the forecast of the future expert schedule is only partly right?
Trace and frame = M-14 EVICT45 (scripts/evict45.py, its loader / items / learned key copied unchanged in substance):
BigMoE's cap45 (the exact REF2 loop at C 160, real DFlash2 16-row checks, six prompts in one process), CHAINED: the cache
carries over, each prompt's prefill is its res0 forced in as one uncounted call, misses counted on prompts 2-6; per layer
a check needs the union of its 16 rows' experts; victims are taken outside the current call's needs (only when res0 alone
is larger than K, i.e. K 147 < 160, is the prefill set itself trimmed by the policy); K = resident MSB-3 slots per layer:
266 (45 GB EXACT, M-14's C) and 147 (24 GB, C_m of the 147 / 46 split).
Policy FH(H, forecast): experts with a forecast use within the next H checks stay (if all candidates are forecast, the
farthest forecast use leaves first, LRU on ties); the others leave by LRU. The forecast never looks past the prompt's own
last check (a drafter forecasts its own continuation; after the prefill it sees the prompt's first H checks).
(1) perfect forecast, H 1 / 2 / 4 / 8 / 16, plus OPT (Belady, full future).
(2) noisy forecast at H 8 / 16: each true future (layer, expert, check) use is kept with probability r = 0.5 / 0.7 / 0.9,
    drawn once per use (a dropped use is never forecast); the dropped uses are replaced by an equal count of false
    positives at the same check, drawn without replacement from that layer's leave-one-prompt-out popularity among the
    experts the check does not use (recall = precision = r per check); 5 seeds.
References: LRU; M-14's learned key (LEARN, re-fit here per K, leave-one-prompt-out; M-14: -7.9 % misses at K 266).
Sensitivities (after the lines): per-look noise (a fresh draw at every check, r 0.7); full-horizon checks only (counted
checks with H checks left in their prompt, all policies on the same checks: a long-generation proxy, the captures are ~29
checks per prompt); M-14's crossing oracle convention (the forecast also sees the next prompt; M-14's 'hor' reproduced); miss-focused recall (added
after the smoke, before the full run): uniform recall is dominated by the uses LRU hits anyway, so r 0.9 on the uses LRU
hits and r_m 0.1 / 0.3 / 0.5 / 0.7 on the uses LRU misses (at that K), 2 seeds, shows which recall the gain needs; and
controls for the false-positive model (popularity false positives are themselves a popularity prior): a popularity-only
forecast (r 0: every forecast entry a popularity draw, no true information), r 0.9 on hits and 0 on LRU misses, and r 0.7
with uniform (not popularity) false positives. Translation for a GPU recall measurement: at the eviction decisions, the
residents' 'used again within H checks' as a classification: the r 0.7 forecast's recall / precision (on FH's trajectory)
next to recency's (used within the last m checks, on LRU's trajectory).
Bytes = misses x 2,162,688 B (MSB-3 parts; the LSB sidecar is outside this replay, as in M-14).
Lines (QUEUE M-38, the orchestrator's, written first): PASS = r 0.7, H 8, K 266: >= 20 % fewer misses than LRU AND
better than the learned key; BETWEEN = 10-20 %; KILL = < 10 % or no better than the learned key.
usage: [BigMoE venv] OMP_NUM_THREADS=1 nice python foresight45.py (one core, a few minutes) -> stdout + data/foresight45.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import bisect, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E, R = 48, 512, 16
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")                  # the capture run's order (cap45_run.txt)
KS = (266, 147)
HS = (1, 2, 4, 8, 16)
RS = (0.5, 0.7, 0.9)
SEEDS = 5
CAPD = 300.0                                                         # next-use distance cap (items) for the labels
MSB = 2162688
BIG = 10 ** 6
SMOKE = os.environ.get("SMOKE") == "1"
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
P = []
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"rows": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["rows"]:
            c["kept"] = r                                            # the commit replay: kept rows
        else:
            c["rows"][l] = ids[roff[n]:roff[n] + r].astype(np.int64)  # [16, 10]
    cks = [checks[s] for s in order if len(checks[s]["rows"]) == NL and checks[s]["kept"] is not None]
    res0 = [np.nonzero(z["res0"][l] >= 0)[0] for l in range(NL)]
    P.append((name, cks, res0))
POP = []                                                             # leave-one-prompt-out popularity [48, 512]
for i in range(len(P)):
    pop, n = np.zeros((NL, E)), 0
    for j, (_, cks, _) in enumerate(P):
        if j != i:
            for c in cks:
                for l in range(NL):
                    pop[l, np.unique(c["rows"][l])] += 1
            n += len(cks)
    POP.append(pop / n)
KEPT = [np.array([c["kept"] for c in cks]) for _, cks, _ in P]
say(f"M-38 FORESIGHT ({time.strftime('%a %Y-%m-%d %H:%M')}): cap45, {sum(len(c) for _, c, _ in P)} live checks over {len(P)} "
    f"prompts ({', '.join(str(len(c)) for _, c, _ in P)} checks), chained (M-14's replay); res0 "
    f"{np.mean([len(x) for _, _, r in P for x in r]):.0f} experts per layer; K {' / '.join(map(str, KS))}")


def items_for(l):
    """the chained call sequence at layer l: (prompt, check or -1, need [n], rows using each [n], kept-row use [n], prefill)"""
    out = []
    for p, (_, cks, res0) in enumerate(P):
        out.append((p, -1, res0[l], np.zeros(len(res0[l])), np.zeros(len(res0[l]), bool), True))
        for t, c in enumerate(cks):
            rw = c["rows"][l]
            need, cnt = np.unique(rw, return_counts=True)
            kr = np.isin(need, rw[:c["kept"]])
            out.append((p, t, need, cnt / R, kr, False))
    return out


ITEMS = [items_for(l) for l in range(NL)]
NI = len(ITEMS[0])
PRE = np.array([it[5] for it in ITEMS[0]])                           # prefill items
PRM = np.array([it[0] for it in ITEMS[0]])                           # prompt of each item
PEND = np.array([max(j for j in range(NI) if PRM[j] == PRM[i]) for i in range(NI)])   # last item of the item's prompt
TRUE = np.zeros((NL, NI, E), bool)
for l in range(NL):
    for i, it in enumerate(ITEMS[l]):
        TRUE[l, i, it[2]] = True
POPI = np.stack([np.stack([POP[PRM[i]][l] for i in range(NI)]) for l in range(NL)]) + 1e-6   # [48, NI, 512]
USES = []
for l in range(NL):
    u = {}
    for i, it in enumerate(ITEMS[l]):
        for e in it[2].tolist():
            u.setdefault(e, []).append(i)
    USES.append(u)


def next_use(l, e, i):
    u = USES[l].get(e)
    if not u:
        return CAPD
    k = bisect.bisect_right(u, i)
    return min(float(u[k] - i), CAPD) if k < len(u) else CAPD


def next_from(FC, within=True):
    """forecast matrix [48, NI, 512] -> the next item j > i whose forecast names e (BIG if none in view); within: the view
    ends at the prompt's last check (the next prompt's prefill and checks are not seen)"""
    nxt = np.full((NL, NI, E), BIG, np.int32)
    cur = np.full((NL, E), BIG, np.int32)
    for i in range(NI - 1, -1, -1):
        nxt[:, i] = cur
        if within and PRE[i]:
            cur = np.full((NL, E), BIG, np.int32)
        else:
            cur = np.where(FC[:, i], i, cur)
    return nxt


def fp_draw(T, w, drop, rng):
    """false positives per row: drop[k] experts outside T[k], without replacement, proportional to w[k] (Gumbel top-k)"""
    g = np.log(w) - np.log(-np.log(rng.random(T.shape)))
    g[T] = -np.inf
    order = np.argsort(-g, axis=-1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(E), order.shape), axis=-1)
    return rank < drop[..., None]


def noisy_fc(r, seed):
    """once-per-use noise: each true check use kept with probability r, the dropped ones replaced by popularity false
    positives at the same check"""
    rng = np.random.default_rng(seed)
    keep = TRUE & (rng.random(TRUE.shape) < r)
    drop = (TRUE & ~keep).sum(-1)
    FC = keep | fp_draw(TRUE, POPI, drop, rng)
    FC[:, PRE] = False                                               # prefills are never forecast (within view only)
    return FC


def split_fc(r_hit, r_miss, lmiss, seed, uniform=False):
    """once-per-use noise with recall r_miss on the uses LRU misses and r_hit on the rest; uniform: false positives drawn
    uniformly among the experts the check does not use"""
    rng = np.random.default_rng(seed)
    keep = TRUE & (rng.random(TRUE.shape) < np.where(lmiss, r_miss, r_hit))
    drop = (TRUE & ~keep).sum(-1)
    FC = keep | fp_draw(TRUE, np.ones_like(POPI) if uniform else POPI, drop, rng)
    FC[:, PRE] = False
    return FC


def look_fn(r, decay=False):
    """per-look noise: a fresh forecast of the checks in view at every decision; decay: recall falls linearly with the
    distance, r + 0.2 at 1 check to r - 0.2 at H checks (mean r over the horizon)"""
    def f(l, i, H, rng):
        j1, j2 = i + 1, min(i + H, PEND[i])
        d = np.full(E, BIG)
        if j2 < j1:
            return d
        T = TRUE[l, j1:j2 + 1]
        rr = np.linspace(r + 0.2, r - 0.2, H)[:len(T), None] if decay else r
        keep = T & (rng.random(T.shape) < rr)
        F = keep | fp_draw(T, POPI[l, j1:j2 + 1], (T & ~keep).sum(-1), rng)
        has = F.any(0)
        return np.where(has, F.argmax(0) + 1, BIG)
    return f


NF = 9


def feats(st, cand, i, l, p, pre_set):
    """[n, NF] features of the candidate experts at call i (M-14's)"""
    rec = np.log1p(i - st["last"][cand])
    dc = np.log1p(st["score"][cand] / st["w"])
    return np.stack([rec, dc, POP[p][l, cand], st["nrows"][cand], st["kept"][cand].astype(float),
                     st["pre"][cand].astype(float), pre_set[cand].astype(float), np.full(len(cand), l / NL),
                     np.ones(len(cand))], 1)


def replay(K, pol, H=0, nxt=None, look=None, model=None, collect=None, seed=0, mark=None, stats=None):
    """misses per prompt (arrays of per-check sums over layers); pol: lru, min, learn, hor (M-14's oracle), fh"""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        st = {"res": np.zeros(E, bool), "last": np.zeros(E), "score": np.zeros(E), "w": 1.0, "nrows": np.zeros(E),
              "kept": np.zeros(E, bool), "pre": np.zeros(E, bool)}
        pre_set = np.zeros(E, bool)
        rng = np.random.default_rng(7919 * l + 17 + 100003 * seed) if look else None
        for i, (p, t, need, nr, kr, pre) in enumerate(ITEMS[l]):
            st["w"] /= 0.99
            if pre:
                pre_set = np.zeros(E, bool); pre_set[need] = True
            hit = st["res"][need]
            if t >= 0:
                miss[p][t] += int((~hit).sum())
                if mark is not None:
                    mark[l, i, need[~hit]] = True
            st["last"][need] = i
            st["score"][need] += st["w"]
            st["nrows"][need] = nr; st["kept"][need] = kr; st["pre"][need] = pre
            st["res"][need] = True
            over = int(st["res"].sum()) - K
            if collect is not None and t >= 0:
                cand = np.nonzero(st["res"])[0]
                cand = cand[~np.isin(cand, need)]
                if cand.size:
                    sel = cand if cand.size <= 64 else np.random.default_rng(i * 131 + l).choice(cand, 64, replace=False)
                    collect.append((p, feats(st, sel, i, l, p, pre_set), np.log1p([next_use(l, e, i) for e in sel.tolist()])))
            if over <= 0:
                continue
            nm = np.zeros(E, bool); nm[need] = True
            cand = np.nonzero(st["res"] & ~nm)[0]
            if cand.size < over:                                     # res0 alone is larger than K (K 147): trim the prefill set
                cand = np.nonzero(st["res"])[0]
            if stats is not None:                                    # residents: used again within H (in view)?
                y = NXW[l, i, cand] - i <= stats["H"]
                stats["n"] += y.size; stats["pos"] += int(y.sum())
                if pol == "fh":
                    yh = nxt[l, i, cand] - i <= stats["H"]
                    stats["f"] += np.array([int((y & yh).sum()), int(yh.sum())])
                for m in (1, 2, 4, 8):
                    yh = i - st["last"][cand] <= m
                    stats[m] = stats.get(m, 0) + np.array([int((y & yh).sum()), int(yh.sum())])
            if pol == "lru":
                vict = cand[np.argsort(st["last"][cand], kind="stable")[:over]]
            elif pol == "learn":
                pred = feats(st, cand, i, l, p, pre_set) @ model
                vict = cand[np.argsort(-pred, kind="stable")[:over]]
            elif pol == "hor":                                       # M-14: LRU, a true next use within H calls protects
                d = np.array([next_use(l, e, i) for e in cand.tolist()])
                key = np.where(d <= H, 1e9, 0.0) + st["last"][cand]
                vict = cand[np.argsort(key, kind="stable")[:over]]
            elif pol == "fh":                                        # forecast within H stays, farthest forecast first
                d = (look(l, i, H, rng) if look is not None else nxt[l, i] - i)[cand].astype(float)
                fc = d <= H
                key = np.where(fc, 1e9 - 1e3 * d, 0.0) + st["last"][cand]
                vict = cand[np.argsort(key, kind="stable")[:over]]
            else:                                                    # Belady
                d = np.array([next_use(l, e, i) for e in cand.tolist()])
                vict = cand[np.argsort(-d, kind="stable")[:over]]
            st["res"][vict] = False
        if os.path.exists(PAUSE):
            while os.path.exists(PAUSE):
                time.sleep(20)
    return miss


def pooled(miss, H=None, ps=range(1, 6)):
    """misses per kept token on prompts 2-6; H: only checks with H checks left in their prompt (full-horizon proxy)"""
    cut = lambda a: a if H is None else a[:max(len(a) - H, 0)]
    return sum(cut(miss[p]).sum() for p in ps) / sum(cut(KEPT[p]).sum() for p in ps)


def learn_run(K):
    smp = []
    replay(K, "lru", collect=smp)
    X = np.concatenate([x for _, x, _ in smp]); Y = np.concatenate([y for _, _, y in smp])
    PID = np.concatenate([np.full(len(y), q) for q, _, y in smp])
    out = [np.zeros(len(c)) for _, c, _ in P]
    for p in range(1, len(P)):
        m = PID != p
        out[p] = replay(K, "learn", model=np.linalg.lstsq(X[m], Y[m], rcond=None)[0])[p]
    return out


if SMOKE:                                                            # a quick check of M-14's references at K 266
    ks, HS, RS, SEEDS = (266,), (8,), (0.7,), 1
else:
    ks = KS
res = {}
for K in ks:
    res[(K, "lru")] = replay(K, "lru")
    res[(K, "min")] = replay(K, "min")
    res[(K, "learn")] = learn_run(K)
    say(f"  K {K}: LRU {pooled(res[(K, 'lru')]):.1f}, learned key {pooled(res[(K, 'learn')]):.1f}, OPT {pooled(res[(K, 'min')]):.1f} "
        f"misses per kept token ({time.time() - t0:.0f} s)")
NX = {"within": next_from(TRUE, True), "cross": next_from(TRUE, False)}
NXW = NX["within"]
for K in ks:
    for H in HS:
        res[(K, "fh", H, "within")] = replay(K, "fh", H, nxt=NX["within"])
        res[(K, "fh", H, "cross")] = replay(K, "fh", H, nxt=NX["cross"])
        if K == 266:
            res[(K, "hor", H)] = replay(K, "hor", H)
    say(f"  K {K}: perfect FH done ({time.time() - t0:.0f} s)")
del NX
for r in RS:
    for sd in range(SEEDS):
        nxt = next_from(noisy_fc(r, 1000 * sd + int(100 * r)), True)
        for K in ks:
            for H in (8, 16):
                if H in HS or not SMOKE:
                    res[(K, "noisy", H, r, sd)] = replay(K, "fh", H, nxt=nxt)
        del nxt
    say(f"  noisy r {r} done ({time.time() - t0:.0f} s)")
for K in ks:
    for H in (8, 16):
        if H in HS or not SMOKE:
            for sd in range(2 if not SMOKE else 1):
                res[(K, "look", H, 0.7, sd)] = replay(K, "fh", H, look=look_fn(0.7), seed=sd)
                res[(K, "decay", H, 0.7, sd)] = replay(K, "fh", H, look=look_fn(0.7, True), seed=sd)
say(f"  per-look noise done ({time.time() - t0:.0f} s)")
RMS = (0.1, 0.3, 0.5, 0.7)
LM = {}
for K in ks:
    LM[K] = np.zeros((NL, NI, E), bool)
    replay(K, "lru", mark=LM[K])
    for rm in (RMS if not SMOKE else (0.3,)):
        for sd in range(2 if not SMOKE else 1):
            nxt = next_from(split_fc(0.9, rm, LM[K], 5000 + 1000 * sd + int(100 * rm)), True)
            for H in (8, 16):
                if H in HS or not SMOKE:
                    res[(K, "split", H, rm, sd)] = replay(K, "fh", H, nxt=nxt)
            del nxt
CTL = {"pop0": (0.0, 0.0, False), "hit9m0": (0.9, 0.0, False), "uni7": (0.7, 0.7, True)}
for K in ks:
    for c, (rh, rm, uni) in CTL.items():
        for sd in range(2 if not SMOKE else 1):
            nxt = next_from(split_fc(rh, rm, LM[K], 9000 + 1000 * sd + len(c), uni), True)
            for H in (8, 16):
                if H in HS or not SMOKE:
                    res[(K, "ctl", c, H, sd)] = replay(K, "fh", H, nxt=nxt)
            del nxt
say(f"  miss-focused recall and controls done ({time.time() - t0:.0f} s)")

say("misses per kept token, prompts 2-6 pooled [MB per kept token = misses x 2.163]; '% LRU' = fewer misses than LRU; "
    "'gap' = share of the LRU -> OPT gap taken")
for K in ks:
    lru, mn, lrn = pooled(res[(K, "lru")]), pooled(res[(K, "min")]), pooled(res[(K, "learn")])
    f = lambda v: f"{v:5.1f} [{v * MSB / 1e6:6.1f} MB] {100 * (lru - v) / lru:5.1f} % LRU, gap {100 * (lru - v) / (lru - mn):4.0f} %"
    say(f"  K {K} ({'45 GB EXACT' if K == 266 else '24 GB'}): LRU {f(lru)} | learned key {f(lrn)} | OPT {f(mn)}")
    for H in HS:
        say(f"    perfect H {H:2d}: {f(pooled(res[(K, 'fh', H, 'within')]))}   (crossing view {pooled(res[(K, 'fh', H, 'cross')]):5.1f}"
            + (f"; M-14's hor {pooled(res[(K, 'hor', H)]):5.1f})" if K == 266 else ")"))
    for H in (8, 16):
        if (K, "noisy", H, RS[0], 0) not in res:
            continue
        for r in RS:
            v = [pooled(res[(K, "noisy", H, r, sd)]) for sd in range(SEEDS)]
            say(f"    noisy H {H:2d} r {r}: {f(float(np.mean(v)))}  (seeds {min(v):.1f}..{max(v):.1f})")
say("sensitivities (after the lines): full-horizon checks only (counted checks with H checks left in their prompt; LRU / "
    "learned key on the same checks), per-look noise (fresh draw every check, r 0.7, 2 seeds), and per-look recall "
    "falling linearly with the distance, 0.9 at 1 check to 0.5 at H checks (mean 0.7, 2 seeds):")
for K in ks:
    for H in (8, 16):
        if (K, "noisy", H, RS[0], 0) not in res:
            continue
        lruH, lrnH = pooled(res[(K, "lru")], H), pooled(res[(K, "learn")], H)
        pf = pooled(res[(K, "fh", H, "within")], H)
        nz = {r: float(np.mean([pooled(res[(K, "noisy", H, r, sd)], H) for sd in range(SEEDS)])) for r in RS}
        lk = [pooled(res[(K, "look", H, 0.7, sd)]) for sd in range(2 if not SMOKE else 1)]
        dk = float(np.mean([pooled(res[(K, "decay", H, 0.7, sd)]) for sd in range(2 if not SMOKE else 1)]))
        say(f"  K {K} H {H:2d} full-horizon: LRU {lruH:5.1f}, learned {100 * (lruH - lrnH) / lruH:5.1f} %, perfect "
            f"{100 * (lruH - pf) / lruH:5.1f} %, noisy " + ", ".join(f"r {r} {100 * (lruH - v) / lruH:5.1f} %" for r, v in nz.items())
            + f" fewer misses than LRU | per-look r 0.7 (all checks): {float(np.mean(lk)):5.1f} = "
            f"{100 * (pooled(res[(K, 'lru')]) - float(np.mean(lk))) / pooled(res[(K, 'lru')]):5.1f} % LRU; depth-decaying "
            f"recall 0.9 -> 0.5 (mean 0.7): {dk:5.1f} = {100 * (pooled(res[(K, 'lru')]) - dk) / pooled(res[(K, 'lru')]):5.1f} % LRU")
say("miss-focused recall (after the lines): r 0.9 on the uses LRU hits, r_m on the uses LRU misses at that K; % fewer misses "
    "than LRU (overall recall in brackets):")
for K in ks:
    ntot, nm_ = TRUE[:, ~PRE].sum(), LM[K].sum()
    for H in (8, 16):
        if (K, "split", H, (RMS if not SMOKE else (0.3,))[0], 0) not in res:
            continue
        lru = pooled(res[(K, "lru")])
        out = []
        for rm in (RMS if not SMOKE else (0.3,)):
            v = float(np.mean([pooled(res[(K, "split", H, rm, sd)]) for sd in range(2 if not SMOKE else 1)]))
            out.append(f"r_m {rm}: {100 * (lru - v) / lru:5.1f} % [{(0.9 * (ntot - nm_) + rm * nm_) / ntot:.2f}]")
        ctl = {c: float(np.mean([pooled(res[(K, "ctl", c, H, sd)]) for sd in range(2 if not SMOKE else 1)])) for c in CTL}
        say(f"  K {K} H {H:2d} (LRU misses = {100 * nm_ / ntot:.1f} % of check uses): " + ", ".join(out)
            + f" | controls: popularity-only forecast {100 * (lru - ctl['pop0']) / lru:5.1f} %, r 0.9 hits / 0 misses "
            f"{100 * (lru - ctl['hit9m0']) / lru:5.1f} %, r 0.7 uniform false positives {100 * (lru - ctl['uni7']) / lru:5.1f} %")
SF, SL = {"H": 8, "n": 0, "pos": 0, "f": np.zeros(2, int)}, {"H": 8, "n": 0, "pos": 0}
replay(266, "fh", 8, nxt=next_from(noisy_fc(0.7, 70), True), stats=SF)
replay(266, "lru", stats=SL)
pr = lambda a, S: f"recall {a[0] / S['pos']:.2f}, precision {a[0] / max(a[1], 1):.2f}"
say(f"residents' 'used again within 8 checks' at the eviction decisions, K 266 (base rate {SF['pos'] / SF['n']:.2f} on FH's "
    f"trajectory, {SL['pos'] / SL['n']:.2f} on LRU's): the r 0.7 forecast {pr(SF['f'], SF)}; recency (used in the last m checks, "
    f"LRU's trajectory): " + ", ".join(f"m {m} {pr(SL[m], SL)}" for m in (1, 2, 4, 8)))
K = 266
if (K, "noisy", 8, 0.7, 0) in res:
    lru, lrn = pooled(res[(K, "lru")]), pooled(res[(K, "learn")])
    v = float(np.mean([pooled(res[(K, "noisy", 8, 0.7, sd)]) for sd in range(SEEDS)]))
    cut, lcut = 100 * (lru - v) / lru, 100 * (lru - lrn) / lru
    call = "PASS" if cut >= 20 and v < lrn else ("KILL" if cut < 10 or v >= lrn else "BETWEEN")
    say(f"Lines: noisy FH r 0.7 H 8 at K 266 = {v:.1f} misses per kept token, {cut:.1f} % fewer than LRU {lru:.1f} (learned key "
        f"{lcut:.1f} %); PASS >= 20 % and better than the learned key, BETWEEN 10-20 %, KILL < 10 % or no better -> {call}")
say(f"({time.time() - t0:.0f} s)")
if not SMOKE:
    open(os.path.expanduser("~/MoonShots/data/foresight45.txt"), "w").write("\n".join(lines) + "\n")
