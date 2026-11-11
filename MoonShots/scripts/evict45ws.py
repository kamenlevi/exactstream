"""M-14b EVICT45-WS (MoonShots, Sun 2026-11-01; follow-up of M-14): the LRU -> Belady gap at the 45 GB cache is
medium-range reuse of the answer's working set (horizon oracle: H 5 49 %, H 10 68 % of the gap), so give the eviction key
in-answer features: uses so far in this prompt, a short-window count (decay 0.8 per call), the mean reuse interval in this
prompt, calls since the prompt started (plus M-14's: recency, decayed count, popularity, rows, kept-row use, prefill flags, layer).
Keys, trained leave-one-prompt-out on Belady labels (log 1 + next-use distance) from LRU states:
  learn    M-14's linear key (reference)
  learnws  the linear key with the in-answer features
  bins     a table: recency bin x in-prompt-uses bin x popularity quintile -> mean label (fallbacks for thin cells)
Same chained replay as M-14 / BigMoE's seed45_replay.py (cap45, res0 forced in per prompt, prompts 2-6 counted).
Lines (QUEUE M-14b, written first): PASS = >= 30 % of the LRU -> MIN gap at K 266, no prompt worse than LRU; KILL = no better
than M-14's linear key (< 16 % of the gap); BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python evict45ws.py (CPU, one core, ~2 min) -> stdout + data/evict45ws.txt"""
import bisect, os, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E, R = 48, 512, 16
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")                  # the capture run's order (cap45_run.txt)
KS = (246, 266, 287)
CAPD = 300.0                                                         # next-use distance cap (items) for the labels
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
say(f"M-14b EVICT45-WS: cap45, {sum(len(c) for _, c, _ in P)} live checks over {len(P)} prompts, chained (BigMoE's seed45 replay); "
    f"res0 {np.mean([len(x) for _, _, r in P for x in r]):.0f} experts per layer")


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


RECB = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55])
IPCB = np.array([1, 2, 3, 5, 9])


def feats(st, cand, i, l, p, pre_set, ws=False):
    """features of the candidate experts at call i: M-14's 9, or with ws=True 13 (+ in-answer), last column = bias"""
    rec = np.log1p(i - st["last"][cand])
    dc = np.log1p(st["score"][cand] / st["w"])
    base = [rec, dc, POP[p][l, cand], st["nrows"][cand], st["kept"][cand].astype(float),
            st["pre"][cand].astype(float), pre_set[cand].astype(float), np.full(len(cand), l / NL)]
    if ws:
        ipc = st["ipc"][cand]
        mri = np.where(ipc >= 2, st["isum"][cand] / np.maximum(ipc - 1, 1), CAPD)
        base += [np.log1p(ipc), st["sd"][cand], np.log1p(mri), np.full(len(cand), np.log1p(i - st["pstart"]))]
    return np.stack(base + [np.ones(len(cand))], 1)


def raw(st, cand, i, l, p):
    """recency, in-prompt uses, popularity for the binned key"""
    return i - st["last"][cand], st["ipc"][cand], POP[p][l, cand]


def replay(K, pol, arg=0, model=None, collect=None):
    """misses per prompt (arrays of per-check sums over layers); collect = list to append (prompt, X, y) samples to"""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        st = {"res": np.zeros(E, bool), "last": np.zeros(E), "score": np.zeros(E), "w": 1.0, "nrows": np.zeros(E),
              "kept": np.zeros(E, bool), "pre": np.zeros(E, bool), "rrpv": np.zeros(E, np.int8),
              "ipc": np.zeros(E), "isum": np.zeros(E), "ilast": np.full(E, -1.0), "sd": np.zeros(E), "pstart": 0}
        pre_set = np.zeros(E, bool)
        for i, (p, t, need, nr, kr, pre) in enumerate(ITEMS[l]):
            st["w"] /= 0.99
            st["sd"] *= 0.8
            if pre:
                pre_set = np.zeros(E, bool); pre_set[need] = True
                st["ipc"][:] = 0; st["isum"][:] = 0; st["ilast"][:] = -1; st["pstart"] = i
            else:
                seen = st["ilast"][need] >= 0
                st["isum"][need[seen]] += i - st["ilast"][need[seen]]
                st["ipc"][need] += 1
            st["ilast"][need] = i
            st["sd"][need] += 1
            hit = st["res"][need]
            if t >= 0:
                miss[p][t] += int((~hit).sum())
            new, old = need[~hit], need[hit]
            if pol == "lip":
                lo = st["last"][st["res"]].min() - 1 if st["res"].any() else float(i)
                st["last"][new] = lo
                st["last"][old] = i
            elif pol == "mid":
                st["last"][new] = i - arg
                st["last"][old] = i
            else:
                st["last"][need] = i
            st["score"][need] += st["w"]
            st["nrows"][need] = nr; st["kept"][need] = kr; st["pre"][need] = pre
            st["rrpv"][old] = 0
            st["rrpv"][new] = arg if pol == "srrip" else 0
            st["res"][need] = True
            over = int(st["res"].sum()) - K
            if collect is not None and t >= 0:
                cand = np.nonzero(st["res"])[0]
                cand = cand[~np.isin(cand, need)]
                if cand.size:
                    sel = cand if cand.size <= 64 else np.random.default_rng(i * 131 + l).choice(cand, 64, replace=False)
                    collect.append((p, feats(st, sel, i, l, p, pre_set), feats(st, sel, i, l, p, pre_set, True),
                                    np.stack(raw(st, sel, i, l, p), 1), np.log1p([next_use(l, e, i) for e in sel.tolist()])))
            if over <= 0:
                continue
            nm = np.zeros(E, bool); nm[need] = True
            cand = np.nonzero(st["res"] & ~nm)[0]
            if pol in ("lru", "lip", "mid"):
                vict = cand[np.argsort(st["last"][cand], kind="stable")[:over]]
            elif pol == "lfu":
                vict = cand[np.argsort(st["score"][cand], kind="stable")[:over]]
            elif pol == "srrip":
                vict = []
                while len(vict) < over:
                    pool = cand[~np.isin(cand, vict)]
                    at3 = pool[st["rrpv"][pool] >= 3]
                    if at3.size:
                        k = min(over - len(vict), at3.size)
                        vict += at3[np.argsort(st["last"][at3], kind="stable")[:k]].tolist()
                    else:
                        st["rrpv"][pool] = np.minimum(st["rrpv"][pool] + 1, 3)
                vict = np.array(vict)
            elif pol == "learn":
                pred = feats(st, cand, i, l, p, pre_set) @ model
                vict = cand[np.argsort(-pred, kind="stable")[:over]]
            elif pol == "learnws":
                pred = feats(st, cand, i, l, p, pre_set, True) @ model
                vict = cand[np.argsort(-pred, kind="stable")[:over]]
            elif pol == "bins":
                pred = model(*raw(st, cand, i, l, p))
                vict = cand[np.argsort(-pred, kind="stable")[:over]]
            elif pol == "hor":                                       # LRU, but a next use within arg calls protects
                d = np.array([next_use(l, e, i) for e in cand.tolist()])
                key = np.where(d <= arg, 1e9, 0.0) + st["last"][cand]
                vict = cand[np.argsort(key, kind="stable")[:over]]
            else:                                                    # Belady
                d = np.array([next_use(l, e, i) for e in cand.tolist()])
                vict = cand[np.argsort(-d, kind="stable")[:over]]
            st["res"][vict] = False
        if os.path.exists(PAUSE):
            while os.path.exists(PAUSE):
                time.sleep(20)
    return miss


def pooled(miss, ps=range(1, 6)):
    return sum(miss[p].sum() for p in ps) / sum(KEPT[p].sum() for p in ps)


res = {}
ARMS = [("lru", 0), ("min", 0)]
for K in KS:
    for pol, a in ARMS:
        res[(K, pol, a)] = replay(K, pol, a)


def binned(Rr, Ip, Pp, Y, qs):
    rb, ib, pb = np.digitize(Rr, RECB), np.digitize(Ip, IPCB), np.digitize(Pp, qs)
    cell = (rb * 6 + ib) * 5 + pb
    s1 = np.bincount(cell, Y, 300); n1 = np.bincount(cell, None, 300)
    c2 = rb * 5 + pb
    s2 = np.bincount(c2, Y, 50); n2 = np.bincount(c2, None, 50)
    t1 = np.where(n1 >= 20, s1 / np.maximum(n1, 1), np.nan)
    t2 = np.where(n2 >= 20, s2 / np.maximum(n2, 1), Y.mean())
    def f(r, ip, pp):
        a_, b_, c_ = np.digitize(r, RECB), np.digitize(ip, IPCB), np.digitize(pp, qs)
        v = t1[(a_ * 6 + b_) * 5 + c_]
        return np.where(np.isnan(v), t2[a_ * 5 + c_], v)
    return f


for K in KS:
    smp = []
    replay(K, "lru", collect=smp)
    PID = np.concatenate([np.full(len(y), q) for q, _, _, _, y in smp])
    X9 = np.concatenate([x for _, x, _, _, _ in smp]); X13 = np.concatenate([x for _, _, x, _, _ in smp])
    RW = np.concatenate([r for _, _, _, r, _ in smp]); Y = np.concatenate([y for _, _, _, _, y in smp])
    out = {k: [np.zeros(len(c)) for _, c, _ in P] for k in ("learn", "learnws", "bins")}
    for p in range(1, len(P)):
        m = PID != p
        w9 = np.linalg.lstsq(X9[m], Y[m], rcond=None)[0]
        w13 = np.linalg.lstsq(X13[m], Y[m], rcond=None)[0]
        qs = np.quantile(RW[m, 2], [0.2, 0.4, 0.6, 0.8])
        fb = binned(RW[m, 0], RW[m, 1], RW[m, 2], Y[m], qs)
        out["learn"][p] = replay(K, "learn", model=w9)[p]
        out["learnws"][p] = replay(K, "learnws", model=w13)[p]
        out["bins"][p] = replay(K, "bins", model=fb)[p]
    for k in out:
        res[(K, k, 0)] = out[k]
    if K == 266:
        coef13 = np.linalg.lstsq(X13, Y, rcond=None)[0]
    say(f"  K {K}: lru {pooled(res[(K, 'lru', 0)]):.1f}, learn {pooled(out['learn']):.1f}, learnws {pooled(out['learnws']):.1f}, "
        f"bins {pooled(out['bins']):.1f}, min {pooled(res[(K, 'min', 0)]):.1f} ({time.time() - t0:.0f} s)")
keys = [("lru", 0), ("learn", 0), ("learnws", 0), ("bins", 0), ("min", 0)]
say("misses per kept token, prompts 2-6 pooled; share of the LRU -> MIN gap taken:")
say("  K    " + " ".join(f"{k[0]:>8s}" for k in keys))
for K in KS:
    lru, mn = pooled(res[(K, "lru", 0)]), pooled(res[(K, "min", 0)])
    say(f"  {K}  " + " ".join(f"{pooled(res[(K,) + k]):8.1f}" for k in keys))
    say(f"       " + " ".join(f"{100 * (lru - pooled(res[(K,) + k])) / (lru - mn):7.0f}%" for k in keys))
FN = ["log1p recency", "log1p decayed count", "popularity", "rows/16", "kept-row use", "last use = prefill", "in prefill set",
      "layer/48", "log1p in-prompt uses", "short-window count", "log1p mean reuse interval", "log1p calls since prompt start", "bias"]
say("  learnws key at K 266 (all prompts; larger = farther predicted next use): " +
    ", ".join(f"{n} {c:+.3f}" for n, c in zip(FN, coef13)))
online = [("learnws", 0), ("bins", 0)]
best = min(online, key=lambda k: pooled(res[(266,) + k]))
lru, mn = pooled(res[(266, "lru", 0)]), pooled(res[(266, "min", 0)])
share = (lru - pooled(res[(266,) + best])) / (lru - mn)
worst = max(res[(266,) + best][p].sum() / res[(266, "lru", 0)][p].sum() - 1 for p in range(1, len(P)))
say("per prompt at K 266, misses per kept token: LRU / learn / best of learnws, bins / MIN")
for p in range(1, len(P)):
    say(f"  {P[p][0][:30]:30s} " + " ".join(f"{res[(266,) + k][p].sum() / KEPT[p].sum():6.1f}" for k in
                                             (("lru", 0), ("learn", 0), best, ("min", 0))))
call = "PASS" if share >= 0.30 and worst <= 0.0 else ("KILL" if share < 0.16 else "BETWEEN")
say(f"Lines: best in-answer key {best[0]} takes {100 * share:.1f} % of the LRU -> MIN gap at K 266 (worst prompt "
    f"{100 * worst:+.1f} % vs LRU; PASS >= 30 % with no prompt worse, KILL < 16 %) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/evict45ws.txt"), "w").write("\n".join(lines) + "\n")
