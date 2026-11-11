"""M-14 EVICT45 (MoonShots, Sun 2026-11-01; SSD-miss wall): how much of the LRU -> Belady gap at the 45 GB default's cache
(BigMoE SEED45: K 266 LRU 43.5 -> MIN 19.2 misses per kept token) do online eviction policies take?
Published: SRRIP (Jaleel 2010), LRU-insertion / DIP (Qureshi 2007), Belady-labelled learned eviction (Hawkeye 2016; FlashMoE
2601.17063). Residency only, so every token stays exact.
Replay = BigMoE's scripts/seed45_replay.py, unchanged in substance: cap45 (the exact REF2 loop at C 160, real DFlash2 16-row
checks, six prompts in one process), CHAINED: the cache carries over, each prompt's prefill is its res0 forced in as one
uncounted call, misses counted on prompts 2-6. Per layer a check needs the union of its 16 rows' experts; victims are taken
outside the current call's needs; K = resident MSB-3 slots per layer.
Policies (per layer):
  LRU, LFU (decayed 0.99)   references (the live loop runs LRU)
  SRRIP-n                   2-bit re-reference values; a hit sets 0, a new expert enters at n; victims = value 3, oldest
                            first, everyone else ages by 1 when none is at 3
  LIP / MID-D               a new expert enters at the LRU end (LIP) or D calls back (MID); a hit moves it to the front
  LEARN                     key = a linear fit of log(1 + next-use distance) on recency, decayed count, popularity (other
                            prompts), rows that used it last time, a kept row used it, the last use was the prefill, in this
                            prompt's prefill set, layer; trained on Belady labels from LRU states of the OTHER prompts
                            (leave-one-prompt-out); the farthest predicted next use leaves first
  MIN                       Belady (future known)
Lines (QUEUE M-14, written first): PASS = the best online policy takes >= 30 % of the LRU -> MIN gap at K 266, no prompt worse
than LRU by > 2 %; KILL = < 10 %; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python evict45.py (CPU, one core, a few minutes) -> stdout + data/evict45.txt"""
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
say(f"M-14 EVICT45: cap45, {sum(len(c) for _, c, _ in P)} live checks over {len(P)} prompts, chained (BigMoE's seed45 replay); "
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


NF = 9


def feats(st, cand, i, l, p, pre_set):
    """[n, NF] features of the candidate experts at call i"""
    rec = np.log1p(i - st["last"][cand])
    dc = np.log1p(st["score"][cand] / st["w"])
    return np.stack([rec, dc, POP[p][l, cand], st["nrows"][cand], st["kept"][cand].astype(float),
                     st["pre"][cand].astype(float), pre_set[cand].astype(float), np.full(len(cand), l / NL),
                     np.ones(len(cand))], 1)


def replay(K, pol, arg=0, model=None, collect=None):
    """misses per prompt (arrays of per-check sums over layers); collect = list to append (prompt, X, y) samples to"""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        st = {"res": np.zeros(E, bool), "last": np.zeros(E), "score": np.zeros(E), "w": 1.0, "nrows": np.zeros(E),
              "kept": np.zeros(E, bool), "pre": np.zeros(E, bool), "rrpv": np.zeros(E, np.int8)}
        pre_set = np.zeros(E, bool)
        for i, (p, t, need, nr, kr, pre) in enumerate(ITEMS[l]):
            st["w"] /= 0.99
            if pre:
                pre_set = np.zeros(E, bool); pre_set[need] = True
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
                    collect.append((p, feats(st, sel, i, l, p, pre_set), np.log1p([next_use(l, e, i) for e in sel.tolist()])))
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
ARMS = [("lru", 0), ("lfu", 0), ("srrip", 1), ("srrip", 2), ("srrip", 3), ("lip", 0), ("mid", 20), ("mid", 60), ("min", 0)]
for K in KS:
    for pol, a in ARMS:
        res[(K, pol, a)] = replay(K, pol, a)
    say(f"  K {K}: " + ", ".join(f"{pol}{a if a else ''} {pooled(res[(K, pol, a)]):.1f}" for pol, a in ARMS) +
        f" ({time.time() - t0:.0f} s)")
# learned key: LRU states, Belady labels, leave-one-prompt-out (one chained run per held-out prompt, its own model)
coef = {}
for K in KS:
    smp = []
    replay(K, "lru", collect=smp)
    X = np.concatenate([x for _, x, _ in smp]); Y = np.concatenate([y for _, _, y in smp])
    PID = np.concatenate([np.full(len(y), q) for q, _, y in smp])
    coef[K] = np.linalg.lstsq(X, Y, rcond=None)[0]
    learn = [np.zeros(len(c)) for _, c, _ in P]
    for p in range(1, len(P)):
        m = PID != p
        w_p = np.linalg.lstsq(X[m], Y[m], rcond=None)[0]
        learn[p] = replay(K, "learn", model=w_p)[p]
    res[(K, "learn", 0)] = learn
    say(f"  K {K}: learn {pooled(learn):.1f} ({time.time() - t0:.0f} s)")
say("misses per kept token, prompts 2-6 pooled; share of the LRU -> MIN gap taken:")
names = [f"{pol}{a if a else ''}" for pol, a in ARMS[:-1]] + ["learn", "min"]
keys = [(pol, a) for pol, a in ARMS[:-1]] + [("learn", 0), ("min", 0)]
say("  K    " + " ".join(f"{n:>8s}" for n in names))
for K in KS:
    lru, mn = pooled(res[(K, "lru", 0)]), pooled(res[(K, "min", 0)])
    say(f"  {K}  " + " ".join(f"{pooled(res[(K,) + k]):8.1f}" for k in keys))
    say(f"       " + " ".join(f"{100 * (lru - pooled(res[(K,) + k])) / (lru - mn):7.0f}%" for k in keys))
FN = ["log1p recency", "log1p decayed count", "popularity", "rows/16", "kept-row use", "last use = prefill", "in prefill set",
      "layer/48", "bias"]
say("  learned key at K 266 (all prompts; larger = farther predicted next use): " +
    ", ".join(f"{n} {c:+.3f}" for n, c in zip(FN, coef[266])))
online = [k for k in keys if k[0] not in ("min", "lru")]
best = min(online, key=lambda k: pooled(res[(266,) + k]))
lru, mn = pooled(res[(266, "lru", 0)]), pooled(res[(266, "min", 0)])
share = (lru - pooled(res[(266,) + best])) / (lru - mn)
worst = max(res[(266,) + best][p].sum() / KEPT[p].sum() / (res[(266, "lru", 0)][p].sum() / KEPT[p].sum()) - 1
            for p in range(1, len(P)))
say("per prompt at K 266, misses per kept token: LRU / best online / MIN")
for p in range(1, len(P)):
    say(f"  {P[p][0][:30]:30s} {res[(266, 'lru', 0)][p].sum() / KEPT[p].sum():6.1f} "
        f"{res[(266,) + best][p].sum() / KEPT[p].sum():6.1f} {res[(266, 'min', 0)][p].sum() / KEPT[p].sum():6.1f}")
call = "PASS" if share >= 0.30 and worst <= 0.02 else ("KILL" if share < 0.10 else "BETWEEN")
say(f"Lines: best online policy {best[0]}{best[1] if best[1] else ''} takes {100 * share:.1f} % of the LRU -> MIN gap at K 266 "
    f"(worst prompt {100 * worst:+.1f} % vs LRU; PASS >= 30 % with <= +2 %, KILL < 10 %) -> {call}")
say("descriptive (after the lines): horizon-limited oracles at K 266 = LRU, but experts needed again within H calls are "
    "protected (H calls = H checks, plus a prefill call at prompt starts); share of the LRU -> MIN gap:")
for H in (1, 2, 3, 5, 10, 20):
    v = pooled(replay(266, "hor", H))
    say(f"  H {H:2d}: {v:5.1f} misses per kept token ({100 * (lru - v) / (lru - mn):3.0f} % of the gap)")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/evict45.txt"), "w").write("\n".join(lines) + "\n")
