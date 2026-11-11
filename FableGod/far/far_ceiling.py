"""FAR ceiling (Fable FAR, Wed 2026-11-04): the TRUE future tokens as a perfect far-ahead draft, mapped through cheap
token -> expert maps, on MoonShots' M-38 frame (scripts/foresight45.py, commit 99df774; loader / replay / next_from copied
unchanged in substance, read-only use of ~/BigMoE/data/cap45 and ~/MoonShots).
Row -> token alignment (probe_shift, all six prompts): row j of a check carries token tokens[off + j] for j < kept (shift 0
gives the highest layer-0 same-token Jaccard, 0.54-0.66, vs <= 0.38 for shifts -1 / +1 / +2). Rows j >= kept are rejected
drafts whose tokens the capture does not hold (a perfect far draft would not have produced them either).
Forecasts (all at H 8, within the prompt, prefills never forecast; FH policy = M-38's):
  PERFECT-ALL   M-38's perfect forecast (every row's true experts)                       [reference, -33.7 % at K 266]
  PERFECT-KEPT  the kept rows' true experts only (the ceiling of any map of known tokens; the rejected rows are the rest)
  ALL-k         a token -> expert table from ALL six prompts' kept rows (top-k by count per token and layer): a
                coverage upper bound for a token-identity table (it has seen the held-out prompt)
  LOPO-k        the same table from the OTHER five prompts (leave-one-prompt-out); unseen tokens -> popularity top-k
  MEMO-m        online: at decision i the token's experts = the union of its last m occurrences in checks <= i of this
                prompt (the engine knows every row's route it computed); unseen -> LOPO-10 (and popularity)
  MEMO-m-only   the same without the LOPO / popularity fallback
Measured on the forecast's own trajectory (M-38's convention): residents' 'used again within 8 checks' recall / precision
at the eviction decisions, misses per kept token (prompts 2-6), % fewer than LRU, and stalled layers per check (layers with
>= 1 miss: the EXACT check's SSD wait is latency-bound, ~1.4 misses per layer [GOAL.md]).
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_ceiling.py [K ...] (one core, ~2-4 min) -> stdout + far_ceiling.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import bisect, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
OUT = os.path.expanduser("~/FableGod/far/far_ceiling.txt")
NL, E, R = 48, 512, 16
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
KS = tuple(int(a) for a in sys.argv[1:]) or (266, 147)
H = 8
BIG = 10 ** 6
CAPD = 300.0
MSB = 2162688
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
P = []                                                               # (name, checks, res0); checks carry 'toks' (kept rows' tokens)
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, ids, tokens = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"], z["tokens"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"rows": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["rows"]:
            c["kept"] = r
        else:
            c["rows"][l] = ids[roff[n]:roff[n] + r].astype(np.int64)
    off = 0
    for s in order:                                                  # token offsets over ALL checks (kept 16 when no commit entry)
        c = checks[s]
        k = c["kept"] if c["kept"] is not None else 16
        c["toks"] = tokens[off:off + k].astype(np.int64)
        off += k
    cks = [checks[s] for s in order if len(checks[s]["rows"]) == NL and checks[s]["kept"] is not None]
    res0 = [np.nonzero(z["res0"][l] >= 0)[0] for l in range(NL)]
    P.append((name, cks, res0))
POP = []
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
say(f"FAR CEILING ({time.strftime('%a %Y-%m-%d %H:%M')}): cap45, {sum(len(c) for _, c, _ in P)} live checks over {len(P)} prompts "
    f"({', '.join(str(len(c)) for _, c, _ in P)}), chained (M-38's frame); K {' / '.join(map(str, KS))}; H {H}")


def items_for(l):
    out = []
    for p, (_, cks, res0) in enumerate(P):
        out.append((p, -1, res0[l], True))
        for t, c in enumerate(cks):
            out.append((p, t, np.unique(c["rows"][l]), False))
    return out


ITEMS = [items_for(l) for l in range(NL)]
NI = len(ITEMS[0])
PRE = np.array([it[3] for it in ITEMS[0]])
PRM = np.array([it[0] for it in ITEMS[0]])
PCHK = np.array([it[1] for it in ITEMS[0]])                          # check index within the prompt (-1 = prefill)
PEND = np.array([max(j for j in range(NI) if PRM[j] == PRM[i]) for i in range(NI)])
TRUE = np.zeros((NL, NI, E), bool)
for l in range(NL):
    for i, it in enumerate(ITEMS[l]):
        TRUE[l, i, it[2]] = True
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
    nxt = np.full((NL, NI, E), BIG, np.int32)
    cur = np.full((NL, E), BIG, np.int32)
    for i in range(NI - 1, -1, -1):
        nxt[:, i] = cur
        if within and PRE[i]:
            cur = np.full((NL, E), BIG, np.int32)
        else:
            cur = np.where(FC[:, i], i, cur)
    return nxt


NXW = next_from(TRUE, True)


def replay(K, pol, nxt=None, look=None, stats=None):
    """misses and stalled layers per prompt (per-check sums over layers); pol: lru, min, fh"""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    stall = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        res = np.zeros(E, bool); last = np.zeros(E)
        for i, (p, t, need, pre) in enumerate(ITEMS[l]):
            hit = res[need]
            if t >= 0:
                nm_ = int((~hit).sum())
                miss[p][t] += nm_; stall[p][t] += nm_ > 0
            last[need] = i
            res[need] = True
            over = int(res.sum()) - K
            if over <= 0:
                continue
            nm = np.zeros(E, bool); nm[need] = True
            cand = np.nonzero(res & ~nm)[0]
            if cand.size < over:
                cand = np.nonzero(res)[0]
            if pol == "lru":
                vict = cand[np.argsort(last[cand], kind="stable")[:over]]
            elif pol == "fh":
                d = (look(l, i) if look is not None else nxt[l, i] - i)[cand].astype(float)
                fc = d <= H
                if stats is not None:
                    y = NXW[l, i, cand] - i <= H
                    stats["n"] += y.size; stats["pos"] += int(y.sum())
                    stats["tp"] += int((y & fc).sum()); stats["fp"] += int((~y & fc).sum())
                key = np.where(fc, 1e9 - 1e3 * d, 0.0) + last[cand]
                vict = cand[np.argsort(key, kind="stable")[:over]]
            else:
                d = np.array([next_use(l, e, i) for e in cand.tolist()])
                vict = cand[np.argsort(-d, kind="stable")[:over]]
            res[vict] = False
        while os.path.exists(PAUSE):
            time.sleep(20)
    return miss, stall


def pooled(a, ps=range(1, 6)):
    return sum(a[p].sum() for p in ps) / sum(KEPT[p].sum() for p in ps)


def pooled_chk(a, ps=range(1, 6)):
    return sum(a[p].sum() for p in ps) / sum(len(a[p]) for p in ps)


# ---- token -> expert tables -------------------------------------------------------------------------------------------
def table(prompts):
    """{token: [48, 512] counts} over the kept rows of the given prompts"""
    T = {}
    for p in prompts:
        for c in P[p][1]:
            for j, tk in enumerate(c["toks"].tolist()):
                a = T.get(tk)
                if a is None:
                    a = T[tk] = np.zeros((NL, E), np.int32)
                for l in range(NL):
                    a[l, c["rows"][l][j]] += 1
    return T


def topk_sets(T, k):
    """{token: bool [48, 512]} of the top-k experts per layer (ties by count, then id)"""
    S = {}
    for tk, a in T.items():
        if k >= E:
            S[tk] = a > 0
        else:
            idx = np.argsort(-a, axis=1, kind="stable")[:, :k]
            m = np.zeros((NL, E), bool)
            np.put_along_axis(m, idx, True, axis=1)
            S[tk] = m & (a > 0)
    return S


def pop_sets(k):
    """popularity top-k per layer, leave-one-prompt-out: [6, 48, 512]"""
    out = np.zeros((len(P), NL, E), bool)
    for p in range(len(P)):
        idx = np.argsort(-POP[p], axis=1, kind="stable")[:, :k]
        np.put_along_axis(out[p], idx, True, axis=1)
    return out


ALLT = table(range(len(P)))
LOPOT = [table([q for q in range(len(P)) if q != p]) for p in range(len(P))]
say(f"tables built: ALL {len(ALLT)} tokens; LOPO coverage of the held-out prompts' kept-row tokens: "
    + ", ".join(f"{np.mean([tk in LOPOT[p] for c in P[p][1] for tk in c['toks'].tolist()]):.2f}" for p in range(len(P)))
    + f" ({time.time() - t0:.0f} s)")

# token determinism per layer: same-token pairs within a prompt (last occurrence's top-10 vs the row's true top-10)
rec_prev, jac_prev, n_pairs, rec_lopo = np.zeros(NL), np.zeros(NL), 0, np.zeros(NL)
n_lopo = 0
LOPO10 = [topk_sets(LOPOT[p], 10) for p in range(len(P))]
for p, (_, cks, _) in enumerate(P):
    seen = {}
    for c in cks:
        for j, tk in enumerate(c["toks"].tolist()):
            if tk in seen:
                prev = seen[tk]
                n_pairs += 1
                for l in range(NL):
                    a, b = set(prev[l].tolist()), set(c["rows"][l][j].tolist())
                    rec_prev[l] += len(a & b) / 10
                    jac_prev[l] += len(a & b) / len(a | b)
            if tk in LOPO10[p]:
                n_lopo += 1
                m = LOPO10[p][tk]
                for l in range(NL):
                    rec_lopo[l] += m[l][c["rows"][l][j]].mean()
            seen[tk] = np.stack([c["rows"][l][j] for l in range(NL)])
say("token determinism of the route (kept rows): per-row recall@10 of the row's true top-10 from the token's PREVIOUS "
    f"occurrence in the prompt ({n_pairs} pairs) / from the LOPO table's top-10 ({n_lopo} rows):")
for g in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 48)):
    sl = slice(*g)
    say(f"  layers {g[0]:2d}-{g[1] - 1:2d}: prev-occurrence recall {rec_prev[sl].mean() / n_pairs:.3f} (Jaccard "
        f"{jac_prev[sl].mean() / n_pairs:.3f}), LOPO top-10 recall {rec_lopo[sl].mean() / max(n_lopo, 1):.3f}")

# per-prompt occurrence lists for the online memo: {p: {token: [(check t, ids [48, 10]), ...]}}
OCC = []
for p, (_, cks, _) in enumerate(P):
    d = {}
    for t, c in enumerate(cks):
        for j, tk in enumerate(c["toks"].tolist()):
            d.setdefault(tk, ([], []))
            d[tk][0].append(t); d[tk][1].append(np.stack([c["rows"][l][j] for l in range(NL)]))
    OCC.append({tk: (ts, np.stack(ids)) for tk, (ts, ids) in d.items()})
ITEM_OF = {}                                                         # (p, t) -> item index
for i in range(NI):
    if PCHK[i] >= 0:
        ITEM_OF[(PRM[i], PCHK[i])] = i


def fc_from_sets(S_of_p, pop=None):
    """per-check forecast matrix from token -> bool[48, 512] sets (S_of_p(p) -> dict); pop: [6, 48, 512] fallback"""
    FC = np.zeros((NL, NI, E), bool)
    for p, (_, cks, _) in enumerate(P):
        S = S_of_p(p)
        for t, c in enumerate(cks):
            i = ITEM_OF[(p, t)]
            for tk in c["toks"].tolist():
                m = S.get(tk)
                if m is None and pop is not None:
                    m = pop[p]
                if m is not None:
                    FC[:, i] |= m
    return FC


def memo_look(m_occ, fallback):
    """per-look forecast: at decision (l, i), the next H checks' tokens through the memo of checks <= i (union of the
    token's last m_occ occurrences), unseen tokens through fallback (dict token -> bool[48, 512]) or nothing"""
    def f(l, i):
        p, t0_ = PRM[i], PCHK[i]
        d = np.full(E, BIG)
        j2 = min(i + H, PEND[i])
        for j in range(i + 1, j2 + 1):
            c = P[p][1][PCHK[j]]
            m = np.zeros(E, bool)
            for tk in c["toks"].tolist():
                occ = OCC[p].get(tk)
                k = bisect.bisect_right(occ[0], t0_) if occ else 0
                if k > 0:
                    m[occ[1][max(0, k - m_occ):k, l].ravel()] = True
                else:
                    fb = fallback(p).get(tk) if fallback is not None else None
                    if fb is not None:
                        m |= fb[l]
            d[m & (d == BIG)] = j - i
        return d
    return f


def run(K, tag, pol, nxt=None, look=None, ref=None):
    S = {"n": 0, "pos": 0, "tp": 0, "fp": 0} if pol == "fh" else None
    miss, stall = replay(K, pol, nxt=nxt, look=look, stats=S)
    m, s = pooled(miss), pooled_chk(stall)
    txt = f"  K {K} {tag:24s}: {m:6.1f} misses / kept token [{m * MSB / 1e6:6.1f} MB], stalled layers / check {s:5.1f}"
    if ref is not None:
        txt += f", {100 * (ref[0] - m) / ref[0]:5.1f} % fewer misses than LRU, stalls {100 * (ref[1] - s) / ref[1]:5.1f} % fewer"
    if S is not None and S["pos"]:
        fpos = S["tp"] + S["fp"]
        txt += (f" | recall {S['tp'] / S['pos']:.2f}, precision {S['tp'] / max(fpos, 1):.2f} (base {S['pos'] / S['n']:.2f}, "
                f"forecast-positive {fpos / S['n']:.2f})")
    say(txt + f" ({time.time() - t0:.0f} s)")
    return m, s


FC_KEPT = np.zeros((NL, NI, E), bool)                                # the kept rows' true experts
for p, (_, cks, _) in enumerate(P):
    for t, c in enumerate(cks):
        i = ITEM_OF[(p, t)]
        for l in range(NL):
            FC_KEPT[l, i, c["rows"][l][:c["kept"]]] = True
POPS = {k: pop_sets(k) for k in (5, 10, 20)}
for K in KS:
    say(f"K {K} ({'45 GB EXACT' if K == 266 else '24 GB' if K == 147 else ''}):")
    lru = run(K, "LRU", "lru")
    run(K, "OPT (Belady)", "min", ref=lru)
    run(K, "PERFECT-ALL (M-38)", "fh", nxt=NXW, ref=lru)
    run(K, "PERFECT-KEPT rows", "fh", nxt=next_from(FC_KEPT), ref=lru)
    for k in (5, 10, 20):
        AS = topk_sets(ALLT, k)
        run(K, f"ALL-{k}", "fh", nxt=next_from(fc_from_sets(lambda p: AS)), ref=lru)
    for k in (5, 10, 20):
        LS = [topk_sets(LOPOT[p], k) for p in range(len(P))]
        run(K, f"LOPO-{k} (+pop {k})", "fh", nxt=next_from(fc_from_sets(lambda p: LS[p], POPS[k])), ref=lru)
        run(K, f"LOPO-{k} only", "fh", nxt=next_from(fc_from_sets(lambda p: LS[p])), ref=lru)
    run(K, "POP-20 only", "fh", nxt=next_from(fc_from_sets(lambda p: {}, POPS[20])), ref=lru)
    for m_occ in (1, 2, 4):
        run(K, f"MEMO-{m_occ} (+LOPO-10 +pop)", "fh", look=memo_look(m_occ, lambda p: LOPO10[p]), ref=lru)
    run(K, "MEMO-2 only", "fh", look=memo_look(2, None), ref=lru)
    run(K, "MEMO-4 (+ALL-10)", "fh", look=memo_look(4, lambda p, A=topk_sets(ALLT, 10): A), ref=lru)
say(f"({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
