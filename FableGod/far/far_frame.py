"""The replay frame shared by far_state.py (copied from far_ceiling.py; M-38's chained cap45 replay, read-only data)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import bisect, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E, R = 48, 512, 16
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
KS = tuple(int(a) for a in sys.argv[1:]) or (266, 147)
H = 8
BIG = 10 ** 6
CAPD = 300.0
MSB = 2162688

if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
P = []
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, ids, tokens = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"], z["tokens"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"rows": {}, "kept": None, "seq": s})
        if s not in order:
            order.append(s)
        if l in c["rows"]:
            c["kept"] = r
        else:
            c["rows"][l] = ids[roff[n]:roff[n] + r].astype(np.int64)
    off = 0
    for s in order:
        c = checks[s]
        k = c["kept"] if c["kept"] is not None else 16
        c["toks"] = tokens[off:off + k].astype(np.int64)
        off += k
    cks = [checks[s] for s in order if len(checks[s]["rows"]) == NL and checks[s]["kept"] is not None]
    res0 = [np.nonzero(z["res0"][l] >= 0)[0] for l in range(NL)]
    P.append((name, cks, res0))
KEPT = [np.array([c["kept"] for c in cks]) for _, cks, _ in P]


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
PCHK = np.array([it[1] for it in ITEMS[0]])
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


LRU_REF = {}
for K in KS:
    miss, stall = replay(K, "lru")
    LRU_REF[K] = (pooled(miss), pooled_chk(stall))
