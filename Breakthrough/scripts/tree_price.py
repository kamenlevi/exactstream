"""TREES price (NewIdeas Sun 06:1x, chat added 08:3x; lines in SCOREBOARD 'TREES LINES'; CPU): exact draft trees vs today's
chain on DFlash2's own lattices (data/trees/lattice_{code,chat,rest}.npz from tree_dump.py: the L1-20 drafter along the
selector's own trajectory; code = Leap's 24 HumanEval captures, chat / agent / mgsm / gsm = MTP Head's D-02a captures).
Per block: each depth's 16 candidates (path-independent: DFlash2 drafts every depth at once) and the selector's first-order
lattice P(candidate j at depth d | candidate i at depth d - 1). A tree of R rows = the anchor + the R - 1 most probable paths:
  tree     k-best under the first-order lattice (each node's children ranked by P(child | that node)): as the selector would
           grow it, siblings entering only where it is unsure
  medusa   the pre-registered ranking: every branch reuses the probabilities along the selector's own path
Kept = the longest true prefix in the tree + 1 (capped at the sample's end, as cand21); the chain = the selector's picks.
GB per check from BigMoE's rows curve (4-bit at C 160; linear between its points); tree rows count like chain rows (MoonShots'
M-10: tree rows cost 0.84-0.91x of chain rows in experts, so this is conservative). Also: chains truncated by the running
product of the pick's probability (L1-14, Leap's t), trees grown while the next path's probability >= tau (EAGLE-2's dynamic
size, <= 16 rows), the top-16 oracle, and an exact-capped check (a depth where the answer's token is not the capture's own
top-1 ends the block there: 0.3-5 % of positions).
Usage: [BigMoE venv] python tree_price.py [lattice.npz ...] -> stdout + data/trees/price.txt"""
import glob, heapq, os, re, sys
import numpy as np

D = os.path.expanduser("~/Breakthrough/data/trees")
FILES = sys.argv[1:] or [f for f in (f"{D}/lattice_{t}.npz" for t in ("code", "chat", "rest")) if os.path.exists(f)]
Z = [np.load(f) for f in FILES]
cat = lambda k: np.concatenate([z[k] for z in Z])
CAND, TRANS, PICK, TRUE, VTOP, NT = cat("cand"), cat("trans"), cat("pick"), cat("true"), cat("vtop"), cat("ntrue")
SCAND, SPROB, DOM = cat("sel_cand"), cat("sel_prob").astype(np.float64), cat("dom")
SAMPLE = np.concatenate([z["sample"] + 1000 * i for i, z in enumerate(Z)])
B, DEP, K = CAND.shape
RG = np.array([1, 2, 4, 8, 12, 16]); GBP = np.array([1.40, 2.29, 3.82, 6.29, 8.32, 10.03])
gb = lambda r: float(np.interp(r, RG, GBP))
ROWS = (16, 12, 8, 6, 4, 3, 2)
TS, TAUS = (0.5, 0.3, 0.1), (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
CAP = np.where(NT == DEP, DEP + 1, NT)                               # tokens a block can keep (the sample's end caps it)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def kbest(first, ordr, srt, rows=None, tau=None, cap=16):
    """k-best paths (tuples of candidate indices): each popped node pushes its best child and its next sibling. first(d, path)
    gives the parent row for depth d's children; ordr / srt [DEP, rows, K] = children order and sorted probabilities"""
    lim = rows - 1 if rows else cap - 1
    out, heap = set(), [(-srt[0, 0, 0], (int(ordr[0, 0, 0]),), 0, 1.0)]
    while heap and len(out) < lim:
        nq, path, r, pp = heapq.heappop(heap)
        if tau is not None and -nq < tau:
            break
        out.add(path)
        d = len(path)
        if d < DEP:
            i = first(d, path)
            heapq.heappush(heap, (nq * srt[d, i, 0], path + (int(ordr[d, i, 0]),), 0, -nq))
        if r + 1 < K:
            i = first(d - 1, path[:-1])
            heapq.heappush(heap, (-pp * srt[d - 1, i, r + 1], path[:-1] + (int(ordr[d - 1, i, r + 1]),), r + 1, pp))
    return out


def walk(T, J, nt):
    d, path = 0, ()
    while d < nt and J[d] < K and path + (int(J[d]),) in T:
        path += (int(J[d]),); d += 1
    return d


# ---------------------------------------------------------------- per block: every variant's (kept, rows)
V = {}                                                                # name -> [B, 2] (kept, rows)


def put(name, b, kept, rows):
    if name not in V:
        V[name] = np.zeros((B, 2))
    V[name][b] = kept, rows


SHAPE = np.zeros((B, 3))                                              # tree16: depth reached, branch points, off-chain nodes
XCAP = np.zeros(B, int)
for b in range(B):
    nt, cap = int(NT[b]), int(CAP[b])
    Jr = np.full(DEP, K); Js = np.full(DEP, K)                        # the true token's index: raw (cand) / sorted (sel_cand)
    for d in range(nt):
        w = np.nonzero(CAND[b, d] == TRUE[b, d])[0]; Jr[d] = w[0] if len(w) else K
        w = np.nonzero(SCAND[b, d] == TRUE[b, d])[0]; Js[d] = w[0] if len(w) else K
    hit = PICK[b, :nt] == TRUE[b, :nt]
    kc = int(np.argmin(hit)) if not hit.all() else nt                  # the chain's accepted drafts
    mm = np.nonzero(TRUE[b, :nt] != VTOP[b, :nt])[0]
    XCAP[b] = (mm[0] if len(mm) else nt) + 1
    tr = TRANS[b].astype(np.float64)
    ordr = np.argsort(-tr, axis=-1, kind="stable"); srt = np.take_along_axis(tr, ordr, -1)
    first1 = lambda d, path: path[-1] if d >= 1 and path else 0     # first-order: the parent's own row (depth 0: the anchor's)
    ords = np.broadcast_to(np.arange(K), (DEP, 1, K)); srs = SPROB[b][:, None, :]
    firstm = lambda d, path: 0                                        # medusa: one row per depth (the selector's own path)
    for R in ROWS:
        rr = min(R, cap)
        put(f"chain {R}", b, min(min(kc, R - 1) + 1, cap), rr)
        T = kbest(first1, ordr, srt, rows=R)
        put(f"tree {R}", b, min(walk(T, Jr, nt) + 1, cap), rr)
        if R == 16:
            depth = max(len(p) for p in T)
            kids = {}
            for p in T:
                kids[p[:-1]] = kids.get(p[:-1], 0) + 1
            chain = {tuple(int(np.nonzero(CAND[b, i] == PICK[b, i])[0][0]) for i in range(j + 1)) for j in range(DEP)}
            SHAPE[b] = depth, sum(v >= 2 for v in kids.values()), len(T - chain)
            put("tree 16 exact-capped", b, min(walk(T, Jr, nt) + 1, cap, XCAP[b]), rr)
            put("chain 16 exact-capped", b, min(kc + 1, cap, XCAP[b]), rr)
        Tm = kbest(firstm, ords, srs, rows=R)
        put(f"medusa {R}", b, min(walk(Tm, Js, nt) + 1, cap), rr)
    cp = np.cumprod(SPROB[b, :, 0])
    for t in TS:
        n = int(np.sum(cp >= t)); put(f"chain t {t}", b, min(min(kc, n) + 1, cap), min(n + 1, cap))
    for tau in TAUS:
        T = kbest(first1, ordr, srt, tau=tau)
        put(f"tree tau {tau}", b, min(walk(T, Jr, nt) + 1, cap), min(len(T) + 1, cap))
    put("oracle 16", b, min(int(np.argmax(np.append(Jr[:nt] >= K, True))) + 1, cap), min(16, cap))

# ---------------------------------------------------------------- per group
REF = {}
try:                                                                  # R-61N's exact tokens per pass (MTP Head, same replay)
    for m in re.finditer(r"== (.+?): \d+ blocks from \d+ answers; exact tokens per pass ([\d.]+)",
                         open(os.path.expanduser("~/TrainLab/data/r61n/chatnt_b16.txt")).read()):
        REF[m.group(1)] = float(m.group(2))
except OSError:
    pass
REF["code"] = 9.54                                                    # Leap's L1-20 verdict (cand21, selector, GPU)
GROUPS = [("code", ("tf_code_HumanEval",)), ("chat (ifeval + multi)", ("tf_ifeval", "tf_multi")), ("ifeval", ("tf_ifeval",)),
          ("multi", ("tf_multi",)), ("agent", ("agent",)), ("mgsm", ("mgsm",)), ("gsm (reference)", ("tf_gsm",))]


def agg(name, m):
    k, r = V[name][m, 0], V[name][m, 1]
    return k.mean(), r.mean(), sum(gb(x) for x in r) / k.sum()


say(f"TREES price: {B} blocks from {', '.join(os.path.basename(f) for f in FILES)}; DFlash2 L1-20 along the selector's trajectory; "
    f"GB per check from BigMoE's curve (4-bit, C 160); rows = the anchor + drafted nodes")
VER = {}
for gname, doms in GROUPS:
    m = np.isin(DOM, doms)
    if not m.any():
        continue
    ck = agg("chain 16", m)[0]
    say(f"\n== {gname}: {m.sum()} blocks from {len(set(SAMPLE[m]))} answers; chain at 16 rows {ck:.2f} kept per check"
        + (f" (reference run: {REF[gname]:.2f})" if gname in REF else "")
        + f"; true token in the top-16 at depth 1 / 4 / 8: "
        + " / ".join(f"{np.mean([np.isin(TRUE[b, d], CAND[b, d]) for b in np.nonzero(m & (NT > d))[0]]):.2f}" for d in (0, 3, 7)))
    say("    rows              " + "".join(f"{R:>8d}" for R in ROWS))
    for v in ("chain", "tree", "medusa"):
        a = [agg(f"{v} {R}", m) for R in ROWS]
        say(f"    {v:6s} kept        " + "".join(f"{x[0]:8.2f}" for x in a))
        say(f"    {v:6s} GB per kept " + "".join(f"{x[2]:8.3f}" for x in a))
    g = {R: agg(f"tree {R}", m)[0] / agg(f"chain {R}", m)[0] - 1 for R in ROWS}
    say("    tree vs chain     " + "".join(f"{100 * g[R]:+7.1f}%" for R in ROWS))
    say("    truncated chains (running pick prob >= t): " + "; ".join(
        f"t {t}: {a[0]:.2f} kept at {a[1]:.2f} rows, {a[2]:.3f} GB/kept" for t in TS for a in [agg(f"chain t {t}", m)]))
    say("    dynamic trees (path prob >= tau, <= 16 rows): " + "; ".join(
        f"tau {t}: {a[0]:.2f} at {a[1]:.2f} rows, {a[2]:.3f}" for t in TAUS for a in [agg(f"tree tau {t}", m)]))
    o, xc, xt = agg("oracle 16", m), agg("chain 16 exact-capped", m), agg("tree 16 exact-capped", m)
    say(f"    oracle top-16 at 16 rows {o[0]:.2f} kept; exact-capped (answer != the capture's top-1 ends the block): chain 16 "
        f"{xc[0]:.2f}, tree 16 {xt[0]:.2f} ({100 * (xt[0] / xc[0] - 1):+.1f} %)")
    sh = SHAPE[m].mean(0)
    say(f"    tree at 16 rows: deepest node at depth {sh[0]:.1f} (the chain's 15), {sh[1]:.1f} branch points, {sh[2]:.1f} of 15 "
        f"nodes off the chain")
    best = lambda pre: min((agg(n, m)[2], n) for n in V if n.split(" ")[0] in pre and "capped" not in n and "oracle" not in n)
    bc, bt, bm = best(("chain",)), best(("tree",)), best(("medusa",))
    gm = agg("medusa 16", m)[0] / ck - 1
    verdict = lambda g16, bx: "PASS" if g16 >= 0.10 and bx[0] <= bc[0] else "KILL" if g16 < 0.05 else "BETWEEN"
    VER[gname] = (verdict(g[16], bt), verdict(gm, bm))
    say(f"    best GB per kept token: chain {bc[0]:.3f} ({bc[1]}), tree {bt[0]:.3f} ({bt[1]}, {100 * (bt[0] / bc[0] - 1):+.1f} %), "
        f"medusa {bm[0]:.3f} ({bm[1]})")
    say(f"    VERDICT at equal rows 16 (PASS >= +10 % kept at equal bytes per kept token, KILL < +5 %): first-order tree "
        f"{100 * g[16]:+.1f} % -> {VER[gname][0]}; pre-registered medusa ranking {100 * gm:+.1f} % -> {VER[gname][1]}")
say("\nSUMMARY " + "; ".join(f"{k}: tree {a} / medusa {b}" for k, (a, b) in VER.items()))
open(f"{D}/price.txt", "w").write("\n".join(lines) + "\n")
