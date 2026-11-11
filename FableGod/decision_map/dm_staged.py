"""Staged verification on the decision map (Breakthrough, Tue 2026-11-03; the BREAKTHROUGH PUSH, the orchestrator 17:0x;
pre-registered SCOREBOARD 54036cf). A cheap pass picks one path out of the drafter's top-3 tree and ONE exact pass checks
only that path: exact by construction (it keeps the path's leading drafts equal to its own argmax, + its bonus).
Per HPD-0 16-row check (anchor A; draft j predicts ids[A + j + 1]): truth = the window's tokens; drafter = DFlash2 L1-20's
16 candidates + teacher-forced selector softmax per depth (TL_WALK_TOP16 tables, d02b_walk_kw_<dom>.npz); a true-path
node's children = its top-3 by that softmax. Along the true path everything is teacher-forced data: the chosen path
follows the truth to depth d iff at every depth < d the truth is a child in the tree and the cheap pass (the arm's top-8
at the node's map row) ranks it first among the children; what the path does off the truth never changes the kept count.
Trees: lat = the width-3 lattice to depth 15; tN = static best-first shapes of N nodes from the truth's per-depth rank
frequencies (leave-one-window-out per domain); chain = 0^15. Self-checks: S1 the chain tree = the walk's k for every arm;
S2 a perfect picker gives staged = oracle on every tree; S3 the table's top-1 chain = the walk's k at every check.
Usage (BigMoE venv): python dm_staged.py [arm ...] -> out/STAGED_read.txt"""
import heapq, json, os, sys
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
OUT = os.path.expanduser(os.environ.get("DM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")))
W, DEPTH, CUT = 3, 15, 0.3
PRIMARY = "p1_oea_k1"
ARMS = sys.argv[1:] or [PRIMARY, "t3_top2_all", "topk2", "p1_top_k2", "t3_all", "oea_k6", "oea_k8", "p1_oea_k1_ku4_t3",
                        "top_k8", "res_c256", "baseline"]
TREES = ("chain", "t32", "t64", "t128", "lat")
ROWS = {"chain": "16", "t32": "33", "t64": "65", "t128": "129", "lat": "all"}
base = json.load(open(os.path.join(OUT, "baseline.json")))
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def load_dom(dom):
    """the domain's checks (window, A, k, m, ranked candidates [m, 16], the truth's rank per depth [m] (16 = outside),
    the truth [m], the drafter's own cf [m]); None without a drafter table or when S3 fails"""
    p = os.path.join(TD, f"d02b_walk_kw_{dom}.npz")
    if not os.path.exists(p): return None
    K, H = np.load(p), np.load(os.path.join(TD, f"d02b_walk_hpd0_{dom}.npz"))
    kn, hn = [str(x) for x in K["names"]], [str(x) for x in H["names"]]
    chk, bad = [], 0
    for nm, b in base.items():
        if b["dom"] != dom: continue
        i, h = kn.index(nm), hn.index(nm); ids = K[f"{i}_ids"]
        assert np.array_equal(ids, H[f"{h}_ids"]) and np.array_equal(K[f"{i}_checks"], H[f"{h}_checks"]), nm
        a, n = b["a"], len(b["tk"])
        for c, (k, m, A) in enumerate(K[f"{i}_checks"].tolist()):
            o = np.argsort(-K[f"{i}_tsm"][c, :m].astype(np.float32), axis=1, kind="stable")
            rk = np.take_along_axis(K[f"{i}_cand"][c, :m], o, axis=1); tru = ids[A + 1:A + 1 + m]
            hit = rk == tru[:, None]; tr = np.where(hit.any(1), hit.argmax(1), 16)
            bad += int((int(np.argmax(tr != 0)) if (tr != 0).any() else m) != k)
            if A < a or A + m - 1 > a + n - 1: continue          # rows the map did not measure (each window's first check)
            chk.append((nm, A, k, m, rk, tr, tru, K[f"{i}_dcf"][c, :m].astype(np.float64)))
    if bad: say(f"S3 FAILED on {dom}: the table's top-1 chain misses the walk's k at {bad} checks; {dom} not read"); return None
    return chk

def shape(chk, n):
    """best-first static tree of n nodes (rank tuples) from the truth's per-depth rank frequencies on these checks"""
    reach, cnt = np.zeros(DEPTH), np.zeros((DEPTH, W))
    for *_, m, rk, tr, tru, dcf in chk:
        for j in range(m):
            reach[j] += 1
            if tr[j] >= W: break
            cnt[j, tr[j]] += 1
    P = cnt / np.maximum(reach, 1)[:, None]
    T, hp = set(), [(-P[0, r], (r,)) for r in range(W)]
    heapq.heapify(hp)
    while hp and len(T) < n:
        v, t = heapq.heappop(hp)
        if -v <= 0: break
        T.add(t)
        if len(t) < DEPTH:
            for r in range(W): heapq.heappush(hp, (v * P[len(t), r], t + (r,)))
    return T

def kids(tree, pre):
    if tree == "lat": return list(range(W))
    if tree == "chain": return [0] if all(r == 0 for r in pre) else []
    return [r for r in range(W) if pre + (r,) in tree]

def walk(tree, rk, tr, tru, m, clp, top):
    """one check -> (oracle depth, staged depth, unresolved picks, picks, tail depth, cut kept depth, cut rows lo, hi).
    clp(j, tokens) = the cheap logprobs of tokens at the depth-j node's row (-inf outside its top-8); top(j) = its argmax"""
    do = ds = L = None; unres = picks = 0; pre = (); cum = 1.0
    for j in range(m):
        ch = kids(tree, pre)
        if do is None and tr[j] not in ch: do = j
        if ds is None:
            if not ch: ds = j
            else:
                lp = clp(j, rk[j, ch]); picks += 1
                if np.all(np.isneginf(lp)): unres += 1; i = 0
                else: i = int(np.argmax(lp))
                cum *= float(np.exp(lp[i]))                    # exp(-inf) = 0: a pick outside the cheap top-8 cuts
                if L is None and cum < CUT: L = j              # the cheap cut: drafts 0 .. j - 1 go to the exact pass
                if ch[i] != tr[j]: ds = j
        if do is not None and ds is not None: break
        pre = pre + (int(tr[j]),)
    do = m if do is None else do; ds = m if ds is None else ds
    dt, pre = 0, ()                                            # tail: follow the cheap argmax while it is a child; when
    for j in range(m):                                         # it is not, it is one more draft and the path ends
        if top(j) != tru[j]: break
        dt = j + 1
        if tr[j] not in kids(tree, pre): break
        pre = pre + (int(tr[j]),)
    if L is not None: return do, ds, unres, picks, dt, min(ds, L), L + 1, L + 1
    return do, ds, unres, picks, dt, ds, min(ds + 1, m) + 1, m + 1

def cheap(arm):
    """per window -> (clp, top): the arm's top-8 at map row A + j; 'truth' = the perfect picker (S2)"""
    R = None if arm == "truth" else json.load(open(os.path.join(OUT, arm + ".json")))
    def mk(nm, A, tru):
        if R is None:
            return (lambda j, toks: np.where(toks == tru[j], 0.0, -np.inf)), (lambda j: int(tru[j]))
        a = base[nm]["a"]; tk, lp = np.array(R[nm]["tk"]), np.array(R[nm]["lp"])
        def clp(j, toks):
            r = A + j - a
            if r >= len(tk): return np.full(len(toks), -np.inf)
            d = dict(zip(tk[r].tolist(), lp[r].tolist()))
            return np.array([d.get(int(x), -np.inf) for x in toks])
        return clp, (lambda j: int(tk[A + j - a, 0]) if A + j - a < len(tk) else -1)
    return mk

def run(got):
    res = {}
    for dom, chk in got.items():
        shp = {nm: {t: shape([c for c in chk if c[0] != nm], int(t[1:])) for t in TREES if t.startswith("t")}
               for nm in sorted({c[0] for c in chk})}
        for arm in ["truth"] + ARMS:
            if arm != "truth" and not os.path.exists(os.path.join(OUT, arm + ".json")): continue
            mk = cheap(arm)
            for tree in TREES:
                acc = np.zeros(11)
                for nm, A, k, m, rk, tr, tru, dcf in chk:
                    do, ds, un, pk, dt, ck, lo, hi = walk(shp[nm][tree] if tree.startswith("t") else tree, rk, tr, tru, m,
                                                          *mk(nm, A, tru))
                    acc += [1, k + 1, do + 1, ds + 1, int(ds == do), un, pk, dt + 1, ck + 1, lo, hi]
                    if tree == "chain": assert do == k and ds == k, f"S1 failed: {dom} {arm} {nm} A {A}"
                    if arm == "truth": assert ds == do, f"S2 failed: {dom} {nm} A {A} {tree}"
                res[(dom, arm, tree)] = acc
        Lc = [int((np.cumprod(c[7]) >= CUT).sum()) for c in chk]   # the chain with DFlash2's live cut (its own cf, t 0.3)
        res[(dom, "livecut")] = np.array([sum(min(c[2], l) + 1 for c, l in zip(chk, Lc)), sum(l + 1 for l in Lc), len(chk)])
    return res

def row(pool, dl, arm, tree, res, V):
    X = [res[(d, arm, tree)] for d in dl if (d, arm, tree) in res]
    if len(X) < len(dl): return
    a = np.sum(X, axis=0); n = a[0]; ch, orc, st = a[1] / n, a[2] / n, a[3] / n
    fid = (st - ch) / (orc - ch) if orc > ch else float("nan")
    V[(pool, arm, tree)] = (st / ch, orc / ch, fid, a[4] / n)
    say(f"| {pool} | {arm} | {tree} | {ROWS[tree]} | {ch:.2f} | {orc:.2f} | {st:.2f} | x{st / ch:.3f} | x{orc / ch:.3f} | "
        f"{fid:.2f} | {a[4] / n:.3f} | {int(a[5])}/{int(a[6])} | {a[7] / n:.2f} | {a[8] / n:.2f} at {a[9] / n:.1f}-{a[10] / n:.1f} |")

if __name__ == "__main__":
    doms = {d: load_dom(d) for d in ("code", "ifeval", "multi")}
    got = {d: c for d, c in doms.items() if c}
    say("STAGED VERIFICATION replay (pre-registered SCOREBOARD 54036cf): HPD-0's 16-row checks, truth = the window, the "
        f"drafter's top-{W} tree, the cheap pass = a map arm's top-8; kept per exact pass = one pass on the chosen path")
    for d, c in doms.items():
        say(f"  {d}: " + (f"{len(c)} checks on {len({x[0] for x in c})} windows (S3 ok)" if c else "not read (no drafter table)"))
    res = run(got)
    say("  S1 (the chain tree = the walk's k, every arm) ok; S2 (a perfect picker: staged = oracle, every tree) ok")
    pools = [(p, [d for d in dl if d in got]) for p, dl in (("code", ["code"]), ("chat", ["ifeval", "multi"]),
                                                             ("ifeval", ["ifeval"]), ("multi", ["multi"]))]
    pools = [(p, dl) for p, dl in pools if dl and (p != "chat" or len(dl) == 2)]
    head = ("| pool | cheap arm | tree | cheap rows | chain | oracle | staged | staged/chain | oracle/chain | fidelity | M2 | "
            "unresolved picks | + tail | cheap cut: kept at exact rows (lo-hi) |")
    V = {}
    say(f"\nThe primary arm ({PRIMARY}) on every tree (kept per check; one exact pass of <= 16 rows on the chosen path)")
    say(head); say("|" + "---|" * 14)
    for p, dl in pools:
        for tree in TREES: row(p, dl, PRIMARY, tree, res, V)
    say("\nEvery arm on T_64 and the lattice")
    say(head); say("|" + "---|" * 14)
    for p, dl in pools:
        for arm in ARMS[1:]:
            for tree in ("t64", "lat"): row(p, dl, arm, tree, res, V)
    say("\nThe shipped chain with DFlash2's live cut (t 0.3 on its own cf), same checks")
    for p, dl in pools:
        kc, rc, nc = np.sum([res[(d, "livecut")] for d in dl], axis=0)
        say(f"  {p}: {kc / nc:.2f} kept at {rc / nc:.1f} exact rows per check = {kc / rc:.3f} kept per exact row "
            f"(the 16-row chain: {np.sum([res[(d, PRIMARY, 'chain')][1] for d in dl if (d, PRIMARY, 'chain') in res]) / nc / 16:.3f})")
    p = (PRIMARY, "t64")
    if ("code",) + p in V and ("chat",) + p in V:
        c, h = V[("code",) + p], V[("chat",) + p]
        if (h[1] < 1.15 and c[1] < 1.05) or (h[2] < 0.5 and c[2] < 0.5): v = "KILL"
        elif h[0] >= 1.25 and h[2] >= 0.75 and c[0] >= 1.05: v = "PASS"
        else: v = "BETWEEN"
        say(f"\nVerdict ({PRIMARY}, T_64): chat staged/chain x{h[0]:.3f}, fidelity {h[2]:.2f}, oracle x{h[1]:.3f}; code staged "
            f"x{c[0]:.3f}, oracle x{c[1]:.3f}, fidelity {c[2]:.2f} -> {v}")
    else:
        say(f"\nVerdict pending: the lines need code and chat ({', '.join(d for d in doms if d not in got)} not read yet)")
    open(os.path.join(OUT, "STAGED_read.txt"), "w").write("\n".join(lines) + "\n")
